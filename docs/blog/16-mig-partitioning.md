# When to Partition a GPU: MIG for Researchers

*Part 16 of a series on building a small reasoning language model end-to-end.*

---

A researcher asked me a good question last week: their AWS g7e instance has an
RTX PRO 6000 Blackwell that can be split into up to 4 partitions using NVIDIA's
Multi-Instance GPU (MIG) feature. When should they use it?

The short answer is: **almost never for training, often for research.** This
post unpacks why.

---

## What MIG Actually Does

MIG (Multi-Instance GPU) partitions a single physical GPU into multiple
hardware-isolated instances. Each instance gets its own slice of the streaming
multiprocessors, VRAM, memory bandwidth, and L2 cache. From the CUDA runtime's
perspective, each slice looks like a separate GPU — its own `cuda:0`, its own
memory space, its own scheduler.

Critically, the isolation is at the hardware level. An OOM on one instance
cannot crash another. A process that hangs a CUDA context on instance 0
doesn't affect instance 1. This is stronger than what you get from CUDA MPS
(which shares a single context with no resource guarantees) or process-level
time-slicing (which is just preemption with overhead).

The catch: partitioning is a product-segmentation feature. NVIDIA gates MIG to
datacenter cards and, as of 2025, exactly one workstation card:

- **Supported:** A100, A30, H100, H200, B100, B200, **RTX PRO 6000 Blackwell
  Server Edition** (new in 2025)
- **Not supported:** All consumer RTX cards — including the 5090 — and prior
  workstation cards like the A6000 and RTX 6000 Ada

The RTX PRO 6000 Blackwell is the interesting one. It's positioned as a
workstation card (96 GB VRAM, PCIe form factor, much cheaper than a full H100)
but inherits MIG from the Blackwell datacenter line. For researchers with
access to a g7e or similar, it's the first time MIG has been practical outside
a full datacenter tier.

---

## The Trap: Treating MIG as "Cheaper GPU"

The wrong mental model for MIG is: "a 1/4 slice is 1/4 of a GPU, so I use it
when I need 1/4 of a GPU."

That's wrong because a single training run that would saturate a full GPU now
runs 4x slower. You haven't saved any GPU-seconds. You've just taken longer.

I saw this firsthand while writing this post. Our 500M model pre-training run
on an RTX 5090 holds the card at 98% compute utilization and 99% of its 600W
power budget. Every streaming multiprocessor is busy every step. Every
watt is doing gradient work. A 1/4 MIG slice would give us 1/4 the SMs and
1/4 the memory bandwidth, turning a 4-day training into a 16-day training for
no benefit. The correct choice there is always the full card.

The right mental model for MIG is different: **MIG converts one underutilized
card into several fully-utilized partitions.**

The question isn't "is my job small?" It's "does one job of mine, running on a
full card, leave the GPU sitting at 30% utilization most of the time?" If yes,
MIG can recover that idle capacity by running other jobs alongside. If no, you
don't want MIG — the full card is the right size for your work.

---

## Six Places MIG Actually Helps

### 1. Parallel hyperparameter sweeps

You have 12 training runs to do: 4 learning rates × 3 random seeds. On a full
card, you run them serially, 12 runs deep. On a 4-way MIG partition, you run
4 at a time, 3 batches deep. Each individual run is slower, but the total
wall-clock time for the sweep is lower because you're exploiting your ability
to parallelize independent work.

This only works when the runs are actually independent. If your sweep has
sequential dependencies (one run's hyperparameters are chosen based on the
previous run's result), partitioning doesn't help — you can't run in parallel
what can't be parallelized.

### 2. Training plus development

The most common research workflow pain: you have a long training run going
(days long), and you want to write and debug the eval script that will run
against its checkpoints. On an un-partitioned GPU, your eval smoke tests have
to wait until training finishes — or you risk crashing the training run by
competing for memory.

On a 4-way partition, you give training 3 slices (75% of the card) and keep
one slice for iteration. You can run the eval script on a stale checkpoint,
iterate on reward computation, debug JSON extraction, and verify your
benchmark harness, all without touching the training run. When training
finishes, you're ready.

### 3. Multi-tenant environments

Shared lab or team GPUs. Each team member or project gets a slice. Someone's
OOM doesn't kill your training. Someone's runaway `while True` doesn't starve
you of compute. Hardware isolation means you stop needing a human
scheduling process.

This was always possible with a job scheduler (SLURM, LSF) on a cluster. What
MIG adds is making it work on a single workstation or a single cloud instance
without a scheduler. For small teams without dedicated cluster infrastructure,
this is huge.

### 4. Inference serving with latency guarantees

You're serving four different small models as separate endpoints. Without
MIG, they share the GPU — which means a spike in traffic on model A causes
latency spikes on model B. Tail latencies get unpredictable. With MIG, each
model lives on its own slice with its own compute and memory budget. One
endpoint's load cannot affect another's response time.

This matters for production SI pipelines: the reasoning model at 500M, a
code SI model at 7B, a verification model, and maybe a lightweight router
can all live on separate slices of one RTX PRO 6000, each with predictable
latency.

### 5. Debugging with a protected baseline

You have a training run that works. You want to try a change, but the change
might break something. On a shared full-card setup, you lose the baseline
when you start the experiment. On MIG, you pin the known-good run on one
slice and experiment on another. If the experiment fails, you still have the
baseline running — a crucial property when the baseline takes days to
produce.

### 6. Fitting jobs to available hardware

The RTX PRO 6000 Blackwell has 96 GB of VRAM. A 500M model training uses
about 9 GB. Running one training job on the full card means 87 GB of VRAM
is sitting idle. Partitioning into 4 × 24 GB slices turns that waste into
four productive workers.

The key number is what fraction of the card your typical job actually uses.
If it's less than 1/2, consider partitioning.

---

## Five Places MIG Makes Things Worse

### 1. Single throughput-bound training runs

We've covered this. If one job saturates the full card, a slice of the card
takes proportionally longer with no benefit. This is the default case for
serious pre-training or long SFT/GRPO runs.

### 2. Anything that needs to span partitions

MIG instances cannot share state with each other. No DDP, no FSDP, no shared
KV cache, no cross-partition communication beyond what you'd do over the
network. If you were planning to do model-parallel or data-parallel training
across partitions of a single card, you can't. Use a single instance or
multiple full GPUs.

### 3. Jobs that exceed the per-slice memory

A 7B model in BF16 weighs 14 GB. On a 1/4 slice of a 96 GB card (24 GB),
that leaves 10 GB for activations, optimizer state, and gradients — tight.
At longer context or larger batches you'll OOM. Either train on the full
card or use a 1/2 slice (48 GB).

This isn't a MIG bug; it's just that partition boundaries don't move to
accommodate your job.

### 4. Bandwidth-bound workloads

Attention at long sequence lengths, large quantization-heavy inference,
and high-batch inference can all be limited by HBM bandwidth rather than
compute. A 1/4 slice has 1/4 the bandwidth. If the full card was already
bandwidth-limited, the slice is 4x worse. Profile before you partition.

### 5. Unpredictable job sizes

MIG partitioning is sticky — on most platforms you must reset the GPU to
repartition. If your workflow is "sometimes one big job, sometimes four
small jobs," the static partition will hurt in one direction or the other.
Pick the configuration that matches your typical week, not your best or
worst week.

---

## A Rule of Thumb

> MIG optimizes for **throughput per wall-clock hour across a portfolio of
> jobs**, not **time to finish one job**.

If you're asking "will MIG make my training finish faster?" — no. A single
training run is always fastest on the full card.

If you're asking "will MIG make my week more productive?" — maybe. Count
how many experiments you run simultaneously on a typical day, and what
fraction of the card each uses. If the product is less than 1.0, MIG is
almost certainly worth the overhead.

---

## The RTX PRO 6000 Blackwell as a Research Workstation

Zoom out: what's unusual about the RTX PRO 6000 Blackwell isn't the raw
performance. It's that NVIDIA finally put MIG in a workstation-class card.

Before 2025, if you wanted MIG you paid for a datacenter card (H100 starts
around $25K). Now you can get it in a workstation card (RTX PRO 6000 Blackwell
starts around $8K), which means researchers with a lab budget rather than a
cluster allocation can have it.

For someone doing the kind of work this project does — multiple experiment
streams, mixed training-plus-inference workloads, serving several small models
at once — the practical capability is:

- **Without MIG:** one GPU that does one thing at a time
- **With MIG:** four 24 GB workstations with hardware isolation

That's a meaningful productivity multiplier if your work looks like
"several small models in motion at once" rather than "one big model training
to convergence."

---

## How to Decide

Three diagnostic questions to run through before partitioning:

1. **What does my GPU utilization look like during a typical workload?**
   Run `nvidia-smi dmon` for an hour of normal work. If utilization averages
   below 50%, MIG can recover throughput. If it's above 80%, you're already
   using the card efficiently — leave it alone.

2. **Do my jobs naturally come in batches of independent work?**
   Hyperparameter sweeps, multi-seed runs, ensemble serving. If yes, MIG
   parallelizes them cleanly. If your work is one-big-job-at-a-time, MIG
   doesn't help.

3. **Does OOM isolation matter to me?**
   Shared lab, teaching cluster, production inference? Hardware isolation is
   worth the overhead. Solo researcher with one card? Probably not.

If you answer "high utilization, single job, solo researcher" to all three,
don't partition. If you answer "low utilization, batched jobs, shared
environment" to all three, definitely partition. Most real cases are in
between, and the honest answer is: try it on a day of actual work and see
whether your productivity improves.

---

## The Small Reasoning Model Case

Where does our project land?

**Pre-training (Phase 0):** single 500M run, saturates the card. Full GPU
every time. We watched the RTX 5090 hold 98% utilization and 99% of its
600W power budget for days. A MIG slice would have been pure loss.

**SFT and GRPO (Phases 1–2):** similar — throughput-bound, saturates the
card. Full GPU.

**Evaluation and development:** this is where MIG would genuinely help.
Running `math_eval.py` and `si_eval.py` concurrently with a training run,
iterating on reward functions while training proceeds, serving a checkpoint
for interactive testing while the next one trains. On the 5090 we had to
serialize these. On an RTX PRO 6000 Blackwell, we could parallelize them.

**Multi-model serving:** the SI pipeline described in
[post 11](11-structured-intent.md) involves a reasoning model, an SI model,
and verification. Running these on separate MIG slices with latency
isolation would be the right production architecture — and a compelling
use case for researchers to adopt MIG even if their training workloads
don't benefit.

So the answer to the original question — "when should I use MIG on my g7e?"
— is: not for the committed training runs, but absolutely for the evaluation,
serving, and experimentation that happens around them. The researcher's
week has more variety than the researcher's headline workload, and MIG
optimizes for that variety.

---

*Detailed criteria with decision flowchart in
[`docs/mig-guidance.md`](../mig-guidance.md).*
