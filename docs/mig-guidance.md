# When to Use MIG (Multi-Instance GPU)

Practical guidance for researchers deciding whether to partition a GPU with MIG
or use the whole card.

---

## What MIG is

Multi-Instance GPU (MIG) is an NVIDIA hardware feature that partitions a single
GPU into multiple isolated instances. Each instance gets its own slice of:

- Streaming multiprocessors (compute)
- Memory (VRAM)
- Memory bandwidth
- L2 cache

Instances are hardware-isolated — one instance's crash, OOM, or runaway job
cannot affect another. This is stronger than CUDA MPS (which shares context
but gives no resource isolation) or time-slicing.

**MIG-capable GPUs:**

- Datacenter Ampere: A100, A30 (up to 7 instances)
- Datacenter Hopper: H100, H200 (up to 7 instances)
- Datacenter Blackwell: B100, B200
- Workstation Blackwell: **RTX PRO 6000 Blackwell Server Edition** (up to 4 instances — new in 2025)

**Not MIG-capable:** All consumer RTX cards (30xx, 40xx, 50xx including 5090),
prior-generation workstation cards (RTX 6000 Ada, A6000, etc.).

---

## Opt for MIG when

### 1. Running parallel experiments
Hyperparameter sweeps, ablations, seed variance studies. 4 runs on 4 MIG slices
finish faster than 4 serial runs on the full card, even though each slice is
slower. **Parallelism beats per-run speed** when you have a portfolio of jobs.

### 2. Mixed workloads during development
You want a long-running training on one slice while you iterate on eval scripts,
run quick smoke tests, or serve an inference endpoint on another slice. Without
MIG, one of those clobbers the other.

### 3. Multi-tenant environments
Shared lab GPU, student workstations, classroom clusters. Hardware isolation
means one user's OOM or runaway job can't kill another's run. CUDA MPS gives
you sharing but not isolation.

### 4. Inference serving
Multiple small models each fitting in a slice, each serving its own endpoint.
**Latency isolation matters:** a request spike on model A shouldn't starve
model B.

### 5. Debugging with a reserved baseline
Pin a known-good training run on one slice as a control while you experiment
on another. The control cannot be perturbed by your mistakes.

### 6. Fitting the job to the hardware
If your model only uses 20 GB, a full 96 GB card is waste. Slicing turns one
card into four productive workers.

---

## Don't use MIG when

### 1. Single throughput-bound training run
The full card wins every time. MIG cannot speed up one job — a job saturating
the full card (98% utilization, 99% power) running on a 1/4 slice takes ~4x
longer for no upside.

### 2. You need cross-GPU coordination
MIG instances cannot share gradients, KV caches, or distributed state. If you
were going to DDP or FSDP across partitions, you can't.

### 3. Your model needs more than 1/N of the memory
A 7B model in BF16 needs 14 GB for weights alone. A 1/4 slice of a 96 GB card
(24 GB) is tight once you add activations and optimizer state. Don't fight the
partition boundary.

### 4. Bandwidth-bound workloads at full-card bandwidth
Attention at long context, quantization-heavy inference, large-batch
throughput serving. Slicing costs you proportional memory bandwidth. If the
full card was already bandwidth-limited, the slice is 4x worse.

### 5. Job scheduling overhead exceeds parallel gain
MIG instances cannot be dynamically reconfigured on most platforms — you must
reboot the GPU to repartition. If your job sizes are unpredictable (mix of
small and full-card jobs through the day), the static partitioning hurts.

---

## Rule of thumb

> MIG optimizes for **throughput per wall-clock hour across a portfolio of
> jobs**, not **time to finish one job**.

- 4–8 experiments running simultaneously → MIG is a productivity multiplier
- 1 big committed training run to convergence → full card every time

---

## When to revisit the decision

Configuration is sticky (repartitioning requires GPU reset), so the decision
should match your typical week, not a single project. Heuristics:

- **Count your GPU seconds by usage pattern.** If most of your time is
  parallel small jobs, partition. If it's one long run followed by analysis,
  don't.
- **Check underutilization.** `nvidia-smi dmon` during your typical workload:
  if GPU util averages < 50%, you're leaving throughput on the table —
  MIG can recover it.
- **Count OOM / crash blast radius.** If one person's mistake takes down the
  whole team's work, isolation is worth the overhead.

---

## Example: RTX PRO 6000 Blackwell on AWS g7e

- 96 GB VRAM, up to 4 MIG instances of 24 GB each
- Good fit for researchers doing lots of small-to-medium experiments
- Effectively **4 × 24 GB workstations for the cost of managing one driver**
- Trade-off: individual runs are slower than the full card, but total
  throughput on a 4-way workload is higher

If you're running the small-reasoning-model 500M pre-training (one committed
run, saturates the card): use the whole g7e, not a slice.

If you're doing GRPO hyperparameter sweeps (4 configs × 3 seeds = 12 runs):
partition into 4 and run 4-at-a-time. 3 batches sequential is faster than
12 serial on the full card, because each run is throughput-bound but you
have idle capacity when running one at a time.
