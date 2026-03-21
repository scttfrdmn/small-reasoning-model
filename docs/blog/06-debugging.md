# Two Deadlocks and a GPU at 98%: Debugging the Training Infrastructure

*Part 6 of a series on building a small reasoning language model end-to-end.*

---

After three weeks of building, we had a working pre-training loop: 500M parameters,
BF16, gradient checkpointing, cosine LR schedule. The data pipeline was producing
39GB of tokenized text. The tokenizer was trained. Everything validated.

We launched training on the RTX 5090 and watched the GPU hit 98% utilization.

For two hours, nothing was logged. The step counter never moved.

This post is about what was happening, why it was hard to diagnose, and how we
fixed it. The root cause turned out to be two separate deadlocks interacting,
each in a different layer of the stack.

---

## What the Symptoms Looked Like

```
step      loss    lr        tok/s        tokens
────────────────────────────────────────────────────────────
```

The table header printed. Then nothing. No error, no hang warning, no OOM.
`nvidia-smi` showed the GPU at 97–98% utilization. CPU was at 100% on one core.
Memory usage was stable. The process was alive.

The misleading part: *the GPU was actually doing work*. It was running forward
passes and backward passes. The CUDA kernel queue was active. But no Python
code was executing — the main thread was completely blocked. The training loop
never reached the `print()` call that logs each step.

We let it run for two hours before killing it.

---

## The First Deadlock: DataLoader + IterableDataset + CUDA

### The configuration

Our training data was a streaming JSONL file tokenized on-the-fly. The DataLoader
was configured as:

```python
DataLoader(
    dataset,           # IterableDataset
    batch_size=4,
    num_workers=2,     # fork 2 worker processes
    pin_memory=True,   # copy tensors to pinned memory for fast GPU transfer
    prefetch_factor=4, # each worker buffers 4 batches ahead
)
```

This is a completely normal DataLoader configuration. It's what you'd find in
any PyTorch tutorial.

### What goes wrong

When `num_workers > 0`, PyTorch forks worker processes to load data in the
background. The fork happens *after* the main process has already initialized
CUDA and allocated GPU memory.

This creates a known but poorly-documented problem: **forking a process that
has already initialized CUDA is undefined behavior in CUDA's memory model**.
The forked worker inherits file descriptors and memory mappings but not the
CUDA context state. If the worker tries to do anything CUDA-related, bad things
happen.

`pin_memory=True` adds a second hazard. The pin-memory thread lives in the main
process and runs a tight loop: wait for tensors from worker queues, pin them to
page-locked memory, put them in the output queue. This thread uses its own
synchronization primitives — futexes — to coordinate with the DataLoader queue.

With `IterableDataset`, there's a third problem: the dataset doesn't know how
many items it has. Workers can't be pre-assigned independent shards (unlike
map-style datasets). PyTorch implements IterableDataset workers with a shared
iterator, protected by a lock.

The combination: CUDA initialization after fork + pin_memory thread futex +
shared IterableDataset lock + worker-process synchronization = futex deadlock.

> **Sidebar: What is a Futex?**
>
> A "futex" (fast userspace mutex) is a Linux synchronization primitive. It's
> the implementation substrate for Python's threading locks, multiprocessing
> semaphores, and most inter-thread/inter-process coordination.
>
> A futex has two states: locked and unlocked. `futex_wait` puts the calling
> thread to sleep until the futex is unlocked. `futex_wake` wakes one or more
> sleeping threads.
>
> In a deadlock, two threads each hold a lock the other is waiting for, so
> both sleep forever: A is `futex_wait`-ing on a lock held by B, while B is
> `futex_wait`-ing on a lock held by A. The process shows 100% CPU in the kernel
> (it's constantly being woken, checking state, sleeping again) but makes no
> progress.
>
> You can see this in `strace -p <pid>`: `futex(addr, FUTEX_WAIT, ...)` repeating
> indefinitely.

### The diagnosis

We used `py-spy` to sample the main thread's stack:

```
Thread 1 (main thread)
  futex_do_wait (syscall)
  _PyThread_acquire_lock
  PyObject_RichCompare
  ... DataLoader internal queue get ...
```

The main thread was stuck waiting for the DataLoader to produce a batch. The
DataLoader worker processes appeared to be running (they had CPU time) but the
pin_memory thread was blocked trying to acquire a lock held by a worker.

### The fix

The fix is anticlimactic: don't use background workers for IterableDataset.

```python
DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,      # load in main process
    pin_memory=False,   # can't pin in main thread without the manager
    prefetch_factor=None,
)
```

With `num_workers=0`, there's no forking, no pin_memory thread, no cross-process
lock contention. Data loading runs synchronously in the main process thread.

The throughput cost is real but modest: we lose the prefetch pipeline that hid
data loading latency. For pre-tokenized binary data read from NVMe (which we
switched to later), the latency is small enough that `num_workers=0` is fine.

We added a detailed comment in the code explaining the failure mode so that
future readers don't silently undo this fix:

```python
# num_workers=0: run data loading in the main process thread.
#
# Why not num_workers > 0 with IterableDataset:
#   PyTorch's DataLoader forks worker processes AFTER the model is already
#   on CUDA.  When pin_memory=True is combined with num_workers > 0 and an
#   IterableDataset, the pin-memory thread and worker processes end up in a
#   futex deadlock — the main process waits on a lock that a worker holds,
#   while the worker waits on pin_memory queue operations.  The bug manifests
#   as 100% CPU + 98% GPU utilisation with zero logged steps.
```

---

## The Second Deadlock: HuggingFace Tokenizers + CUDA

We fixed the DataLoader configuration and relaunched. The same symptoms appeared.

Identical step counter. GPU at 98%. CPU at 100%. Nothing logged.

This was the same surface presentation but a different root cause.

### The configuration

Our dataset was tokenizing text on-the-fly using the HuggingFace `tokenizers`
library:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer_output/tokenizer.json")

# In the dataset's __iter__:
for text in corpus:
    tokens = tokenizer.encode(text).ids  # called in main thread
    yield tokens
```

With `num_workers=0`, this tokenization runs in the main thread, interleaved
with the forward/backward pass.

### What goes wrong

The HuggingFace `tokenizers` library is written in Rust and uses the `rayon`
library for data parallelism. Rayon creates a thread pool at first use.
Rayon's thread pool uses futex-based work-stealing synchronization.

PyTorch's CUDA runtime also maintains background threads for CUDA stream
management, memory management, and event tracking. These threads also use
futex-based synchronization.

When `tokenizer.encode()` is called from the Python main thread while CUDA
background threads are active, both thread pools are contending on futexes.
Under the right race condition — which, on the RTX 5090 with the specific
workload, triggered almost immediately — the Rust rayon thread pool and the
CUDA runtime threads enter a cross-system futex deadlock.

This is not documented anywhere we could find.

> **Sidebar: The Python GIL and Why It Doesn't Help Here**
>
> Python's Global Interpreter Lock (GIL) ensures that only one Python thread
> runs Python bytecode at a time. This prevents data races in pure Python code.
>
> But the GIL doesn't protect C/Rust extension code. When Python calls into the
> `tokenizers` Rust library, the GIL is *released* for the duration of the call
> (this is standard practice for extension modules that do significant work, to
> avoid blocking other Python threads). The Rust rayon thread pool runs without
> the GIL.
>
> Similarly, PyTorch's CUDA runtime threads are C++ threads that don't hold the
> GIL. They use their own synchronization (CUDA events, CUDA streams, internal
> mutexes).
>
> The GIL only serializes Python code. When you have Rust futexes contending with
> CUDA C++ futexes, the GIL is irrelevant. The deadlock lives entirely in the
> C/Rust/CUDA layer.

### The diagnosis

The strace output looked superficially identical to the first deadlock. But with
the DataLoader fix in place, the only place `tokenizer.encode()` could be called
was the main thread's `next(iter(dataset))` call.

We added `TOKENIZERS_PARALLELISM=false` as an environment variable:

```bash
TOKENIZERS_PARALLELISM=false python training/pretrain.py ...
```

This environment variable tells the HuggingFace tokenizers library to disable
rayon's thread pool and tokenize single-threaded. It's a documented mitigation
for tokenizer + fork interactions (they warn about it in their docs), but the
interaction with CUDA threads is not documented.

With `TOKENIZERS_PARALLELISM=false`, the rayon thread pool was never created,
and the deadlock disappeared. Training ran normally.

### The permanent fix

`TOKENIZERS_PARALLELISM=false` is a workaround. The correct fix is to eliminate
on-the-fly tokenization entirely.

We wrote `data/tokenize_dataset.py` — a pre-processing script that tokenizes
the entire 39GB corpus to flat uint16 binary files:

```python
# Pre-tokenize: JSONL → flat uint16 .bin file
tokens = tokenizer.encode(text).ids   # done ONCE, offline
arr = np.array(tokens, dtype=np.uint16)
arr.tofile(output_file)               # sequential write, no copy
```

The training-time dataset reads directly from the binary file with `np.memmap`:

```python
class BinaryTokenDataset(IterableDataset):
    def __init__(self, data_dir, max_seq_len, split="train"):
        bin_path = Path(data_dir) / f"{split}.bin"
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
```

With pre-tokenization, `tokenizer.encode()` is never called during training.
There's no Rust rayon thread pool. There's no interaction with CUDA threads.
The deadlock cannot occur.

> **Sidebar: Why np.memmap Instead of Loading the File?**
>
> `np.memmap` maps a file into the process's virtual address space without
> actually loading it into RAM. When you access a page of the file (a 4KB
> region of the array), the OS loads it from disk on demand.
>
> For our 9.5B-token training file (19.7GB), loading the full array into RAM
> would require 19.7GB of available memory. With memmap, the OS manages which
> pages stay in RAM based on access patterns. The "active" 4MB of tokens being
> streamed stays in RAM; the rest stays on disk.
>
> This is why we could train on ceres (128GB RAM, 1.5TB NVMe) without the 20GB
> binary file consuming most of memory. The OS page cache handles caching
> naturally; we don't need to manage it explicitly.

---

## The Third Bug: A Logging Issue

This wasn't a deadlock but a different kind of invisible failure: the log file
was always 0 bytes.

We launched training with:

```bash
python training/pretrain.py --config 500m ... 2>&1 | tee logs/pretrain.log
```

The `tee` command is supposed to write to stdout and the log file simultaneously.
But the log file stayed at 0 bytes for the first ~100 seconds, then suddenly
had 5MB of content.

The cause: Python's stdout is block-buffered when piped to a non-terminal.
The default buffer is 8KB. When Python writes `print("step 1, loss 3.24")`,
the bytes go into an 8KB internal buffer. Nothing is written to the pipe (and
thus to `tee`'s output file) until the buffer fills or an explicit flush happens.

On training step 0, we do `print(..., flush=True)`. That flush empties the
buffer and produces the first output. Until then: silence.

The fix is one line in the launch script:

```bash
PYTHONUNBUFFERED=1 python training/pretrain.py ... | tee logs/pretrain.log
```

`PYTHONUNBUFFERED=1` disables Python's output buffering entirely: every
`print()` writes to the pipe immediately. Log files now update in real time.

---

## What We Changed

After these three fixes, the training loop ran cleanly:
- `num_workers=0, pin_memory=False` in the DataLoader
- Pre-tokenized binary data (no on-the-fly tokenization)
- `PYTHONUNBUFFERED=1` in the launch script

The GPU was still at ~98% utilization. But now step 1 appeared after 12 seconds,
step 10 after 2 minutes, and the training run proceeded normally for 43 hours to
completion.

---

## Lessons

**1. GPU utilization is not throughput.**
98% GPU utilization tells you the GPU is computing. It doesn't tell you whether
those computations are producing training progress. Always instrument the step
counter, not just GPU stats.

**2. Deadlocks between language runtimes are almost never documented.**
The Rust/rayon + CUDA futex interaction is not in the PyTorch docs, not in the
HuggingFace tokenizers docs, and not easily findable. The symptom (main thread
blocking) looks identical to DataLoader deadlocks. Systematic elimination
(fix one possible cause at a time, retest) is the only reliable approach.

**3. Pre-tokenization is just better.**
On-the-fly tokenization makes training pipelines more convenient but introduces
a language-runtime boundary (Python calling Rust calling C++) in a hot loop on
the main training thread. Pre-tokenization moves that boundary offline, where
deadlocks are recoverable (just re-run) rather than catastrophic.

**4. PYTHONUNBUFFERED=1 should be in every ML training script.**
This should be a default. The number of people who have stared at an empty
log file wondering if their training script started is larger than it needs to be.

---

*Next: [Part 7 — Phase 1 SFT: Loss Masking and Teaching a Model to Think](07-sft.md)*
