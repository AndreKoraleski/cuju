# Backends

The **backends** of cuju are the two compiled libraries that run its environments, a CUDA backend
for NVIDIA GPUs and a CPU backend that distributes environments over a pool of host threads. Both
are compiled from one header-only C++ source and execute the same floating-point operations in the
same order, so they produce bitwise identical outputs, which also equal those of Google Research
Football, or GRF.

The backend of a batch is chosen at creation through the `device` option. On the GPU each
environment occupies one warp, and a step is a single kernel launch on the current PyTorch stream,
which CUDA graphs and `torch.compile` can capture. On the CPU a thread pool steps the environments,
with options for asynchronous steps, thread pinning, NUMA placement and page-locked outputs. On
both backends the outputs are PyTorch tensors that view the buffers of the batch. A run spans
several GPUs either with one process per GPU or with `cuju.MultiDeviceBatch` in a single process.

## Architecture

The simulation, comprising the physics, the animations, the built-in AI, the referee, the
observations and the rewards, is written once in header-only C++. It is compiled twice, by nvcc
into `libcuju_cuda.so` for NVIDIA GPUs and by a host C++ compiler into `libcuju_cpu.so`, which
are `cuju_cuda.dll` and `cuju_cpu.dll` on Windows and `libcuju_cpu.dylib` on macOS. Both libraries
export the same C API, and the Python package loads the one that `device` names.

Because both backends execute the same operations in the same order, the outputs of a batch on the
GPU equal its outputs on the CPU bit for bit, and both equal the outputs of GRF, as described in
[Exactness](exactness.md). Work developed on a CPU therefore carries over unchanged to training on
a GPU, and a result obtained on one backend can be reproduced on the other.

The two backends differ in library, hardware and execution model.

| | `cuda` | `cpu` |
| --- | --- | --- |
| Library | `libcuju_cuda.so`, `cuju_cuda.dll` | `libcuju_cpu.so`, `cuju_cpu.dll`, `libcuju_cpu.dylib` |
| Platforms | Linux on x86-64, Windows, WSL2 | Linux on x86-64 and on Arm, Windows, macOS, WSL2 |
| Hardware | NVIDIA GPU, compute capability 7.5 or higher | x86-64 processor with AVX2, BMI2 and FMA, or any 64-bit Arm processor |
| One environment runs on | one warp of 32 threads | one thread at a time |
| One step is | one kernel launch | one pass of the thread pool |
| Outputs live in | device memory, as CUDA tensors | host memory, as CPU tensors |

Every platform gives the same bits on each backend, as the shared digests of the self-check confirm,
and [Installation](installation.md) lists the platforms with their wheels and requirements.

## Device selection

The `device` option accepts a backend name, a CUDA device or `'auto'`, and each value places the
batch on one device.

| Value | Runs on |
| --- | --- |
| `'cuda'` | the current CUDA device of PyTorch |
| `'cuda:N'`, `torch.device('cuda', N)`, or an integer `N` | GPU `N` |
| `'cpu'` | the CPU |
| `'auto'` | the current CUDA device where PyTorch sees a GPU and the CUDA library is installed, else the CPU |

The default is `'cuda'`. The value `'auto'` selects once, at creation, and suits code that runs on
machines with and without a GPU, such as quick starts and notebooks. Only an explicit `'auto'`
selects, so a program that names `'cuda'` on a machine without a GPU raises `DeviceError` at
creation and reports the problem at once. The attribute `batch.device` records the chosen backend,
`'cuda'` or `'cpu'`, and `batch.torch_device` the `torch.device` of the buffers, such as `cuda:0` or
`cpu`. Every call of a batch runs on its own device, independent of the device that is current at
the time of the call.

## Comparison

The GPU backend is designed for training at scale. Its throughput rises with the number of
environments until the GPU is full, reaching 32,582 steps per second for 16,384 environments of
`5_vs_5` on an RTX 4090, and its observations remain in device memory beside the policy, so a step
involves no transfer over the PCIe bus.

The CPU backend serves development, evaluation and small batches. A single environment of `5_vs_5`
steps 2,018 times per second on one core of an AMD EPYC 7763, and the backend runs on any machine
with a supported processor. Its results equal those of the GPU, so a test on the CPU predicts the
behaviour on the GPU. A program may hold batches on both backends at once, for example evaluating
on the CPU while training on the GPU. Measured figures for both backends appear in
[Performance](performance.md).

## GPU backend

On the GPU each environment runs on one warp. The 32 threads of the warp divide the work of a tick
where it parallelizes naturally, such as over the players of a team, the candidate animations of a
player or the cells of the SMM, and execute the sequential parts in turn. A step is a single kernel
launch that advances every environment, writes its outputs and resets the environments whose
episode ended, so resets need no additional launch and delay no other environment.

`step` is enqueued on the current CUDA stream of PyTorch and returns without waiting for the device.
Subsequent work on the same stream observes the outputs, so a training loop confined to one stream
needs no synchronization. When the actions are already in device memory, a step is one launch with
no copy, which allows a CUDA graph to capture it together with the policy. PyTorch warms a module
up on a side stream before capturing it, and each replay of the resulting graph performs one step
of the batch and computes the next actions.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4096, device='cuda')
policy = torch.nn.Linear(115, 19).cuda()
observation = batch.reset()
actions = torch.zeros((4096, 4), dtype=torch.int32, device='cuda')

# PyTorch warms a module up on a side stream before it captures it.
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(side):
    for _ in range(3):
        observation, _, _, _ = batch.step(actions)
        actions.copy_(policy(observation).argmax(-1))
torch.cuda.current_stream().wait_stream(side)

graph = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(graph):
    observation, reward, done, information = batch.step(actions)
    actions.copy_(policy(observation).argmax(-1))
for _ in range(1000):
    graph.replay()  # one step of the batch and the next actions, in one replay
```

The script `examples/cuda_graph_step.py` runs this loop and reports its speed. A replay produces the
same bits as stepping without a graph. Graph capture covers the calls that keep their data on the
device, while the calls that copy through the host or wait for the device run between replays.

| Captured by a graph | Run between replays |
| --- | --- |
| `step` with the actions in a tensor on the batch's GPU | `step` with actions on the host, which the library checks and copies at the call |
| `reset_done` | `reset()`, and `reset` with `seeds` or `reverse`, which wait for the device |
| `reset(mask=...)` with the mask on the device, once every environment has started | `get_state` and `set_state`, which copy to and from the host |

The first call on the GPU loads the kernel and therefore takes longer than later calls, and
timings conventionally exclude it.

### Compilation with `torch.compile`

`step` and `reset_done` are registered as the custom operators `cuju::step` and
`cuju::reset_done`, so a function that steps a batch compiles with `fullgraph=True`, and
`mode='reduce-overhead'` places the step and the policy in one CUDA graph.

```python
@torch.compile(mode='reduce-overhead', fullgraph=True)
def act_and_step(observation):
    actions = policy(observation).argmax(-1).to(torch.int32)
    return batch.step(actions)
```

`reset`, `raw()`, `get_state` and `set_state` run outside the trace, and each of them causes a
graph break.

The operators require PyTorch 2.4 or later, whose `torch.library.custom_op` defines them. With an
earlier PyTorch, such as 2.2 on macOS on Intel processors, a compiled function breaks its graph at
each step and runs the step eagerly, with the same results, so `fullgraph=True` fails there. On
Windows, PyTorch's wheels ship without Triton, which Inductor needs for code on the GPU, so there
`torch.compile` raises `TritonMissing` for a function that steps a GPU batch, and the eager step
gives the same results.

The CUDA graph needs the batch's output buffers marked as static inputs of the compiler. A batch
marks them at a `reset` or an eager `step` once PyTorch has loaded the compiler, which the call to
`torch.compile` does, and the compiler takes seconds to load, so a batch never loads it itself. The
order that keeps the graph is therefore `torch.compile` first and a `reset` of the batch after it,
or one more `reset` after `torch.compile` for a batch reset before. With the opposite order the
outputs stay exact, the step runs outside the graph, and Inductor reports that it skips CUDA graphs
for mutated inputs.

The benefit of a graph depends on the composition of a step. A graph removes the overhead of
launching each kernel from Python, for the batch and the policy alike. With a small policy, or a
batch of a few hundred environments, launch overhead forms a large share of a step, and a graph
removes much of it. With thousands of environments the step kernel dominates, and a graph saves
little. A graph also fixes the shapes and the memory of everything it contains, which constrains
the training loop built around it.

### GPU options

The GPU options control how the work runs and leave every output unchanged. They apply to the GPU
alone.

**`allocator='torch'`** draws the device memory of the batch from the caching allocator of PyTorch,
so that `torch.cuda.memory_allocated`, the related statistics and
`set_per_process_memory_fraction` include it. The default, `'cuda'`, allocates with `cudaMalloc`,
outside the accounting of PyTorch. With either value, the local memory that the driver reserves for
the kernels remains outside PyTorch.

Under the default, the memory of a batch is accounted separately and returns to the driver when the
batch closes. Under `'torch'`, a batch and a model on the same GPU share one memory budget, the
memory statistics of PyTorch cover both, a memory fraction limits the batch as well, and the memory
of a closed batch stays in the cache of PyTorch for later tensors instead of returning to the
driver. The second behaviour benefits long runs that create and close batches repeatedly, such as an
evaluation batch every few updates, since the cache reuses the same memory each time.

**`check_actions=True`** checks actions that already lie in GPU memory against the action set
before each step, as the library checks actions in host memory, and raises `ActionRangeError`
before any environment steps. The check reads its result on the host, so each step waits for the
device, and a CUDA graph capture skips it. By default such actions go to the step as they are, and
an index out of range plays as `idle`, as described in
[Rewards and actions](rewards-and-actions.md). The option suits the first runs of a new policy or
action pipeline, and a long run leaves it off once the actions are known to stay in range.

### Debugging and profiling

A GPU batch enqueues its work on the stream and returns, so a failure on the device, such as a read
of an invalid address, surfaces at a later call that waits, often far from its cause. The option
**`debug=True`** makes every `step`, `step_async`, `reset_done` and `reset` wait for the device
before it returns, so that such a failure raises `DeviceError` at the call that caused it, with the
CUDA call that reported it. It also checks the actions as `check_actions` does. A CUDA graph
capture admits no wait for the device, so a capture skips both the waits and the check. The waits
remove the overlap of the batch with the rest of the program, so `debug` serves the search for a
fault, and a training run leaves it off. A failure that ends the CUDA context, such as the trap of
an undefined path, still ends it, and the process restarts. The CPU waits at every call anyway, so
it ignores the option.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4096, device='cuda', debug=True)
```

For profiling, a source build with the CMake option `CUJU_NVTX` wraps each call of the C API in an
NVTX range, which Nsight Systems shows on a timeline beside the kernels of the batch and of
PyTorch. The ranges belong to the domain `cuju` and carry the names of the calls, `create`,
`reset`, `reset_masked`, `reset_done`, `step`, `step_async`, `step_wait`, `get_state` and
`set_state`. The ranges cover the CPU library as well as the CUDA library, and they place the time
of each call of the batch among the other work of a training step, such as the policy, the update
and any copies. NVTX 3 is header-only and comes with the CUDA toolkit, and a build that finds no
NVTX headers warns and builds the libraries without ranges. Each NVTX call returns at once unless a
profiler is attached, and the kernels and outputs are the same with and without the option. The
release wheel is built without it, so profiling with ranges uses a source build, as described in
[Installation](installation.md).

```sh
CMAKE_ARGS="-DCUJU_NVTX=ON" pip install .
nsys profile --trace=cuda,nvtx python train.py
```

## CPU backend

On the CPU a pool of threads runs the environments. Each thread claims the next unclaimed
environment and runs its whole step, which balances the load when matches differ in cost, as they do
in a mixed batch. The calling thread participates, and `step` returns when every environment has
finished.

The `threads` option sets the size of the pool, including the calling thread. The default,
`threads=0`, uses one thread per logical processor in the affinity mask of the process, capped by
the CPU quota of its cgroup, rounded up. Slurm and `docker run --cpuset-cpus` set the affinity mask
and `docker run --cpus` sets the quota, so that `docker run --cpus=2.5` yields three threads. On
Windows and macOS, `threads=0` uses one thread per logical processor of the machine. The outputs
are identical for every number of threads.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r4', environment_count=192, device='cpu', threads=12)
```

Two mechanisms reduce the overhead of the pool. A worker that has finished a step polls for the next
one for about 50 microseconds before sleeping, so a tight loop of steps finds the threads awake. The
large buffers, such as the worlds of the environments, request transparent huge pages from the
Linux kernel, which reduces the cost of address translation.

On x86-64 the library targets the `x86-64-v3` level, with AVX2, BMI2 and FMA, which the Intel Core
processors since Haswell, released in 2013, and the AMD processors since Excavator, released in
2015, support, and MSVC builds it with `/arch:AVX2`. On Arm, Apple silicon included, it targets the
base Armv8-A, which every 64-bit Arm processor implements. The variable `CUJU_CPU_FLAGS` sets the
compiler flags at build time, with the default
`-O3 -march=x86-64-v3 -fno-math-errno -fno-semantic-interposition` on Linux on x86-64 and the
defaults of each other platform listed in [Installation](installation.md#source-build). The build
rejects flags that change the rounding of floating-point operations, such as `-ffast-math`, and a
build with other flags, such as `-march=native`, is checked with `pytest tests` before use.

### CPU options

The CPU options likewise leave every output unchanged. The last three apply to the CPU alone, and on
the GPU `step_async` behaves as `step`. Each option improves throughput under some conditions and
reduces it under others, so each starts switched off and is enabled where its conditions hold.

**`step_async(actions)` and `step_wait()`** divide a step in two. The CPU steps in a thread of the
batch while the calling code continues, and `step_wait` returns the result that `step` would. The
typical use is a CPU batch that feeds a learner on the GPU, which updates while the CPU plays the
next step. The gain is bounded by the shorter of the two durations, so it is largest when the
learner and the step take similar time. The outputs are complete once `step_wait` returns, and any
other call on the batch first waits for the step.

**`pin_threads=True`** binds each thread of the pool to a processor of its own, while the calling
thread stays where the operating system places it. On a dedicated machine, pinning keeps threads on
their cores and caches, which reduces the variance of the step time. On a shared machine it
increases the step time, since a pinned thread remains on a core that another process occupies, and
the whole step waits for it. Pinning applies under Linux, and on Windows and macOS the threads stay
where the operating system places them.

**`numa=True`** forms one group of threads per NUMA node, each stepping a fixed range of
environments whose memory its own threads touch first. On servers with two or more sockets, memory
attached to another socket has higher latency, and this placement keeps each environment on the
node that holds its memory. On a single-node machine, and on Windows and macOS, the option leaves
the pool as it is, so code that runs on every kind of machine can enable it unconditionally.

**`pin_memory=True`** page-locks the output buffers, so that `tensor.to('cuda', non_blocking=True)`
copies them asynchronously. It requires a CUDA GPU that PyTorch sees. It is intended for runs that
send the observations of every step to a GPU, usually together with `step_async`, in a loop that
overlaps the CPU step with a learner update. Page-locked memory stays resident in physical memory,
which makes the option costly for a very large batch on a machine with little memory.

```python
batch = cuju.BatchEnvironment(
    '5_vs_5_l4_r0', environment_count=512, device='cpu', pin_threads=True, pin_memory=True
)
observation = batch.reset()
for _ in range(steps):
    batch.step_async(actions)  # the CPU steps
    learner_update()  # while the GPU learns
    observation, reward, done, information = batch.step_wait()
    observation_gpu = observation.to('cuda', non_blocking=True)
```

## Integration with PyTorch

Every output is a PyTorch tensor that views a buffer of the batch, a CUDA tensor on the GPU and a
CPU tensor on the CPU. Each call returns the same tensors without copying, and the next `reset`,
`step` or `set_state` overwrites them, so outputs that must persist are cloned.

Actions are accepted as a tensor on the device of the batch or as any object that `torch.as_tensor`
reads. A contiguous `int32` tensor on the device of the batch is read in place. Other integer types
incur a conversion and an allocation at every step, which applies to the `int64` output of `argmax`
and `Categorical.sample`, and copying such actions into an `int32` buffer allocated once avoids
that cost. Actions elsewhere, in host memory or on another GPU, pass through the host. Actions on the
host are copied and checked, while actions on the GPU of the batch are read in place and checked
only with `check_actions=True` or `debug=True`, as described in
[Rewards and actions](rewards-and-actions.md).

`step` releases the global interpreter lock of Python while the library runs, so other Python
threads, such as one stepping a second batch, run concurrently.

cuju sets the floating-point mode it requires on every call and restores the mode of the caller
afterwards, so programs that flush subnormal numbers to zero, as some numerical libraries do, obtain
the same results.

A batch frees its memory when the last tensor that views it is released, or at once on `close()`.
A tensor retained after its batch was dropped continues to hold valid values, whereas the tensors of
a batch closed with `close()` become invalid at that call, and every later call on the batch raises
`ClosedError`. Explicit closing suits the GPU, where a batch holds gigabytes and the last tensor that
views it can be hard to locate. A `with` statement closes the batch at the end of its block, an
exception included, so the memory returns even when the training code raises.

```python
with cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4096, device='cuda') as batch:
    observation = batch.reset()
    for _ in range(1000):
        observation, reward, done, information = batch.step(policy(observation))
```

## Processes

One batch per process and device minimizes the interaction between batches. A process may hold
several batches, in which case each CPU batch has its own pool and the pools compete for the same
cores.

A CPU batch created before `os.fork()` remains usable in the child, which starts its own pool. If
another thread of the parent was stepping the batch at the time of the fork, the first call in the
child raises an error. A CUDA batch belongs to the process that created it, as its CUDA context
does, so a child process creates a batch of its own. Creating batches after the fork serves both
backends and also avoids the copy-on-write of the buffers. Each worker of multiprocessing or Ray
creates its own batch inside the worker, since a batch holds device memory and threads that stay
with their process, and pickling a batch raises `TypeError` with that advice.

## Several GPUs

A run spans several GPUs in one of two layouts, depending on where the learner runs.

**One process per GPU.** When each GPU trains its own copy of the model, as with the distributed
data parallel of PyTorch, each process holds a batch on its own GPU. The GPU is named with
`device=f'cuda:{local_rank}'`, and `first_index` assigns each rank its own range of the seed rule,
so that the ranks together play the environments of one large batch.

```python
import os

rank = int(os.environ['LOCAL_RANK'])
count = 4096
batch = cuju.BatchEnvironment(
    '5_vs_5_l4_r0', environment_count=count, device=f'cuda:{rank}', first_index=rank * count
)
```

In this layout every process keeps its observations, actions and policy on its own device, and only
the gradients cross between GPUs, which makes it the layout that scales furthest.

**One process over several devices.** When one learner consumes more environments than one device
can play, `cuju.MultiDeviceBatch` holds one batch per device and steps them together.

```python
batch = cuju.MultiDeviceBatch('5_vs_5_l4_r0', ['cuda:0', 'cuda:1'], environment_count=32768)
observation = batch.reset()
```

Environment `i` plays as environment `i` of a single batch of 32,768, regardless of the split. A
step is launched on every device before any is awaited, so the GPUs step concurrently and a CPU part
steps in a thread of its own. The outputs of `reset` and `step` are gathered on `output_device`, by
default the first device, at the cost of one copy per step, and `batch.parts` exposes each part as a
`BatchEnvironment` with its own outputs and no copy. A CPU part with `pin_memory=True` copies its
outputs to a GPU without waiting, and its next step, reset or state waits only until that copy has
run. The `reset` method takes `seeds` and `reverse`,
as for a single batch. A masked reset goes through `batch.parts`, each part taking the mask of its
own range.

The `devices` list may mix GPUs and the CPU and may repeat a device, and `split` sets the number of
environments of each part, by default an even split. Since a step waits for the slowest part, a
split in proportion to the speed of each device balances the step, with the rates in
[Performance](performance.md) as a starting point.

cuju 0.1.0 was checked on machines with a single GPU, and the batches on further GPUs of a machine
run the same code on another device. Reports from machines with several GPUs are welcome as issues.

## See also

- [API reference](api.md), for every option of both backends in table form
- [Concepts](concepts.md), for batches, bundles and environments
- [Episodes and seeds](episodes-and-seeds.md), for the seed rule that `first_index` shifts
- [Performance](performance.md), for the throughput and memory of each backend
