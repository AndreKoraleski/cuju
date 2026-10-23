# Performance

The **performance** of cuju is its throughput in environment steps per second, measured against
Google Research Football, or GRF, on the same machine. On one processor of an AMD EPYC 7763, a
single environment of `5_vs_5` with four agents a side runs 9.75 times as many steps per second as
one GRF process, and on 31 processors the ratio is 6.62. On an NVIDIA RTX 4090, 16,384 environments
of the same scenario reach 32,582 steps per second, about 15% fewer than the 31 processors of the
same machine, with the observations, the actions and the policy kept in device memory.

GPU throughput rises with the number of environments until the device is full, and device memory,
at about 0.37 MiB per environment plus a fixed reservation for the kernels, bounds the size of a
batch. On small GPUs under Windows and WSL2, memory beyond the capacity of the device spills into
shared system memory, which slowed two concurrent training processes on a 4 GB laptop GPU by a
factor of about ten. `cuju.memory_estimate` gives the memory of a batch and the reservation for the
kernels before the batch is created, and a GPU batch that requests more than the free memory of its
device warns at creation.

## Method

Throughput is counted in environment steps per second over the whole batch. With four agents on a
side, one environment step moves four agents, and it counts once.

Each figure is the median of interleaved runs. The configurations alternate within each round, so
that heat and background load affect them alike, and the percentage beside each figure is the spread
over the rounds, `(max - min) / median`. Each run discards its first 100 steps and times the next
1,000, episode resets included, as they occur in training. Random actions come from a fixed
generator, so every run plays the same matches. GRF runs on the same machine, unrendered, with the
same scenario, agents and representation, one process per environment. The script
`examples/benchmark.py` reproduces every row.

The machine is a cloud instance with an NVIDIA RTX 4090 and an AMD EPYC 7763 of the Zen 3
generation, whose container quota gives the process 31 processors. cuju is the release wheel,
`cuju 0.1.0`, and GRF is commit 3d9e754 built with its own flags, with gym 0.25.2 and numpy 1.26.4.
The `11_vs_11_stochastic` runs have one agent, on the left.

## CPU throughput

On a single processor, one environment of cuju and one GRF process, both pinned to the same
processor, were compared over three rounds.

| Scenario | GRF, one process | cuju, one environment | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 207 (4.9%) | 2,018 (1.5%) | 9.75x |
| `11_vs_11_stochastic`, 1 agent | 163 (1.7%) | 914 (1.9%) | 5.61x |

On the whole quota, 31 threads of 16 environments each were compared with 31 GRF processes started
together, over three rounds.

| Scenario | GRF, 31 processes | cuju, 31 threads | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 5,819 (0.2%) | 38,497 (0.7%) | 6.62x |
| `11_vs_11_stochastic`, 1 agent | 4,391 (0.4%) | 25,088 (0.2%) | 5.71x |

The ratio narrows from one processor to 31. For `5_vs_5`, each GRF process keeps about 91% of its
single-processor rate when 31 run together, and each cuju thread about 62%. The `5_vs_5` rows come
from three commands, for GRF on one processor, for cuju on one processor and for cuju on all 31.

```sh
taskset -c 0 python examples/benchmark.py --grf --scenario 5_vs_5 --left 4 --right 4
taskset -c 0 python examples/benchmark.py --scenario 5_vs_5_l4_r4 --device cpu --environments 1 --threads 1
python examples/benchmark.py --scenario 5_vs_5_l4_r4 --device cpu --environments 496 --threads 31
```

The all-core GRF figure is the sum of 31 copies of the first command run at once. The 11-a-side rows
use `--scenario 11_vs_11_stochastic --left 1 --right 0` for GRF and
`--scenario 11_vs_11_stochastic_l1_r0` for cuju.

On a laptop with an AMD Ryzen 5 7535HS of 12 threads under WSL2, a build of the release tree for
the same commands gave the following rates over one round.

| Scenario | cuju, one environment | cuju, 12 threads |
| --- | --- | --- |
| `5_vs_5`, 4 agents a side | 859 | 6,146 |
| `11_vs_11_stochastic`, 1 agent | 439 | 3,724 |

## GPU throughput

On the RTX 4090, the machine code of the wheel for compute capability 8.9 was measured over two
rounds, which agree within 0.6%.

| Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 9,920 (0.6%) | 22,661 (0.2%) | 32,582 (0.0%) |
| `11_vs_11_stochastic`, 1 agent | 8,679 (0.0%) | 17,885 (0.2%) | 24,221 (0.0%) |

The `5_vs_5` figure at 16,384 environments comes from one command, and `--scenario` and
`--environments` select the others.

```sh
python examples/benchmark.py --scenario 5_vs_5_l4_r4 --device cuda --environments 16384
```

On the GPUs of Google Colab, the same wheel and method were run for one round each, so these figures
carry no spread. Each runtime has an Intel Xeon host with 12 processors, or eight for the T4.

| GPU (compute capability) | Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- | --- |
| A100-SXM4-40GB (8.0) | `5_vs_5`, 4 agents a side | 5,462 | 8,940 | 10,334 |
| | `11_vs_11_stochastic`, 1 agent | 4,350 | 6,342 | 6,981 |
| L4 (8.9) | `5_vs_5`, 4 agents a side | 6,048 | 9,772 | 11,270 |
| | `11_vs_11_stochastic`, 1 agent | 4,503 | 6,627 | 7,316 |
| T4 (7.5) | `5_vs_5`, 4 agents a side | 2,761 | 3,601 | 3,859 |
| | `11_vs_11_stochastic`, 1 agent | 1,954 | 2,369 | 2,335 |

With one warp per environment, throughput favours high clock rates over a large number of slower
cores, so the A100, at 1.41 GHz, stays near the L4 and well behind the 4090. The T4 is saturated at
4,096 environments of `11_vs_11_stochastic`.

The RTX 3050 Laptop GPU of the same laptop, with 4 GB and compute capability 8.6, was measured with
the same build over one round. Its memory holds 4,096 environments of either scenario, while 16,384
need about 6 GiB of buffers besides the local memory of the kernels.

| GPU (compute capability) | Scenario | 1,024 environments | 4,096 |
| --- | --- | --- | --- |
| RTX 3050 Laptop GPU, 4 GB (8.6) | `5_vs_5`, 4 agents a side | 3,332 | 3,982 |
| | `11_vs_11_stochastic`, 1 agent | 2,384 | 2,718 |

GRF runs on the CPU, so the GPU figures compare with the CPU of the same machine. At 16,384
environments the 4090 steps `5_vs_5` about 15% slower than 31 processors, and `11_vs_11_stochastic`
about 3% slower, and on the laptop the 12 threads step both scenarios faster than its GPU. The GPU
earns its place through what it keeps in device memory. The observations, the actions and the policy
stay there, so a training loop copies nothing between the batch and the network, and the CPU remains
available for the rest of the work, such as the optimizer, logging and evaluation.

## Batch size

On the GPU, throughput rises with the number of environments until the device is full and then
levels off. A search for the batch size starts conveniently at 4,096 environments and doubles the
count until the steps per second stop rising or the memory runs out. Past the point where the rate
levels off, more environments add memory and latency per update without adding throughput.

Each environment of `5_vs_5_l4_r0` with `simple115v2` occupies about 0.37 MiB of device memory, so
16,384 environments occupy about 6 GiB. Creating the first batch of a process on a GPU also makes
the driver reserve the local memory of the kernels, which it then holds until the process ends. The
driver keeps a stack of 1 KB for every thread that the GPU can run at a time from the creation of
its context, and the first launch raises it to the local memory of the largest kernel, 49,776 bytes
per thread in the build for the RTX 4090, so the reservation adds about 48 KB per thread. That
amounts to about 8.9 GiB on an RTX 4090, which runs 196,608 threads at a time, and to about 1.1 GiB
on a 4 GB RTX 3050 Laptop GPU, which runs 24,576. By default the library allocates outside the
allocator of PyTorch, so `torch.cuda.memory_allocated` leaves this memory out, while
`torch.cuda.mem_get_info` counts all of it. With `allocator='torch'`, PyTorch counts the buffers of
the batch, as described in [Backends](backends.md).

`cuju.memory_estimate` returns the bytes that a batch would request, for the same arguments as
`BatchEnvironment`, without allocating them, and `batch.memory_bytes` returns the bytes that a
created batch has requested. Both count the buffers, the worlds and the shared tables of the batch.
The result of `memory_estimate` also carries the reservation for the kernels as its attribute
`local_memory_bytes`, which the library computes from the local memory of each kernel, the current
stack of the device and the threads that the GPU runs at a time. It is 0 once the process has
created a batch on the GPU, and always 0 on the CPU. On an RTX 4090 the figure, 9,585,033,216 bytes,
came within 1 to 5 MiB, or 0.06%, of the drop in free memory that `torch.cuda.mem_get_info` reported
around the first creation, once the creation of a second batch of the same size was subtracted. The
difference between the estimates for two environment counts gives the memory of one environment, and
the estimate for a planned count, together with `local_memory_bytes`, the context of PyTorch and the
model, shows before any allocation whether the batch fits the device. The estimate reads the bundles
and animation tables as a creation does, so it takes a fraction of a second. [Recipes](recipes.md)
gives a search for the largest batch that fits a budget.

On the CPU, about 16 environments per thread keep every core busy. `threads=0` uses every processor
that the process may run on, within the CPU quota of its cgroup, as described in
[Backends](backends.md).

The representation also affects memory. An observation of `simple115v2` holds 115 `float32` values,
460 bytes per agent, while the SMM of `extracted` at the default 96 by 72 cells holds four channels
of one byte each, 27,648 bytes per agent, and stacking multiplies either by four. An observation of
the SMM is therefore 60 times the size of a vector observation, which matters for rollout buffers
that hold every step of an update. [Observations](observations.md) gives the layout of each
representation.

The memory of a batch by scenario and representation was measured on the RTX 3050 Laptop GPU with
`allocator='torch'`, which counts every buffer of the batch. Each batch also takes about 31 MiB
regardless of its size, and the figures leave out the local memory of the kernels, the context of
PyTorch and the model. The SMM costs more than its observation alone, since the batch also keeps the
last observation of each ended episode in `final_observation`.

| Scenario | Representation | MiB per environment | 1,024 environments | 4,096 |
| --- | --- | --- | --- | --- |
| `5_vs_5_l4_r4` | `simple115v2` | 0.37 | 407 MiB | 1,535 MiB |
| | `extracted` | 0.78 | 833 MiB | 3,240 MiB |
| | `raw` | 0.37 | 405 MiB | 1,527 MiB |
| `11_vs_11_stochastic_l1_r0` | `simple115v2` | 0.36 | 401 MiB | 1,509 MiB |
| | `extracted` | 0.41 | 456 MiB | 1,723 MiB |
| | `raw` | 0.36 | 401 MiB | 1,511 MiB |

## Small and shared GPUs

A GPU with a few gigabytes of memory, such as that of a laptop, fills quickly, since the batch, the
local memory of the kernels, the context of PyTorch and the model share it. When the memory runs
out, a Linux driver fails the allocation, and creating the batch raises an error that names the
memory it requested, the memory the batch already held and the free memory of the GPU. A Windows
driver under WDDM, the driver model that WSL2 also uses, may instead move memory into the shared
system memory of the machine, which the GPU reads over the PCIe bus. The allocation then succeeds,
and every kernel that touches that memory runs many times slower. Two training processes that each
held a batch on a 4 GB RTX 3050 Laptop GPU under WSL2 both ran about ten times slower than one
process alone.

A GPU batch compares at creation the memory it requested with the free memory of the GPU
immediately before, and when the request is larger it issues a `UserWarning` that names both
amounts in MiB. With `allocator='torch'`, the memory that the cache of PyTorch holds unused counts
as free, since the batch may take it. Under a Linux driver such a request fails, and the creation
raises the error that names the memory, so the warning matters on Windows and WSL2, where the batch
is created and runs anyway, and there it is the main sign of a spill. A run that shows the warning
is resized before training begins, with fewer environments or a smaller model on the same GPU. For
the first batch of the process on a GPU, the comparison also counts the local memory that the
creation reserves for the kernels, and the warning names that share. Later batches on the same GPU
find the reservation held and compare their own memory alone.

One process with one batch per GPU, sized to the free memory that `torch.cuda.mem_get_info()[0]`
reports less room for PyTorch and the model, avoids the spill. `cuju.memory_estimate` with the
planned arguments gives the size of the batch for that comparison before any allocation, and its
`local_memory_bytes` the reservation that a first batch adds for the kernels. On a laptop,
evaluation on the CPU with `device='cpu'` beside training on the GPU takes the place of a second GPU
batch. A spill appears in the Windows Task Manager as shared GPU memory greater than zero, together
with a drop in steps per second of an order of magnitude. With `CUJU_LOG=1`, each batch logs at
creation the device memory it took, so a log of the run shows how the memory of the GPU was spent.

## Timing

The first steps after a reset run faster than later ones, because the players spread out and the
built-in AI has more to consider as a match develops, so a short window after kick-off overstates
the rate, and timings span hundreds of steps. The first call on the GPU loads the kernel and is
excluded from timings. An 11-a-side match costs more than a five-a-side one, roughly in proportion
to its number of players, as the single-processor rates of 2,018 and 914 steps per second show.

## See also

- [Backends](backends.md), for the execution model and the options of each backend
- [Exactness](exactness.md), for the equality of the CPU and GPU outputs that the figures compare
- [Installation](installation.md), for the wheel and the GPUs it covers
- [Recipes](recipes.md), for training loops that keep a batch on the GPU
