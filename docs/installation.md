# Installation

**Installation** of cuju takes one of three forms, a wheel from PyPI, a build from source, or a
build inside a container image for Docker or Apptainer, and each ends with `cuju check`,
which confirms that the install plays the matches of Google Research Football, or GRF, bit for bit.
PyPI holds a wheel for Linux on x86-64 and on Arm, for Windows, and for macOS on Apple silicon and
on Intel processors, and the wheel for Linux on x86-64 serves WSL2 as well. A source build serves
GPUs outside the wheels, other compilers and other flags.

cuju requires Python 3.10 or later and PyTorch 2.3 or later, or PyTorch 2.2 on macOS on Intel
processors. On x86-64 the CPU must support AVX2, BMI2 and FMA, and every 64-bit Arm processor runs
the CPU backend. The GPU backend runs on Linux, on Windows and in WSL2, and it requires an NVIDIA GPU
of compute capability 7.5 or higher and a driver for CUDA 12, with the CUDA runtime linked
statically in place of a toolkit. pip installs PyTorch along with cuju when it is missing, and
installing a chosen build of PyTorch first selects the build that cuju then uses.

## Requirements

Each requirement follows from the platforms and the hardware on which cuju is built and checked.

| | Needed | Why |
| --- | --- | --- |
| Operating system | one of the platforms in [Platforms](#platforms) | the libraries are built and checked there |
| CPU on x86-64 | AVX2, BMI2 and FMA, the level `x86-64-v3`, so Intel since Haswell and AMD since Excavator | the x86-64 CPU library is compiled for them, and exactness was checked with them |
| CPU on Arm | any 64-bit Arm processor, Apple silicon included | the Arm CPU library is compiled for the base Armv8-A, which every such processor implements |
| GPU, for `device='cuda'` | NVIDIA, compute capability 7.5 or higher, and a driver for CUDA 12, release 525.60.13 or later on Linux | the oldest architecture checked |
| Python | 3.10 or later | |
| PyTorch | 2.3 or later, or 2.2 on macOS on Intel processors, and a build that sees the GPU for `device='cuda'` | outputs are PyTorch tensors on the device of the batch, and 2.2 is the last release of PyTorch for macOS on Intel processors |

On Linux, the three instruction sets of x86-64 appear in `/proc/cpuinfo`, and the first command
prints all three on a supported CPU. On macOS on an Intel processor, the second prints `1` for each.

```sh
grep -o -w 'avx2\|bmi2\|fma' /proc/cpuinfo | sort -u   # expect all three
sysctl hw.optional.avx2_0 hw.optional.bmi2 hw.optional.fma
```

cuju checks the processor itself when it loads the packaged CPU library on x86-64. It reads the
three flags from `/proc/cpuinfo` on Linux and from `sysctl` on macOS, and on Windows, where the
system reports AVX2 alone, it checks AVX2. A processor that lacks one of them raises `DeviceError`
with the missing names, in place of the illegal instruction that the library would otherwise
cause. A library that `CUJU_CPU_LIBRARY` names is a build of its own and loads without the check.
The Arm library uses only the base Armv8-A, so it loads on every 64-bit Arm processor. On both
architectures each call of the CPU library sets the default floating-point mode of the C library,
rounding to nearest and keeping subnormal numbers, and restores the caller's mode afterwards, so a
mode that another library sets leaves every output unchanged.

Compute capability 7.5 is the oldest GPU architecture on which cuju was checked for exactness, that
of the T4 and the RTX 20 series. Earlier architectures, such as the P100 at 6.0 that Kaggle
notebooks offer beside the T4, lie outside that range, and on Kaggle the T4 runs cuju.

The CUDA library links the runtime of CUDA 12.9 statically, so it runs without a CUDA toolkit and
independently of the CUDA version of the PyTorch build. It needs only the driver. NVIDIA's
minor-version compatibility runs a CUDA 12 program on any driver for CUDA 12, under Linux from
release 525.60.13 on.

## Platforms

Each platform has a wheel on PyPI. The CPU library needs no GPU, so a machine without one runs
`device='cpu'` on every platform.

| Platform | Wheel | Backends |
| --- | --- | --- |
| Linux on x86-64 | `manylinux_2_28_x86_64`, for glibc 2.28 or later, so RHEL 8, Ubuntu 20.04 and later | CPU and CUDA |
| Linux on Arm, aarch64 | `manylinux_2_28_aarch64`, for glibc 2.28 or later | CPU |
| Windows on x86-64 | `win_amd64`, with `cuju_cpu.dll` and `cuju_cuda.dll` | CPU and CUDA |
| macOS on Apple silicon, arm64 | `macosx_11_0_arm64`, for macOS 11 or later | CPU |
| macOS on Intel processors, x86-64 | `macosx_10_15_x86_64`, for macOS 10.15 or later, with PyTorch 2.2 | CPU |
| Windows through WSL2 | the wheel for Linux on x86-64 inside a WSL2 distribution, with the NVIDIA driver installed on Windows | CPU and CUDA |

Every platform passed the self-check and the public tests for 0.1.0, on the CPU everywhere and on
the GPU under Linux, in WSL2 and under Windows, where the CUDA library built with CUDA 12.9 passed
all 23 cases on an RTX 3050. The expected digests are one file, the same on every platform, so a
pass on any platform means the bits of Linux on x86-64 and of GRF. The checks covered Linux on
x86-64 with GCC and Clang 18 and on Arm with Clang 18, Windows with the MSVC of Visual Studio 2022
and 2026, and macOS 14 on Apple silicon and macOS 15 on an Intel processor with Apple Clang.

On macOS on Intel processors, PyTorch ends with release 2.2, which predates the custom operators
that `torch.compile` traces, so there a compiled function breaks its graph at each step and runs
the step eagerly with the same results, as [Backends](backends.md) describes. The CPU options
`pin_threads` and `numa` place threads under Linux alone, and on Windows and macOS the threads stay
where the system puts them.

On every platform, [`cuju check`](#install-check) reports whether the build plays GRF's
matches, and the same command confirms each new machine.

## Installation methods

The three methods differ in the GPUs they cover and in the part of the build that the installer
controls.

The wheel from PyPI installs with one command and covers every GPU in the wheel's table of
architectures, as well as any machine that uses the CPU alone. It is also the one install whose
machine code was tested on those GPUs, which makes it the method for work where exactness matters
and the GPU is one of the wheel's architectures.

A build from source covers GPUs outside the wheel, such as the H100 and the RTX 50 series, as well
as builds with another compiler or other flags. It takes a few minutes and, for the CUDA library, a
CUDA toolkit, and it places the check of exactness with the person who builds, through
`cuju check --long` and `pytest tests/`.

Docker runs a source build inside an image, for clusters, for continuous integration and for
machines that keep no CUDA toolkit, and Apptainer runs the same image on clusters that offer
Apptainer in place of Docker.

Every method ends with `cuju check`, which reports in one to two minutes whether the
install works and plays the matches of the release, as timed in [Install check](#install-check).
Where the scripts folder of a user installation is missing from the path, as can happen on Windows,
`python -m cuju check` runs the same command.

## PyTorch

cuju declares `torch>=2.3` as a dependency, and `torch>=2.2` on macOS on Intel processors, so
`pip install cuju` installs PyTorch when it is missing. The build that pip takes from PyPI by
default is a CUDA build on Linux, WSL2 included, and a CPU-only build on Windows and macOS. Package
metadata names a version of PyTorch and leaves the choice of index to pip, so the other builds,
which pytorch.org serves from indexes of its own, are reached by installing them directly. On Linux
with an NVIDIA driver that supports the CUDA version of PyPI's default build, plain
`pip install cuju` therefore gives a working install of both backends in one command. On Windows,
the GPU backend needs a CUDA build of PyTorch from pytorch.org, installed before cuju.

Installing PyTorch first, from the index that pytorch.org lists for the system, selects a
particular build, such as one for another CUDA version, a CPU-only build or a ROCm build. pip then
keeps the PyTorch that is present, since it satisfies the requirement, and installs cuju beside it.
On Windows with an NVIDIA GPU, for example, the two commands install a CUDA build first and cuju
second.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install cuju
```

The build of PyTorch matters for the GPU backend. cuju's CUDA library carries its own CUDA runtime,
so it works with a PyTorch built for any CUDA version, but a GPU batch places its buffers on a
device that PyTorch manages, so `device='cuda'` needs a build for which `torch.cuda.is_available()`
is true. The CPU backend runs with every build of PyTorch.

## Wheel from PyPI

The wheel installs with pip.

```sh
pip install cuju
```

The extra `spaces` adds Gymnasium, which builds the spaces of GRF's interface in `cuju.env` and
`cuju.gfootball`, the `GFootball-...` ids of `gymnasium.make`, and the spaces of a batch, such as
`batch.single_observation_space`. The batch itself steps without it.

```sh
pip install 'cuju[spaces]'
```

The extra `render` adds the browser renderer, the package cuju-render, which
[Rendering](rendering.md) describes. It installs into the folder of cuju and releases with it,
so the extra always takes a renderer that fits the installed cuju.

```sh
pip install 'cuju[render]'
```

The wheels for Linux on x86-64 and for Windows carry both libraries, and the wheels for Linux on
Arm and for macOS carry the CPU library. The CUDA library holds machine code for the four GPU
architectures that were checked for exactness.

| Compute capability | GPUs |
| --- | --- |
| 7.5 | T4, RTX 20 series |
| 8.0 | A100 |
| 8.6 | A10, RTX 30 series |
| 8.9 | L4, L40, RTX 40 series |

The wheel carries machine code alone. It leaves out PTX, the intermediate code that a driver can
compile for later GPUs at load, so that every GPU on which the wheel runs executes code that was
checked. Other GPUs, such as the H100 at 9.0, the B200 at 10.0 or the RTX 50 series at
12.0, run cuju from a source build. With the wheel, a CUDA batch on such a GPU fails with CUDA's
report that no kernel image is available for the device, and `cuju check` names the
GPU's compute capability and the build it needs. `device='cpu'` works on every machine.

## Source build

A source build compiles both libraries for the machine from a clone of the repository.

```sh
git clone https://github.com/AndreKoraleski/cuju
cd cuju
pip install .
```

A source build needs CMake 3.24 or later and the C++ compiler of its platform, GCC 12 or later or
Clang 18 on Linux, the MSVC of Visual Studio 2022 or later on Windows, and Apple Clang or another
Clang on macOS. The CUDA library needs the CUDA toolkit 12.0 or later, or 12.8 or later for compute
capability 10.0 and 12.0, which earlier toolkits lack. On Linux, when nvcc is missing, the build
skips the CUDA library and reports it. A build of the CPU library alone, on purpose, turns the CUDA
library off.

```sh
CMAKE_ARGS=-DCUJU_CUDA=OFF pip install .
```

Clang builds a CPU library about 10% faster than GCC's, with the same results, and the wheel's CPU
library is built that way. A source build does the same by naming Clang for the C++ code and keeping
GCC as the host compiler of nvcc.

```sh
CXX=clang++ CUDAHOSTCXX=g++ pip install .
```

The CUDA library targets the GPU of the build machine, `native`. A build machine with another GPU,
or none, names the architectures explicitly.

```sh
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" pip install .
```

The build options are passed in `CMAKE_ARGS`, and each has a default that reproduces the wheel's
build where the hardware allows.

| Option | Default | Effect |
| --- | --- | --- |
| `CUJU_CUDA` | `AUTO` | `AUTO` builds the CUDA library when nvcc is found, `ON` requires it, and `OFF` skips it |
| `CUJU_CPU` | `ON` | build the CPU library |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | the GPU architectures of the CUDA library |
| `CUJU_CPU_FLAGS` | the platform's, in the table below | the flags of the CPU library, and the build refuses flags that change floating-point results |
| `CUJU_NVTX` | `OFF` | `ON` wraps the calls of the C API in NVTX ranges for Nsight Systems, in both libraries, with the NVTX 3 headers of the CUDA toolkit |

The default of `CUJU_CPU_FLAGS` depends on the compiler and the architecture. Every default keeps
each floating-point operation as written, and the build adds the flags of exactness to it, such as
`-ffp-contract=off` for GCC and Clang and `/fp:precise` for MSVC, together with `-mfma` on x86-64
outside MSVC, so that each `fma` call of the ported math functions compiles to one instruction.

| Platform | Default of `CUJU_CPU_FLAGS` |
| --- | --- |
| Linux on x86-64 | `-O3 -march=x86-64-v3 -fno-math-errno -fno-semantic-interposition` |
| Linux on Arm | `-O3 -fno-math-errno -fno-semantic-interposition` |
| Windows on x86-64 | `/O2 /arch:AVX2` |
| macOS on Intel processors | `-O3 -march=x86-64-v3 -fno-math-errno` |
| macOS on Apple silicon | `-O3 -fno-math-errno` |

The wheel is built with `CUJU_NVTX` off. A build with it on produces the same kernels and outputs,
and each range costs a push and a pop, which return at once unless a profiler is attached. A build
that finds no NVTX headers warns and leaves the ranges out. [Backends](backends.md) describes the
ranges and their use with Nsight Systems.

```sh
CMAKE_ARGS="-DCUJU_NVTX=ON" pip install .
```

A build with non-default flags or compilers is followed by the tests of the
[install check](#install-check), which establish whether the build is still exact. The same holds
for a build for an architecture below compute capability 7.5, such as 7.0 for the V100, which lies
outside the range that the project checks, and `cuju check --long` on that GPU
establishes whether it plays GRF's matches.

### Windows

On Windows the build uses the MSVC of Visual Studio 2022 or later, with the C runtime linked
statically, so `cuju_cpu.dll` and `cuju_cuda.dll` load without a Visual C++ runtime beside them.
`pip install .` in a clone builds the CPU library. The CUDA library is built on request, with
`-DCUJU_CUDA=ON`, from the x64 Native Tools Command Prompt of Visual Studio 2022 with the CUDA
toolkit and its Visual Studio integration installed, and with Ninja as the generator, as the wheel
is built.

```bat
pip install ninja
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DCUJU_CUDA=ON
pip install .
```

CUDA 12.9 accepts the MSVC of Visual Studio 2022 and refuses that of Visual Studio 2026, so a build
of the CUDA library takes Visual Studio 2022. The CPU library builds with either. Windows builds
for x86-64.

### macOS

On macOS the build uses Apple Clang, from the Xcode Command Line Tools, or another Clang, and
builds the CPU library, `libcuju_cpu.dylib`. It targets one architecture, that of the host or the
one that `CMAKE_OSX_ARCHITECTURES` names, which is also the architecture of the Python that loads
the library. `MACOSX_DEPLOYMENT_TARGET` sets the oldest macOS that the library supports, 11.0 for
the wheel on Apple silicon and 10.15 for the wheel on Intel processors.

```sh
MACOSX_DEPLOYMENT_TARGET=11.0 pip install .
```

## Docker image

The repository's `Dockerfile` builds an image with both libraries and the Python package, on Ubuntu
24.04 with CUDA 12.9.1. The build runs without a GPU, so it names the architectures, with `-real`
for machine code alone, as in the wheel. Its default is the wheel's four architectures,
`75-real;80-real;86-real;89-real`, and a build for the RTX 40 series alone names `89-real`.

```sh
docker build --build-arg CUDA_ARCHITECTURES=89-real -t cuju:0.2.1 .
docker run --rm --gpus all cuju:0.2.1 python3 -m cuju.check
```

The image installs PyTorch from PyPI by default. The build argument `TORCH_INDEX_URL` names another
index, such as `https://download.pytorch.org/whl/cu126` for a host whose driver is older than the
CUDA version of PyPI's build.

```sh
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t cuju:0.2.1 .
```

The image needs only the NVIDIA driver on the host, since the libraries link the CUDA runtime
statically. It keeps the source distribution and the public tests in `/usr/share/cuju`, which a
published image keeps or offers under the terms in the License section of the
[README](../README.md#license), and it checks itself offline with `python3 -m cuju.check` or with
the tests.

```sh
docker run --rm --gpus all cuju:0.2.1 python -m pytest -p no:cacheprovider /usr/share/cuju/tests
```

## Apptainer image

The repository's `apptainer.def` makes an Apptainer image from the image that the `Dockerfile`
builds, for clusters that run Apptainer, version 1.2 or later. By default it takes the image
`cuju:0.2.1` from the local Docker daemon, so the Docker image is built first with that tag.

```sh
docker build -t cuju:0.2.1 .
apptainer build cuju.sif apptainer.def
```

An image from a registry, such as a site's own, comes in through the build arguments `BOOTSTRAP`
and `IMAGE`, so a cluster without Docker builds from an image that another machine pushed.

```sh
apptainer build --build-arg BOOTSTRAP=docker \
    --build-arg IMAGE=registry.example.org/cuju:0.2.1 cuju.sif apptainer.def
```

The image runs Python with its arguments, and `--nv` brings in the host's NVIDIA driver. Its test
runs the public tests from `/usr/share/cuju/tests` offline, and without `--nv`, as during the
build, the CUDA cases are skipped. `apptainer run-help cuju.sif` prints the same commands.

```sh
apptainer run --nv cuju.sif my_training.py
apptainer test --nv cuju.sif
```

## Windows through WSL2

WSL2 runs cuju on Windows with an Ubuntu distribution and the NVIDIA driver installed on Windows
itself, outside WSL, since WSL2 uses the Windows driver. Inside WSL, cuju installs as on Linux, PyPI
gives the CUDA build of PyTorch, and the CPU library works the same way.

`nvidia-smi` inside WSL lists the GPU when the Windows driver works, and
`nvidia-smi --query-gpu=name,compute_cap --format=csv` gives the compute capability to compare with
the wheel's table of architectures. The check of the CPU's instruction sets in `/proc/cpuinfo` works
inside WSL as well. A virtual environment, created with `python3 -m venv ~/cuju-venv`, holds PyTorch
and cuju.

Code and virtual environments belong in the Linux file system under `~/`, where file access stays
inside WSL, since on a Windows drive as WSL mounts it every file access crosses into Windows. WSL2
receives part of the machine's memory and processors by default. `memory=` and `processors=` under
`[wsl2]` in `%UserProfile%\.wslconfig` change its share, and `threads=0` uses the processors that WSL
has.

On a GPU with a few gigabytes of memory, one process with one batch at a time keeps the GPU fast.
The Windows driver moves memory that the GPU lacks into shared system memory, where a Linux driver
would fail the allocation, and every step then slows by a factor of about ten, as measured in
[Performance](performance.md).

## Install check

Two commands check an install.

```sh
python -c "import cuju; b = cuju.BatchEnvironment('5_vs_5', environment_count=4, device='cpu'); print(b.reset().shape)"
cuju check
```

The first command prints `torch.Size([4, 1, 115])`, four environments, one agent each, and the 115
numbers of `simple115v2`. The second ships in the wheel. It prints the versions, the processor, the
libraries and the GPUs it finds, checks the shipped data, plays the digest cases on the CPU and on
every GPU, and ends with `PASSED` or `FAILED`. Each line of its output is explained in
[Exactness](exactness.md). The check reports each backend it finds and plays its cases there, so on
a machine with a GPU the CUDA cases appear among the lines, and `--device cuda` requires them. From
a clone of the repository, `pytest tests/` checks more.

The check took about a minute on a machine with an RTX 4090 and an AMD EPYC 7452, 9 seconds for the
CPU cases and 53 for the GPU cases, and about 100 seconds on a laptop with an RTX 3050, 16 and 84.
The CPU cases depend on the processor and its threads. Each GPU case steps four environments, a
batch too small to fill a GPU, so the GPU cases depend on the time of one step of a small batch,
and they run again on every GPU of the machine. The first GPU batch also loads the kernels, and the
line of each case gives its time.

## Environment variables

cuju reads five environment variables, which override the libraries it loads, its logging, the
bundle cache and the NUMA layout.

| Variable | Effect |
| --- | --- |
| `CUJU_CUDA_LIBRARY` | the path of the CUDA library to load, in place of the one in the package |
| `CUJU_CPU_LIBRARY` | the path of the CPU library to load |
| `CUJU_LOG` | `1` logs one line per batch created, through the logger `cuju`, with the environments, the device, the memory, the library and the time it took |
| `CUJU_BUNDLE_CACHE` | `1`, or a directory, to keep the bundles built from scenario files, as in [Custom scenarios](custom-scenarios.md) |
| `CUJU_NUMA_NODES` | the NUMA node of each processor in order, such as `0,0,1,1`, in place of what the system reports, for `numa=True` under Linux |

The two library variables load a rebuilt library without a reinstall of the package.

## Troubleshooting

Each common install failure has a cause and a fix.

| Symptom | Cause | Fix |
| --- | --- | --- |
| CUDA reports that no kernel image is available for the device | the wheel holds no code for that GPU | a source build with `CMAKE_CUDA_ARCHITECTURES` set to it |
| an error that the processor lacks AVX2, BMI2 or FMA | a CPU older than Haswell or Excavator, or a VM that hides the flags | a machine with those instructions, or the flags enabled in the VM |
| an error that PyTorch sees no CUDA GPU | a CPU-only build of PyTorch, or a driver that the build of PyTorch does not accept | a CUDA build of PyTorch installed first, as in [PyTorch](#pytorch), or `device='cpu'` |
| `FileNotFoundError` for the CUDA library | the package holds the CPU library alone, as the wheels for Linux on Arm and for macOS do, or was built without CUDA | `device='cpu'`, or on Linux and Windows the toolkit installed and a rebuild |
| nvcc on Windows reports an unsupported Microsoft Visual Studio version | CUDA 12.9 refuses the MSVC of Visual Studio 2026 | the build run from the x64 Native Tools Command Prompt of Visual Studio 2022, as in [Windows](#windows) |
| `torch.compile` on a GPU under Windows fails with `TritonMissing` | PyTorch's wheels for Windows ship without Triton, which Inductor needs for GPU code | the step called eagerly, with the same results |
| nvcc fails on an unsupported host compiler | nvcc 12.0 to 12.3 supports GCC up to 12, and Ubuntu 24.04 defaults to 13 | `CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=g++-12"` |
| CMake cannot detect the GPU on a cluster login node | `native` needs a GPU at build time | `CMAKE_CUDA_ARCHITECTURES` set explicitly |
| a batch on the CPU is slower than expected in a container | another process shares the processors that `threads=0` counts | `threads` set to the processors the container was given |
| the first GPU step is slow | CUDA loads the kernel | expected, with timing from the second step |
| "the GPU is out of memory", with the bytes needed and free | the batch does not fit next to what else holds the GPU | the batch created before other large allocations, a call to `torch.cuda.empty_cache()`, or fewer environments |
| every step about ten times slower on Windows or WSL2 | the driver moved GPU memory into shared system memory | one batch per GPU and fewer environments, as in [Performance](performance.md) |
| any other failure | | an issue with the output of `cuju check`, since issues are welcome |

## See also

- [Exactness](exactness.md), the meaning of each line of `cuju check`
- [Backends](backends.md), the GPU and CPU backends and their fit with PyTorch
- [Performance](performance.md), the throughput of each backend and the cost of shared memory on Windows
- [Custom scenarios](custom-scenarios.md), scenario files and the bundle cache
- [Common topics](faq.md), short answers on diagnostics, illegal instructions and missing kernel images
