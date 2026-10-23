# Exactness

In cuju, **exactness** is the property that every output equals, bit for bit, the output of Google
Research Football, or GRF, for the same seed and actions. The claim covers every state, observation,
reward and done flag, against GRF 2.10.3 without rendering on a reference build, GCC for x86-64
Linux with glibc 2.39 on a processor with FMA, and it holds for both backends and any number of
threads.

The claim rests on code that performs GRF's floating-point operations in GRF's order, on ports of
glibc's math functions, and on internal testing. Two public tools check it on any machine,
`python -m cuju.check`, which compares a build with the digests of its release, and
`examples/compare_with_grf.py`, which plays cuju next to a GRF installation. Issues reporting a
difference are welcome, and a reproduced difference is treated as a bug.

## Claim

The claim is stated in one sentence.

> For the same seed and actions, every state, observation, reward and done flag of cuju equals, bit
> for bit, what GRF 2.10.3, commit 3d9e754, produces without rendering, as built by GCC for x86-64
> Linux with glibc 2.39 on a processor with FMA. The two backends give the same bits, whatever the
> number of threads. The checkpoint reward takes the precision of the numpy beneath that GRF through
> the option `numpy_legacy`.

"Every state" means the whole state of the match, as GRF's engine serializes it, with every player,
the ball, the officials, the animations in progress, the built-in AI's memory and the clocks. Equal
states at every step are a stricter test than equal observations, since the observation shows only
a small part of the state and two different matches could share it for a while.

The claim is empirical. It rests on testing, and testing, however extensive, is finite. It is stated
narrowly, naming the version, the build and the outputs, so that a single counterexample refutes it
and the conditions of a counterexample are explicit. The known output differences are the three
cases listed in [Differences from GRF](differences-from-grf.md#output-differences), each with its
bound.

## Reference build

GRF's engine is C++ that calls the math library for sines, cosines, arc tangents and powers. The
last bits of those functions differ between C libraries, and sometimes between versions of one
library. A different last bit in one `atan2f` changes which animation a player picks, and from there
the whole match. The matches of GRF therefore depend on the platform GRF was built on, and the claim
names one, GCC on x86-64 Linux with glibc 2.39, the toolchain of Ubuntu 24.04.

cuju carries ports of glibc 2.39's own implementations of those functions, compiled into both
backends in place of the math library of the system, so its results are the same on every machine
it runs on. A GRF built against another glibc can play other matches, in which case cuju differs
from that build while it equals the reference. `examples/compare_with_grf.py` shows which case a
given GRF installation falls in.

## Scope

The claim covers GRF without rendering, the reference build, the GPU architectures of the wheel,
and the options of cuju that extend GRF.

cuju equals GRF without rendering. A GRF that renders plays other matches, since its referee acts
out each stoppage and its first rendered reset draws one more random number, so the representation
`pixels`, which requires rendering, stays with GRF. The claim refers to the reference build of GRF,
for the reasons given above.

The wheel carries machine code for the GPU architectures it was checked on. Another GPU runs a
source build, which `python -m cuju.check --long` checks.

The checkpoint reward of `rewards='scoring,checkpoints'` depends on the numpy beneath GRF, since
GRF adds it with numpy, and numpy 2 changed how a Python float combines with a `float32`. The
option `numpy_legacy` selects the precision, `True` for a GRF on numpy before 2, the default of a
batch, and `False` for a GRF on numpy 2 or later. Each setting equals its GRF bit for bit, so the
claim holds for the reward with the setting that matches the numpy of the GRF under comparison.
`create_environment` and `examples/compare_with_grf.py` take the setting from the installed numpy.
The scoring reward, the observations, the states and the done flags are the same under both
settings, as described in
[Rewards and actions](rewards-and-actions.md#precision-under-numpy-1-and-numpy-2).

Options that only read, such as `with_raw`, leave every result unchanged. Resets of some
environments with `reset(mask=...)`, or with chosen seeds, play GRF's episodes, each equal to one
GRF run.

## Determinism in the code

Floating-point arithmetic is deterministic when every operation is the same operation in the same
order. The code follows that rule in six ways.

Expressions follow GRF's source, including its evaluation order and its choice of `float` or
`double` for each value. The compilers fuse a multiplication and an addition into one FMA
instruction only where GRF's code does, with `-ffp-contract=off` for the host and `--fmad=false`
for nvcc, and the build stops with an error at flags that would loosen this, such as `-ffast-math`.
On the GPU, where approximate division and square roots are the default, the build selects the IEEE
versions.

The math functions are those of glibc 2.39, ported for the host and the device. Where GRF depends
on the behaviour of x86, such as the result of a conversion from `float` to `int` for values out of
range, or the sign of zero, the code reproduces that behaviour on both devices. Every call into the
library computes in the default floating-point mode, so a program that flushes subnormal numbers to
zero gets the same bits.

Where GRF's code has a quirk, the port keeps it, and a comment marked `GRF:` in the source explains
each one.

## Testing

cuju is tested internally against GRF. Issues are welcome, and a difference from GRF is the most
useful report. The repository provides two checks that run on any machine, a self-check of a build
and a script that plays cuju next to a GRF installation.

The repository also holds a public suite, independent of GRF. `pytest tests/` checks expected
digests of every shipped scenario, the shapes, the independence from the number of threads, the
agreement of the two backends, and a state round trip, among other properties. A digest is a
SHA-256 over every output of a short run, so one changed bit anywhere changes it. The digests fix
the outputs of the release. A passing suite therefore shows that a build reproduces the release bit
for bit, with its compiler, flags and GPU. The comparison with GRF is the role of
`examples/compare_with_grf.py`.

The expected digests are one file, the same on every platform. For 0.1.0 the self-check and the
public tests passed on Linux on x86-64 and on Arm, on Windows, on macOS on Apple silicon and on Intel
processors, and in WSL2, built by GCC and Clang, by MSVC and by Apple Clang, and on the GPU under
Linux, WSL2 and Windows. Each of these platforms therefore plays the bits of the reference build.
On Arm, where the conversion of a float to an integer and the floating-point environment differ
from x86-64, the code follows the behaviour of x86-64 explicitly.

Two properties of the batch matter for mixed batches and for the `difficulty` option, and both hold
by construction. The first is that an environment plays the same bits whatever else its batch
holds, whether other bundles, other agent counts, a wider agent axis, another batch size or another
number of threads. Each environment keeps its own world and its own random generator, and its step
reads only those and the shared tables, which nothing writes, so its match depends on nothing its
neighbours do. The second is that `difficulty=[(l, r)]` equals a scenario file. It plays exactly the
matches of GRF's file of that scenario with `left_team_difficulty = l` and
`right_team_difficulty = r`, as cuju builds it, and that file, as GRF ships it, builds byte for byte
into the shipped bundle.

## Checking a build

The check that a build requires depends on what could have changed. A wheel on a GPU that it covers
requires only the self-check, which takes one to two minutes, as timed in
[Install check](installation.md#install-check). A source build, other compiler flags or a GPU
architecture outside the wheel require the long cases and the tests as well, since a compiler can
change a bit without any error. A suspected fault in cuju itself is settled by a comparison with a
GRF installation, because every other check compares cuju with its own release. The checks range
from the self-check to a full comparison with GRF.

### Self-check

After any install, the check is `python -m cuju.check`. It ships in the wheel and needs only
PyTorch. It prints the versions, the processor and its flags, the CPU quota, each library with its
SHA-256, the GPU code that the CUDA library holds and every GPU with the code it runs. It then
checks the shipped bundles and animations against `data/SHA256SUMS`, plays the digest cases on the
CPU and on every visible GPU, and prints `PASSED` or `FAILED`. Its exit status is 0 when everything
that ran matched, and 1 otherwise. Four forms of the command cover the usual cases.

```sh
python -m cuju.check                  # every short case, the CPU and every GPU
python -m cuju.check --device cuda    # one backend, which must then be present
python -m cuju.check --long           # also whole matches, several minutes on the CPU
python -m cuju.check --list           # the diagnostics and the cases, without playing
```

A failing digest means that the compiler, the flags or the GPU changed the results, and that build
differs from the release. On a GPU whose architecture lies outside the wheel, a source build also
passes `--long` before use.

After a source build, `pytest tests/` runs the same digests and more. Skipped tests, such as the
CUDA cases on a machine without a GPU, count as unchecked rather than passed.

### Package files

`python/cuju/data/SHA256SUMS` lists the SHA-256 of `animations.bin` and of every shipped bundle, in
`sha256sum`'s format, and `sha256sum -c SHA256SUMS` in the installed `cuju/data` directory checks
them without Python. The bundles are those that GRF's own scenario files build, byte for byte.

### Any configuration

`cuju.digest(assets, **options)` is the self-check's digest of any configuration.
`cuju.ActionLog(assets, **options)` creates a batch and journals every call that changes a match,
with its arguments and a running digest. `save(path)` writes the log, and
`cuju.replay(path, device=...)` plays it again on either backend and names the first call whose
outputs differ. A log made on the GPU replays on the CPU, as in the example. The options that change
no output, `device`, `library`, `threads`, `pin_memory`, `allocator`, `pin_threads`, `numa`, `debug`
and `check_actions`, stay out of the log, so the machine that replays it passes its own to
`cuju.replay`, and a log that an earlier version wrote with them replays without them.

```python
import cuju
import torch

log = cuju.ActionLog('5_vs_5_l4_r0', environment_count=8, device='cuda')
log.reset()
for _ in range(1000):
    log.step(torch.randint(0, 19, (8, 4), dtype=torch.int32, device='cuda'))
log.save('run.zip')
assert cuju.replay('run.zip', device='cpu') == log.hexdigest()
```

### Comparison with GRF

Against a GRF installation, the same seed and actions play in both, as described in [Episodes and
seeds](episodes-and-seeds.md#seeds), or a state moves from one to the other and the next steps are
compared, as described in [States](states.md#format). `examples/compare_with_grf.py` does both, by
seed and from a state that GRF saves midway. It compares bit for bit the observation of every agent
in the representation that `--representation` selects, `raw` included and compared key by key in
GRF's types, the rewards, the done flags and the match part of the state at the interval that
`--state-interval` sets. `--agent submission.py:agent` plays a given policy, or a Kaggle submission,
in both engines. The script prints the first difference and exits with 1, or with 0 when everything
is equal.

```sh
python examples/compare_with_grf.py --scenario 5_vs_5 --left 4 --right 0 --steps 3000
```

For a trained policy, its win rate and goal difference against the built-in AI are compared between
the two engines over the same seeds given to `reset(seeds=...)`. When the engines agree, these agree
as well.

## Reporting a difference

Issues are welcome for any difference between cuju and GRF. A difference usually appears thousands
of steps into a run, and its cause is found most directly from one step before it, so the state at
the last step where the two engines agree carries the most information. A report of a difference is
most useful with seven items.

- The whole output of `python -m cuju.check`, which names the versions, the device, the libraries
    and their SHA-256, and whether the build reproduces the digests.
- The library, the wheel's or a source build with its compiler and flags.
- The call that created the batch, with every option.
- The batch `seed` and the index of the environment, and any `seeds` passed to `reset`.
- The state from `get_state` at the last step where the two agree.
- The actions from there on, the step where the two first differ, and what differs, the
    observation, the reward, the done flag or the state.
- The build of the GRF under comparison, its version, the operating system, glibc and GCC.

A difference that reproduces is a bug, and its fix brings cuju to GRF's result.

## See also

- [Differences from GRF](differences-from-grf.md)
- [Episodes and seeds](episodes-and-seeds.md)
- [States](states.md)
- [Backends](backends.md)
- [Common topics](faq.md)
