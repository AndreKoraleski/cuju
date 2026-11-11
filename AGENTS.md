# Notes for coding agents

cuju is a bit-exact, batched port of Google Research Football, GRF 2.10.3 at commit 3d9e754, for
CUDA and the CPU, from one C++ source, with a PyTorch-facing Python package. For the same seed and
actions, every state, observation, reward and done flag equals GRF's, bit for bit, on both backends.
Part one is for projects that use cuju. Part two is for changes to cuju itself.

## Part one, using cuju

### Install and check

```sh
pip install cuju           # brings PyTorch along when it is missing
cuju check                 # ends with PASSED when the machine plays GRF's matches
```

`pip install cuju` works directly on Linux on x86-64 and on Arm, Windows, macOS and WSL2. The CPU
backend runs on all of them, with AVX2, BMI2 and FMA on x86-64, and the CUDA backend on Linux on
x86-64, Windows and WSL2. PyPI's default PyTorch is a CUDA build on Linux and CPU-only on Windows
and macOS. Install PyTorch first from the index of pytorch.org only to pick another build, such as
another CUDA version or a CUDA build for Windows. Intel macOS takes PyTorch 2.2, where
`torch.compile` breaks the graph at each step, since the operators need PyTorch 2.4. Run
`cuju check` after installing and before trusting results on a new machine.

### The batch

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4096, device='cuda')
device = batch.torch_device
observation = batch.reset()
actions = torch.zeros((batch.environment_count, batch.agents), dtype=torch.int32, device=device)
observation, reward, done, information = batch.step(actions)
```

- Name a scenario as `<scenario>_l<left>_r<right>`, the agents of each side. `5_vs_5_l4_r0` plays
    four agents on the left against GRF's built-in AI.
- Pass `device` as `cuda`, `cuda:N`, a `torch.device`, `cpu` or `auto`. The default is `cuda`.
    `auto` takes the current CUDA device where PyTorch sees one, else the CPU. Both backends give
    the same bits.
- Read `batch.torch_device` for the device that holds every output.

### Shapes

| Tensor                               | Shape and type                                                  |
| ------------------------------------ | --------------------------------------------------------------- |
| observation, `simple115v2`           | `[environment_count, agents, 115]` `float32`                    |
| observation, `extracted`             | `[environment_count, agents, height, width, 4]` `uint8`         |
| observation, `raw`                   | a dictionary of `[environment_count, agents, ...]` tensors      |
| reward                               | `[environment_count, agents]` `float32`                         |
| done                                 | `[environment_count]` `uint8`, index with `done.bool()`         |
| `information['score_reward']`        | `[environment_count]` `int32`                                   |
| `information['final_observation']`   | the observation of the step that ended each episode             |
| actions                              | `[environment_count, agents]` `int32`                           |

`stacked=True` multiplies the last axis by 4. `squeeze=True` drops the agent axis for batches of one
agent each.

### Rules of use

- Clone each output that a later call needs. Every `reset` and `step` returns the same tensors and overwrites them.
- Build actions as `int32` on `batch.torch_device`. Such actions take no copy, and any other
    tensor is copied each step.
- Leave the reset to the step. The step that ends an episode starts the next. Use
    `batch.reset(mask=...)` to restart chosen environments and `batch.reset(seeds=...)` to seed
    them.
- Use `batch.get_state(index)` and `batch.set_state(index, state)` for GRF's state format.
- Close a batch with `batch.close()` when its memory is needed back.

### GRF code

`cuju.gfootball` mirrors GRF's import paths. Replace `gfootball` with `cuju.gfootball` in the
imports and GRF code runs unchanged, one match per environment. `cuju.env.create_environment` is
GRF's function with a `device` argument. Move only the training loop to `cuju.BatchEnvironment` for
speed.

### Further reading

- [docs/index.md](docs/index.md), the start of the documentation.
- [docs/api.md](docs/api.md), every public name.
- [docs/migrating-from-grf.md](docs/migrating-from-grf.md), every GRF name and its counterpart.
- [docs/rendering.md](docs/rendering.md), the renderer, the package cuju-render, and how cuju
    loads it.
- [docs/recipes.md](docs/recipes.md), [docs/backends.md](docs/backends.md) and
    [docs/performance.md](docs/performance.md), speed on each device.
- [examples/](examples), runnable scripts and the quickstart notebook.

## Part two, developing cuju

Read [CONTRIBUTING.md](CONTRIBUTING.md) before a change. This part is the short form.

### Exactness

- Keep every output bit for bit what it is with every option at its default. Make a new feature
    opt-in, with no effect when off. Leave out anything that changes what an episode is.
- Follow GRF's source, with the same operations in the same order, its choice of `float` or
    `double` for each value, and its quirks, each marked with a `GRF:` comment.
- Keep floating-point results fixed. Fast math, contraction and the reordering of a simulation
    expression are all ruled out, so the build keeps `-ffp-contract=off` and `--fmad=false` and
    refuses flags that change results. Leave that refusal in place.
- Call the math functions ported from glibc 2.39 in `include/cuju/libm/`, never the system's.
- Make every change work on both backends. Whatever runs on one runs on the other with the same
    bits.
- Keep `pytest tests/` passing on the CPU after a change to `include/` or `src/`, and on a GPU where
    one is available. A changed digest is a changed result, so leave `python/cuju/data/digests.json`
    as it is.
- Run `python ci/data_checksums.py` to rewrite `SHA256SUMS` after changing
    `python/cuju/data/animations.bin` or a shipped bundle.

### Layout

The engine is header-only, in `include/cuju/`. `simulation/` is GRF's game, `batch/` the batch and
its outputs, `state/` GRF's state format, `cpu/` and `gpu/` the two backends. Write logic once, in
the shared headers, and let a backend only schedule it. `src/` holds the C API and the two library
sources, and `python/cuju/` the package.

In `python/cuju/`, `__init__.py` exports the public names and each public module sits at the top.
`_batch/` holds the implementation of `BatchEnvironment`, from loading the library to the pickles of
a state. `gfootball/` stands in for GRF's package, and `data/` holds the bundles, the animations and
the digests. `state.py` reads single fields of a state, `_renderer.py` names the renderer, the
package cuju-render, and `__main__.py` is the command `cuju`, one entry per subcommand.

### Build and test

```sh
pip install .                                  # both libraries where nvcc is found
CMAKE_ARGS=-DCUJU_CUDA=OFF pip install .       # the CPU library alone
pytest tests/                                  # the public suite, a couple of minutes
cuju check                                     # the shipped self-check
ruff check && ruff format --check              # Python
clang-format-18 --dry-run -Werror <files>      # C++ and CUDA
```

### Style

- Indent with four spaces everywhere, in C++, Python, Markdown lists and code blocks.
- In C++, name constants in `UPPER_SNAKE_CASE`, variables in `camelCase`, functions and types in
    `PascalCase`. Write `//` comments only, sparse, saying why. Prefix GRF facts with `GRF:`,
    exactness notes with `Exact:`, speed notes with `Speed:`.
- In Python, follow PEP 8 through `ruff`, with line length 100, single quotes and Google-style
    docstrings. Raise the classes of `cuju.errors` rather than bare built-ins.
- Start new files with `Copyright 2026 The cuju Authors` and `SPDX-License-Identifier: Apache-2.0`.
- Write documentation in plain, precise sentences, with GRF's names for GRF's things. Prose uses
    only words, commas and periods.

### Leave alone unless the pull request states a reason

- `python/cuju/data/`, the animations, the bundles, the digests and their checksums.
- `include/cuju/libm/`, LGPL ports of glibc, and `LICENSE`, `NOTICE`, `LICENSES/`.
- The floating-point guards in `CMakeLists.txt` and `include/cuju/floating_point.h`.
- The C API's layout. Append fields of `cuju_configuration` only, raise
    `CUJU_CONFIGURATION_VERSION` with each addition, and start each new function on its own line in
    `src/*_api.h`.
- The ownership of views. A view or buffer that a batch hands out holds `batch._native`, never the
    batch, so that a dropped batch is freed.

### Commits

Write one line in the imperative, at most 72 characters, with no body, such as
`Port the flight of the ball`.
