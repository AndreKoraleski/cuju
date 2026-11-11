# cuju

**cuju** is a Python library that runs batches of matches of Google Research Football, or GRF, on
one NVIDIA GPU or on the cores of a CPU, with every match equal, bit for bit, to the match that GRF
2.10.3 plays for the same seed and actions. The GPU and CPU backends are compiled from one C++
source, and a single call advances every match of a batch and returns its observations, rewards and
done flags as PyTorch tensors.

As of 0.1.0, cuju ships the 19 scenarios of GRF, keeps the names of GRF's options, and offers GRF's
own interface for one environment beside the batched one. Release 0.1.0 runs on Linux on x86-64
and on Arm, on Windows, on macOS on Apple silicon and on Intel processors, and in WSL2, with a wheel
on PyPI for each. The CPU backend runs on every one of them and the CUDA backend on Linux on
x86-64, on Windows and in WSL2, as [Installation](installation.md) describes platform by platform.

## Installation and check

cuju installs from PyPI with pip, which also installs PyTorch when it is missing. On Linux with an
NVIDIA driver that supports the CUDA version of PyPI's default build of PyTorch, the one command
suffices, and [Installation](installation.md) gives the reasons to install a particular build of
PyTorch first. A check that ships with the package then confirms the install.

```sh
pip install cuju
cuju check
```

The check plays a set of fixed cases on the CPU and on every GPU it finds, compares each with the
digest of a build that matches GRF, and ends with `PASSED` when every case agrees.

## Usage

### CPU backend

A minimal loop creates a batch of 64 matches on the CPU and steps it 100 times with random actions.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu')
observation = batch.reset()  # [64, 4, 115], environments, agents, features

for _ in range(100):
    actions = torch.randint(0, 19, (64, 4), dtype=torch.int32)
    observation, reward, done, information = batch.step(actions)
    # reward [64, 4], done [64], and ended episodes have already restarted

print(observation.shape, reward.shape, done.shape)
# torch.Size([64, 4, 115]) torch.Size([64, 4]) torch.Size([64])
```

`5_vs_5_l4_r0` is GRF's five-a-side scenario with four agents on the left team and the built-in AI
on the right. Every environment runs its own match, and one `step` advances all 64. A match lasts
3,001 steps, so after 100 steps every episode is still running.

### GPU backend

The same code runs on the GPU with `device='cuda'` and the actions on the device. `device='auto'`
selects the GPU where PyTorch sees one and the CUDA library is installed, and the CPU elsewhere, and
`batch.torch_device` reports the choice. The GPU reaches its throughput with thousands of
environments, so a GPU batch is usually created with several thousand.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=8192, device='cuda')
observation = batch.reset()  # a CUDA tensor, [8192, 4, 115]
actions = torch.zeros((8192, 4), dtype=torch.int32, device='cuda')
observation, reward, done, information = batch.step(actions)
```

### Shared output buffers

Every call returns the same tensors and overwrites them in place. A loop that appends each
observation to a list therefore stores one tensor 100 times.

```python
history = []
for _ in range(100):
    observation, reward, done, information = batch.step(actions)
    history.append(observation)  # wrong, every entry shows the last step
```

An observation is kept with `history.append(observation.clone())`. The batch reuses its buffers
because, on the GPU, copying thousands of observations at every step would cost more than the
simulation itself, and a training loop that needs a copy makes it once, into its own rollout
buffer.

### Closing a batch

A batch holds its buffers until `close()`, or until the end of a `with` block, which closes it even
when the block raises.

```python
with cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu') as batch:
    observation = batch.reset()
```

## Terminology

The interface of cuju uses a small set of terms with fixed meanings.

| Term | Meaning |
| --- | --- |
| environment | One match with its episode logic, what GRF's `create_environment` returns. |
| batch | The environments that one call to `step` advances together, a `cuju.BatchEnvironment`. |
| bundle | A scenario turned into the data that a batch loads, a `.cjb` file. One ships per GRF scenario. |
| agent | A player whose actions come from the caller's policy. The built-in AI drives every other player. |
| episode | The steps from a reset to the end of a match or a drill. |
| step | One call to `step`, one action per agent, then a round of 10 ticks of 10 ms by default, repeated while the ball is out of play. |
| representation | The encoding of an observation, `simple115`, `simple115v2`, `extracted` or `raw`. |
| SMM | The super mini map, the `extracted` representation, four planes of 72 by 96 cells. |
| difficulty | A number from 0 to 1 per team, which scales its players and slows its built-in AI. |
| state | The whole match of one environment as bytes in GRF's format, from `get_state`. |

[Concepts](concepts.md) defines further terms and gives the reasons for the shape of the API.

## Topics

The topics of cuju fall into three groups. [Concepts](concepts.md), [Scenarios](scenarios.md),
[Observations](observations.md) and [Rewards and actions](rewards-and-actions.md) describe the
match, what an agent sees and what it is paid for. [Migrating from GRF](migrating-from-grf.md) and
[Differences from GRF](differences-from-grf.md) cover the port of code written for GRF, which keeps
every option name. [Backends](backends.md), [Performance](performance.md) and
[Recipes](recipes.md) cover throughput, with measured figures and the patterns that keep a device
busy.

Each topic has an article of its own.

| Article | Subject |
| --- | --- |
| [Concepts](concepts.md) | Environments, batches, bundles and agents, and the reasons for the shape of a batched API. |
| [Installation](installation.md) | The requirements of cuju, its platforms and the ways to install it. |
| [Migrating from GRF](migrating-from-grf.md) | The cuju name for each GRF option, method and wrapper. |
| [Scenarios](scenarios.md) | The matches a batch plays, the agents on each side and batches that mix scenarios. |
| [Custom scenarios](custom-scenarios.md) | Scenario files written in GRF's format. |
| [Observations](observations.md) | What an agent sees, in what shape and at what cost. |
| [Rewards and actions](rewards-and-actions.md) | What an agent is paid for and what it can do. |
| [Episodes and seeds](episodes-and-seeds.md) | The end of an episode and the match that each environment plays. |
| [States](states.md) | Saving, restoring and moving a match between cuju and GRF, and reading single fields. |
| [Rendering](rendering.md) | Watching matches in a browser, saved or live, with the package cuju-render. |
| [Backends](backends.md) | The GPU and CPU backends and their fit with PyTorch. |
| [Performance](performance.md) | Measured throughput and the size of a batch. |
| [Recipes](recipes.md) | Patterns for wiring a batch into training. |
| [Exactness](exactness.md) | The meaning of bit-exact in cuju and the way it is checked. |
| [Differences from GRF](differences-from-grf.md) | The places where cuju differs from GRF, and the reasons. |
| [Common topics](faq.md) | Short answers on results, errors, scaling and compatibility. |
| [API reference](api.md) | Every public name, its type and its default. |

## See also

- [README](../README.md), the overview of the project, its features and its license
- [Changelog](../CHANGELOG.md), the changes in each release
- [Common topics](faq.md), short answers to the questions that come up most
