# API reference

The **application programming interface** of cuju comprises the Python package `cuju` and a C API
that both native libraries export. Every operation on a batch is reached from
`cuju.BatchEnvironment`, which creates a batch, steps and resets it, exposes its raw observations
and states, and raises the exceptions of `cuju.errors`. `cuju.MultiDeviceBatch` spreads one batch
over several devices, and submodules add the interfaces and tools of Google Research Football, or
GRF, around it.

Each option of a batch has a type, a default and, where one exists, a counterpart among the options
of GRF. In the shapes of tensors, `E` stands for `environment_count` and `A` for `batch.agents`.

## Batch creation

`cuju.BatchEnvironment(assets, **options)`

The options fall into four groups. The scenario, episode and output options determine which matches
a batch plays and what it returns, so two runs are comparable when they agree on all three groups.
The run options determine how the work is done and leave every output unchanged, so they may differ
between machines.

### Scenario options

The scenario options select the matches, their agents and their difficulty.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `assets` | `str` or `list[str]` | required | `env_name` | A shipped scenario with agents, such as `5_vs_5_l4_r0`, a bare scenario name, which gives one left agent, the path of a bundle, or the path of a scenario file. A list holds several. Described in [Scenarios](scenarios.md). |
| `assignment` | integers or `None` | `None` | | The bundle index of each environment, `[E]`. `None` gives every environment the first bundle. |
| `agent_counts` | list of pairs or `None` | `None` | `number_of_left_players_agent_controls`, `number_of_right_players_agent_controls` | One `(left, right)` pair per bundle, each `None` for what the name gives. |
| `difficulty` | list of pairs or `None` | `None` | `left_team_difficulty`, `right_team_difficulty` | One `(left, right)` pair per bundle, each from 0 to 1 or `None` for the scenario's. |
| `physics_steps_per_frame` | `int` | `10` | `physics_steps_per_frame` | The ticks in one round of a step. |
| `animations` | `str` or `None` | `None` | | The path of `animations.bin`. `None` takes the shipped tables. |

### Episode options

The episode options set the number of environments, their seeds and the sequence of episodes.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `environment_count` | `int` | `1024` | | The number of environments. |
| `seed` | `int` | `0` | | Environment `index` plays as GRF after `random.seed(seed * 2**32 + first_index + index)`, from `0` to `2**64 - 1`. Described in [Episodes and seeds](episodes-and-seeds.md). |
| `first_index` | `int` | `0` | | The index of environment 0 in the seed rule. Every index stays below `2**32`. |
| `episode_number` | `int` | `0` | `episode_number` | The number of the episode before the first. Its parity selects the form. |
| `auto_reset` | `bool` | `True` | | Starts the next episode inside the step that ends one. |
| `game_engine_random_seed` | `int`, sequence or `None` | `None` | `game_engine_random_seed` | One seed for every episode. A sequence gives one per environment, `-1` to draw. |

### Output options

The output options set the representation, the rewards and the action set.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `representation` | `str` | `'simple115v2'` | `representation` | `simple115`, `simple115v2`, `extracted` or `raw`. Described in [Observations](observations.md). |
| `with_raw` | `bool` | `False` | | Also writes the `raw` arrays, for `raw()` and `final_raw()`. |
| `stacked` | `bool` | `False` | `stacked` | Stacks the last four observations along the last axis. |
| `channel_dimensions` | `(int, int)` | `(96, 72)` | `channel_dimensions` | The width and height of the SMM. |
| `squeeze` | `bool` | `False` | the single-agent wrappers | Drops the agent axis. Requires one agent in every environment. |
| `rewards` | `str` | `'scoring'` | `rewards` | `scoring` or `scoring,checkpoints`. Described in [Rewards and actions](rewards-and-actions.md). |
| `numpy_legacy` | `bool` | `True` | | Adds the checkpoint reward in the precision of numpy before version 2. |
| `action_set` | `str` | `'default'` | `action_set` | `default` with 19 actions, `v2` with 20 or `full` with 33. |

### Run options

The run options select the device and the execution resources and leave every output unchanged.
[Backends](backends.md) describes each of them.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `device` | `str`, `int` or `torch.device` | `'cuda'` | `cuda`, `cuda:N`, `cpu` or `auto`. |
| `threads` | `int` | `0` | The CPU threads, the caller included. `0` takes one per processor the process may run on, up to its cgroup CPU quota. |
| `library` | `str` or `None` | `None` | The path of the library of `device`. `None` takes `library_path(device)`. |
| `allocator` | `str` | `'cuda'` | The source of GPU buffers, `cuda`, or `torch` for the caching allocator of PyTorch. |
| `pin_threads` | `bool` | `False` | Binds each CPU worker to a processor of its own. |
| `numa` | `bool` | `False` | Forms one group of CPU workers per NUMA node, each with its own environments. |
| `pin_memory` | `bool` | `False` | Page-locks the output buffers of a CPU batch for asynchronous copies to the GPU. |
| `check_actions` | `bool` | `False` | Checks actions in GPU memory against `[-action_count, action_count)` before each step, at the cost of a wait for the device. The CPU checks every action anyway. |
| `debug` | `bool` | `False` | Waits for the device after every `step`, `step_async`, `reset_done` and `reset` of a GPU batch, so that a failure on the device raises at the call that caused it, and checks the actions as `check_actions` does. The CPU ignores it. |

The two checking options leave the outputs of every step that runs unchanged. They differ from the
other run options in one respect, since an action out of range in GPU memory then raises an error
where it would otherwise play as `idle`.

The constructor raises `ChoiceError` for an unknown `representation`, `action_set`, `device` or
`allocator`, `ConfigurationError` for malformed or inconsistent options, `MissingFileError` for an
asset that names nothing or a missing library, `DeviceError` for a GPU that PyTorch does not see or
a CPU without AVX2, BMI2 or FMA, and `LibraryError` when the library refuses the batch. A GPU batch
that runs out of memory at creation raises `LibraryError` with the memory it requested, the memory
it already held and the free memory of the GPU. A GPU batch whose request exceeds the free memory of
the GPU at the moment of creation, yet is granted, issues a `UserWarning` that names both amounts.
For the first batch of the process on a GPU, the request includes the local memory that the creation
reserves for the kernels. With `allocator='torch'`, the memory that the cache of PyTorch holds
unused counts as free. Under Windows and WSL2 the warning is the main sign that the driver placed
the batch in shared system memory, as described in [Performance](performance.md).

`cuju.memory_estimate(assets, **options)` returns the bytes that a batch with the same arguments
would request, without creating it, and `batch.memory_bytes` returns the bytes that a batch has
requested so far. Both count device memory on the GPU and host memory on the CPU, the buffers,
worlds and tables of the batch. The estimate is an `int` with the attribute `local_memory_bytes`,
the local memory that the driver reserves for the kernels when the process creates its first batch
on the GPU and then holds until the process ends. The library computes it from the local memory of
each kernel, the current stack of the device and the threads that the GPU runs at a time, and it is
0 once the process has created a batch on that GPU, and always 0 on the CPU. The estimate equals the
`memory_bytes` of the batch right after its creation. It reads the bundles and animation tables as a
creation does, takes a fraction of a second, and raises the errors that the constructor raises for
the same arguments. A few later calls add a small buffer each, on the GPU the first `reset` with
seeds, a reverse flag or a host mask, and on the CPU the first `step_async`.

```python
import cuju

per_environment = cuju.memory_estimate(
    '5_vs_5_l4_r0', environment_count=2, device='cpu'
) - cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1, device='cpu')
```

A batch dropped without `close()` is freed with its last tensor, and `close()` frees it at once.
Explicit closing suits the GPU, where the memory of a batch is large and the release of its last
tensor can be hard to locate. A batch is also a context manager, so
`with cuju.BatchEnvironment(...) as batch:` closes it at the end of the block, an exception
included, and entering a closed batch raises `ClosedError`.

## Loop methods

A program calls `reset` once, then `step` in a loop. The other methods serve a step split in two on
the CPU, the reset of ended environments without the automatic reset, and the current observation
without a step.

| Method | Returns | Meaning |
| --- | --- | --- |
| `reset(seeds=None, reverse=None, mask=None)` | observation | Starts the next episode of every environment, or of those where `mask` is true. `seeds` takes one per environment, as a list, tuple, array or tensor, or one integer for all, each from `0` to `2**32 - 1` or `-1` to draw. `reverse` takes the team-order flag in the same forms, read at the first reset of an environment. |
| `step(actions)` | `(observation, reward, done, information)` | Runs one step. `actions` is `[E, A]` of integers, or `[E]` with `squeeze`, and a contiguous `int32` tensor on the device of the batch is read in place. Releases the GIL while the library runs. |
| `step_async(actions)`, `step_wait()` | `None`, then as `step` | A step in two halves. On the CPU it runs in a thread of the batch meanwhile, and on the GPU `step_async` enqueues the step as `step` does. |
| `reset_done()` | `None` | Starts the next episode of every environment that ended, for `auto_reset=False`. |
| `observation()` | observation | The current observation, as the last call returned it. |
| `close()` | `None` | Frees the batch at once. Repeated calls are harmless. |
| `__enter__()`, `__exit__()` | the batch, then `None` | The context manager, which closes the batch at the end of a `with` block, an exception included. |

`step` returns the observation, the reward, the done flags and a dictionary of information with two
entries, whose shapes and types depend on the representation.

| Output | Shape and type |
| --- | --- |
| observation | `[E, A, 115]` `float32` for the vector representations, `[E, A, height, width, 4]` `uint8` for `extracted`, a dictionary of tensors for `raw`. A stack multiplies the last axis by 4 |
| reward | `[E, A]` `float32` |
| done | `[E]` `uint8` |
| `information['final_observation']` | as the observation, meaningful where `done == 1` |
| `information['score_reward']` | `[E]` `int32`, the scoring reward of the left team |

`step` raises `ActionRangeError` for an action out of range in host memory, or in GPU memory with
`check_actions` or `debug`, `TypeError` for floating-point actions, `ArgumentError` for actions of
another shape, `UndefinedPathError` after an environment reached a path that GRF leaves undefined,
`DeviceError` when the device fails in a `debug` batch, `ClosedError` on a closed batch, and
`LibraryError`, a `RuntimeError`, before the first reset. `step_async` checks its actions in the
same way, and `step_async`, `reset` and `reset_done` raise `DeviceError` when the device fails in a
`debug` batch. Every returned tensor shares memory with the batch and is overwritten by the next
`reset`, `step` or `set_state`. Actions, masks and seeds in host memory, page-locked or not, may be
changed as soon as `step`, `step_async` or `reset` returns, since a GPU batch copies page-locked
ones before the device reads them.

`step` and `reset_done` are also the PyTorch operators `cuju::step` and `cuju::reset_done`, so a
function that steps a batch compiles under `torch.compile` with `fullgraph=True`.

The errors of `step` fall into two kinds. A wrong type, a wrong shape, an action out of range in
host memory and a closed batch are errors of the calling code, raised before any environment steps,
so the batch keeps its state and the corrected call succeeds. Actions already on the GPU go to the
step as they are by default, so that a step never waits for the device, and an action out of range
there plays as `idle`. With `check_actions=True`, or with `debug=True`, the batch checks those
actions on the device before the step and raises `ActionRangeError` naming the first action out of
range, its agent and its environment, before any environment steps. The check reads its result on
the host, so every step waits for the device, and a CUDA graph capture skips it. As with actions in
host memory, the slots of padded agents go unchecked.

`UndefinedPathError` reports a match that reached a path that GRF leaves undefined. On the CPU the
library raises it from the step. On the GPU the device traps, which ends the CUDA context of the
process, so the next CUDA call reports the failure and the process restarts. A long run therefore
saves states with `get_state` at intervals, as described in
[Episodes and seeds](episodes-and-seeds.md) and [States](states.md). A GPU batch enqueues its work
and returns, so a failure on the device usually surfaces at a later call that waits. With
`debug=True` the batch waits after every step and reset and raises `DeviceError` at the call that
caused the failure, which locates a fault at the cost of the waits.

## Raw observations

Two methods return the `raw` arrays of GRF as tensors.

| Method | Needs | Returns |
| --- | --- | --- |
| `raw()` | `representation='raw'` or `with_raw=True` | The `raw` arrays of the last step, a dictionary of tensors. |
| `final_raw()` | the same | The `raw` arrays of the step that ended each episode. |

## States

Two methods save and load the state of one environment.

| Method | Meaning |
| --- | --- |
| `get_state(index, stacked=False)` | The state of one environment as `bytes`, in the format of GRF. `stacked=True` adds the stack of four of GRF. |
| `set_state(index, state, stacked=False)` | Sets one environment to a state of cuju or of GRF, rewriting the returned tensors in place. |

`set_state` raises `StateError` for bytes that fail to load, a state of another scenario, agents,
difficulties or end flags, naming the fields that differ, or a stack that differs from that of the
batch, and `EnvironmentIndexError` for an index outside the batch. The format is described in
[States](states.md).

## Attributes

A batch exposes its configuration and layout as attributes.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `device` | `str` | `cuda` or `cpu`, the backend |
| `torch_device` | `torch.device` | the device of the buffers, such as `cuda:0` |
| `environment_count` | `int` | the number of environments |
| `threads` | `int` | the CPU threads, the caller included, 0 on the GPU |
| `agents` | `int` | the room for agents in each environment |
| `players` | `int` | the room for players in each team array of `raw` |
| `action_count` | `int` | the actions of the action set |
| `left_agents`, `right_agents` | `[E]` `int32` | the agents of each team of each environment |
| `agent_mask` | `[E, A]` `bool` | the agents that exist |
| `left_players`, `right_players` | `[E]` `int32` | the players of each team, from the view of a left agent |
| `difficulty` | `[E, 2]` `float32` | the difficulties that play, the scenario's where nothing overrides them |
| `squeeze`, `stacked`, `channel_dimensions`, `auto_reset`, `check_actions`, `debug` | | the options of the batch |
| `memory_bytes` | `int` | the bytes of memory that the batch has requested so far, device memory on the GPU and host memory on the CPU |
| `single_observation_space`, `single_action_space` | spaces | one environment, as the spaces of GRF, with a `None` observation space for `raw` |
| `observation_space`, `action_space` | spaces | the whole batch |

The spaces require `gymnasium`, installed with the extra `spaces`, or `gym`.

## Several devices

`cuju.MultiDeviceBatch(assets, devices, environment_count=1024, split=None, output_device=None, assignment=None, game_engine_random_seed=None, **options)`
holds one batch per device and steps them together. Environment `i` plays as environment `i` of one
`BatchEnvironment`, whatever the split, through `first_index`. `devices` may mix GPUs and the CPU,
`split` sets the environments of each part, by default an even split, and `options` are those of
`BatchEnvironment` except `device` and `first_index`, which each part takes from `devices` and
`split`. The outputs of `reset` and `step` are gathered on `output_device`, by default the first
device, and `batch.parts` holds the parts, each with its own outputs. Its `reset` takes `seeds` and
`reverse`, and `get_state` and `set_state` take the index of an environment over the whole batch.
`close()` closes every part, after which every other call raises `ClosedError`. The layouts are
described in [Backends](backends.md).

## Module functions

The package also exports functions and classes for scenarios, digests and action logs.

| Name | Returns or does |
| --- | --- |
| `cuju.bundled_scenarios()` | the names of the shipped scenarios, such as `5_vs_5` |
| `cuju.bundled_assets()` | the names of the shipped bundles, such as `5_vs_5_l0_r0` |
| `cuju.library_path(device='cuda')` | the path of the library that a batch on `device` loads |
| `cuju.memory_estimate(assets, **options)` | the bytes that `BatchEnvironment(assets, **options)` would request at creation, without creating it, with the reservation of a first GPU batch for the kernels as `local_memory_bytes` |
| `cuju.__version__` | the version of the package |
| `cuju.Digest()` | a running SHA-256 of tensors, dictionaries, lists and bytes, with `update`, `copy` and `hexdigest` |
| `cuju.digest(assets, device='cpu', steps=100, seed=7, environment_count=4, **options)` | the self-check digest of a configuration |
| `cuju.ActionLog(assets, **options)` | a batch whose calls are journaled, with `reset`, `step`, `reset_done`, `set_state`, `save(path)` and `hexdigest()` |
| `cuju.replay(path, device=None, **options)` | plays an action log again and returns its digest, raising at the first call that differs |

## Exceptions

Every class lives in `cuju.errors`, is exported by `cuju`, and derives from `cuju.Error` and from a
built-in type, so that `except ValueError` and similar clauses also catch it.

The double ancestry allows handlers at three levels of precision. `except cuju.Error` catches every
exception that cuju raises and no other, as suits a training script that logs and restarts. A
specific class, such as `StateError` around `set_state`, catches one failure. Code written against
the built-in types, such as code written for GRF, keeps catching the type it catches.

Each class derives from `cuju.Error` and from the classes in its second column.

| Class | Also a | Raised for |
| --- | --- | --- |
| `Error` | `Exception` | the base of all |
| `ConfigurationError` | `ValueError` | an option refused at creation. `scenarios.ScenarioError` subclasses it |
| `ChoiceError` | `ConfigurationError`, `KeyError` | an unknown `representation`, `action_set`, `device` or `allocator` |
| `ArgumentError` | `ValueError` | an argument of a call, such as actions of the wrong shape, or `reset(seeds=, reverse=, mask=)` of the wrong length or range |
| `ActionRangeError` | `IndexError` | an action outside `[-action_count, action_count)` |
| `EnvironmentIndexError` | `IndexError`, `RuntimeError` | an environment index outside the batch |
| `StateError` | `ValueError`, `RuntimeError` | a state that fails to load into its environment |
| `ClosedError` | `RuntimeError` | any call on a closed batch |
| `DeviceError` | `RuntimeError` | a GPU that PyTorch does not see, a CPU without AVX2, BMI2 or FMA, a CUDA failure |
| `UndefinedPathError` | `RuntimeError` | an environment that reached a path that GRF leaves undefined |
| `LibraryError` | `RuntimeError` | every other failure the library reports |
| `MissingFileError` | `FileNotFoundError` | a missing scenario file, bundle or library |

## Modules

Each submodule is imported by name, as in `import cuju.dumps`, and imports the packages it depends
on, such as numpy for `cuju.kaggle`, at that point. The core of the package depends on PyTorch
alone. `cuju.env` and `cuju.gfootball` need Gymnasium, from the extra `spaces`, or GRF's gym.
`import cuju` leaves Gymnasium and gym unloaded and registers the `GFootball-*-v0` ids with
Gymnasium, and with gym before 0.26, as soon as either is imported, before or after cuju, as
importing `gfootball` registers them in GRF.

| Module | Names |
| --- | --- |
| `cuju.scenarios` | `build_bundle(scenario, out, name=None, device='cpu', library=None, animations=None)`, `describe(scenario, episodes=...)`, `load_scenario(path)`, `cached_bundle(scenario, directory=None, device='cpu', library=None, animations=None)`, `cache_directory()`, `ScenarioError`, and the builder API of GRF, `Builder`, `ScenarioConfig`, `PlayerRole`, `Team`. Described in [Custom scenarios](custom-scenarios.md) |
| `cuju.env` | `create_environment(env_name, ..., *, device='cpu', threads=1, api='gym')` with the arguments and defaults of GRF, the gym 0.21 interface of GRF by default and the Gymnasium interface with `api='gymnasium'`, and `register_environments()` for the `GFootball-*-v0` ids. Described in [Migrating from GRF](migrating-from-grf.md) |
| `cuju.dumps` | `DumpWriter(batch, logdir, environments=None, write_full_episode_dumps=True, write_goal_dumps=False, dump_frequency=1)` with `record(actions)`, `close()` and `paths`, and `read_dump(path)` |
| `cuju.players` | `Bot(batch, agent)` and `Lazy(batch, agents)`, the `bot` and `lazy` players of GRF, each with `act()`, and `Bot.reset(mask=None)` |
| `cuju.raw` | `grf_observation(raw, environment, agent, own_players, other_players)`, `grf_observations(batch, environment, agents=None, final=False)` and `team_sizes(batch, environment, agent)`, the raw dictionaries of GRF in the types of GRF |
| `cuju.wrappers` | `MultiAgentToSingleAgent(batch, left_players, right_players, action_set=None, stacked=False, squeeze=False)`, the wrapper of GRF for scenarios that set `control_all_players` |
| `cuju.kaggle` | `players_raw(batch, environment, agent, final=False)`, `observation(batch, environment, agent, step=0, final=False)`, `KaggleAgent(function, batch, agent, configuration=None, agents=None)` and `evaluate(left, right=None, episodes=1, scenario='11_vs_11_kaggle', seed=0, device='cpu', threads=0, configuration=None)`, the observations and agents of Kaggle, and matches of Kaggle agents, each the first episode of its environment, as Kaggle plays them |
| `cuju.gfootball` | the import paths of GRF, `env` with `create_environment`, `config.Config`, `football_env.FootballEnv`, `football_action_set`, `observation_preprocessing`, `scenario_builder`, `wrappers`, `player_base`, `players`, `script_helpers`, and `replay`, `dump_to_txt` and `scenarios`, so that GRF code runs with its imports switched. Described in [Migrating from GRF](migrating-from-grf.md) |
| `cuju.check` | `cuju check`, with `[--long] [--device all\|cpu\|cuda] [--case NAME] [--list]` |
| `cuju.state` | `layout()`, every field of the match in memory by path, `gather(state, fields, frame='rest')`, the bytes of chosen fields of a state, `read(state, fields, frame='rest')`, the same as numpy arrays, `Field` and `FRAMES`. Described in [States](states.md#reading-fields) |
| `cuju.render` | `Recorder(batch, environment=0, title=None, teams=None, scenario=None)` with `capture(done=None)`, `save()` and `saved`, `watch(batch, environment=0, title=None, teams=None, backlog=False)` and `LiveSource`, `add_team(folder, name=None, format='11v11')`, `play(left, right, scenario='11_vs_11_kaggle', seed=None, title=None, device='cpu', ability=(None, None))`, `serve(port=8780, open_browser=False)` and `home()`, and the command `cuju render [serve\|team\|play\|sample]`, also `python -m cuju.render`. Described in [Rendering](rendering.md) |

`Recorder.capture(done)` saves an episode when it ends, and `Recorder.save()` saves the frames
captured so far. Both return a path or `None`. `LiveSource.close()` withdraws a source, and
`watch` replaces any source already attached to that batch. `home()` returns the renderer folder.

`play` runs a requested match and returns its recording path. Each team's saved control settings
choose the count, dynamic or fixed assignment, indices or roles, goalkeeper and action set.
The scenario may be bundled or one of the six renderer GRF MARL scenarios. Each entry of `ability`
sets GRF's team difficulty from 0 to 1, with `None` keeping that side's scenario value.
The browser also provides Play a game with keyboard and controller input, described in
[Rendering](rendering.md#playing-a-game).

## Command line

The package installs the command `cuju`, which `python -m cuju` also runs, with one subcommand per
tool. `cuju render` runs the renderer, `cuju check` the self-check, and `cuju replay` and
`cuju dump-to-txt` GRF's scripts for dumps, each with the arguments of its module, which
`python -m` also runs. `cuju help` lists the subcommands, `cuju help NAME` describes one, and
`cuju --version` prints the version. Each subcommand is an entry of `cuju.__main__.COMMANDS`, a
module whose `main(arguments)` parses the rest of the command line, so a later tool joins the
command as one more entry.

```sh
cuju help
cuju render serve --open
cuju check --device cpu
```

## Environment variables

The library variables select a rebuilt library without reinstalling the package, and the others
control logging, caching, NUMA placement and the renderer. All of them leave the outputs unchanged.

| Variable | Meaning |
| --- | --- |
| `CUJU_CUDA_LIBRARY` | the path of the CUDA library to load |
| `CUJU_CPU_LIBRARY` | the path of the CPU library to load |
| `CUJU_LOG` | `1` logs one line per batch created, through the logger `cuju` |
| `CUJU_BUNDLE_CACHE` | `1` or a directory, to cache the bundles built from scenario files |
| `CUJU_NUMA_NODES` | the NUMA node of each processor, for `numa=True` |
| `CUJU_RENDER` | `on`, or the index of an environment, gives every batch a live source for the renderer, and an empty value, `off`, `false` or `no` none, with an index past a batch's count taken as 0 |
| `CUJU_RENDER_BACKLOG` | any value but an empty one, `0`, `off` or `no` makes every live source read every step, so that a viewer starts with 10 seconds of history |
| `CUJU_RENDER_HOME` | the folder of the renderer's recordings, teams and live sources, `~/.cuju/render` by default |

## C API

Both libraries export the same C functions, declared in `src/environment_api.h` and the other
`src/*_api.h` headers, which define their exact signatures. The Python package is a thin layer over
them. The C API serves programs that run outside Python, such as an engine in another language or a
training system with its own runtime. It covers the engine alone, and the additions of the Python
package, such as the tensors, the spaces and the interfaces of GRF, are then built by the caller. A
call sequence proceeds in four stages.

1. `cuju_environment_create(&configuration)` with a `cuju_configuration` whose `version` is
    `CUJU_CONFIGURATION_VERSION`. It returns a handle, or null on failure.
2. `cuju_environment_reset`, then `cuju_environment_step` in a loop, with
    `cuju_environment_reset_done`, `cuju_environment_get_state` and `cuju_environment_set_state` as
    needed.
3. `cuju_environment_buffer(handle, name)` returns the address of a named output buffer, which stays
    valid and in place for the life of the handle.
4. `cuju_environment_destroy(handle)`.

A function that fails returns a non-zero status or a null pointer, and `cuju_last_error()` returns
the reason. On the CUDA library, the calls that take a stream run on that stream. `step` returns
without waiting unless asked to synchronize, `reset_done` never waits, and `reset`, `get_state` and
`set_state` wait for the device. The headers state the contract of each call. The `version` field of
`cuju_configuration` names the layout of the structure, and `cuju_environment_create` refuses a
configuration whose version differs from the `CUJU_CONFIGURATION_VERSION` of the library, which is
2 in cuju 0.1.0.

Three calls read the state format without a batch. `cuju_state_layout(&size)` returns the layout
of the match in memory as text, one field per line with its path, offset, size and type.
`cuju_state_gather(bytes, size, frame, offsets, sizes, count, out)` loads a match, the bytes of a
state after its pickle length and pickle, and copies the chosen fields into `out`, in the rest frame
for `frame` 0 or the frame of the pitch for 1. `cuju_state_padding(bytes, size, mask)` marks the
bytes of a match that GRF leaves uninitialized, so two states compare without them.

Four further calls serve memory planning and debugging.
`cuju_memory_estimate(&configuration, &bytes)` writes the bytes that `cuju_environment_create`
would request for the same configuration, without allocating them, and
`cuju_environment_memory_bytes(handle)` returns the bytes that a batch has requested so far.
`cuju_local_memory_bytes(&bytes)` writes the local memory that a creation on the current CUDA
device adds for the kernels, which is 0 once a batch exists there and on the CPU.
`cuju_environment_synchronize(handle, stream)` waits for the work of the batch on `stream` and fails
as the device failed, and it returns at once on the CPU. A library built with the CMake option
`CUJU_NVTX` wraps the creation, the resets, the steps and the state calls in NVTX ranges, as
described in [Backends](backends.md).

## See also

- [Concepts](concepts.md), for batches, bundles and environments
- [Recipes](recipes.md), for complete programs built on the package
- [Common topics](faq.md), for the behaviours that most often prompt questions
