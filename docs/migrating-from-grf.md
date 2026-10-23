# Migrating from GRF

**Migration from GRF** moves code written for Google Research Football, or GRF, onto cuju's engine
through one of three entry points, GRF's import paths in `cuju.gfootball`, GRF's
`create_environment` in `cuju.env`, and the batch, `cuju.BatchEnvironment`. The matches are the
same in all three, and every option keeps GRF's name. The first two entry points play one
environment per call with GRF's interface and run GRF's code as it stands, and the batch plays
thousands of environments per call and takes a rewrite of the training loop.

The batch changes three properties of a loop. Its outputs are shared tensors that the next step
overwrites, its episodes reset inside `step` with the last observation in `final_observation`, and
every output carries an environment axis. GRF's dumps, scripted players and scenario configuration
carry over to both kinds of environment. A result made with cuju compares directly with one made on
GRF when five matters agree, the options, the count of steps, the size of an update, the evaluation
and the version of GRF.

## Entry points

cuju has three entry points, which trade compatibility with GRF's code for throughput.

**GRF's import paths, in `cuju.gfootball`.** The imports change and the rest of the code stays as it
is. The code keeps calling `create_environment`, the wrappers and `observation_preprocessing`, as
it did with GRF.

**GRF's `create_environment`, in `cuju.env`.** The same function under a shorter path takes three
keyword-only arguments beyond GRF's, `device`, `threads` and `api`, the last of which chooses
Gymnasium's interface.

**The batch, `cuju.BatchEnvironment`.** Thousands of environments advance per call, and their
outputs arrive as PyTorch tensors. The throughput of cuju lies here, and the loop is written for a
batch.

The first two play one environment per call, as GRF does, so they keep GRF's architecture and the
throughput of one environment at a time. In return, GRF's code runs on them as it stands.
Evaluation scripts, a Kaggle submission, a notebook that inspects raw observations or a test that
compares with published numbers keep working, and code with known results confirms that cuju
reproduces them. The batch serves the part of a project where the time goes, the collection of
experience for training.

A port usually proceeds in two stages. The imports switch first, and the existing code runs, so that
any difference appears in code whose behaviour is known. The training loop then moves to a batch, a
rewrite of one function, while the rest of the project stays on GRF's interface for as long as it is
convenient. Both kinds of environment play GRF's matches, so the two coexist in one project.

## Import paths

`cuju.gfootball` mirrors GRF's import paths, `gfootball.env`, `gfootball.env.config.Config`,
`FootballEnv`, `football_action_set`, `observation_preprocessing`, `scenario_builder`,
`script_helpers`, `player_base`, `gfootball.scenarios`, `gfootball.replay` and `dump_to_txt`. Its
`wrappers` module holds every wrapper of GRF's module, each running GRF's code over cuju's
environment, and `create_environment` applies them itself from its arguments, as GRF's does. A
stack of wrappers built by hand over a raw environment therefore gives the outputs of
`create_environment` for the same arguments, bit for bit. Its `get_state` and `set_state` carry the
counts of `CheckpointRewardWrapper` and the frames of `FrameStack` in the state, as GRF's do, so the
stack and `create_environment` write equal bytes and load each other's states. Code written for GRF
runs unchanged once its imports point there, except where it renders or reaches into GRF's
internals, as listed in [Counterparts of GRF's other features](#counterparts-of-grfs-other-features).

These paths, and `cuju.env` beneath them, build GRF's spaces with Gymnasium, which the extra
`spaces` installs. As in GRF, `import cuju.gfootball` binds `cuju.gfootball.env`, which loads at
its first use, so the import itself succeeds on an installation without Gymnasium.

`FootballEnv(config)` plays the players of a `Config` as GRF's does, with at most one `agent`.
Without an agent, as in GRF's `play_game.py`, its players play every step of `env.step([])`, and
`reset` and `step` return GRF's observation of the match, the dictionary that GRF's dumps hold,
with the scoring reward as a `float32` scalar array. A configuration with no player at all leaves
the batch no raw observation to read, so its observation is `None`.

```sh
pip install 'cuju[spaces]'
```

A short program in GRF opens an environment and computes its SMM.

```python
import gfootball.env as football_env
from gfootball.env import observation_preprocessing

env = football_env.create_environment('11_vs_11_easy_stochastic', representation='raw')
observation = env.reset()
smm = observation_preprocessing.generate_smm(observation)
```

With cuju, the same code differs in its imports alone.

```python
import cuju.gfootball.env as football_env
from cuju.gfootball.env import observation_preprocessing

env = football_env.create_environment('11_vs_11_easy_stochastic', representation='raw')
observation = env.reset()
smm = observation_preprocessing.generate_smm(observation)
```

These paths run on cuju alone, and with GRF installed as well, the two run side by side in one
script. Each draws its seeds from Python's `random`, so two environments in one
process shift each other's seeds, as two GRF environments do, a behaviour covered in
[Episodes and seeds](episodes-and-seeds.md).

### Counterparts of GRF's other features

Almost all of GRF's interface works through these paths. The remaining features of GRF have
counterparts in cuju or in GRF itself, and they fall into three groups.

| GRF | Counterpart | Reason |
| --- | --- | --- |
| `tracker_setup` | `get_state` at the steps to inspect | GRF's debugging tracker of its engine |
| `env.unwrapped._env` and the private attributes of the players | the public API, or `get_state` | GRF's internals, with no core object per environment in cuju |
| `render`, `write_video`, the representations `pixels` and `pixels_gray`, `PixelsStateWrapper` | dumps, which GRF's `dump_to_video` renders | rendering takes place in GRF, from dumps |
| the `keyboard` player and `real_time` | GRF's own `keyboard` player, in a GRF that renders | interactive play needs a renderer |
| the `ppo2_cnn` player | the policy loaded in the caller's code and played as an agent | a TensorFlow checkpoint player |
| `create_remote_environment` | an environment from `create_environment`, for evaluation on the caller's machine | a client of GRF's online game server |
| importing a scenario module, as `import gfootball.scenarios.<name>` | the scenario's name, or `cuju.scenarios` with a custom scenario file | a batch plays bundles rather than importable modules |

Rendering accounts for the largest group. Videos, pixel observations and interactive play come from
GRF, which renders cuju's dumps. For training this costs little, since a rendered GRF plays other
matches than an unrendered one, as listed in [Differences from GRF](differences-from-grf.md), so
every result trained on GRF's vector or SMM observations came from an unrendered match. A dump that
GRF renders into a video shows the match that was played.

GRF's internals form the second group. Code that reaches into `env.unwrapped._env` or the private
attributes of GRF's players depends on GRF's object layout, which has no counterpart in a batch. In
practice such code usually seeks one of two things, a value of the match, which `get_state` and the
`raw` observation hold, or a way to set the match up, which a scenario file or `set_state` gives. The
tracker belongs here too, and states saved at the steps of interest replace it.

The third group holds features that belong to GRF's wider ecosystem. Remote play went through GRF's
online game server, and evaluation in cuju runs on the caller's machine with `create_environment` or
a batch. The `ppo2_cnn` player is a TensorFlow checkpoint, which loads in the caller's code and
plays as an agent. A scenario is a name or a file, which a batch turns into a bundle.

## GRF's interface for one environment

`cuju.env.create_environment` takes GRF's arguments with GRF's defaults and returns one environment
with GRF's interface, the one of gym 0.21 that GRF 2.10.3 is written for. `reset()` returns the
observation, and `step(action)` returns `(observation, reward, done, info)`. It plays the same
matches as GRF's after the same `random.seed(...)`, since it draws its seeds from Python's `random`
as GRF does. It is also the function behind `cuju.gfootball.env.create_environment`. An environment
with a stacked SMM and checkpoint rewards takes one step in a few lines.

```python
import random

import cuju.env

random.seed(0)
env = cuju.env.create_environment(
    '11_vs_11_easy_stochastic',
    representation='extracted',
    stacked=True,
    rewards='scoring,checkpoints',
)
observation = env.reset()
observation, reward, done, info = env.step(12)
```

Three keyword-only arguments come on top of GRF's, `device`, which is `'cpu'` by default,
`threads`, which is 1, and `api`. The CPU is the default because one environment occupies one warp
of a GPU and leaves the rest of the device idle, while on the CPU one environment already runs
several times as fast as a GRF process.

`api='gymnasium'` returns Gymnasium's interface instead, where `reset()` returns
`(observation, info)` and `step` five values. `import cuju` registers the ids
`GFootball-<scenario>-<representation>-v0`, with the representations `SMM`, `simple115` and
`simple115v2`, for every shipped scenario, with Gymnasium as soon as it is imported, and
`gymnasium.make('GFootball-11_vs_11_kaggle-simple115v2-v0')` gives that interface. gym's interface
suits code written for GRF, and Gymnasium's suits libraries that expect it.

`get_state`, `set_state`, `observation()`, which returns GRF's raw dictionaries of the agent's
players, and `close` work as GRF's do, and `env.batch` is the batch of one environment underneath,
for access to a batch method from GRF's interface. With `representation='raw'`, `observation_space`
is `None`, as GRF's raw observation has no space. `seed` returns `None` and `reward_range` is
unbounded, as gym's `Env` gives them to GRF's environment, and `disable_render` does nothing.
`render=True`, `write_video=True` and the `Pixels` ids raise `ValueError`, since rendering takes
place in GRF, from dumps.

The environment is a Gymnasium `Env` whose `reset` and `step` return gym's values. Its spaces are
Gymnasium's when Gymnasium is installed, so a check such as
`isinstance(space, gymnasium.spaces.Box)` holds, and the same check against gym's classes fails.
GRF's wrappers come from `cuju.gfootball.env.wrappers`, which run over this interface.

This interface serves existing code, the check of a port and comparisons, and the batch serves
training. A loop over many such environments, one per process as GRF is usually scaled, combines
GRF's architecture with cuju's engine and leaves most of the batch's throughput unused.

### Dumps and replays

GRF writes a dump on a call to `env.write_dump(name)`, and so does cuju's environment. It writes at
most the last 100 steps of the episode as a GRF dump, `<logdir>/<name>_<time>.dump` with the date and
time in GRF's format, and returns the path without `.dump`, as GRF's does. Each name writes once an
episode, and a second call returns `None`. After the episode ends it raises `RuntimeError` until
`reset()`, where GRF raises `AttributeError`.

The options `write_full_episode_dumps` and `write_goal_dumps` write GRF's dumps as well, with GRF's
names, `<logdir>/episode_done_<time>.dump` for an episode and `score_<time>.dump` or
`lost_score_<time>.dump` for a goal, each with the time at which GRF starts the dump. The step that
finishes a dump lists it in `info['dumps']`, as GRF does, with one dictionary for each dump, such as
`{'dump': '/tmp/dumps/episode_done_20261027-142233123456.dump', 'name': 'episode_done'}`, and a goal
that ends an episode lists its goal dump first.

A dump holds every observation of the steps it covers, and keeping all of them during play would
cost memory at every step for a file written only on occasion. The environment therefore keeps the
actions of each step alone, and plays them again in a batch of one environment to write the dump on
request. The replay of a long episode takes a moment on the CPU, and the `time` of each step in the
dump is the time of that replay rather than of the original step. The match itself is the same, bit for bit.

A dump can also be checked without rendering. `python -m cuju.gfootball.replay --trace_file=<dump>`
plays a dump's actions again from its configuration and compares every step with the dump, bit for
bit. It prints the first difference and exits 1, or the steps found equal and exits 0.
`cuju.gfootball.replay.check(trace_file)` returns `(steps, ended, difference)` for use in a script,
and `ScriptHelpers().replay(dump, render=False)` is GRF's method, which raises `ValueError` naming
the first difference. Both take dumps of cuju and of GRF that start at the start of an episode. The
check confirms that a dump written by GRF on a given machine plays the same match in cuju.

### Custom players

GRF loads scripted players from `extra_players`, and so does cuju. A name other than `agent`, `bot`
and `lazy`, as in `extra_players=['<name>:left_players=1']`, loads
`cuju.gfootball.env.players.<name>`, else the module `<name>` itself, such as
`my_package.my_player`, and builds its class `Player(player_config, env_config)`, as GRF does. Its
`take_action(observations)` receives GRF's raw observation of each of its players and returns one
action each, and `reset()` follows every `env.reset()`. `cuju.gfootball.env.player_base.PlayerBase`
is GRF's base class, so a player written for GRF imports it from there, and the `replay` player
plays a dump's actions. A missing module or class, or a wrong number of actions, raises
`ValueError`.

Such a player runs in Python once per step, as in GRF, which suits an opponent in evaluation. For
training, an opponent plays as agents of a batch, or as the ported
[`bot` and `lazy` players](#scripted-players-and-control_all_players), which act for every
environment at once.

### Scenario configuration

`Config.ScenarioConfig()` builds the scenario of the current episode as GRF's `scenario_builder`
does, from GRF's scenario files, which cuju ships, or from a custom scenario file. `NewScenario()`
and `set_scenario_value(key, value)` behave as GRF's do. `scenario_builder.Scenario(config)` is
GRF's builder of a configuration's scenario, which draws the seed of the episode from Python's
`random` unless the configuration holds one, and its `ScenarioConfig()` returns the scenario. Code
that reads a scenario's settings, such as the length of a half or whether it ends on a goal, keeps
working. A dump written after 150 steps and the start of the second half of `11_vs_11_kaggle` each
take a line.

```python
import cuju.env
from cuju.gfootball.env import config

env = cuju.env.create_environment('11_vs_11_easy_stochastic', logdir='/tmp/dumps')
env.reset()
for _ in range(150):
    env.step(0)
path = env.write_dump('shutdown')  # '/tmp/dumps/shutdown_<time>'

configuration = config.Config({'level': '11_vs_11_kaggle'})
configuration.ScenarioConfig().second_half  # 1500
```

Two details differ. A bundle holds no scenario file, so its `ScenarioConfig()` raises
`NotImplementedError`. The positions keep the scenario's double values, where GRF's engine rounds
them to `float32`, so comparisons of positions use a tolerance.

## Batched training loop

The batch replaces a list of GRF environments, or a pool of GRF processes, with one object. The
mapping of names is mostly mechanical. A port centres on three changes, which the
[pitfalls of a port](#pitfalls-of-a-port) discuss in detail. Outputs are shared buffers, the episode resets
inside the step, and every output has an environment axis.

### Loop comparison

With GRF, one environment runs per process.

```python
import gfootball.env as football_env

env = football_env.create_environment(
    env_name='5_vs_5',
    representation='simple115v2',
    rewards='scoring,checkpoints',
    number_of_left_players_agent_controls=4,
)
observation = env.reset()
for _ in range(1000):
    action = policy(observation)  # [4] actions
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
```

With cuju, 1,024 environments run in one process.

```python
import cuju
import torch

batch = cuju.BatchEnvironment(
    '5_vs_5_l4_r0',
    environment_count=1024,
    device='cpu',
    representation='simple115v2',
    rewards='scoring,checkpoints',
)
observation = batch.reset()
for _ in range(1000):
    actions = policy(observation)  # [1024, 4] actions, a tensor
    observation, reward, done, information = batch.step(actions)
    # No reset: environments whose episode ended have already started the next one.
    # information['final_observation'][done.bool()] holds their last observations.
```

The policy receives 1,024 observations at once, the batched input that a neural network processes
in one pass, and the `if done` branch disappears, since the batch resets its own environments. The
number of agents moves into the name, `_l4_r0`, the way a batch names its agents, as described in
[Scenarios](scenarios.md).

### Arguments of `create_environment`

Every argument of GRF's `create_environment` has a place in the batch. Most keep their names, and
the agent counts move into the scenario's name or `agent_counts`.

| GRF argument | `BatchEnvironment` | Notes |
| --- | --- | --- |
| `env_name` | the first argument | a shipped scenario by name, a bundle by path, or a scenario file |
| `stacked` | `stacked` | same |
| `representation` | `representation` | same names, the default is `simple115v2` where GRF's is `extracted`, and `pixels` comes from GRF's rendering |
| `rewards` | `rewards` | same, `scoring` or `scoring,checkpoints` |
| `write_goal_dumps`, `write_full_episode_dumps`, `dump_frequency`, `logdir` | `cuju.dumps.DumpWriter` | written on request, outside the step, as in [Dumps from a batch](#dumps-from-a-batch) |
| `render`, `write_video` | GRF's `dump_to_video`, from dumps | rendering takes place in GRF |
| `extra_players` | `cuju.players.Bot`, `cuju.players.Lazy` | GRF's `bot` and `lazy` players, ported, as in [Scripted players](#scripted-players-and-control_all_players) |
| `number_of_left_players_agent_controls` | the `_l<left>` of the name, or `agent_counts` | same meaning, any count from 0 to the team size |
| `number_of_right_players_agent_controls` | the `_r<right>` of the name, or `agent_counts` | same meaning |
| `channel_dimensions` | `channel_dimensions` | same |
| `other_config_options` | the keyword arguments of the constructor | flattened into the constructor |

The one default that changes is the representation. The batch defaults to `simple115v2` because
most training on GRF reads a vector, and an SMM of thousands of environments takes gigabytes, as
measured in [Observations](observations.md). A port that names its representation is unaffected by
the default.

### `other_config_options`

GRF passes its lower-level options in a dictionary. The batch takes them as keyword arguments, with
the same names where they mean the same thing.

| GRF key | cuju | Notes |
| --- | --- | --- |
| `action_set` | `action_set` | same, `default`, `v2`, `full` |
| `physics_steps_per_frame` | `physics_steps_per_frame` | same |
| `game_engine_random_seed` | `game_engine_random_seed` | same, and a sequence sets one per environment |
| `episode_number` | `episode_number` | same |
| `reverse_team_processing` | `reset(reverse=...)` | the team-order flag, read at an environment's first reset |
| `level` | the first argument | the scenario |
| `players` | the agent counts, `cuju.players.Bot`, `cuju.players.Lazy`, and custom players through `create_environment` | the `keyboard` player plays in a GRF that renders |
| `real_time` | the speed of the hardware | a batch runs as fast as the hardware allows |
| `display_game_stats`, `custom_display_stats` | GRF's rendering of dumps | rendering takes place in GRF |
| `render_resolution_x`, `render_resolution_y` | GRF's rendering of dumps | rendering takes place in GRF |
| `video_format`, `video_quality_level`, `write_video` | GRF's rendering of dumps | rendering takes place in GRF |
| `dump_full_episodes`, `dump_scores`, `tracesdir` | `cuju.dumps.DumpWriter` | as in [Dumps from a batch](#dumps-from-a-batch) |

`reverse_team_processing` is the only option whose place changes in meaning. In GRF it is drawn from
the seed and set by hand mainly to replay a run, so the batch takes it at the reset where a replay
happens, as covered in [Episodes and seeds](episodes-and-seeds.md).

### Environment methods

The methods map one to one, with an environment axis added to everything they take and return.

| GRF | cuju | Notes |
| --- | --- | --- |
| `env.reset()` | `batch.reset()` | resets every environment, `reset(mask=...)` some, and `reset(seeds=..., reverse=...)` replays chosen seeds |
| `env.step(action)` | `batch.step(actions)` | `actions` is `[environment_count, agents]`, and it returns `(observation, reward, done, information)` for the whole batch |
| `env.observation()` | `batch.observation()` | the current observation |
| `env.get_state(to_pickle)` | `batch.get_state(index)` | GRF's format, as in [States](states.md) |
| `env.set_state(state)` | `batch.set_state(index, state)` | GRF's format, in both directions |
| `env.close()` | `batch.close()` | frees the buffers |
| `env.observation_space` | `batch.single_observation_space` | `batch.observation_space` for the batch |
| `env.action_space` | `batch.single_action_space` | `batch.action_space` for the batch |
| `env.render()`, `env.disable_render()` | GRF's `dump_to_video`, from dumps | rendering takes place in GRF |
| `env.write_dump(name)` | `env.write_dump(name)` in `cuju.env`, `cuju.dumps.DumpWriter` for a batch | the last steps of an episode, or whole episodes and goals |
| `env.tracker_setup(start, end)` | `get_state` at the chosen steps | the tracker is part of GRF's engine |
| `info['score_reward']` | `information['score_reward']` | same, `[environment_count]` |
| `len(obs['left_team'])`, `len(obs['right_team'])` | `batch.left_players`, `batch.right_players` | per environment, from a left agent's view |

States are per environment, so `get_state` and `set_state` take an index. They copy between the
device and the host, which makes them the one part of the mapping to keep out of the hot loop.

### Wrappers

GRF builds its observations by wrapping the environment. A batch computes them inside the step, on
the device, so each wrapper becomes an option. The values are the same, and the cost of a wrapper
in Python disappears.

| GRF wrapper | cuju | Notes |
| --- | --- | --- |
| `Simple115StateWrapper` | `representation='simple115'` or `'simple115v2'` | computed inside the batch |
| `SMMWrapper` | `representation='extracted'` | computed inside the batch |
| `PixelsStateWrapper` | GRF itself | pixel observations come from a renderer |
| `FrameStack` | `stacked=True` | four observations, the oldest first |
| `CheckpointRewardWrapper` | `rewards='scoring,checkpoints'` | same reward, same state key |
| `SingleAgentObservationWrapper`, `SingleAgentRewardWrapper` | `squeeze=True` | drops the agent axis for one agent, which GRF does by itself when one agent plays and cuju only when asked |
| `GetStateWrapper` | built in | `get_state` and `set_state` are methods of the batch |
| `PeriodicDumpWriter` | `cuju.dumps.DumpWriter` | as in [Dumps from a batch](#dumps-from-a-batch) |
| `MultiAgentToSingleAgent` | `cuju.wrappers.MultiAgentToSingleAgent` | for scenarios that set `control_all_players`, as in [Scripted players](#scripted-players-and-control_all_players) |

The classes themselves remain in `cuju.gfootball.env.wrappers` for code on GRF's interface, where
they wrap one environment at a time, as GRF's do. `MultiAgentToSingleAgent` there takes a batch as
well and then gives the batched class. Over one environment it gives the players that the policy
does not move GRF's `builtin_ai` action, which a cuju environment plays with the action set `v2` or
`full`, so the environment under it takes `other_config_options={'action_set': 'v2'}`, whose first
19 actions are those of `default`.

A custom wrapper that transforms observations ports best as a function applied to the batch's
tensors after each step, where it runs once for all environments rather than once per environment.

### Gym names

The batch spells out a few names that Gym abbreviates. The objects are the same.

| Gym and GRF | cuju |
| --- | --- |
| `env` | `batch`, a `BatchEnvironment` |
| `num_envs` | `environment_count` |
| `obs` | `observation` |
| `info` | `information` |
| `final_obs` (Gymnasium's vector environments) | `information['final_observation']` |

### Shapes

The outputs of a batch carry an environment axis in front of GRF's shapes.

| Output | GRF, one agent | GRF, several agents | cuju |
| --- | --- | --- | --- |
| observation (`simple115v2`) | `[115]` | `[agents, 115]` | `[environment_count, agents, 115]` |
| reward | scalar | `[agents]` | `[environment_count, agents]` |
| done | bool | bool | `[environment_count]` of `uint8` |
| action | int | `[agents]` | `[environment_count, agents]` |

GRF drops the agent axis by itself when one agent plays, and cuju keeps it unless `squeeze=True` is
passed. cuju keeps it by default because a batch can mix scenarios with different numbers of
agents, and code whose agent axis comes and goes is a common source of silent broadcasting bugs.
With `squeeze=True` and one agent per environment, cuju's shapes become `[environment_count, 115]`,
`[environment_count]` and `[environment_count]`, GRF's single-agent shapes with an environment axis,
and the spaces become GRF's for one agent, such as `Discrete(19)` in place of `MultiDiscrete([19])`.

## Dumps from a batch

`cuju.dumps.DumpWriter` writes GRF's dump files for chosen environments of a batch. They are the
same pickles, with the same keys and values, that GRF's `write_full_episode_dumps` and
`write_goal_dumps` write, so GRF's tools read them, and GRF's `dump_to_video` turns them into
videos. A writer that follows two environments of a batch for one match writes their episodes and
goals.

```python
import cuju
import cuju.dumps
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu', with_raw=True)
writer = cuju.dumps.DumpWriter(batch, 'dumps', environments=[0, 1], write_goal_dumps=True)
batch.reset()
for _ in range(3001):
    actions = torch.randint(0, 19, (64, 4), dtype=torch.int32)
    batch.step(actions)
    writer.record(actions)  # after every step, with the same actions
writer.close()  # writes the episodes still running
print(writer.paths)
```

`record(actions)` follows every step. It reads the batch's buffers after the step and leaves every
match as it plays. It waits for the device and copies the rows of the followed environments alone, so on a GPU
it adds a wait at every step, and its cost grows with the environments it follows. A writer
therefore follows a few chosen environments.

The files are `episode_done_environment_<i>_episode_<n>.dump`, and for goals
`score_environment_<i>_episode_<n>_step_<frame>.dump` and `lost_score_...`. GRF names a dump by its
kind and the time to the microsecond, which serves one environment, and the environments of a batch
can start dumps in the same microsecond, so the writer of a batch names each dump by its
environment and episode instead. `dump_frequency` selects
episodes as GRF's `PeriodicDumpWriter` does. GRF also limits dumps by wall-clock time, to at most
one full episode every 10 seconds, a limit sized for GRF's speed. A batch plays many episodes a
second, so the writer writes every selected episode, and `dump_frequency` is the setting that bounds
the space taken on disk.

The steps hold GRF's observation only if the batch writes `raw`, through `with_raw=True` or
`representation='raw'`. GRF's `replay` needs none, and `dump_to_video` does. `with_raw=True`
therefore serves dumps meant for video, and a dump meant for replay needs no raw observation.

An episode whose difficulty differs from its scenario's names its level
`<scenario>_difficulty_<left>_<right>` and writes that GRF scenario file next to the dumps. GRF's
tools find it once it is copied into GRF's `gfootball/scenarios/`, since GRF looks scenarios up by
name. `cuju.dumps.read_dump(path)` reads a dump, cuju's or GRF's, without GRF or numpy, and runs no
code from it, which makes it safe for dumps from other sources.

In GRF, `ScriptHelpers().replay(path, render=False)` replays a dump of cuju bit for bit, with GRF's
default `physics_steps_per_frame=10`, and `python -m cuju.gfootball.replay` checks one without GRF.
A rendering replay drifts from the match, as rendering does, so its video can part from the match
that the dump was written from.

## Scripted players and `control_all_players`

GRF ships two scripted players and a wrapper for scenarios in which one policy controls a whole
team. cuju ports all three to act on every environment of a batch at once.

**`cuju.players.Bot(batch, agent)`** is GRF's `bot` player, ported exactly in float64. `act()`
returns its actions, an index of the `full` set, for every environment at once. It needs
`action_set='full'`, as GRF's bot asserts, and raw arrays. Like GRF's, it keeps its memory across
episodes. It provides a scripted opponent besides the built-in AI, for variety in training and
evaluation.

**`cuju.players.Lazy(batch, agents)`** is GRF's `lazy`, which plays `idle`. A policy that has
learned to score beats a team that stands still, which makes `Lazy` a baseline opponent for any
policy.

**`cuju.wrappers.MultiAgentToSingleAgent(batch, left_players, right_players)`** is GRF's wrapper for
scenarios that set `control_all_players`. A policy plays the designated player of each side, and the
built-in AI plays every other agent. The batch needs every controllable player of the played sides
as agents, raw arrays and the `v2` or `full` set, for `builtin_ai`. A batch accepts such a scenario
when each side has either no agent or one per controllable player, as GRF's engine asserts.

`create_environment` uses all of them as GRF does, from `extra_players` such as
`['bot:right_players=1']` and from a scenario file's `control_all_players`, so code that names them
there runs unchanged. The two combine as in GRF. With `control_all_players`, the wrapper covers the
players of the agent, and the players of `extra_players`, such as `['lazy:right_players=11']`, play
the other slots with their own actions.

## Pitfalls of a port

Most pitfalls of a port raise no error, and their effects appear in the results of training, so
each is worth checking once in a ported loop.

**Storing observations.** Code that appends `obs` to a list stores the same tensor many times,
since each step overwrites it. `observation.clone()`, or a write into a preallocated buffer, keeps
each step.

**The done flag.** The reset happens inside `step`, and the observation of a done environment is
the next episode's. Bootstrapping code that reads the observation after the last step reads
`information['final_observation']` instead.

**Time limits.** As in GRF, the end of the match time is a normal end of the episode, signalled by
the done flag alone.

**The default representation.** Code that relied on GRF's default `extracted` gets `simple115v2`
until it names the representation.

**Padded agents.** In a mixed batch, rewards and losses averaged over agents are first masked with
`batch.agent_mask`, so that the zeros of padded slots stay out of the average.

**Controllable players of `5_vs_5`.** Four players per team are controllable in `5_vs_5`, since the
scenario marks its goalkeepers as fixed to the built-in AI, as in GRF.

**One agent.** GRF drops the agent axis by itself when one agent plays, and cuju keeps it.
`BatchEnvironment('academy_empty_goal_close')` gives `[environment_count, 1, 115]` and
`MultiDiscrete([19])`, and `squeeze=True` gives GRF's shapes and spaces.

**`done` as `uint8`.** PyTorch deprecates indexing with a `uint8` mask, so the index is written
`final_observation[done.bool()]` in place of `final_observation[done]`. With
`representation='raw'`, `final_observation` is a dictionary of tensors, each indexed in turn, as
`{key: value[done.bool()] for key, value in final_observation.items()}`.

**The number of environments.** `environment_count` defaults to 1,024, so a ported call without it
creates 1,024 environments.

**The same matches in every run.** The default `seed=0` plays the same matches every time, where an
unseeded GRF draws different ones in each run. A different `seed` per run gives different matches.

**gym and gymnasium.** The spaces of a batch come from gymnasium whenever it is installed, even when
gym is installed too, so a check such as `isinstance(space, gym.spaces.Box)` is written against
`gymnasium.spaces.Box`.

**GRF's PPO example.** `gfootball/examples/run_ppo2.py` runs OpenAI Baselines on TensorFlow 1.15,
which installs on Python 3.7 or older, and cuju runs on Python 3.10 or later. The learner is
therefore ported to PyTorch or kept in a process of its own.

## Reproduction of GRF results

A result made with cuju compares directly with one made on GRF when the setup matches. Exactness
makes such comparisons possible, and a batch can still change the shape of training while every
match is GRF's. Five matters decide whether the setups match.

**The options.** GRF's scripts that reproduce the PPO results of its paper,
`gfootball/examples/repro_scoring_easy.sh` and `repro_checkpoint_easy.sh`, play
`11_vs_11_easy_stochastic` with one agent on the left, the stacked SMM, the default action set, and
`scoring` or `scoring,checkpoints`. cuju expresses the same setup as one batch, and the bare name
gives one left agent, as GRF's `create_environment` does.

```python
batch = cuju.BatchEnvironment(
    '11_vs_11_easy_stochastic',
    environment_count=16,
    device='cpu',
    representation='extracted',
    stacked=True,
    rewards='scoring,checkpoints',
    squeeze=True,
)
```

**Counting steps.** GRF counts environment steps. One `env.step` is one step of one match, however
many agents play in it, and the step counts of papers, such as the `num_timesteps` of Baselines'
PPO, add them up over the parallel environments. One `batch.step` is therefore `environment_count`
such steps, and the agents of each environment add nothing to the count.

**The size of an update.** PPO learns from `num_envs × nsteps` steps per update, 16 × 512 = 8,192
in GRF's scripts. A batch of 4,096 environments with the same `nsteps` collects 2,097,152 steps per
update, so the same number of steps gives 256 times fewer updates, and a learning curve plotted
against steps changes while every match is GRF's. A comparison of curves keeps `num_envs × nsteps`
and the minibatches.

**The evaluation.** Evaluation against the built-in AI uses fixed seeds, given to
`reset(seeds=...)`, and reports an interval with the win rate. The same seeds play the same matches
in GRF.

**The version.** Exactness holds against GRF 2.10.3, commit 3d9e754. Many published results were
made on older versions of GRF, and comparisons with those are statistical, over many episodes,
since cuju's match-by-match agreement is established against 2.10.3.

A larger update makes a different experiment. A batch makes it cheap to collect far more experience
per update than GRF's scripts did, since the extra environments cost little, and the larger update
can improve the policy reached within a given time. A reproduction of a curve keeps the update at
GRF's size and gains in wall-clock time alone. A search for the best policy within a time budget
changes the size of the update on purpose and reports the change.

## See also

- [Differences from GRF](differences-from-grf.md), every behaviour that differs beyond the names
- [Episodes and seeds](episodes-and-seeds.md), the automatic reset, seeds and `reverse_team_processing`
- [States](states.md), `get_state` and `set_state` in GRF's format
- [Scenarios](scenarios.md), the naming of agents and batches that mix scenarios
- [Custom scenarios](custom-scenarios.md), scenario files and `control_all_players`
- [Recipes](recipes.md), patterns for wiring a batch into training
