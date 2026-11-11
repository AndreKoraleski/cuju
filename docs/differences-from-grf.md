# Differences from GRF

The **differences from Google Research Football**, or GRF, are the points at which cuju behaves
otherwise than GRF 2.10.3 while it reproduces GRF bit for bit within the scope of its
[exactness](exactness.md) claim. They lie in the interface of a batch, in the inputs that cuju
validates, in a bound of its engine and in the parts of GRF it ports. Behaviour outside these
differences follows GRF, and issues reporting any further difference are welcome.

Three computations are known to give other bits than GRF. Two of them arise in no match that a
batch accepts, and the third concerns GRF's loader of states, where cuju continues the saved match
exactly. The interface is that of a batch of environments returning PyTorch tensors, with an
automatic reset inside the step and with GRF's single-environment interface available as
`cuju.env.create_environment`. cuju validates scenarios and states with named errors, keeps a
bounded history of positions, and plays without rendering, so matches are viewed in cuju's own
renderer, which reads states without changing them, or in GRF.

## Output differences

The known differences in output concern three computations, each with a bounded reach, which the
table compares.

| Computation | cuju | GRF | Reach |
| --- | --- | --- | --- |
| the time a player of the built-in AI needs to reach the ball, an unsigned 32-bit count of milliseconds, in three tests of the AI | converts the time straight to `float` | converts the time to `float` through `int` | from 2^31 milliseconds, while the engine draws the time from the ball's predictions a few seconds ahead, so no match reaches it |
| the reciprocal inside the two `Normalize` helpers of the vector class, and any other float operation, for a NaN input | gives the GPU's NaN bit pattern on the GPU, and the host's on the CPU as GRF does | gives the host's NaN bit pattern | no output, since the engine passes no NaN there |
| the frame in which each of the AI's mental images stores its ball predictions, after `set_state` | sets the frame of every image from the state and continues the saved match exactly | keeps the frame outside the state, and when the environment holds fewer images than the state, adds the missing ones with their predictions mirrored, so the AI reads wrong predictions until new images replace the added ones, about every 100 milliseconds, and the match drifts | the steps after a `set_state` whose state has `reverse_team_processing` on, a non-deterministic scenario with an odd engine seed, into a GRF environment that holds fewer images than the state, such as a freshly reset one |

Every other output equals GRF's, bit for bit, within the scope of the [exactness](exactness.md)
claim. The third row is the reason a comparison of the two engines from a state loads GRF's state
into cuju, as described in [States](states.md#format).

### Variation within GRF

Two behaviours vary between runs or machines of GRF itself and affect comparisons.

GRF's state bytes vary from run to run. GRF writes the padding between the fields of its engine's
structures as it finds it in memory, so two identical GRF runs give states that differ in a few
hundred padding bytes. cuju writes zeros there, so its states of one match are identical byte for
byte, and states of the two engines are compared by their fields or by loading them.

GRF's `bot` player varies with the processor. It computes norms and dot products with numpy, whose
BLAS may fuse a multiplication and an addition on a processor with AVX-512, where GRF's own bot can
then decide differently in the last bit. `cuju.players.Bot` computes as numpy does without fusing,
as on an AVX2 processor, so it gives the same decisions on every machine.

## Interface

cuju is a batch of environments, and the differences of its interface follow from that. The table
gives each with its reason.

| GRF | cuju | Reason |
| --- | --- | --- |
| one environment per `create_environment` | `BatchEnvironment` with `environment_count` environments | to run thousands of matches per call, with `cuju.env.create_environment` giving GRF's interface for one |
| observations as numpy arrays | PyTorch tensors that view the buffers of the batch, overwritten by each call | copying would cost more than the simulation on the GPU |
| one observation per agent, the agent axis dropped for one agent | `[environment_count, agents, ...]`, padded to the largest number of agents of the batch, with `squeeze=True` dropping the axis for one agent | a mixed batch needs one shape |
| default `representation='extracted'` | default `simple115v2` | a vector occupies 460 bytes per agent, and an SMM of thousands of environments occupies gigabytes |
| `raw` values in numpy's types | `float32`, `int32`, `bool` and `uint8` tensors, holding GRF's values | one fixed type per array on every device |
| `info` with `score_reward` | `information` with `score_reward` and `final_observation` | the automatic reset hands over the last observation of an episode in the step that ends it |
| a step after the end of an episode raises | the step that ends an episode starts the next one with `auto_reset=True`, the default, and with `auto_reset=False` the environment stays at its end, with a reward of 0, until `reset_done` or `reset` | stopping a batch for the few environments that ended would idle the rest |
| `create_environment` resets in its constructor, so a step before `reset` plays the first episode | a batch steps after its first `reset`, or after `set_state` of every environment, and raises `LibraryError` before | the batch allocates its worlds without playing |
| actions are always checked | actions in host memory are checked as GRF checks them, and actions already on the GPU play as `idle` when out of range unless `check_actions=True` checks them too | a check on the GPU makes every step wait for the device, so it is an option |
| the pickle inside a state is written with Python's default protocol, wrapper keys first | protocol 2, the episode logic first | the batch writes the pickle itself, and both sides read each other's states |
| dumps are limited by wall-clock time, a full episode at most every 10 seconds | `DumpWriter` writes every selected episode and goal | a batch plays many episodes per second of wall-clock time |
| a dump is named by its kind and the time to the microsecond, `<kind>_<date>-<time>.dump` | the same names from the dump options of `cuju.env.create_environment`, and names by environment, episode and goal step from `DumpWriter` over a batch | the environments of a batch can start dumps in the same microsecond, where their names would collide |

`cuju.env.create_environment` follows GRF's interface for one environment, with three differences.
A step after the end of an episode raises `RuntimeError`, where GRF raises `AssertionError`. An
action given as one of GRF's `CoreAction`s belongs to the environment's action set, and one from a
larger set raises `ValueError` with the name of the set to create. With no agent on any side,
`simple115` is an empty array of shape `(0, 115)`, where GRF's is `(0,)`, so the observation keeps
its last axis.

`cuju.gfootball.env.football_env.FootballEnv` plays a configuration without an `agent` player as
GRF's does, with two differences. A step given actions raises `ValueError`, where GRF fails an
assertion. A configuration with no player at all gives `None` as its observation, where GRF gives
the observation of the match, since a batch reads that observation from the raw observations of
its agents.

## Input validation

cuju validates scenarios when it builds them and states when it loads them. For inputs on which GRF
crashes, hangs or silently changes its behaviour, it raises a named error, as the table lists.

| Input | GRF | cuju |
| --- | --- | --- |
| a scenario with a team of more than 11 players | crashes | `ScenarioError` |
| a scenario whose team's first player is not a goalkeeper | hangs in open play | `ScenarioError` |
| a scenario that draws from `random` or NumPy's global generator | runs, and shifts the seeds of every later episode | `ScenarioError` |
| a scenario that reads `controllable_left_players`, `controllable_right_players` or `dynamic_player_selection` while it builds | freezes the value of the first read, before the agents are set | `ScenarioError` |
| a scenario that depends on more than the parity of the episode number, or whose team sizes change with it | runs | `ScenarioError` |
| a deterministic scenario that turns `reverse_team_processing` on | runs | `ScenarioError` |
| a scenario with `control_all_players` and agents that would switch players | fails an assertion in its engine | `LibraryError` with GRF's condition |
| a state whose scenario differs from the environment's outside the constant part, such as in its formation entries | stops only when the constant part differs, in the ball, the team sizes, the agent counts, the difficulties, the flags and the duration | `StateError` for any difference but the seed, naming up to three fields |
| with `rewards='scoring,checkpoints'`, a state without checkpoint counts | `KeyError` | `StateError` |
| a stacked state loaded into an unstacked environment, or an unstacked state into a stacked one | ignores the stack of the first, and raises `KeyError` for the second | `StateError` for both |

## Engine

The engine of cuju bounds one quantity that GRF lets grow, and it equals one reference build of
GRF. At paths that GRF leaves undefined, it stops with a named error.

### Position history

GRF keeps one position per player for each second between kick-offs, in a list that grows without
bound. A batch keeps 512 seconds of that history, which holds every shipped scenario, and accepts at
creation every match of up to 51,199 ticks of play, `game_duration` times `physics_steps_per_frame`,
which is 5,119 steps at the default of 10 ticks a step. A longer match raises an error at creation,
and `build_bundle` warns at build time and names the largest `physics_steps_per_frame` that fits.
The bound keeps the state of every environment at a fixed size, which a batch of thousands of
environments on a GPU requires.

### Reference build

cuju equals GRF as GCC builds it for x86-64 Linux with glibc 2.39, on a processor with FMA. A GRF
built for another platform or C library can play other matches, for the reasons given in
[Exactness](exactness.md#reference-build).

### Undefined paths

GRF leaves a few paths of its engine undefined, such as a set piece without a taker, and at such a
path it has no result for cuju to equal. cuju stops there, on the CPU with `cuju.UndefinedPathError`
naming the environment and GRF's reason, and on the GPU with a trap of the device. Every shipped
scenario avoids them, as far as is known. The behaviour on each device, and the checkpoints that
protect a long run, are described in [Episodes and seeds](episodes-and-seeds.md#undefined-paths).

## Rendering, players and wrappers

GRF is both a training environment and a game with rendering, video and human input, and cuju ports
the training environment. Each interactive feature of GRF has a counterpart in cuju or stays in GRF,
as the table shows.

| GRF | cuju |
| --- | --- |
| `render`, `write_video` and the representation `pixels` | matches play without rendering, and a match is viewed in cuju's renderer, described in [Rendering](rendering.md), which reads states and leaves the match as it is, or by writing a dump and replaying it in GRF, or by loading a state into a GRF that renders, since a rendered GRF plays other matches than an unrendered one |
| the tracker, `tracker_setup` | stays in GRF |
| keyboard and gamepad control and the bundled `ppo2_cnn` player | stay in GRF |
| the `replay` player, which plays a dump's actions | `cuju.gfootball.env.players.replay`, loaded from `extra_players` as in GRF |
| the `bot` and `lazy` players | `cuju.players` |
| the observation wrappers | representations and options of the batch, `stacked`, `channel_dimensions` and `squeeze`, and GRF's classes in `cuju.gfootball.env.wrappers` for one environment |
| `CheckpointRewardWrapper` | `rewards='scoring,checkpoints'`, and GRF's class in `cuju.gfootball.env.wrappers` |
| `PixelsStateWrapper` | stays in GRF, since it requires rendering, and cuju's class raises `NotImplementedError` on construction |

Training on GRF's vector and SMM observations runs without rendering, the game that cuju plays, so
a policy trained in cuju meets in an unrendered GRF the game that it trained on. Every option of GRF has a mapping in [Migrating from GRF](migrating-from-grf.md).

## See also

- [Exactness](exactness.md)
- [Migrating from GRF](migrating-from-grf.md)
- [Observations](observations.md)
- [Episodes and seeds](episodes-and-seeds.md)
- [States](states.md)
- [Common topics](faq.md)
