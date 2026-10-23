# Observations

In cuju, an **observation** is the data that each agent receives after every step of a batch. cuju
computes the four observation representations of Google Research Football, or GRF, named
`simple115`, `simple115v2`, `extracted` and `raw`, inside the batch and on the device that runs it,
under GRF's names and with GRF's values. Each representation is a PyTorch tensor, or a dictionary
of tensors, with one row per environment and one entry per agent.

The default representation is `simple115v2`, a vector of 115 numbers per agent with a fixed slot for
each player. The super mini map, `extracted`, encodes the pitch as image planes for convolutional
networks and occupies 60 times the memory of a vector, and `raw` holds every value of GRF's
observation by name. Every agent observes its own team as the left team, so the observation of an
agent on the right team is turned by 180 degrees, as in GRF. Coordinates are scaled from metres to
raw units that extend slightly beyond ±1 at the goal lines. In a batch that mixes scenario bundles,
every output is padded to the largest bundle.

## Representations

A representation is the encoding of an observation, selected at creation with the option
`representation`. The four representations differ in shape, type and content.

| Representation | Shape | Type | Content |
| --- | --- | --- | --- |
| `simple115` | `[environment_count, agents, 115]` | `float32` | GRF's 115 numbers, in GRF's first layout |
| `simple115v2` | `[environment_count, agents, 115]` | `float32` | the same numbers with a fixed slot for each player |
| `extracted` | `[environment_count, agents, height, width, 4]` | `uint8` | the SMM, four planes drawn on a grid |
| `raw` | a dictionary of tensors `[environment_count, agents, ...]` | by array | every value of GRF's observation, by name |

The default is `simple115v2`, while GRF's `create_environment` defaults to `extracted`. The default
reflects the memory of the two encodings. A vector occupies 460 bytes per agent, while the two SMM
buffers of a batch of 16,384 environments with four agents occupy 3.6 GB. A batch of 16
environments of four agents with the SMM returns observations of 72 by 96 cells.

```python
import cuju

batch = cuju.BatchEnvironment(
    '5_vs_5_l4_r0', environment_count=16, device='cpu', representation='extracted'
)
observation = batch.reset()
print(observation.shape, observation.dtype)  # torch.Size([16, 4, 72, 96, 4]) torch.uint8
```

### Choice of representation

The representations differ in the information they retain, in the networks they suit and in their
memory cost.

`simple115v2` occupies 460 bytes per agent and suits a multilayer perceptron, and each of its
numbers has a fixed meaning. It contains the positions and directions of every player and of the
ball, the team in possession, the active player and the game mode. Tiredness, yellow cards, roles,
the score, the time remaining and the sticky actions appear only in `raw`, so a policy that reads
`simple115v2` alone has no information about the end of the match or the fatigue of a player. The
omission has little effect in most scenarios and can matter in 11-a-side play with a time limit.

`extracted`, the SMM, encodes the positions on a grid. It suits convolutional networks, which can
learn spatial patterns such as space behind a defender without hand-made features. The PPO
baselines of the GRF paper were trained on a stack of four SMMs, which `extracted` with
`stacked=True` reproduces. Per agent the SMM is 60 times larger than a vector, and with thousands of
environments on a GPU that factor determines whether a batch fits in memory.

`raw` contains every value of GRF's observation by name. It serves feature engineering, reward
shaping, logging and Kaggle submissions, which receive the same dictionary. Networks rarely read it
directly. A common arrangement trains on a vector or the SMM and reads raw values alongside it with
`with_raw=True`, which adds the writing of every raw array to each step.

`simple115` serves the reproduction of work that used it. Its layout depends on the number of
players in a team.

## Vector representations

Both vector representations are GRF's `Simple115StateWrapper`. They hold the positions and
directions of both teams, the position, direction and owner of the ball, the active player and the
game mode, 115 numbers in all. In `simple115v2` each quantity occupies a fixed range of indices,
which the table lists.

| Indices in `simple115v2` | Numbers | Content |
| --- | --- | --- |
| 0 to 21 | 22 | positions of the agent's team, `x` then `y` of each of 11 player slots |
| 22 to 43 | 22 | directions of the agent's team, in the same slots |
| 44 to 65 | 22 | positions of the other team |
| 66 to 87 | 22 | directions of the other team |
| 88 to 90 | 3 | position of the ball, `x`, `y` and `z` |
| 91 to 93 | 3 | direction of the ball |
| 94 to 96 | 3 | owner of the ball, one-hot over none, the agent's team and the other team |
| 97 to 107 | 11 | the active player, one-hot over the agent's team, all zero without one |
| 108 to 114 | 7 | the game mode, one-hot, in the order of the values of `game_mode` |

Player `k` of the agent's team has its position at indices `2k` and `2k + 1`, and frame `k` of a
stack adds `115 * k` to every index.

The two representations differ only when a team has fewer than 11 players. `simple115` writes the
four player arrays one after the other and fills the rest of the 88 player numbers with -1, so in
`5_vs_5` the directions of the agent's team start at index 10, where `simple115v2` holds the
position of the sixth player slot. `simple115v2` gives every array its 11 slots and fills the unused
slots of each with -1, so a given index always holds the same quantity. GRF introduced
`simple115v2` for that reason, and cuju adopts it as the default for the same reason. In both
layouts the ball, the owner, the active player and the game mode occupy indices 88 to 114.

## Super mini map

The super mini map, or SMM, is the representation `extracted`. It draws the pitch on four planes of
`height` by `width` cells, one each for the agent's team, the other team, the ball and the active
player. A cell holds 255 where something stands and 0 elsewhere. The option
`channel_dimensions=(width, height)` sets the size, 96 by 72 by default, as in GRF.

The batch draws the SMM itself, so the planes are ready on the device with the rest of the
observation. As in GRF, the observation of a right agent is turned by 180 degrees, so every agent
has its own goal on the left.

## Raw observation

The raw observation gives every value of GRF's observation as a dictionary of tensors under GRF's
names. Each tensor has the shape `[environment_count, agents, ...]`, and an agent observes its own
team as the left team, as in GRF. Each array has a fixed shape after the agent axis and a fixed
type, which the table gives.

| Array | Shape after the agent axis | Type |
| --- | --- | --- |
| `ball`, `ball_direction`, `ball_rotation` | `[3]` | `float32` |
| `ball_owned_team`, `ball_owned_player` | none | `int32` |
| `left_team`, `left_team_direction` | `[players, 2]` | `float32` |
| `left_team_tired_factor` | `[players]` | `float32` |
| `left_team_yellow_card`, `left_team_active` | `[players]` | `bool` |
| `left_team_roles` | `[players]` | `int32` |
| the same six arrays for `right_team` | as for `left_team` | as for `left_team` |
| `score` | `[2]` | `int32` |
| `steps_left`, `game_mode` | none | `int32` |
| `active`, `designated` | none | `int32` |
| `sticky_actions` | `[10]`, or `[13]` for `full`, one flag per sticky action | `uint8` |

The values equal GRF's, held in these types. An owner of the ball or an active player of -1 means
none. The order of `sticky_actions` is that of the action set, given in
[Rewards and actions](rewards-and-actions.md#sticky-actions). The remaining arrays take the values
that GRF defines for them, which the second table describes.

| Array | Values |
| --- | --- |
| `ball_owned_team` | -1 for none, 0 for the agent's own team, 1 for the other team |
| `ball_owned_player` | the index of the owner in its own team's arrays, -1 for none |
| `game_mode` | 0 normal, 1 kick-off, 2 goal kick, 3 free kick, 4 corner, 5 throw-in, 6 penalty |
| `left_team_roles`, `right_team_roles` | the role of each player, from 0 for the goalkeeper to 9 for the centre forward |
| `left_team_active`, `right_team_active` | true for a player on the pitch, false for a player sent off, who keeps the slot, and for a padded slot |
| `left_team_tired_factor` | GRF's tiredness of each player, 0 when fresh |
| `active` | the player of the agent's team that the agent controls |
| `designated` | the player of the agent's team that a single agent of that team would control, equal to `active` for a team of one agent |
| `score` | the goals of the agent's own team, then those of the other team |
| `steps_left` | the steps until the match time runs out |

The roles are 0 goalkeeper, 1 centre back, 2 left back, 3 right back, 4 defensive midfielder,
5 central midfielder, 6 left midfielder, 7 right midfielder, 8 attacking midfielder and 9 centre
forward. In a batch of `5_vs_5_l1_r1`, each team array holds five players for each of the two
agents.

```python
batch = cuju.BatchEnvironment(
    '5_vs_5_l1_r1', environment_count=4, device='cpu', representation='raw'
)
raw = batch.reset()
print(raw['left_team'].shape)  # torch.Size([4, 2, 5, 2]), 4 environments, 2 agents, 5 players
print(raw['score'][0, 0])  # tensor([0, 0], dtype=torch.int32)
```

### Pitch coordinates

GRF's engine measures the pitch in metres, with the centre spot at the origin, and divides each
coordinate for the observation, `x` by 54.4 and `y` by -83.6, so `y` also changes sign. The ball's
`z`, its height, stays in metres. The pitch measures 110 m by 72 m, so the lines fall slightly
beyond the round numbers that GRF's documentation gives. Each main feature of the pitch has a
position in metres and one in raw units.

| Feature | Metres | Raw units |
| --- | --- | --- |
| goal line | `x` = ±55 | `x` = ±55 / 54.4 = ±1.0110 |
| touchline | `y` = ±36 | `y` = ±36 / 83.6 = ±0.4306 |
| goal, half its width | 3.7 | 3.7 / 83.6 = 0.0443 |
| goal, height, the ball's `z` | 2.5 | 2.5 |
| centre spot | 0 | 0 |

`x` runs from about -1.011 at the goal line of the agent's team to 1.011 at the other, and `y` from
-0.4306 at the top touchline to 0.4306 at the bottom one. GRF's documentation rounds these to ±1
and ±0.42, and the engine uses the unrounded values, `PITCH_HALF_LENGTH`, `PITCH_HALF_WIDTH` and
`GOAL_HALF_WIDTH` in `include/cuju/game_definitions.h`. Players can stand beyond the lines, as a
goalkeeper on his goal line does at about `x = -1.011`. GRF's engine decides when a goal counts, and
the observation shows a goal as a change of `score`.

The two axes have different units. One raw unit of `x` is 54.4 m and one of `y` is 83.6 m, so a
distance computed from raw coordinates shrinks every sideways metre to about two thirds of a
lengthwise one. GRF's own features, such as the zones of its checkpoint reward, are measured in
these units, and policies trained on them adapt to the distortion. A feature in metres, such as the
distance from a player to the ball for a pressing reward, requires the coordinates scaled back
first, `x * 54.4` and `y * 83.6`.

The values extend beyond ±1. Clipping or normalizing observations to the interval from -1 to 1 cuts
off the goal lines and every player beyond them, so a goalkeeper on his line and one a metre in
front of it become indistinguishable. Normalization with the bounds of the pitch keeps them apart,
and since the values are already of order one, the observation is also usable without
normalization.

`y` grows toward the bottom touchline, the direction of the actions whose names contain `bottom`. A
plot of the pitch drawn with `y` upward therefore appears upside down.

### Orientation of right agents

Every agent observes its own team as the left team, defending the goal at `x = -1.011`. For a right
agent, GRF turns the pitch by 180 degrees in `flip_observation` of its `observation_rotation.py`,
and cuju gives the same values key by key. The turn affects each key of the raw observation in a
fixed way, which the table sets out.

| Key | For a right agent |
| --- | --- |
| `ball`, `ball_direction` | `x` and `y` change sign, `z` stays |
| `ball_rotation` | as the engine gives it, since GRF leaves it unturned |
| `ball_owned_team` | 0 and 1 swap, as 0 is always the agent's own team, and -1 stays |
| `ball_owned_player` | the same index into the owning team's arrays |
| `left_team*` | the right team's arrays, with `x` and `y` of positions and directions negated, and tired factor, yellow cards, active flags and roles as they are |
| `right_team*` | the left team's arrays, turned the same way |
| `score` | the two entries swap, so the agent's own goals come first |
| `active`, `designated` | indices into the agent's own team, which is `left_team` |
| `sticky_actions` | the direction flags turn with the pitch, so a right agent that presses `right` sees the flag of `right`, while `sprint`, `dribble` and the `full` set's flags stay |
| `game_mode`, `steps_left` | the same for both sides |

`simple115v2`, `simple115` and the SMM are computed from the turned view and follow the same rules.
A right agent's actions turn back in the same way, as described in [Rewards and
actions](rewards-and-actions.md#sticky-actions), so one policy can play either side. A scenario file
gives each team's positions in that team's own view, so `x = -1` places a right-team player at the
right team's own goal.

The turn serves self-play. Every agent faces the same way and plays the same game from its own point
of view, so a single network learns one game rather than two mirrored ones, and it can play the left
team in one environment and the right in the next without knowing which. Training on both sides in
self-play doubles the experience that each match gives that network.

The observations of a left and a right agent of one environment lie in two frames as a result.
Negating the `x` and `y` of a right agent's positions again returns them to the left team's frame,
as a plot or a centralized critic requires. The turn is also a near symmetry of the game rather
than an exact one. `ball_rotation` is the one array that GRF leaves unturned, so its meaning depends
on the side, and a feature built from it transfers between sides less well than the rest. The sides
also differ in the game itself, most visibly in difficulty, since most scenarios give the right
team weaker players, as listed in [Scenarios](scenarios.md). A policy that plays both sides
therefore meets two different opponents through one kind of observation.

### Agents and players

When a team has as many agents as controllable players, each agent keeps one player for the whole
episode. The agents take the team's controllable players in order, so `raw['active']` stays fixed.
In `5_vs_5`, whose goalkeepers stay with the built-in AI, four agents a side control players 1 to 4.

With fewer agents than controllable players, the agents switch players during play, as in GRF's
dynamic player selection. Each agent controls the player that GRF makes active for it, usually the
one nearest the ball, and `raw['active']` changes during the episode. In the 11-a-side scenarios
every player is controllable, the goalkeeper included, so a single agent can end up in goal. With
one agent, `designated` equals `active`.

An agent can have no player, and `agent_mask` still counts it. This occurs in two cases. With more
agents than controllable players, the last agents of the team have none, so in `5_vs_5_l5_r0`
agent 4 has `active == -1` and an all-zero active one-hot in `simple115v2`. When a team keeps one
player per agent and a player is sent off, the agent of that player has `active == -1` for the rest
of the episode.

`raw['active'] != -1` marks the agents that control a player after the last step, as in GRF, and a
loss multiplied by it counts only those agents.

### Padded slots in a mixed batch

A batch that mixes bundles pads every output to the largest bundle, with `batch.agents` agent slots
per environment and `batch.players` player slots per team array. Each kind of padded slot holds
fixed values, which the table lists.

| Slot | Holds |
| --- | --- |
| an agent slot beyond the environment's agents | zeros in every observation and a reward of 0, and the batch ignores its actions |
| its `raw` | zeros, so `active` reads 0 rather than -1 and its roles read 0, the goalkeeper's role, so its mask is `agent_mask` rather than `active` |
| a player slot beyond its team's players | zeros, so position and direction 0, `*_active` false and role 0 |
| the same slot in `simple115v2` | -1, as GRF fills a team of fewer than 11 |

The real sizes of each environment are the attributes `left_players`, `right_players`,
`left_agents` and `right_agents`, CPU tensors of `[environment_count]`. The mask of the left team's
real players follows from them in one line.

```python
import torch

left_players = torch.arange(batch.players) < batch.left_players[:, None]
```

The mask has the shape `[environment_count, players]`, and the right team's mask is formed in the
same way from `right_players`. It is moved to the observation's device once, with
`.to(observation.device)`. A right agent observes the teams swapped, so its own `left_team` takes
the right team's mask. A padded player and a player sent off both have `*_active` false, and the
mask tells them apart.

### Raw values alongside another representation

Most training reads a vector observation, while reward shaping, logging and evaluation often use a
raw value as well, such as the score or the owner of the ball. With `with_raw=True`, the batch also
writes the raw arrays. `batch.raw()` returns them for the last step, and `batch.final_raw()` for the
step that ended each episode. Writing every raw array adds to the cost of each step, which a short
timing of the batch at its intended size measures.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4, device='cpu', with_raw=True)
observation = batch.reset()
print(observation.shape, batch.raw()['ball_owned_team'].shape)
# torch.Size([4, 4, 115]) torch.Size([4, 4])
```

## Stacking

With `stacked=True`, the batch keeps the last four observations of each agent along the last axis,
the oldest first, as GRF's `stacked` option does. The last axis then holds 460 numbers for
`simple115` and `simple115v2`, and 16 planes for `extracted`. A reset fills all four places with its
first observation, and the stack of the step that ended an episode goes to `final_observation`.

GRF stacks the encoded representations, and cuju follows it, so `stacked=True` takes `simple115`,
`simple115v2` or `extracted` and raises `cuju.ConfigurationError` with `representation='raw'`.

A stacked batch saves and loads states with `stacked=True`, in GRF's format, which pickles the stack
inside the state.

## Single-agent batches

GRF drops the agent axis when an environment has one agent. `squeeze=True` does the same for a batch
whose environments all have exactly one agent. The observation and the reward lose their agent axis,
and `step` takes actions of shape `[environment_count]`. The raw observation keeps its agent axis,
as in GRF. A squeezed batch of eight environments of `academy_empty_goal_close` returns
observations of shape `[8, 115]`.

```python
batch = cuju.BatchEnvironment(
    'academy_empty_goal_close', environment_count=8, device='cpu', squeeze=True
)
print(batch.reset().shape)  # torch.Size([8, 115])
```

## Gymnasium spaces

`single_observation_space` and `single_action_space` describe one environment, as GRF's
`observation_space` and `action_space` do for the same options. `observation_space` and
`action_space` describe the whole batch, as in Gymnasium's vector environments. They require
`gymnasium`, installed with `pip install 'cuju[spaces]'`, or `gym` when Gymnasium is absent. The
observation spaces of `raw` are `None`, as GRF defines its raw observation as a dictionary without
a space.

## Memory and buffers

A vector observation occupies 460 bytes per agent, and an SMM of the default size 27,648 bytes per
agent, 60 times more. The batch keeps two of each, the current observation and
`final_observation`. A batch of 16,384 environments with four agents therefore holds 3.6 GB of SMM,
and four times that with `stacked=True`. A smaller `environment_count` or `channel_dimensions` fits
the SMM on a GPU with little memory.

Every output of a batch is a view of its buffers. Each call overwrites the tensors it returned
before, `raw()` and `final_raw()` included, and so does `set_state`, so a tensor kept across calls
is cloned first. The views cost no copy, which matters most on the GPU, where a copy of every
output at every step would take longer than the simulation.

Padded agents are masked with `agent_mask` and padded players with the mask of real players, since
their zeros otherwise read as players at the origin or as goalkeepers.

As in GRF, the directions in the observation are divided by `physics_steps_per_frame`, so a change
of that option changes their scale.

## See also

- [Rewards and actions](rewards-and-actions.md)
- [Episodes and seeds](episodes-and-seeds.md)
- [States](states.md)
- [Scenarios](scenarios.md)
- [Differences from GRF](differences-from-grf.md)
- [Common topics](faq.md)
