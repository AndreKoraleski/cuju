# Rewards and actions

In cuju, a **reward** is the value that each agent receives after a step, and an **action** is the
index that each agent chooses for the next step from an action set. Both follow Google Research
Football, or GRF. The two rewards of GRF, `scoring` and the checkpoint reward, are computed in the
same way, and the three action sets, `default`, `v2` and `full`, have the same actions at the same
indices.

The reward of an agent on the right team has the sign of the right team. The checkpoint reward
depends in its last bits on the version of numpy under GRF, and the option `numpy_legacy` selects
the matching precision. Movement and a few other actions are sticky and stay pressed until
released. Actions in host memory are checked as GRF checks them, and actions already on the GPU go
to the device as they are, with an index out of range played as `idle`, unless the option
`check_actions` checks them as well.

## Rewards

The option `rewards` names the rewards as GRF's option does. It accepts two values, which the table
describes.

| `rewards` | Reward of an agent at each step |
| --- | --- |
| `scoring` | 1 when its team scores, -1 when the other team scores, 0 otherwise |
| `scoring,checkpoints` | the scoring reward plus the checkpoint reward |

The reward has the shape `[environment_count, agents]`, one `float32` value per agent. As in GRF, a
batch requires `scoring`, so `rewards='checkpoints'` alone raises `cuju.ConfigurationError`, a
`ValueError`, where GRF fails an assertion. A part other than `scoring` and `checkpoints`, such as a
misspelt name, draws a warning, since GRF ignores such a part without a message.

### Checkpoint reward

GRF's checkpoint reward, `CheckpointRewardWrapper`, rewards carrying the ball toward the goal. It
divides the distance from the ball to the opponent's goal into 10 zones, with thresholds from 0.99
down to 0.19 in the raw units of the pitch. Each time an agent's active player owns the ball inside
a zone it has not yet reached, the agent receives 0.1 for that zone and for every farther zone it
has skipped. A goal by the agent's team pays the zones that remain, so an agent whose team scores
always collects the full 1.0 of checkpoints in that episode. The count of collected zones belongs
to the agent and resets with the episode.

The reward provides a learning signal before a policy scores at all, which matters in the
11-a-side scenarios, where a random policy almost never scores. It also shapes behaviour, as
policies trained with it tend to dribble forward. Kurach and colleagues compare the two rewards in
the GRF paper of 2020. In the example, 64 environments of `academy_empty_goal_close` play the
action `shot` with both rewards.

```python
import cuju
import torch

batch = cuju.BatchEnvironment(
    'academy_empty_goal_close', environment_count=64, device='cpu', rewards='scoring,checkpoints'
)
batch.reset()
actions = torch.full((64, 1), 12, dtype=torch.int32)  # 12 is `shot`
for _ in range(20):
    observation, reward, done, information = batch.step(actions)
```

### Reward of a right agent

A right agent plays for the right team, and its reward has the sign of its own team, 1 when the
right team scores, as in GRF. `information['score_reward']` holds the raw scoring signal of each
environment, `[environment_count]` of `int32`, from the left team's point of view and before the
checkpoint reward, as GRF's `info['score_reward']` does. The `score` of the `raw` representation
is from each agent's own side, with its own team's goals first.

Each agent's own scoring reward, apart from the checkpoints, follows from signing `score_reward` by
side.

```python
is_right = (torch.arange(batch.agents) >= batch.left_agents[:, None]) & batch.agent_mask
sign = torch.where(is_right, -1, 1) * batch.agent_mask  # 0 for padded agents
own_score = information['score_reward'][:, None] * sign.to(reward.device)  # [environments, agents]
```

With `rewards='scoring'`, `own_score` equals the reward.

### Sums and bounds

Self-play with agents on both sides gives each side its own reward. With equal numbers of agents on
each side and without checkpoints, the rewards of all agents of an environment sum to zero at every
goal, as in a zero-sum game, so a training curve of their sum stays flat. With other counts the sum
of a goal is the difference of the counts, so in `5_vs_5_l2_r3` a goal sums to 1 or -1, and
checkpoints add their own.

The checkpoint reward at a goal adds the zones not yet collected to the scoring reward of 1, so a
scoring step pays an agent at most 2.0.

### Precision under numpy 1 and numpy 2

GRF adds the checkpoint reward with numpy, and numpy 2 changed how a Python float combines with a
numpy `float32`, in the proposal NEP 50. The same reward can therefore differ in its last bits
between a GRF that runs on numpy 1 and one that runs on numpy 2. `numpy_legacy=True`, the default,
adds in the precision of numpy before version 2, and `numpy_legacy=False` follows numpy 2. GRF pins
no numpy version, so the matching setting depends on the numpy installed with the GRF under
comparison. The expression `numpy_legacy=int(numpy.__version__.split('.')[0]) < 2`, evaluated where
that GRF runs, gives it. For training, the two settings differ only in the last bits of a
`float32`.

## Actions

An action is an index into the action set. `step` takes one action for each agent of each
environment, `[environment_count, agents]` of integers. The option `action_set` selects one of
three sets, each extending the one before, and `batch.action_count` gives the size of the chosen
set.

| `action_set` | Actions | Contents |
| --- | --- | --- |
| `default` | 19 | GRF's `action_set_v1` |
| `v2` | 20 | the 19, then `builtin_ai`, which hands the player to the built-in AI for that step |
| `full` | 33 | the 20, then `keeper_rush`, `pressure`, `team_pressure`, `switch` and the release of each button |

The default set numbers its 19 actions from 0 to 18, as the table shows.

| Index | Action | Index | Action |
| --- | --- | --- | --- |
| 0 | `idle` | 10 | `high_pass` |
| 1 | `left` | 11 | `short_pass` |
| 2 | `top_left` | 12 | `shot` |
| 3 | `top` | 13 | `sprint` |
| 4 | `top_right` | 14 | `release_direction` |
| 5 | `right` | 15 | `release_sprint` |
| 6 | `bottom_right` | 16 | `sliding` |
| 7 | `bottom` | 17 | `dribble` |
| 8 | `bottom_left` | 18 | `release_dribble` |
| 9 | `long_pass` | | |

The 13 actions that `full` adds, in order from index 20, are `keeper_rush`, `pressure`,
`team_pressure`, `switch`, `release_long_pass`, `release_high_pass`, `release_short_pass`,
`release_shot`, `release_keeper_rush`, `release_sliding`, `release_pressure`,
`release_team_pressure` and `release_switch`.

### Choice of action set

`default` is the set of GRF's papers and of the Kaggle competition, so a policy trained on it is
comparable with published results and plays in a Kaggle submission unchanged, and its 19 actions
keep the policy's output small. `v2` adds one action, `builtin_ai`, for an agent that hands its
player to the built-in AI at times, as in a curriculum or a partly learned team. `full` adds the
defensive buttons and the explicit releases. Each added action spreads exploration over one more
choice, so the larger set suits work that needs those buttons, such as GRF's `bot` player, which
requires `full`.

In `v2` and `full`, index 19 lets the built-in AI play the agent's player for one step. The action
serves curricula and debugging, and as GRF's own action it makes the player behave exactly as GRF's
AI would. A side whose agents all play 19 still has its agents, with their observations and
rewards, so its matches differ from the first step from those of a side without agents, such as
`_r0`. [Recipes](recipes.md) compares the two.

### Sticky actions

The eight directions, `sprint` and `dribble` are sticky, and in `full` also `keeper_rush`,
`pressure` and `team_pressure`. Once taken, each stays pressed, as on a gamepad. A direction holds
until `release_direction` or another direction replaces it, the other buttons hold until their own
release action, and `idle` leaves every one pressed. The directions turn with the side, as the
observation does. For every agent, `left` points toward its own goal and `right` toward the
opponent's, so a policy can play either side without knowing which one it plays.

The `sticky_actions` array of the `raw` representation shows which are pressed for each agent, one
flag per sticky action in the order of the action set, as in GRF, 10 flags for `default` and `v2`
and 13 for `full`. Each flag has a fixed index, which the table gives.

| Index | Sticky action |
| --- | --- |
| 0 to 3 | `left`, `top_left`, `top`, `top_right` |
| 4 to 7 | `right`, `bottom_right`, `bottom`, `bottom_left` |
| 8 | `sprint` |
| 9 | `dribble` |
| 10 to 12 | `keeper_rush`, `pressure`, `team_pressure`, in `full` only |

### Index range and checks

GRF looks up each action in a Python list, so an index from `-action_count` to -1 counts from the
end of the set, and cuju accepts the same indices, from -19 to 18 for `default` and from -33 to 32
for `full`. An index outside that range raises `cuju.ActionRangeError`, an `IndexError`, before any
environment steps, and a floating-point action raises `TypeError`, as GRF accepts only integers.

On the GPU, actions that already lie on the device go to the step as they are, which keeps every
step free of a wait for the device. An index out of range there plays as `idle` for that agent,
where GRF would raise, so the batch plays a match that GRF never plays. Actions stay in range by
construction when they come from an `argmax` or a sample over `action_count` logits.

The option `check_actions=True` checks the actions on the GPU as well, against the same range, and
raises `ActionRangeError` naming the first action out of range, its agent and its environment,
before any environment steps. The check reads its result on the host, so every step waits for the
device, which suits the first runs of a new policy or action pipeline more than a long training run.
A CUDA graph capture skips the check, since a capture admits no wait. The option `debug=True`
includes the same check together with a wait after every step, as described in
[Backends](backends.md). A CPU batch checks every action at the call in any case and runs the same
code, so it serves for debugging as well.

The batch ignores the actions of padded agents, the slots beyond an environment's own agents in a
mixed batch, on both devices.

## See also

- [Observations](observations.md)
- [Episodes and seeds](episodes-and-seeds.md)
- [Recipes](recipes.md)
- [Differences from GRF](differences-from-grf.md)
- [Common topics](faq.md)
