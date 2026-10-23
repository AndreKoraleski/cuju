# Scenarios

In cuju, a **scenario** is the description of a match in Google Research Football, or GRF, the teams,
their players and roles, the position of the ball, the length of an episode and the difficulty of
each team. cuju ships the 19 scenarios of GRF as bundles, from full 11-a-side matches to short
academy drills, and plays any number of agents on either side of each, from none up to the size of
the team.

A suffix such as `_l4_r0` on a scenario's name sets the agents of each team, and with fewer agents
than controllable players each agent controls the player that GRF makes active. One batch holds
several bundles at once, mixing scenarios, team sizes and sides, and a pair of difficulties per
bundle overrides the scenario's own. Some drills reward a constant action, so a result on them is
weighed against such a baseline.

## Shipped scenarios

Each of the 19 scenarios has a team size, a length in steps and a difficulty for each team, all taken
from GRF.

| Scenario | Players | Steps | Difficulty, left and right | What it is |
| --- | --- | --- | --- | --- |
| `11_vs_11_stochastic` | 11 v 11 | 3,001 | 1.0, 0.6 | A full match against the built-in AI |
| `11_vs_11_easy_stochastic` | 11 v 11 | 3,001 | 1.0, 0.05 | The same against a weak right team |
| `11_vs_11_hard_stochastic` | 11 v 11 | 3,001 | 1.0, 0.95 | The same against a strong right team |
| `11_vs_11_competition` | 11 v 11 | 3,001 | 1.0, 1.0 | A full match with both teams at full strength |
| `11_vs_11_kaggle` | 11 v 11 | 3,001 | 1.0, 1.0 | The match of the Kaggle competition, with a second half from step 1,500 |
| `5_vs_5` | 5 v 5 | 3,001 | 0.05, 0.05 | A full match of five-a-side |
| `1_vs_1_easy` | 1 v 1 | 501 | 0, 0 | One player against one |
| `academy_empty_goal_close` | 2 v 1 | up to 401 | 1.0, 0.6 | Score from close to an empty goal |
| `academy_empty_goal` | 2 v 1 | up to 401 | 1.0, 0.6 | Score into an empty goal from the middle of the pitch |
| `academy_run_to_score` | 2 v 6 | up to 401 | 1.0, 0.6 | Run from the middle to an empty goal, with five defenders chasing |
| `academy_run_to_score_with_keeper` | 2 v 6 | up to 401 | 1.0, 0.6 | The same, with a goalkeeper |
| `academy_pass_and_shoot_with_keeper` | 3 v 2 | up to 401 | 1.0, 0.6 | Two attackers at the edge of the box, the one with the ball marked and the one in the centre free |
| `academy_run_pass_and_shoot_with_keeper` | 3 v 2 | up to 401 | 1.0, 0.6 | The same, but the one with the ball is free and the one in the centre is marked |
| `academy_3_vs_1_with_keeper` | 4 v 2 | up to 401 | 1.0, 0.6 | Three attackers against one defender and a goalkeeper |
| `academy_corner` | 11 v 11 | up to 401 | 1.0, 0.6 | Score from a corner |
| `academy_counterattack_easy` | 11 v 11 | up to 401 | 1.0, 0.6 | A counterattack, four attackers against one defender |
| `academy_counterattack_hard` | 11 v 11 | up to 401 | 1.0, 0.6 | A counterattack, four attackers against two defenders |
| `academy_single_goal_versus_lazy` | 11 v 11 | up to 3,001 | 1.0, 0.6 | Score once against a team whose outfield players stand still and intercept only nearby balls |
| `test_example_multiagent` | 4 v 2 | up to 101 | 1.0, 0.6 | Three attackers against one defender, with the right team's goalkeeper placed in the left team's goal, the scenario of GRF's example of training several agents |

The players include the goalkeepers. A step count is the length of an episode as GRF counts it. Each
academy ends early on a goal, when the ball goes out of play, or, in every academy but the corner,
when the other team gains possession, so its episodes are usually much shorter.
`test_example_multiagent` ends on the same three events. The academies and `test_example_multiagent`
keep GRF's default difficulties, 1.0 for the left team and 0.6 for the right. In `11_vs_11_kaggle`,
play restarts after step 1,500 with a kick-off by the team that did not kick off the first half, and
the teams keep their ends, so no agent's view of the pitch flips. `5_vs_5` and the `11_vs_11_*`
scenarios alternate the kick-off from one episode to the next, by the parity of the episode number,
while the academies, `1_vs_1_easy` and `test_example_multiagent` play the same form in every
episode, a behaviour covered in [Episodes and seeds](episodes-and-seeds.md).

`cuju.bundled_scenarios()` lists the names. The bundles are built from GRF's own files of these
scenarios, which are in [GRF's repository at commit 3d9e754][grf-scenarios] and make the starting
point for a variant of one, written as a [custom scenario](custom-scenarios.md).

[grf-scenarios]: https://github.com/google-research/football/tree/3d9e754720a95621bba6475c4d3b0d56fe919014/gfootball/scenarios

## Constant-action baselines

A single repeated action solves some drills. A policy that always shoots, whatever the observation
says, scores in 98% of the episodes of `academy_empty_goal_close` and in 91% of those of
`academy_empty_goal`, measured over 64 environments and 400 steps, so a high scoring rate there says
little about what a policy has learned. In `academy_pass_and_shoot_with_keeper`,
`academy_run_to_score_with_keeper` and `academy_3_vs_1_with_keeper` the same policy scored four
goals in over 2,000 episodes, so scoring there takes a learned policy. A result on any scenario is
therefore weighed against a constant-action baseline measured on the same scenario, which costs a
few seconds on the CPU and shows how much of a scoring rate the policy itself earns.

## Agents on each side

A name `<scenario>_l<left>_r<right>` asks for that scenario with `left` agents on the left team and
`right` on the right, each from 0 to the players of the team, as GRF's
`number_of_left_players_agent_controls` and `number_of_right_players_agent_controls` do. A few
names show the range of the suffix.

| Name | Agents |
| --- | --- |
| `5_vs_5` | one on the left, as GRF's `create_environment` gives by default |
| `5_vs_5_l4_r0` | four on the left against the built-in AI |
| `5_vs_5_l4_r4` | four on each side, for self-play |
| `5_vs_5_l0_r0` | none, the built-in AI plays both teams |
| `11_vs_11_stochastic_l11_r0` | a whole team of agents |

The agents of an environment come in a fixed order, the left agents first. A right agent sees the
pitch turned by 180 degrees, its actions turn the same way, and its rewards have the sign of its own
team, as set out in [Rewards and actions](rewards-and-actions.md). When a team has as many agents as
controllable players, each agent keeps its player for the whole episode. An agent without a player,
such as the fifth of `5_vs_5_l5_r0`, whose goalkeepers are not controllable, still counts in
`agent_mask`, and its `active` in `raw` is -1.

**Fewer agents than players.** When a team has more controllable players than agents, each agent
controls the player that GRF makes active for it, which changes during the match. It is usually the
player nearest the ball, chosen by GRF's own logic, and in the 11-a-side scenarios that can be the
goalkeeper. The built-in AI drives everyone else. This is GRF's dynamic player selection, and the
way GRF's single-agent benchmarks work.

**Goalkeepers.** A scenario can mark players as not controllable. `5_vs_5` does so for both
goalkeepers, so four players of each team can be controlled there. `5_vs_5_l5_r0` still gives five
agents, one of which never has a player of its own, exactly as in GRF. In the 11-a-side scenarios
every player, the goalkeeper included, can be controlled.

One bundle per scenario serves every number of agents. The batch sets up the agents when it loads
the bundle, as GRF does when it builds a match.

## Mixed batches

A batch can hold several bundles. `assignment` gives each environment the index of its bundle, so
one batch mixes scenarios, team sizes and sides, and self-play, play against the built-in AI and
evaluation share one launch.

```python
import cuju
import torch

batch = cuju.BatchEnvironment(
    ['5_vs_5_l4_r4', '5_vs_5_l4_r0', '11_vs_11_stochastic_l1_r0'],
    environment_count=96,
    device='cpu',
    assignment=torch.arange(3).repeat_interleave(32),
)
print(batch.agents)  # 8, the most agents of any bundle
print(batch.agent_mask.sum(1))  # 8 for the first 32, 4 for the next 32, 1 for the last 32
```

Every output has room for `batch.agents` agents. `batch.left_agents` and `batch.right_agents` give
the agents of each environment, and `batch.agent_mask` the slots that exist. The other slots output
zeros, and the batch ignores their actions. The raw team arrays have room for `batch.players`
players, the real players first and zeros after them, and `batch.left_players` and
`batch.right_players` give the number of real players in each environment, as detailed in
[Observations](observations.md).

## Difficulty

Each team has a difficulty from 0 to 1, which the scenario sets. `difficulty` overrides it for the
teams of each bundle, a pair per bundle, with `None` to keep the scenario's value. A pair plays
exactly GRF's matches of the scenario file with `left_team_difficulty` and `right_team_difficulty`
set to it.

```python
batch = cuju.BatchEnvironment(
    ['5_vs_5_l4_r0', '5_vs_5_l4_r0'],
    environment_count=64,
    device='cpu',
    assignment=torch.arange(2).repeat_interleave(32),
    difficulty=[(None, None), (None, 0.95)],
)
```

The difficulty belongs to the team, as in GRF, rather than to the built-in AI. It scales the
statistics of every player of the team by `0.3 + 0.7 * difficulty`, the players of agents included,
and the players that the built-in AI drives react up to 100 milliseconds later as it falls. Lowering
the right team's difficulty thus weakens the opponents, and lowering the left team's weakens the
agents' own players too. `batch.difficulty` reports the difficulties that play in each environment,
the scenario's where nothing overrides them. It is a copy, so a write into it leaves the batch
unchanged.

**Self-play.** `11_vs_11_competition`, `11_vs_11_kaggle`, `5_vs_5` and `1_vs_1_easy` give both
teams the same difficulty. In the other scenarios, agents on the right play with weaker players than
those on the left. In `11_vs_11_stochastic` that is 0.6 against 1.0, which scales their statistics
by 0.72 against 1.0. An even match sets both sides, as `difficulty=[(1.0, 1.0)]`.

## Cost, length and naming

An 11-a-side environment costs more than a five-a-side one, since the built-in AI and the animation
of every player run each tick, and [Performance](performance.md) gives the figures for both. A mixed
batch steps as fast as its most expensive environments allow on the GPU, and spreads the cost evenly
on the CPU.

Academy episodes often end within tens of steps. The automatic reset starts the next one in the
same step, and comparisons with GRF's results count episodes rather than steps.

Every name of the form `<scenario>_l<left>_r<right>`, such as `academy_corner_l3_r2`, loads the one
shipped bundle of its scenario, which holds the match without agents, and the batch sets up the
agents that the name asks for. Matches beyond the shipped scenarios are written as scenario files,
in GRF's format, as [Custom scenarios](custom-scenarios.md) describes.

## See also

- [Custom scenarios](custom-scenarios.md), scenario files written the way GRF writes them
- [Episodes and seeds](episodes-and-seeds.md), the end of an episode and the form it plays
- [Rewards and actions](rewards-and-actions.md), the rewards and actions of agents on each side
- [Observations](observations.md), the shapes and padded slots of a mixed batch
- [Performance](performance.md), the cost of each scenario on each backend
