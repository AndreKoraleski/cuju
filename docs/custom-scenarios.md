# Custom scenarios

In cuju, a **custom scenario** is a scenario file in the format of Google Research Football, or GRF, a
Python file that defines `build_scenario(builder)`, which a batch builds into a bundle and plays
without GRF installed. The builder computes what GRF's engine computes when it builds a match, so the
batch plays GRF's episodes for that file bit for bit.

A build takes about a second, and `cuju.scenarios.build_bundle` or the bundle cache keeps its result
for later batches. A scenario file runs once, when its bundle is built, and a bundle holds one form
for the even episodes and one for the odd. The builder accepts a scenario that GRF plays and that a
bundle can hold, and raises `cuju.scenarios.ScenarioError` with the reason for any other.

## Scenario files

GRF describes a scenario with a Python file that defines `build_scenario(builder)`. The function
configures the match, adds the players of each team and places the ball. A file written for GRF runs
unchanged. A drill of 400 steps with two players on each team shows the form of such a file.

```python
# my_scenario.py
from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(0.6, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, 0.1, e_PlayerRole_CB)
```

The position that `AddPlayer` gives is where a player starts, at the reset and at kick-off. The spot
the built-in AI keeps it near during play, and its place at the other set pieces, come from GRF's
own 11-player formation, taken by the player's order in its team, rather than from that position.
The player still plays the role the file gives.

`from . import *` gives the file GRF's names for the roles and the teams, as it does inside GRF, so
GRF's own scenario files build as they stand. A file that names a package for these imports takes
them from `cuju.gfootball.scenarios`, which holds the same names. GRF's `gfootball.scenarios` holds
the role objects of GRF's compiled engine instead, and the builder answers them with a
`ScenarioError` that names the role it expected.

A batch takes the path in place of a bundle name. A scenario file gets one agent on the left, as
GRF's `create_environment` gives, and `agent_counts` sets others.

```python
import cuju

batch = cuju.BatchEnvironment(
    'my_scenario.py', environment_count=256, device='cpu', agent_counts=[(2, 0)]
)
```

## Building once

Each time a batch takes a scenario file, it builds the bundle first, which takes about a second.
`cuju.scenarios.build_bundle` writes the bundle to a file once, and a batch then loads it like any
shipped bundle.

```python
from cuju import scenarios

scenarios.build_bundle('my_scenario.py', 'my_scenario_l0_r0.cjb')
batch = cuju.BatchEnvironment(
    'my_scenario_l0_r0.cjb', environment_count=256, device='cpu', agent_counts=[(1, 0)]
)
```

The bundle holds the scenario without agents, `l0_r0`, and the batch sets up the agents it asks for
when it loads it. A bundle path plays the agents that the bundle holds, whatever the file is called,
so a batch of this bundle without `agent_counts` has none, and `batch.agents` is 0. The two backends
build the same bytes, and a build is deterministic, so a bundle can be cited by its SHA-256.
`name=` sets the name stored in the bundle.

With the environment variable `CUJU_BUNDLE_CACHE` set, a batch builds each scenario file once and
keeps its bundle in a cache. `1` takes `~/.cache/cuju/bundles`, under `XDG_CACHE_HOME` if set, and
any other value names the directory. The key is the file's bytes, its name, the animations, the
library and the builder, so an edited file builds again, and concurrent processes never read a
partial bundle. A batch given its own `animations` builds its scenario files with those tables, in
the cache and without it. The cache is off by default because a file that reads anything else, such as
`os.environ` or another file, would keep its first bundle for good. `cuju.scenarios.cached_bundle`
uses the cache explicitly, `cuju.scenarios.cache_directory()` names it, and the cache is safe to
delete at any time.

## What the builder computes

The builder does what GRF's engine does when it builds a match, from the same file. It computes the
statistics of every player, the formation of each team, the officials, the skeletons of the players
and the state right after the reset. A batch therefore plays GRF's episodes for that file, bit for
bit, with no GRF installation. The computation is cuju's own, checked against GRF's.

A scenario may read `builder.EpisodeNumber()`, as some of GRF's do, to change from one episode to the
next. A bundle holds one form for the even episodes and one for the odd, which covers every scenario
that GRF ships. `episode_number`, as GRF's option of that name, sets the number of the episode before
the first.

A scenario file runs only while its bundle is built, which for a batch means at its creation. So
everything that the file reads besides the parity of the episode number, such as `os.environ`,
another file or a variable of a module, is read then and stays fixed for the life of the batch. GRF
runs the file again at every reset, so a file that changes its scenario that way plays on in cuju as
it was built. Such reads are outside the builder's view, so the builder takes whatever they return
at that moment.

A deterministic scenario, with `builder.config().deterministic = True`, plays the seed that it sets,
42 by default, at every episode, as in GRF.

## Conditions on a scenario

The builder raises `cuju.scenarios.ScenarioError` for a scenario that GRF refuses or crashes on, and
for one that a bundle cannot hold. A scenario builds when it meets eight conditions, each with a
reason in GRF's engine or in the format of a bundle.

| Condition | Reason |
| --- | --- |
| at most 11 players per team | GRF's engine crashes on a larger team |
| a goalkeeper as the first player of each team | GRF's engine hangs in open play without one |
| values of the types that GRF's bindings convert, such as numbers for coordinates | GRF raises on other types, such as a string for a coordinate |
| a scenario that depends on the episode number through its parity alone | a bundle holds one form for the even and one for the odd episodes. The builder runs episodes 1 to 8 and a pair of episodes at each power of ten from 100 to one billion, so a scenario that changes only beyond those plays its early form |
| the same team sizes in every episode | a bundle holds one set of team sizes |
| Python's `random` module and NumPy's global generator left untouched | GRF draws the seed of each episode from `random` right after the scenario, so a draw would change every later seed |
| `controllable_left_players`, `controllable_right_players` and `dynamic_player_selection` left unread while the scenario builds | GRF computes those at the first read, before it sets the agents, and keeps them for the match |
| `reverse_team_processing` left off in a deterministic scenario | a bundle holds no field for it |

A scenario that sets `control_all_players` plays when each side has either no agent or one per
controllable player, the condition that GRF's engine asserts. GRF plays such a scenario through its
`MultiAgentToSingleAgent` wrapper, which `cuju.wrappers.MultiAgentToSingleAgent` ports, as covered in
[Migrating from GRF](migrating-from-grf.md).

`build_bundle` also warns when a form lasts longer than a batch plays at GRF's default of 10 ticks a
step, 5,119 steps, and names the largest `physics_steps_per_frame` that fits.

## Security

A scenario file is Python, and building it runs it with the permissions of the user who builds it,
exactly as GRF runs it. A scenario file is therefore trusted to the same degree as any
other Python program, and [SECURITY.md](../SECURITY.md) sets out the security policy of the project.

## Build time and variety

Building takes about a second, from about half a second to under two seconds for the shipped
scenarios and for `my_scenario.py` on the machines measured. A batch rebuilds a scenario file at every creation, so a program that creates
many batches builds once with `build_bundle` or sets `CUJU_BUNDLE_CACHE`.

GRF's own test scenarios under `gfootball/scenarios/tests` are scenario files too, and each of them
runs when it meets the [conditions on a scenario](#conditions-on-a-scenario).

Variety in positions comes from several files, or from two forms selected by the parity of the
episode, whose bundles mix in one batch as [Scenarios](scenarios.md) shows. Several files give as
many starting positions as there are files, each built once, and they keep the seeds of every
episode as GRF draws them, which positions drawn from `random` inside one file would shift.

## See also

- [Scenarios](scenarios.md), the shipped scenarios and batches that mix bundles
- [Migrating from GRF](migrating-from-grf.md), `control_all_players` and GRF's scripted players
- [Differences from GRF](differences-from-grf.md), these conditions together with every other known difference
- [Installation](installation.md), the environment variables, `CUJU_BUNDLE_CACHE` among them
