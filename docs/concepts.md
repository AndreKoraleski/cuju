# Concepts

In cuju, a **batch** is a set of environments of Google Research Football, or GRF, that one call to
`step` advances together, and almost every difference between cuju's interface and GRF's follows
from it. A batch holds thousands of environments on one NVIDIA GPU or on the cores of a CPU, writes
its outputs into buffers that it owns, and returns them as PyTorch tensors. Both backends run the
same C++ code and produce GRF's bits for the same seed and actions.

The batched design gives the interface five properties. Outputs are shared buffers, episodes reset
inside the step, every environment has room for the same number of agents, options are fixed at
creation, and the seed of each environment depends on its index alone. A batch is built from three
kinds of data, the bundles of its scenarios, the animation tables that all scenarios share, and one
world per environment. The rest of cuju's vocabulary, from the environment to the padded slot,
keeps GRF's meanings where GRF has the term.

## Terms

The terms of cuju's interface have fixed meanings.

| Term | Meaning |
| --- | --- |
| environment | One match together with its episode logic, the score, the rewards, the end of the episode and the next reset. It is what `gfootball.env.create_environment` returns in GRF. |
| batch | The environments that one call to `step` advances together. A `cuju.BatchEnvironment` holds one batch. |
| backend | Where a batch runs, `cuda` on one NVIDIA GPU or `cpu` on the cores of the host. |
| scenario | GRF's description of a match, the teams, their players and roles, where the ball starts, how long the match lasts and how hard each team plays. `5_vs_5` and `academy_corner` are scenarios. |
| scenario file | A Python file that defines `build_scenario(builder)`, the way GRF writes its scenarios. |
| bundle | A scenario turned into the data that a batch loads, the static data of the match, the state right after a reset and the skeletons of the players. Bundles end in `.cjb`. |
| player | Anyone on the pitch who belongs to a team. |
| agent | A player whose actions come from the caller's policy rather than from the built-in AI. |
| active player | The player that an agent controls at a given moment. With fewer agents than players, it changes during a match. |
| built-in AI | GRF's AI, which drives every player without an agent. |
| tick | One update of the simulation, 10 milliseconds of match time. |
| step | One call to `step`, a round of `physics_steps_per_frame` ticks, 10 by default, which repeats while the ball is out of play, as in GRF. |
| episode | The steps from a reset to the end of a match. |
| representation | The encoding of an observation, `simple115`, `simple115v2`, `extracted` (the SMM) or `raw`. |
| action set | The actions an agent can take, `default` (19), `v2` (20) or `full` (33). |
| difficulty | A number from 0 to 1 for each team, which scales the statistics of its players and slows the reactions of its built-in AI. |
| form | The version of a scenario that an episode plays. A bundle holds an even and an odd form, by the parity of GRF's episode number. The two forms of `5_vs_5` and of the 11-a-side matches alternate the kick-off, and those of the academies, `1_vs_1_easy` and `test_example_multiagent` are the same. |
| state | The whole match of one environment as bytes in GRF's format, from `get_state`. |
| padded slot | Room in an output for an agent or a player that an environment does not have, in a batch that mixes bundles. It holds zeros. |

## From one environment to a batch

GRF runs one match per process. Each `step` crosses from Python into the C++ engine, runs 10 ticks,
encodes an observation and returns. On one processor of an AMD EPYC 7763, this gives about 207
steps per second for `5_vs_5` with four agents a side and 163 for `11_vs_11_stochastic`, as
measured in [Performance](performance.md), and training on GRF is usually scaled by starting one
process per core and collecting from all of them.

cuju keeps GRF's matches and changes how many run at once. A batch holds thousands of environments,
and one call advances all of them. On the GPU each environment runs on one warp of 32 threads, and
the whole batch is one kernel launch. On the CPU a pool of threads takes the environments one by
one, which keeps the cores busy when some matches cost more than others. On both backends the
observations, rewards and done flags land in buffers that the batch owns and are read as PyTorch
tensors without a copy.

The two backends run the same C++ code, written once and compiled for the device and for the host.
They give the same bits, and both give GRF's bits. For the same seed and actions, every state,
observation, reward and done flag equals what GRF produces. The precise claim and the check that
confirms it on a given machine are the subject of [Exactness](exactness.md).

## Properties of the batched interface

Five properties of the interface follow from the cost of running thousands of matches together.

**Shared output buffers.** A batch writes its observations into the same memory at every step, and
`reset` and `step` return tensors that view that memory. Copying thousands of observations each step
would cost more than the simulation on the GPU, so the batch returns views. The next step overwrites
what the last one returned, and an observation that has to outlive its step is cloned. A clone
taken at the reset keeps the first observation after the step has overwritten the buffer.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=8, device='cpu')
observation = batch.reset()
kept = observation.clone()  # a copy that the next step leaves alone
actions = torch.zeros((8, 4), dtype=torch.int32)
observation, reward, done, information = batch.step(actions)
# False, the step overwrote the buffer, and `kept` holds the reset
print(torch.equal(kept, observation))
```

**Reset inside the step.** In a batch of thousands of matches, a few end at almost every step.
Stopping the whole batch to reset them would leave most of the hardware idle, so the step that ends
an episode starts the next one in the same call. Its done flag is 1, its observation is the first
of the next episode, and the last observation of the episode that ended is held in
`information['final_observation']`. The automatic reset and the option `auto_reset=False`, which
leaves the reset to the caller, are covered in [Episodes and seeds](episodes-and-seeds.md).

**A fixed number of agent slots.** The outputs are tensors with a fixed shape,
`[environment_count, agents, ...]`. A batch can mix scenarios with different numbers of agents, so
`agents` is the largest number of any bundle, and `batch.agent_mask` marks the agents that exist in
each environment. The other slots output zeros, and the batch ignores their actions.

**Options fixed at creation.** Everything that shapes the buffers is fixed when the batch is
created. The seed of an environment's next episode remains open and is set at its reset through
`reset(seeds=)`.

**Seeds by environment index.** Environment `index` of a batch with seed `seed` plays the episodes
that GRF plays after `random.seed(seed * 2**32 + first_index + index)`, where `first_index` is 0 by
default and offsets the indices when a run is split over machines. The same index plays the same
matches whatever the size of the batch, which makes runs reproducible across machines of different
sizes.

## Components of a batch

A batch is made of three kinds of data.

**Bundles.** A batch loads one bundle for each entry of the list of scenarios that it is given. The
package ships one bundle for each of GRF's 19 scenarios, and a name such as `5_vs_5_l4_r0` asks for
that scenario with four agents on the left and none on the right, which the batch sets up when it
loads the bundle. The shipped bundles are listed in [Scenarios](scenarios.md).

**Animation tables.** One file, `animations.bin`, serves every scenario. It holds GRF's player
animations, the motion data from which the engine chooses how a player runs, turns, kicks and
falls.

**Worlds.** Each environment has one world, the state of its match, which the batch allocates once
and updates in place.

Loading a batch parses the tables and builds its worlds, which takes a moment. After that, a step
reads only memory that the batch already holds. A batch keeps its buffers until `close()` or the end
of a `with` block, and every tensor it returned views those buffers until then.

## Performance characteristics

The first call on the GPU takes longer than the rest while CUDA loads the kernel, so timings start
after a few steps. The GPU reaches its throughput with thousands of environments, since each
environment occupies one warp and a batch of a handful leaves most of the device idle. On the CPU, one environment of cuju already runs
several times as many steps per second as one GRF process, 9.75 times for `5_vs_5` on one processor
of the machine measured. Figures for both backends are given in [Performance](performance.md).

Every output shares memory with the batch. Of the five properties of the interface, this one
changes the behaviour of code ported from GRF without raising an error, and a clone keeps any
output that has to last beyond its step.

## See also

- [Exactness](exactness.md), the meaning of bit-exact and the check that confirms it
- [Episodes and seeds](episodes-and-seeds.md), the automatic reset and the seed of each environment
- [Scenarios](scenarios.md), the shipped bundles and the naming of agents
- [Performance](performance.md), the throughput of each backend by batch size
- [Differences from GRF](differences-from-grf.md), the places where cuju differs from GRF on purpose
