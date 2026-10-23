# States

In cuju, a **state** is the complete record of one match at one step, serialized as bytes in the
format of Google Research Football, or GRF. `get_state` saves the state of one environment, and
`set_state` loads a state into any environment of the same configuration, or into GRF, which reads
the same bytes. A loaded state continues bit for bit as the match that saved it.

A state captures a moment, while a seed captures an episode from its kick-off. Seeds replay whole
episodes without copies, and states start play in the middle of a match, branch continuations from
it and move matches between cuju and GRF, at the cost of a copy between the device and the host and
up to about 110 KB per state. A state holds GRF's pickle of the episode logic and GRF's serialized
engine state, and `set_state` reads the pickle with a restricted unpickler.

## Saving and restoring

`get_state(index)` returns the state of one environment as `bytes`. `set_state(index, state)` sets
an environment to such bytes, its own or another's. In the example, environment 1 continues from
the state of environment 0 after 100 steps.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4, device='cpu')
observation = batch.reset()
actions = torch.zeros((4, 4), dtype=torch.int32)
for _ in range(100):
    observation, reward, done, information = batch.step(actions)
state = batch.get_state(0)  # environment 0 after 100 steps
batch.set_state(1, state)  # environment 1 continues from the same moment
```

`set_state` writes into the tensors that the batch has already returned. Every output of a batch is
a view of its buffers, which the next call overwrites, and `set_state` is such a call. After it, the
`observation` returned by the last step already shows the loaded state in row 1, and the reward,
done flag and `score_reward` of that environment read 0, while `final_observation` keeps what it
held. The reward and the done flag of a step are therefore copied before a state is loaded into an
environment that the step ended. From there, the same actions make the two environments play the
same match, bit for bit.

A batch can also start from states instead of `reset`. Once every environment has a state, `step`
runs.

## States and seeds

Two mechanisms restore a match, for different purposes. A seed names an episode from its kick-off.
An environment given the seed with `reset(seeds=...)` and the same actions replays the episode bit
for bit, with no copy and no wait, as described in [Episodes and seeds](episodes-and-seeds.md). A
state names a moment. It holds everything the match is at one step, so it needs no actions to reach
it, and it is the means of starting in the middle of a match and of moving a match between cuju and
GRF.

Evaluation over fixed matches, a sweep that compares policies on the same opponents and the
reproduction of a whole episode all start at a kick-off, and for them a seed costs nothing. A state
serves where the moment matters more than the start, to study one situation, to branch several
continuations from it, to begin training from it, or to hand it to GRF. Each call copies between
the device and the host, and a state occupies up to about 110 KB, which suits occasional use. A
batch that saved every environment at every step would spend its time copying.

## Format

The bytes follow GRF's layout, the length of a pickle as an `int32`, the pickle of the episode
logic, then the serialized match.

The pickle holds what GRF's Python side keeps between steps, the previous score difference, the
previous game mode and the team that owned the ball, under GRF's class
`gfootball.env.football_env_core.EnvState`. With `rewards='scoring,checkpoints'` it also holds the
checkpoints that each agent has collected, under `CheckpointRewardWrapper`, and with GRF's stack of
four, the stacked observations under `FrameStack`, as GRF stores them.

The match is GRF's engine state, with every player, the ball, the officials, the animations in
progress, the memory of the AI and the clocks, in the layout that GRF's engine writes.

Because the format is GRF's, a state moves between the two engines in both directions, as the
example shows with GRF installed.

```python
import gfootball.env as football_env

env = football_env.create_environment(
    '5_vs_5', representation='simple115v2', number_of_left_players_agent_controls=4
)
env.reset()
env.set_state(batch.get_state(0))  # GRF continues the match that cuju played
batch.set_state(2, env.get_state())  # and cuju continues the one that GRF played
```

A state from GRF continues in cuju exactly as the match that saved it. A state from cuju continues
in GRF as GRF's own states do, with one property of GRF's loader to keep in mind. Directly after
`set_state`, a GRF environment that holds fewer of the AI's mental images than the state, as a
freshly reset one does, can drift from the saved match when the state has `reverse_team_processing`
on, as described in [Differences from GRF](differences-from-grf.md#output-differences). A
comparison of the two engines from a state therefore moves the state from GRF to cuju, as
`examples/compare_with_grf.py` does.

The two sides agree on the scenario and on the options that shape the state, such as the rewards.
The bytes are equal in meaning, and they can differ in bits in two places. The pickle that cuju
writes uses protocol 2 and puts the episode logic first, where GRF's uses Python's default protocol,
and each side reads the other's. GRF writes the padding between the fields of its engine's
structures as it finds it in memory, so two identical GRF runs give match bytes that differ in a
few hundred padding bytes, while cuju writes zeros there. States are therefore compared field by
field or after loading, rather than as raw bytes. A GRF state loaded into cuju and saved again comes
back byte for byte.

## Scope of a state

A state holds the match, and three related parts belong to the environment or to its options, the
random generator, the scenario and the stack.

The random generator that draws the seeds of later episodes belongs to the environment. After
`set_state`, the current episode continues exactly, and the next episode draws its seed from the
generator of the environment that loaded the state. GRF behaves the same way, since its seeds come
from Python's `random` module, which lies outside its states. If the environment had not reset
before, that next episode is its first, as defined in [Episodes and
seeds](episodes-and-seeds.md#seeds), and otherwise it is the next of its sequence.

A state belongs to its scenario, its agent counts, its team difficulties and its end flags.
`set_state` loads states of the same configuration and raises `cuju.StateError` for any other,
naming up to three fields that differ, such as `right_agents is 0 in the state and 1 in the
environment`. GRF likewise loads a state only into an environment with the same agent counts. A
state of another difficulty loads into an environment of a bundle with that difficulty, set with
`difficulty` at creation.

A batch with `stacked=True`, GRF's stack of four, saves and loads with
`get_state(index, stacked=True)` and `set_state(index, state, stacked=True)`, which put the stack in
the state as GRF's `FrameStack` does. Such states move to and from a GRF created with
`stacked=True`. A stacked batch takes stacked states and an unstacked batch unstacked ones, and the
other kind raises `StateError`.

The environment of `cuju.env.create_environment` writes the same entries from its options. With
`rewards='scoring,checkpoints'` and `stacked=True` its `get_state()` holds the checkpoints and the
stack, and GRF's wrappers from `cuju.gfootball.env.wrappers`, built by hand over a raw environment,
add theirs as GRF's do, `CheckpointRewardWrapper` its counts and `FrameStack` its frames. The
entries are written the same way from either source, so a stack built by hand and
`create_environment` with the same arguments give equal bytes, and each loads the other's states.

One detail lies outside the state in both engines. When a goalkeeper is sent off, the player who
takes the keeper's place is part of the team data of the environment rather than of the match. A
state loaded into another environment therefore takes that environment's hand-over, in GRF and in
cuju alike.

## Safety

A state is a pickle, and unpickling untrusted data can run code. `set_state` reads the pickle with a
restricted unpickler that accepts only GRF's episode class, plain values and, for a stacked state
or a state of `create_environment`, numpy arrays in a `deque`, and raises `StateError` for anything
else. It remains a deserializer
that outside bytes reach, so states are loaded from trusted sources. [SECURITY.md](../SECURITY.md)
describes how to report a problem.

## Uses

States serve four common purposes. In search, a saved state branches into several continuations in
other environments, of which the best is kept. One state copied into hundreds of environments rolls
out in one step each. Each `set_state` costs milliseconds, so a search loads once and rolls out for
many steps rather than reloading at every step.

For starting positions, training begins from chosen moments, such as a corner or a counterattack,
by loading saved states into environments that have ended. Loading into a few environments at a
time keeps the copies small, since a batch that resets every environment from states spends its
time copying. A moment that can be written as a scenario starts at lower cost from a scenario file,
since a reset from a bundle copies nothing, as described in [Custom
scenarios](custom-scenarios.md).

For debugging, a state saved before a surprising event replays it any number of times.

For viewing, a match from cuju is shown in GRF, either from a dump of the match, described in
[Migrating from GRF](migrating-from-grf.md#dumps-from-a-batch), or from a state loaded into a GRF
that renders. Rendering changes how GRF plays, as described in [Differences from
GRF](differences-from-grf.md#rendering-players-and-wrappers), so the rendered continuation becomes
a different match from the moment rendering starts.

## Cost

`get_state` and `set_state` copy between the device and the host, and on the GPU each call first
waits for the whole device. On a laptop processor of twelve threads, `get_state` took about half a
millisecond and `set_state` from about 4 ms for `5_vs_5` to about 8 ms for `11_vs_11_stochastic`,
and both take longer on a busy processor. Both suit occasional use rather than every environment at
every reset.

A state of `5_vs_5` occupies about 48 KB at kick-off and 78 KB after 600 steps, and one of
`11_vs_11_stochastic` 77 KB and 108 KB. A state grows during play, since GRF keeps one position per
player for each second between kick-offs. A thousand states occupy about 100 MB.

## See also

- [Episodes and seeds](episodes-and-seeds.md)
- [Exactness](exactness.md)
- [Custom scenarios](custom-scenarios.md)
- [Migrating from GRF](migrating-from-grf.md)
- [Differences from GRF](differences-from-grf.md)
- [Common topics](faq.md)
