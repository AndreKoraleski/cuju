# Episodes and seeds

In cuju, an **episode** is one match of an environment, from a reset to the end that Google Research
Football, or GRF, defines, and a **seed** determines which match of GRF the episode plays. A batch
resets an environment inside the step that ends its episode, so thousands of environments play
without waiting for one another, and the step reports the last observation of the finished episode
in `information['final_observation']`. With `auto_reset=False`, an environment instead stays at its
end until it is reset, as in GRF.

Environment `index` of a batch created with `seed` plays the matches that GRF plays after
`random.seed(seed * 2**32 + first_index + index)`, independently of the size of the batch, the
backend and the number of threads. A batch reproduces GRF's draws from Python's `random` module,
including the team order of each run and the alternation of the two forms of a scenario between
even and odd episodes. Seeds can also be set per environment at a reset, which replays any earlier
episode bit for bit.

## Life of an episode

A batch starts with `reset`, which starts the first episode of every environment and returns the
first observation. From then on, `step` advances every environment by one step.

An episode ends inside `step`, at the point where GRF would end it, when the match time runs out or,
in a scenario that ends on a goal, when the ball goes out of play or the other team wins the ball.
The same call then starts the next episode of that environment. A batch of thousands of matches
holds a few that end at almost every step, and a separate `reset` would stop all the others for
them.

The step that ends an episode reports that end in four places. Its done flag for that environment
is 1, and its observation is already the first of the next episode. `information['final_observation']`
holds the last observation of the episode that ended, and the reward is that of the step that ended
it, a goal included. In the example, `last` holds the observation that ended each episode where one
ended, and the current observation elsewhere.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('academy_empty_goal_close', environment_count=256, device='cpu')
observation = batch.reset()
actions = torch.full((256, 1), 12, dtype=torch.int32)  # `shot`
for _ in range(50):
    observation, reward, done, information = batch.step(actions)
    ended = done.bool()[:, None, None]  # broadcasts over the agents and the 115 numbers
    last = torch.where(ended, information['final_observation'], observation)
```

`torch.where` runs without waiting for the device, whereas a boolean index such as
`information['final_observation'][done.bool()]` makes the host wait on the GPU, since it has to
count its elements first.

`final_observation` is meaningful where `done == 1`. Elsewhere it holds whatever the last end of
that environment left there, so it is selected with the done flags. `done` is `uint8`, and
`done.bool()` turns it into a mask, since PyTorch deprecates indexing with `uint8`. With
`representation='raw'`, `final_observation` is a dictionary, and the selection applies to each of
its arrays, as in
`{key: value[done.bool()] for key, value in information['final_observation'].items()}`.

The automatic reset follows the convention of Gymnasium's vector environments before version 1.0.
GRF resets only on a call to `reset`, and its environment raises at a step after the end of an
episode.

### Manual resets

`auto_reset=False` gives GRF's behaviour. An environment whose episode ended stays at its end. Every
later step returns its last observation, its done flag of 1 and a reward of 0, until a call to
`reset_done`, which resets every environment that ended, or to `reset`, which resets them all. The
mode suits evaluation, which requires exactly one episode per environment.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu', auto_reset=False)
batch.reset()
finished = torch.zeros(64, dtype=torch.bool)
while not finished.all():
    observation, reward, done, information = batch.step(torch.zeros((64, 4), dtype=torch.int32))
    finished |= done.bool()
```

`reset_done` reads the done flags where the batch keeps them, so on the GPU it also runs without
waiting for the device. With `auto_reset=False`, environments that ended keep returning
`done == 1` until a reset, so an accumulated mask such as `finished` counts each end once.

### Partial resets

`reset(mask=...)` resets the environments where the mask is true, one boolean per environment, given
as a bool or integer tensor, a list or an array. Each starts its next episode as GRF's `reset()`
does, with its next seed, and the others keep their episode, outputs and done flags untouched. With
the mask on the batch's device, the call copies nothing and waits for nothing, and a CUDA graph can
capture it once every environment has started. In the example, each environment counts its own
steps, and the mask cuts every episode that has run for more than 500 steps.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu')
batch.reset()
steps_in_episode = torch.zeros(64, dtype=torch.int64)
for _ in range(600):
    observation, reward, done, information = batch.step(torch.zeros((64, 4), dtype=torch.int32))
    steps_in_episode = torch.where(done.bool(), 0, steps_in_episode + 1)
    long = steps_in_episode > 500
    batch.reset(mask=long)
    steps_in_episode[long] = 0
```

A masked reset cuts the episode it ends and sets no done flag, since GRF's episode has not ended.
Masked resets also stagger the episodes of a batch, as in [Recipes](recipes.md#staggered-episodes).

### Time limits

GRF signals every end of an episode with the same done flag, and a match ends when its time runs
out. The time remaining is part of the state, `steps_left` in `raw`, while `simple115`,
`simple115v2` and the SMM hold no clock, so a policy that reads them has no information about the
coming limit. A learner that treats the time limit as a truncation rather than a terminal state
reads the reason for each end from the raw values.

The value `steps_left == 0` of the step that ended the episode, read from `final_raw()`,
distinguishes the time limit from other ends. As in GRF, a match of `game_duration` 3,000 ends at
its 3,001st step.

### First reset

A batch steps after `reset`, or after `set_state` of every environment, which starts the episode of
each from the state it loads. A step before either raises `cuju.LibraryError`, a `RuntimeError`,
since the batch allocates its worlds without playing.

### Undefined paths

GRF leaves a few paths of its engine undefined, such as a set piece without a taker, and at such a
path GRF has no result for cuju to equal. No shipped scenario is known to reach one. When an
environment reaches one, the batch stops with `cuju.UndefinedPathError`, a `RuntimeError`.

On the CPU, the step raises the error in the same call and names the environment and GRF's reason.
On the GPU, the device traps, which ends the CUDA context of the process, so the next CUDA call
reports the fault and the process restarts. A long training run plays far more steps than any
test, so it saves regular checkpoints of its model and optimizer, from which a restarted process
continues.

## Seeds

Every environment plays GRF's matches, and the seed selects them by one rule.

> Environment `index` of a batch created with `seed` plays the episodes of the environment that
> `gfootball.env.create_environment` creates after Python's
> `random.seed(seed * 2**32 + first_index + index)`.

`first_index` is 0 by default. A batch with `first_index=k` plays environments `k` to
`k + environment_count - 1` of the rule, exactly as a larger batch plays them, which splits one run
over machines, as described in [Backends](backends.md#several-gpus). Every index stays below 2**32.

The rule holds because GRF draws every random number of a match from one seed per episode, and
draws that seed from Python's `random` module, which a batch reproduces exactly, with the same
Mersenne Twister and the same draws in the same order. GRF draws `random.randint(0, 2000000000)`
each time it builds a scenario. `create_environment` builds three scenarios before the first
episode, one for a configuration that reads the scenario, one for the configuration of the
environment and one for the reset inside the creation of its core, and every reset builds one more.
The first episode of an environment is therefore the one that GRF plays after its first
`env.reset()`, from the fourth draw. GRF also allows a step directly after `create_environment`,
which plays the episode of the third draw, and a batch always starts from the episode of the fourth.

The rule has three consequences. The first is reproducibility, since the same seed and actions give
the same outputs on the GPU and on the CPU, with any number of threads. The second is independence
of the batch size. Environment 7 plays the same matches in a batch of 64 and in a batch of 4,096,
so a run reproduces on a machine of another size, provided the environments compared have the same
indices. The third is a direct comparison with GRF. One environment is compared with GRF by calling
`random.seed(seed * 2**32 + index)`, creating GRF's environment with the same options, calling its
`reset()` and feeding both the same actions. That Python process draws nothing else from `random`
in the meantime, since a policy that calls `random.choice`, or a second GRF environment in the same
process, shifts the seeds of every later episode.

### Team order

GRF processes the two teams in an order that its flag `reverse_team_processing` sets, and the order
changes the outcome of simultaneous events. In an environment, the flag stays fixed for the whole
run, taken from the parity of the second seed that GRF draws, as in GRF. The seed reproduces it
with everything else, so it requires no separate handling. Some scenarios also swap the kick-off
from one episode to the next, according to the [episode
number](#episode-number-and-the-two-forms).

### Setting the seeds

Three settings choose the matches, from coarse to fine, as the table lists.

| Setting | Effect |
| --- | --- |
| `seed=` at creation | the seeding rule, for every environment |
| `game_engine_random_seed=` at creation | GRF's option. Every episode of an environment plays that one seed, and nothing is drawn. An integer serves every environment, and a sequence gives one per environment, -1 where it draws as usual |
| `reset(seeds=..., reverse=...)` | the seed of the next episode of each environment, and the team-order flag of its run, to replay an earlier run |

`reset(seeds=...)` takes one seed per environment as a list, a tuple, a numpy array or an integer
tensor, or a single integer for every environment. Each is from 0 to 4,294,967,295, or -1, which
draws that environment's seed as `seeds=None` would. `reverse` takes one flag per environment in the
same forms, 0 or 1, `False` or `True`, or one for all. Without `reverse`, the flag follows the
parity of the seed, as in GRF. The flag takes effect at the first reset of an environment, since it
holds for the whole run. A value of the wrong length or range raises `cuju.ArgumentError`, a
`ValueError`. A floating-point seed or flag raises `TypeError`, even an integral one such as 5.0, as
GRF refuses a float for its seed.

An environment draws from its generator only for a seed that it needs. A reset with a seed draws
nothing, and at the first reset of an environment it also skips the three draws of the creation,
so the next reset without seeds plays the first draw of the generator rather than the fifth.

The default `seed=0` plays the same matches in every run, whereas an unseeded GRF process seeds
`random` from the operating system and plays other matches each time. The runs of a sweep therefore
take distinct values of `seed`. `game_engine_random_seed` makes every episode of an environment the
same match, which suits debugging and fixed evaluation sets, while training draws a fresh seed for
every episode. The value of `environment_count` leaves what environment `index` plays unchanged,
while the bundle, the options, `seed` and `first_index` each change it.

### Replay of one episode

Episode `k` of environment `index`, counting from 1, plays draw `2 + k` of the generator that
`random.seed(seed * 2**32 + first_index + index)` starts, counting the draws from 0, and the
team-order flag of the run is the parity of draw 1. A separate batch of one environment replays the
episode from them, with `reset(seeds=..., reverse=...)` at its first reset.

That batch takes the options of the batch that played the episode, with `episode_number` set to the
original's plus `k - 1`, which puts its first episode at the parity of episode `k`. The parity
matters for the scenarios with two forms. With the same actions, the replay plays the episode bit
for bit. In the example, the second episode of environment 3 of a batch with `seed=5` starts again
in a batch of its own.

```python
import random

generator = random.Random(5 * 2**32 + 3)
draws = [generator.randint(0, 2000000000) for _ in range(5)]
replay = cuju.BatchEnvironment(
    'academy_3_vs_1_with_keeper', environment_count=1, device='cpu', episode_number=1
)
observation = replay.reset(seeds=[draws[4]], reverse=[draws[1] % 2])
```

### Episode number and the two forms

Some scenarios read the parity of the episode through `builder.EpisodeNumber()`, and a bundle holds
one form of the scenario for the even and one for the odd episodes. `5_vs_5` and the `11_vs_11_*`
scenarios give the kick-off to the left team in the even form and to the right team in the odd one.
`1_vs_1_easy` reads the parity too, with two identical forms, and the academies and
`test_example_multiagent` have one form for both parities.

An environment alternates the two forms from one episode to the next, as a GRF environment does
over its resets. `episode_number`, as GRF's option of that name, sets the parity. With the default
0, the first episode plays the even form, as a fresh GRF environment does after its first
`reset()`, and with 1 it plays the odd form. A scenario that changes with the episode number beyond
its parity, such as a ladder of levels, needs more than the two forms of a bundle, and the builder
raises `ScenarioError` for it, as described in [Custom scenarios](custom-scenarios.md). Such a
ladder plays as one scenario file per level, each built into its own bundle.

## See also

- [States](states.md)
- [Backends](backends.md)
- [Recipes](recipes.md)
- [Custom scenarios](custom-scenarios.md)
- [Differences from GRF](differences-from-grf.md)
- [Common topics](faq.md)
