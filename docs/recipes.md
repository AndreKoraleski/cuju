# Recipes

**Recipes** for cuju are short programs and patterns that connect a batch to reinforcement learning
code. They cover a minimal training loop, rollout buffers, bootstrapping at the end of an episode,
masks for mixed batches, staggered episodes, self-play, evaluation against the built-in AI, training
for the Kaggle competition, seeding, the size of a batch, the number of batches per process, and
the interface of Google Research Football, or GRF. The learning algorithm comes from separate
training code or a reinforcement learning library, and cuju supplies the environment side.

Three properties of a batch underlie the recipes. The tensors it returns are overwritten by the next
call, an episode that ends restarts inside the step, and a mixed batch pads its outputs. Each
property causes silent errors in training code that overlooks it. The most frequent are storing a
returned tensor instead of a copy, which makes every stored observation equal to the last one,
reading `final_observation` where `done == 0` instead of selecting with `done.bool()`, and averaging
over absent agents instead of weighting with `batch.agent_mask`.

Each recipe calls `policy(observation)`, a function that returns `int32` actions of shape
`[environment_count, agents]` on the device of the batch. A stand-in that returns idle actions is
`policy = lambda observation: torch.zeros(observation.shape[:2], dtype=torch.int32, device=observation.device)`.

## Minimal training loop

The shortest complete loop is an advantage actor-critic, or A2C, on the CPU in
`academy_empty_goal_close`, an academy scenario that a policy solves by learning to shoot. It
serves as a starting point for checking a setup and for adaptation, with untuned hyperparameters.

```python
import cuju
import torch

environments, horizon, updates, gamma = 64, 16, 300, 0.99
batch = cuju.BatchEnvironment(
    'academy_empty_goal_close', environment_count=environments, device='cpu', squeeze=True
)
model = torch.nn.Sequential(
    torch.nn.Linear(115, 128), torch.nn.Tanh(), torch.nn.Linear(128, batch.action_count + 1)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

observation = batch.reset().clone()
goals = episodes = 0
for update in range(updates):
    log_probabilities, values, entropies, rewards, dones = [], [], [], [], []
    for _ in range(horizon):
        output = model(observation)  # the action logits, then the value
        distribution = torch.distributions.Categorical(logits=output[:, :-1])
        action = distribution.sample()
        observation, reward, done, information = batch.step(action.to(torch.int32))
        observation = observation.clone()  # the next step overwrites the batch's tensors
        log_probabilities.append(distribution.log_prob(action))
        values.append(output[:, -1])
        entropies.append(distribution.entropy())
        rewards.append(reward.clone())
        dones.append(done.float())
        goals += int((information['score_reward'] == 1).sum())
        episodes += int(done.sum())
    # The end of an episode counts as terminal, as most work on GRF treats it.
    with torch.no_grad():
        target = model(observation)[:, -1]
    returns = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        target = reward + gamma * (1 - done) * target
        returns.append(target)
    advantage = torch.stack(returns[::-1]) - torch.stack(values)
    loss = (
        -(torch.stack(log_probabilities) * advantage.detach()).mean()
        + 0.5 * advantage.pow(2).mean()
        - 0.01 * torch.stack(entropies).mean()
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (update + 1) % 50 == 0:
        print(f'update {update + 1}: {goals / max(episodes, 1):.2f} goals per episode')
        goals = episodes = 0
```

The loop plays 300 updates of 16 steps in 64 environments, 307,200 environment steps in all, which
take a few minutes on the processor of a laptop. The goals per episode that it prints rise as the
policy learns to shoot. `squeeze=True` gives the shapes of single-agent code, `[environments, 115]`
for the observation and `[environments]` for the reward and the actions.

Two lines are essential to its correctness. `observation.clone()` keeps the next step from
overwriting the input that the gradient still needs, and `(1 - done)` stops the return at the end of
an episode, which the batch has already restarted. On the GPU, the loop takes `device='cuda'` for
the batch and `model.cuda()`. The `.to(torch.int32)` of the actions then allocates a tensor at every
step, which a copy into an `int32` buffer allocated once avoids, as described in
[Backends](backends.md). With 64 environments the GPU is slower than the CPU, and GPU runs use
thousands of environments.

## Rollout buffers

Since every step overwrites the tensors it returns, a rollout copies them into storage allocated
once, rather than appending clones to a list.

```python
steps, environments = 128, 1024
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=environments, device='cpu')
agents = batch.agents
observations = torch.empty((steps, environments, agents, 115))
rewards = torch.empty((steps, environments, agents))
dones = torch.empty((steps, environments), dtype=torch.uint8)

observation = batch.reset()
for t in range(steps):
    observations[t].copy_(observation)
    # Random actions in place of a policy.
    actions = torch.randint(0, 19, (environments, agents), dtype=torch.int32)
    observation, reward, done, information = batch.step(actions)
    rewards[t].copy_(reward)
    dones[t].copy_(done)
```

On the GPU the buffers are allocated on the device as well, and the whole rollout stays there.

A list of clones is also correct, and for a short rollout on the CPU the difference is small. On the
GPU, preallocated buffers save one allocation per tensor per step, hold the rollout in one block of
memory that a learner slices without `torch.stack`, and keep the memory of a run constant from the
first update, so an out-of-memory error occurs at the start of a run rather than hours into it.

## Bootstrapping from the final observation

When an episode ends in the middle of a rollout, the observation that `step` returns already belongs
to the next episode. A value target that bootstraps at the end of the match takes the last
observation of the ended episode from `final_observation`.

```python
observation, reward, done, information = batch.step(actions)
ended = done.bool().view(-1, *[1] * (observation.dim() - 1))  # broadcasts over the agents
next_observation = torch.where(ended, information['final_observation'], observation)
# value(next_observation) is the value after this step, for every environment
```

`torch.where` keeps the step free of device waits. A boolean index such as
`observation[done.bool()]` makes the host wait on the GPU at every step, since the size of its
result depends on the data.

Whether to bootstrap at the end of a match is a modelling choice with a cost either way. The end of
the match time is a cut rather than a state of the game that the policy can influence, so in
principle the value after it comes from bootstrapping, as for any truncated episode. The vector
observations and the SMM contain no clock, however, so a value function that reads them cannot
anticipate the end, and a target that is cut at some times and bootstrapped at others from similar
observations is noisy. Most work on GRF treats the end as terminal, and comparisons with that work
follow the same convention. Bootstrapping is consistent when the value function receives the time
left, `steps_left` from `with_raw=True`, which distinguishes the two cases.
[Episodes and seeds](episodes-and-seeds.md) describes how an end at the time limit is told apart
from the other ends.

## Masks for absent agents

In a batch that mixes bundles with different numbers of agents, the outputs of absent agents are
zero, and means over agents exclude them by weighting with `batch.agent_mask`.

```python
mask = batch.agent_mask.to(reward.device).float()  # [environments, agents]
mean_reward = (reward * mask).sum() / mask.sum()
```

Losses are masked the same way before reduction. [Observations](observations.md) lists the contents
of every padded slot.

## Staggered episodes

Every environment of a batch kicks off at its first reset, so the episodes also end together, and
every few thousand steps a rollout consists entirely of kick-offs. An on-policy update learns from
the states of one rollout, and when those states are the opening seconds of a match in one update
and the closing seconds in another, the gradient varies with the phase of the match rather than
with the behaviour of the policy. Cutting each first episode at a random length spreads the ends of
the episodes for the rest of the run.

```python
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4096, device='cuda')
observation = batch.reset()
generator = torch.Generator(device='cuda').manual_seed(0)
# One episode length per environment.
cut = torch.randint(1, 3001, (4096,), device='cuda', generator=generator)
step = torch.zeros(4096, dtype=torch.int64, device='cuda')
for t in range(3001):
    observation, reward, done, information = batch.step(policy(observation))
    step += 1
    batch.reset(mask=step == cut)  # each environment once, at its own step, without a wait
```

A masked reset starts the next episode of each chosen environment as the `reset()` of GRF would, so
every later episode remains a GRF episode of the seed rule. The cut episodes report no `done`, so
their last step is treated as a time limit or dropped. Academy scenarios that end within tens of
steps stagger themselves within a few episodes, so the technique applies to the full-length
scenarios.

## Self-play

A bundle with agents on both sides, such as `5_vs_5_l4_r4`, places the left agents first and the
right agents after them. Each side sees the pitch from its own end and receives the reward of its
own team, so one policy can play both sides.

The position of the first right agent varies between environments, so the actions of two policies
are routed with the layout of the batch rather than with fixed slices of the agent axis.

```python
def random_policy(observation):  # a stand-in, a policy maps observations to actions
    return torch.randint(0, 19, observation.shape[:2], dtype=torch.int32)


batch = cuju.BatchEnvironment(
    ['5_vs_5_l4_r4', '5_vs_5_l2_r3'],
    environment_count=512,
    device='cpu',
    assignment=[index % 2 for index in range(512)],
)
slot = torch.arange(batch.agents)
is_right = (slot >= batch.left_agents[:, None]) & batch.agent_mask  # [environments, agents]
observation = batch.reset()
current, past = random_policy, random_policy
actions = torch.where(is_right, past(observation), current(observation))
observation, reward, done, information = batch.step(actions)
```

The batch ignores the actions of padded agents, so any value serves there.

The choice of opponent is the main design decision of self-play. A policy that plays against its
current self can cycle, learning to beat its latest version while losing the ability to beat older
ones. A pool of past snapshots for `past`, sampled at each rollout, is the usual remedy, and
evaluation against the built-in AI is the usual measure of progress against a fixed opponent. A
mixed batch accommodates both, since self-play and play against the built-in AI can share one batch.

### The built-in AI as opponent

Two mechanisms place the built-in AI on a side, and they produce different matches.

A side with no agents, as in `5_vs_5_l4_r0`, is the scenario of GRF with the built-in AI as the
other team. The papers and leaderboards of GRF evaluate against this form, and it is the form used
for evaluation.

Action 19, `builtin_ai`, in the `v2` and `full` sets, hands the player of an agent to the built-in
AI for that step, while the agent keeps its observations and rewards. `torch.where(is_right, 19, actions)`
turns the right side of any environment of a self-play batch into a built-in AI side, so
one batch serves both self-play and play against the AI. Its matches differ from those of a bundle
without agents from the first step, since the agents still exist, so it serves training while
evaluation uses the first form.

Action 0, `idle`, plays a team that idles, the equivalent of the `lazy` player of GRF.

## Evaluation against the built-in AI

Evaluation uses fixed matches that can be replayed. With `auto_reset=False` each environment plays
exactly one episode, and a dedicated `seed` fixes the matches of the evaluation.

```python
batch = cuju.BatchEnvironment(
    '11_vs_11_stochastic_l1_r0', environment_count=100, device='cpu', auto_reset=False, seed=12345
)
observation = batch.reset()
finished = torch.zeros(100, dtype=torch.bool)
goals_for = torch.zeros(100, dtype=torch.int32)
goals_against = torch.zeros(100, dtype=torch.int32)
while not finished.all():
    observation, reward, done, information = batch.step(policy(observation))
    live = ~finished
    score = information['score_reward']
    goals_for += (score == 1) & live
    goals_against += (score == -1) & live
    finished |= done.bool()
print('wins', int((goals_for > goals_against).sum()), 'of 100')
```

The same `seed` replays the same 100 matches, so two policies meet the same opponents.

The number of matches follows from the difference to be detected. A win rate measured over 100
matches has a 95% confidence interval of about ±10 percentage points near 50%, which leaves policies
at 45% and 55% indistinguishable, while 1,000 matches narrow the interval to about ±3 points.
Matches in a batch are inexpensive, so evaluations with cuju can use more matches than is customary
with GRF, and the interval is reported with the rate. With `auto_reset=False` the environments that
finish early idle until the last one ends, which costs little in a full match, since every full
match lasts its 3,001 steps.

## Kaggle competition

The GRF competition on Kaggle plays `11_vs_11_kaggle` with one agent a side, and each agent receives
the `raw` observation of GRF. A batch plays the same scenario.

`11_vs_11_kaggle_l1_r1` plays a ladder match with one agent a side, and `11_vs_11_kaggle_l1_r0`
plays against the built-in AI at difficulty 1.0, a fixed opponent that stands in for the agents of
the ladder. A submission reads `obs['players_raw'][0]`, a dictionary of the raw arrays of GRF as
lists. `cuju.kaggle.players_raw(batch, environment, agent)` builds the list that holds that
dictionary from a batch with `raw`, and
`cuju.kaggle.observation(batch, environment, agent, step)` builds the whole observation that a
Kaggle agent receives.

```python
import cuju.kaggle


def agent(obs):  # stands in for the agent function of a submission, which runs unchanged
    return [0]


batch = cuju.BatchEnvironment(
    '11_vs_11_kaggle_l1_r1', environment_count=4, device='cpu', representation='raw'
)
batch.reset()
obs = cuju.kaggle.observation(batch, environment=0, agent=1)  # the right agent's view
action = agent(obs)
```

`cuju.kaggle.KaggleAgent(function, batch, agent)` runs the `agent(obs)` function of a submission
over every environment of a batch and returns its actions as a tensor, so a submission plays against
a trained policy, or against itself, at the speed of the batch. It calls the function once per
environment and step, in Python, as Kaggle does. `cuju.kaggle.evaluate(left, right, episodes=100)`
plays 100 matches of two Kaggle agent functions, or of one against the built-in AI when `right` is
`None`, and returns the goals of each match. Its `episodes` defaults to one match. Every match is
the first episode of its environment, the even form with its own team order, as Kaggle plays it.

The wrappers of GRF, applied to `players_raw` with its lists converted back to numpy arrays,
reproduce the observations of cuju. `Simple115StateWrapper.convert_observation([entry], True)`
equals `simple115v2`, and `observation_preprocessing.generate_smm([entry])` equals `extracted`, bit
for bit, so a submission that applies these wrappers to `players_raw` sees what training saw.

The values agree while the types differ. The raw arrays of cuju are `float32`, whereas a Kaggle
agent holds the same numbers as Python floats, which numpy converts to `float64`. A feature computed
in `float32` during training, such as a distance, can therefore differ in its last bits from the
same numpy code in the submission, and computing such features in `float64`, with `.double()`, makes
the two agree.

[Observations](observations.md) gives the meaning of each raw array and the view of a right agent,
and [Scenarios](scenarios.md) describes the second half of `11_vs_11_kaggle`.

## Seeding

A single `seed` per run determines every match. Environment `index` plays its own sequence of
matches, independent of the batch size and of the number of threads. The default, `seed=0`, plays
the same matches in every run, so the runs of a sweep each take their own `seed`.

An experiment on several machines or GPUs gives each the same `seed` and its own range of
environments with `first_index`. Machine `m` of `n`, each with `environment_count=c`, passes
`first_index=m * c`, and together the machines play exactly the environments of one batch of
`n * c`.

`reset(seeds=)` sets the seed of the next episode of each environment, which replays a single
match. `game_engine_random_seed` makes every episode of an environment the same match, which serves
debugging, while training relies on distinct matches.

## Sizing a batch

`cuju.memory_estimate` takes the arguments of `BatchEnvironment` and returns the bytes that the
batch would request, without allocating them, so a program can choose the number of environments
from the free memory of its GPU before it creates the batch. The first batch of a process on a GPU
also reserves the local memory of the kernels, which the estimate gives as `local_memory_bytes`, so
the budget sets that reservation aside once. The search in this program doubles the count from 1,024
while the next count fits the budget, and stops at 16,384, the largest count that
[Performance](performance.md) measures.

```python
import cuju
import torch

scenario = '5_vs_5_l4_r0'
# A first batch reserves the local memory of the kernels, once per process.
kernels = cuju.memory_estimate(scenario, environment_count=1).local_memory_bytes
reserve = 2 * 2**30  # room for the model, its optimizer and the rollouts
budget = torch.cuda.mem_get_info()[0] - reserve - kernels
count = 1024
while count < 16384 and cuju.memory_estimate(scenario, environment_count=2 * count) <= budget:
    count *= 2
batch = cuju.BatchEnvironment(scenario, environment_count=count)
print(f'{count} environments, {batch.memory_bytes / 2**20:.0f} MiB')
```

Each call of `memory_estimate` reads the bundles and the animation tables, so the search takes a
fraction of a second per count. The reserve depends on the model and the rollout buffers, and an
observation of `extracted` takes 60 times the memory of a vector observation, as described in
[Performance](performance.md). On the CPU the same call with `device='cpu'` counts host memory,
which bounds a large batch of `extracted` observations on a machine with little memory.

## Batches per process

The arrangement with the fewest interactions is one batch per process and GPU, with as many
environments as fit. More environments per batch cost less than more batches. On an RTX 4090,
16,384 environments of `5_vs_5` run 32,582 steps per second and 1,024 environments run 9,920, so
one step of the large batch takes about 30% of the time of the 16 steps of the small one that cover
the same number of environments. On the
CPU, one batch with `threads=0` already uses every core. On a small GPU under Windows or WSL2, a
single process with one batch avoids the tenfold slowdown that two processes with a batch each cause
when the driver spills device memory into system memory, as described in
[Performance](performance.md).

## Compiled step

`torch.compile(mode='reduce-overhead')` places the policy and the step in one CUDA graph once the
batch's buffers are marked as static inputs, which a `reset` after `torch.compile` does. The
compiled function therefore comes before the reset.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cuda')


@torch.compile(mode='reduce-overhead', fullgraph=True)
def act_and_step(observation):
    return batch.step(policy(observation))


with torch.no_grad():
    observation = batch.reset()  # after torch.compile, which marks the buffers
    for _ in range(1000):
        observation, reward, done, information = act_and_step(observation)
```

A batch reset before `torch.compile` plays the same bits, and its step runs outside the graph until
one more `reset` marks the buffers. [Backends](backends.md#compilation-with-torchcompile) describes
the operators behind the compiled step.

## GRF interface

For code written against the environment of GRF, `cuju.env.create_environment` takes the arguments
of GRF and returns one environment with the gym 0.21 interface of GRF, or the Gymnasium interface
with `api='gymnasium'`. It builds its spaces with Gymnasium, from `pip install 'cuju[spaces]'`.
[Migrating from GRF](migrating-from-grf.md) gives the details.

## See also

- [API reference](api.md), for the signatures of every name used in the recipes
- [Concepts](concepts.md), for batches, bundles and environments
- [Rewards and actions](rewards-and-actions.md), for the action sets and `builtin_ai`
- [States](states.md), for saving and restoring a single environment
