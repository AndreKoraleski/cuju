# Examples

The **examples** of cuju are single-file Python programs that show the library at work, from a first
rollout to a complete trainer of proximal policy optimization, or PPO. Each runs from the root of
the repository with the CPU backend by default and on an NVIDIA GPU with `--device cuda`, apart from
`compare_with_grf.py`, which plays on the CPU alone, and `cuda_graph_step.py`, which needs a GPU.
Each takes its settings from command-line flags, and draws every random number from a seeded
generator, so a run repeats exactly. They need PyTorch and cuju alone, with three additions noted in
the table. `--help` lists the flags of each.

The programs fall into three groups. The first steps a batch and measures it, the second trains and
evaluates policies, and the third works with single matches through states, dumps and the protocol
of the Kaggle competition. The notebook `cuju_quickstart.ipynb` covers the first steps on Colab's
free GPU.

## Index

| Example | Shows | Command |
| --- | --- | --- |
| [random_rollout.py](random_rollout.py) | the shortest loop, random actions in a batch, with the episodes, goals and speed | `python examples/random_rollout.py --device cuda --environments 4096` |
| [mixed_batch.py](mixed_batch.py) | self-play, play against the built-in AI and an 11-a-side match in one batch, with the agent mask | `python examples/mixed_batch.py` |
| [benchmark.py](benchmark.py) | environment steps per second, as [Performance](../docs/performance.md) reports them, and GRF's for comparison, which needs `gfootball` | `python examples/benchmark.py --device cuda --environments 16384` |
| [cuda_graph_step.py](cuda_graph_step.py) | a step and a policy captured in one CUDA graph, on the GPU only | `python examples/cuda_graph_step.py --environments 4096` |
| [ppo.py](ppo.py) | a complete PPO trainer in the style of CleanRL, one network shared by every agent, with generalized advantage estimation, the clipped objective and value loss, and weights saved as a state dictionary | `python examples/ppo.py --device cuda` |
| [evaluate.py](evaluate.py) | fixed matches of saved weights, or of random actions, against the built-in AI, with wins, draws, losses, goals and a 95% interval on the win rate | `python examples/evaluate.py --weights ppo.pt --matches 1000` |
| [self_play.py](self_play.py) | agents on both sides, the left learning and the right playing a frozen copy that is refreshed every few updates | `python examples/self_play.py --device cuda --refresh 10` |
| [states.py](states.py) | a match saved with `get_state` at a chance to shoot and loaded with `set_state` into many environments, which compare a shot with passes and runs | `python examples/states.py --branches 64` |
| [dumps.py](dumps.py) | GRF dumps of chosen environments written with `cuju.dumps.DumpWriter` and read back with `read_dump`, with a summary of the match | `python examples/dumps.py --follow 0 1 --logdir dumps` |
| [multi_gpu.py](multi_gpu.py) | one batch spread over several devices with `cuju.MultiDeviceBatch`, checked against the same batch on one device | `python examples/multi_gpu.py --device cuda --split 896 128` |
| [kaggle_agent.py](kaggle_agent.py) | an `agent(obs)` function of a Kaggle submission played against the built-in AI or itself with `cuju.kaggle`, which needs `numpy` | `python examples/kaggle_agent.py --matches 8` |
| [compare_with_grf.py](compare_with_grf.py) | the same matches in cuju and in an installed GRF, compared bit for bit at every step, which needs `gfootball` | `python examples/compare_with_grf.py --scenario 5_vs_5 --left 4` |

## Training and evaluation

The three training programs share one network. `ppo.py` defines it and saves its weights with
`torch.save(model.state_dict(), path)`, `evaluate.py` loads them with
`torch.load(path, weights_only=True)`, and `self_play.py` can start from them with `--weights`. A
typical session trains, measures the result against a random baseline on the same matches, and
moves on to self-play.

```sh
python examples/ppo.py --device cuda --save ppo.pt
python examples/evaluate.py --device cuda --weights ppo.pt
python examples/evaluate.py --device cuda --random
python examples/self_play.py --device cuda --weights ppo.pt
```

With its defaults, `ppo.py` trains on `academy_empty_goal_close` in 512 environments for about two
million environment steps. On one RTX 4090 the run took 2 minutes 39 seconds at 9,400 to 12,500
steps per second, and the goals per episode rose from 0.33 in the first update to 0.72 after 655,000
steps and 0.99 at the end. `evaluate.py` then scored in 993 of 1,000 fixed matches with sampled
actions and in all 1,000 with `--greedy`, against 189 of 1,000 for random actions.

The defaults of `ppo.py` suit the academy scenario it starts with. A full match, such as
`5_vs_5_l4_r0`, gives far fewer goals per step, and the flag `--rewards scoring,checkpoints` adds
GRF's checkpoint rewards, which reward progress toward the goal.

## See also

- [Recipes](../docs/recipes.md), for the patterns behind these programs, from rollout buffers to
    evaluation
- [API reference](../docs/api.md), for every name that the examples use
- [States](../docs/states.md), for the format and the cost of `get_state` and `set_state`
- [Migrating from GRF](../docs/migrating-from-grf.md), for dumps and the interfaces of GRF
- [Backends](../docs/backends.md), for the devices, the threads and several GPUs
