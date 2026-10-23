# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Captures one step of a batch and a policy in a CUDA graph and replays it.

    python examples/cuda_graph_step.py [--environments 4096] [--steps 1000]

The step and the policy run on the current stream with the actions on the device, so one replay
of the graph is one step of every environment and the actions of the next.
"""

import argparse
import time

import torch

import cuju


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--environments', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=1000)
    arguments = parser.parse_args()

    batch = cuju.BatchEnvironment(
        '5_vs_5_l4_r0', environment_count=arguments.environments, device='cuda'
    )
    policy = torch.nn.Sequential(
        torch.nn.Linear(115, 256), torch.nn.ReLU(), torch.nn.Linear(256, batch.action_count)
    ).cuda()
    observation = batch.reset()
    actions = torch.zeros((arguments.environments, batch.agents), dtype=torch.int32, device='cuda')

    # PyTorch warms up the policy on a side stream before a capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        for _ in range(3):
            observation, _, _, _ = batch.step(actions)
            actions.copy_(policy(observation).argmax(-1))
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        observation, _, _, _ = batch.step(actions)
        actions.copy_(policy(observation).argmax(-1))

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(arguments.steps):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    steps = arguments.steps * arguments.environments
    print(f'{steps / elapsed:.0f} environment steps per second, the policy included')


if __name__ == '__main__':
    main()
