# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Spreads one batch over several devices and checks it against the same batch on one device.

    python examples/multi_gpu.py --device cuda [--devices cuda:0 cuda:1] [--environments 8192]
    python examples/multi_gpu.py --device cpu [--split 300 724]

`cuju.MultiDeviceBatch` holds one `BatchEnvironment` per device, steps them together and gathers
their outputs on the first device. Environment `i` plays as environment `i` of a single batch,
whatever the split, so the script steps a single batch on `--device` with the same random actions
and compares every observation, reward and done flag, bit for bit. `--devices` defaults to every
visible GPU with `--device cuda`, to the GPU and the CPU together on a machine with one GPU, and to
two CPU parts with `--device cpu`. `--split` sets the environments of each part, since a GPU plays
many more than a CPU in the same time.
"""

import argparse
import time

import torch

import cuju


def default_devices(device):
    """Returns every visible GPU, a GPU and the CPU, or two CPU parts."""
    if device == 'cpu':
        return ['cpu', 'cpu']
    gpus = [f'cuda:{index}' for index in range(torch.cuda.device_count())]
    return gpus if len(gpus) > 1 else [*gpus, 'cpu']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r0')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--devices', nargs='+', help='one entry per part, such as cuda:0 cpu')
    parser.add_argument('--split', type=int, nargs='+', help='environments of each part')
    parser.add_argument('--environments', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--threads', type=int, default=0, help='CPU threads of each CPU part')
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    devices = arguments.devices or default_devices(arguments.device)
    generator = torch.Generator(device=arguments.device).manual_seed(arguments.seed)
    options = {'environment_count': arguments.environments, 'seed': arguments.seed}
    with cuju.BatchEnvironment(arguments.scenario, device=arguments.device, **options) as single:
        spread = cuju.MultiDeviceBatch(
            arguments.scenario,
            devices,
            split=arguments.split,
            output_device=arguments.device,
            threads=arguments.threads,
            **options,
        )
        try:
            for part, (start, end) in zip(spread.parts, spread.ranges, strict=True):
                print(f'{part.torch_device}: environments {start} to {end - 1}')
            same = torch.equal(single.reset(), spread.reset())
            shape = (arguments.environments, single.agents)
            elapsed = 0.0
            for _ in range(arguments.steps):
                actions = torch.randint(
                    0,
                    single.action_count,
                    shape,
                    dtype=torch.int32,
                    device=arguments.device,
                    generator=generator,
                )
                expected = single.step(actions)
                start = time.perf_counter()
                observed = spread.step(actions)
                if arguments.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                same &= all(
                    torch.equal(one, other)
                    for one, other in zip(expected[:3], observed[:3], strict=True)
                )
        finally:
            spread.close()

    steps = arguments.steps * arguments.environments
    print(f'the parts play the matches of the single batch, bit for bit: {same}')
    print(f'{steps / elapsed:,.0f} environment steps per second over {len(devices)} parts')


if __name__ == '__main__':
    main()
