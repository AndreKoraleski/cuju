# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Copies the shipped player models of the renderer out of KayKit's free character packs.

The packs are Kay Lousberg's KayKit Character Pack Skeletons 1.1 and Adventurers 2.0 and KayKit
Character Animations 1.1, free tiers, under CC0, from kaylousberg.itch.io. The script takes the
paths of their zip files, copies each character's binary glTF file into
`python/cuju/render/web/assets/models/` with the mapping from GRF's joints to the packs' shared rig,
`Rig_Medium`, beside it, and copies the animation sets that celebrations use into
`python/cuju/render/web/assets/animations/`.

    python ci/render_models.py KayKit_Skeletons_1.1_FREE.zip KayKit_Adventurers_2.0_FREE.zip \
        KayKit_Character_Animations_1.1.zip
"""

from __future__ import annotations

import argparse
import json
import struct
import zipfile
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / 'python' / 'cuju' / 'render' / 'web' / 'assets'
_OUT = _OUT / 'models'
# KayKit: every character of both packs shares the rig `Rig_Medium`.
JOINTS = {
    'player': 'root',
    'body': 'hips',
    'middle': 'chest',
    'neck': 'head',
    'left_shoulder': 'upperarm.l',
    'left_elbow': 'lowerarm.l',
    'right_shoulder': 'upperarm.r',
    'right_elbow': 'lowerarm.r',
    'left_thigh': 'upperleg.l',
    'left_knee': 'lowerleg.l',
    'left_ankle': 'foot.l',
    'right_thigh': 'upperleg.r',
    'right_knee': 'lowerleg.r',
    'right_ankle': 'foot.r',
}
_PACKS = {
    'Skeletons': 'KayKit Character Pack Skeletons 1.1',
    'Adventurers': 'KayKit Character Pack Adventurers 2.0',
    'Character_Animations': 'KayKit Character Animations 1.1',
}
# The animation sets of `Rig_Medium` whose clips serve celebrations, idles and reactions.
_ANIMATIONS = (
    'General',
    'MovementBasic',
    'MovementAdvanced',
    'Simulation',
    'Special',
    'CombatMelee',
)


def _bones(data: bytes) -> set[str]:
    """Returns the names of the joints of every skin of a binary glTF file."""
    (length,) = struct.unpack_from('<I', data, 12)
    document = json.loads(data[20 : 20 + length])
    nodes = document['nodes']
    return {nodes[joint]['name'] for skin in document['skins'] for joint in skin['joints']}


def main() -> None:
    """Copies the characters of the zip files that the command line names."""
    parser = argparse.ArgumentParser()
    parser.add_argument('packs', nargs='+', type=Path, help='the zip files of the packs')
    parser.add_argument('--out', type=Path, default=_OUT, help='the folder of the models')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    animations = options.out.parent / 'animations'
    animations.mkdir(parents=True, exist_ok=True)
    for path in options.packs:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            license_text = next(
                archive.read(name).decode() for name in names if name.endswith('License.txt')
            )
            if 'Creative Commons Zero, CC0' not in license_text:
                raise SystemExit(f'expected the CC0 license in {path}')
            pack = next(key for key in _PACKS if key in names[0])
            for name in names:
                parts = name.split('/')
                stem = Path(name).stem
                if name.endswith('.glb') and stem.startswith('Rig_Medium_'):
                    if stem.removeprefix('Rig_Medium_') in _ANIMATIONS:
                        data = archive.read(name)
                        target = animations / f'kaykit-{stem.lower().replace("_", "-")}.glb'
                        target.write_bytes(data)
                        print(target.stem, len(data))
                    continue
                # Only the medium mannequin shares the rig of the characters.
                if stem == 'Mannequin_Large' or not name.endswith('.glb'):
                    continue
                if len(parts) < 2 or 'haracters' not in name.lower():
                    continue
                data = archive.read(name)
                missing = set(JOINTS.values()) - _bones(data)
                if missing:
                    raise SystemExit(f'expected the bones {sorted(missing)} in {name}')
                stem = Path(name).stem.lower()
                identifier = f'kaykit-{stem.replace("_", "-")}'
                (options.out / f'{identifier}.glb').write_bytes(data)
                title = Path(name).stem.replace('_', ' ')
                description = {
                    'id': identifier,
                    'name': title,
                    'author': 'Kay Lousberg, www.kaylousberg.com',
                    'source': _PACKS[pack],
                    'license': 'CC0 1.0',
                    'joints': JOINTS,
                }
                text = json.dumps(description, indent=4) + '\n'
                (options.out / f'{identifier}.json').write_bytes(text.encode())
                print(identifier, len(data))


if __name__ == '__main__':
    main()
