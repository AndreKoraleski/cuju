# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""A development server for the browser viewer, with the viewer's API on data held in memory.

It serves the web folder of the renderer and answers its API as the real server does, with the
same checks, status codes and errors, but keeps recordings, teams and requests in memory. It
starts with two synthetic recordings from render_sample.py, runs one live source that replays
synthetic episodes forever at 10 steps per second and greets each viewer with the last 10 seconds,
and plays each requested match as a synthetic one. Nothing here depends on cuju.

Usage:
    python ci/render_mock_server.py [--port 8765]
Then open http://127.0.0.1:8765/ in a browser.
"""

import argparse
import contextlib
import functools
import gzip
import http.server
import io
import json
import mimetypes
import os
import re
import secrets
import struct
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_sample

WEB_ROOT = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'python', 'cuju', 'render', 'web'
    )
)
# The scenarios that cuju ships, as the real server lists them.
SCENARIOS = [
    '11_vs_11_competition',
    '11_vs_11_easy_stochastic',
    '11_vs_11_hard_stochastic',
    '11_vs_11_kaggle',
    '11_vs_11_stochastic',
    '1_vs_1_easy',
    '5_vs_5',
    'academy_3_vs_1_with_keeper',
    'academy_corner',
    'academy_counterattack_easy',
    'academy_counterattack_hard',
    'academy_empty_goal',
    'academy_empty_goal_close',
    'academy_pass_and_shoot_with_keeper',
    'academy_run_pass_and_shoot_with_keeper',
    'academy_run_to_score',
    'academy_run_to_score_with_keeper',
    'academy_single_goal_versus_lazy',
    'test_example_multiagent',
]
TYPES = {
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.json': 'application/json',
    '.glb': 'model/gltf-binary',
    '.webp': 'image/webp',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
}
LOOPBACK = ('127.0.0.1', 'localhost', '::1')
MAXIMUM_BODY = 110 * 1024 * 1024
MAXIMUM_SUBMISSION = 100 * 1024 * 1024
MAXIMUM_UNPACKED = 500 * 1024 * 1024
MAXIMUM_FILES = 10000
MAXIMUM_IMAGE = 8 * 1024 * 1024
MAXIMUM_MODEL = 30 * 1024 * 1024
IDENTIFIER = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')

BUILTIN = 'builtin'
BUILTIN_ENTRY = {'id': BUILTIN, 'name': "GRF's built-in AI", 'kind': 'builtin', 'created': None}
FOLDER_MAGIC = b'CUJDIR1\n'
MAIN = 'main.py'
TEAM_FOLDER = 'cuju-team'
META = f'{TEAM_FOLDER}/team.json'
MODEL = 'model.glb'
PATTERNS = ('plain', 'stripes', 'hoops', 'halves', 'sash', 'checkers')
SKINS = ('grf', 'light', 'tan', 'medium', 'dark')
HAIRS = ('grf', 'bald', 'short01', 'short02', 'medium01', 'long01', 'long02')
HAIR_COLORS = ('grf', 'black', 'brown', 'darkblonde', 'blonde', 'red')
NUMBERS = ('white', 'black', 'secondary')
PARTS = ('primary', 'secondary', 'shorts', 'socks')
FORMATS = {'5v5': 5, '11v11': 11}
IMAGE_SLOTS = (
    'shirt-front',
    'shirt-back',
    'shirt-front-edit',
    'shirt-back-edit',
    'sponsor-1',
    'sponsor-2',
    'sponsor-3',
    'sponsor-4',
    'sponsor-1-edit',
    'sponsor-2-edit',
    'sponsor-3-edit',
    'sponsor-4-edit',
)
IMAGE_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp'}
# The sounds a team may hold, each played when one of its players does that, or its fans sing.
SOUND_SLOTS = ('goal', 'shot', 'pass', 'tackle', 'save', 'kick', 'celebration', 'chant')
MAXIMUM_SOUND = 2 * 1024 * 1024
SOUND_TYPES = {'.ogg': 'audio/ogg', '.mp3': 'audio/mpeg', '.wav': 'audio/wav'}
JOINTS = tuple(render_sample.NODES[1:])
KIT = {
    'pattern': 'plain',
    'primary': [214, 48, 49],
    'secondary': [255, 255, 255],
    'shorts': [255, 255, 255],
    'socks': [214, 48, 49],
}
GOALKEEPER = {
    'primary': [46, 204, 113],
    'secondary': [20, 20, 20],
    'shorts': [20, 20, 20],
    'socks': [46, 204, 113],
}
BUILTIN_KIT = {
    'pattern': 'plain',
    'primary': [9, 132, 227],
    'secondary': [255, 255, 255],
    'shorts': [9, 132, 227],
    'socks': [255, 255, 255],
}
# The header teams of a live source, which names none of its own.
DEFAULT_TEAMS = [
    {'name': 'Left', 'color': [214, 48, 49], 'secondary': [255, 255, 255]},
    {'name': 'Right', 'color': [9, 132, 227], 'secondary': [255, 255, 255]},
]
CLASH = 120
# The frames of a requested match, as a full match of 3000 steps gives.
MATCH_FRAMES = 3001
# A viewer that joins a live source first receives this many seconds of frames.
BACKLOG_SECONDS = 10
# Member paths are checked against this folder as the real server checks them against its own.
UNPACK_ROOT = Path(tempfile.gettempdir()) / 'cuju-mock-team'


class ArgumentError(ValueError):
    """A request that the real server refuses with its own ArgumentError."""


# The teams folder, each team a mapping of the paths of its files to their bytes.
FOLDERS = {}
# The library, each match its summary and its compressed recording.
MATCHES = {}
LIVE = {}
LOCK = threading.RLock()


def locked(function):
    """Runs a function while it holds the lock of all state."""

    @functools.wraps(function)
    def run(*arguments):
        with LOCK:
            return function(*arguments)

    return run


def new_identifier(prefix):
    return f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(2)}'


def valid_identifier(identifier):
    return bool(IDENTIFIER.fullmatch(identifier))


def player_style(index):
    return {
        'name': '',
        'number': index + 1,
        'skin': 'grf',
        'hair': 'grf',
        'hair_color': 'grf',
        'model': 'team',
        'celebration': 'team',
    }


def default_style(format='11v11', kit=None):
    colors = kit or KIT
    return {
        'model': 'grf',
        'kit': dict(colors),
        'goalkeeper': dict(GOALKEEPER),
        'badge': '',
        'number': 'white',
        'colors': {},
        # The fans wear the kit's colors unless the team picks colors of their own.
        'crowd': {'colors': [list(colors['primary']), list(colors['secondary'])]},
        # The celebrations the page picks from after a goal, empty for every one it knows.
        'celebrations': [],
        'players': [player_style(index) for index in range(FORMATS[format])],
    }


def color(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(
            isinstance(part, int) and not isinstance(part, bool) and 0 <= part <= 255
            for part in value
        )
    ):
        raise ArgumentError(f'expected {name} as three integers from 0 to 255, found {value!r}')
    return [int(part) for part in value]


def choice(value, choices, name):
    if value not in choices:
        raise ArgumentError(f'expected {name} as one of {", ".join(choices)}, found {value!r}')
    return value


def players(value, current, models=()):
    """Returns the roster with the changes of value, a list of player dictionaries."""
    if not isinstance(value, list) or len(value) > len(current):
        raise ArgumentError(f'expected at most {len(current)} player entries, found {value!r}')
    roster = [dict(player) for player in current]
    for index, changes in enumerate(value):
        if not isinstance(changes, dict):
            raise ArgumentError(f'expected player {index} as a dictionary, found {changes!r}')
        for key, entry in changes.items():
            if key == 'name':
                if not isinstance(entry, str) or len(entry) > 24:
                    raise ArgumentError(
                        f'expected a name of at most 24 characters, found {entry!r}'
                    )
                roster[index][key] = entry
            elif key == 'number':
                if not isinstance(entry, int) or not 0 <= entry <= 99:
                    raise ArgumentError(f'expected a shirt number from 0 to 99, found {entry!r}')
                roster[index][key] = entry
            elif key == 'skin':
                roster[index][key] = choice(entry, SKINS, 'the skin')
            elif key == 'hair':
                roster[index][key] = choice(entry, HAIRS, 'the hair')
            elif key == 'hair_color':
                roster[index][key] = choice(entry, HAIR_COLORS, 'the hair color')
            elif key == 'celebration':
                # team picks among the team's celebrations, any other value is the player's own.
                roster[index][key] = celebration(entry)
            elif key == 'model':
                # team follows the team's model, any other value gives the player one of its own.
                roster[index][key] = choice(
                    entry, ('team', 'grf', 'custom', *models), 'the player model'
                )
            else:
                raise ArgumentError(f'expected an entry of a player, found {key!r}')
    return roster


HEX = re.compile(r'#[0-9a-f]{6}')
# A celebration is named by the page, such as knee_slide, or clip and an animation of a model.
CELEBRATION = re.compile(r'[a-z0-9_]{1,40}(:[A-Za-z0-9_. -]{1,60})?')


def celebration(value):
    if not isinstance(value, str) or not CELEBRATION.fullmatch(value):
        raise ArgumentError(f'expected the name of a celebration, found {value!r}')
    return value


# A model's colors to replace, at most this many per model.
MAXIMUM_RECOLORS = 32


def recolors(value, models):
    """Returns the colors that replace others on each model, as #rrggbb pairs by model."""
    if not isinstance(value, dict):
        raise ArgumentError(f'expected the colors as a dictionary by model, found {value!r}')
    result = {}
    for model, pairs in value.items():
        choice(model, models, 'the model of the colors')
        if not isinstance(pairs, dict) or len(pairs) > MAXIMUM_RECOLORS:
            raise ArgumentError(f'expected at most {MAXIMUM_RECOLORS} colors for {model!r}')
        result[model] = {}
        for old, new in pairs.items():
            if not (
                isinstance(old, str)
                and isinstance(new, str)
                and HEX.fullmatch(old.lower())
                and HEX.fullmatch(new.lower())
            ):
                raise ArgumentError(f'expected colors as #rrggbb, found {old!r} and {new!r}')
            result[model][old.lower()] = new.lower()
    return result


def style(changes, base, models=()):
    """Returns the complete style base with changes applied, refusing any unknown entry."""
    result = json.loads(json.dumps(base))
    changes = changes or {}
    if not isinstance(changes, dict):
        raise ArgumentError(f'expected a style as a dictionary, found {changes!r}')
    for key, value in changes.items():
        if key in ('kit', 'goalkeeper'):
            if not isinstance(value, dict):
                raise ArgumentError(f'expected {key} as a dictionary, found {value!r}')
            for part, entry in value.items():
                if key == 'kit' and part == 'pattern':
                    result[key][part] = choice(entry, PATTERNS, 'the pattern')
                elif part in PARTS:
                    result[key][part] = color(entry, f'{key} {part}')
                else:
                    raise ArgumentError(f'expected a part of {key}, found {part!r}')
        elif key == 'model':
            result[key] = choice(value, ('grf', 'custom', *models), 'the model')
        elif key == 'number':
            result[key] = choice(value, NUMBERS, 'the numbers')
        elif key == 'badge':
            if not isinstance(value, str) or len(value) > 3:
                raise ArgumentError(
                    f'expected a badge of at most three characters, found {value!r}'
                )
            result[key] = value
        elif key == 'players':
            result[key] = players(value, result[key], models)
        elif key == 'colors':
            result[key] = recolors(value, ('custom', *models))
        elif key == 'celebrations':
            if not isinstance(value, list) or len(value) > 32:
                raise ArgumentError('expected a list of at most 32 celebrations')
            result[key] = [celebration(item) for item in value]
        elif key == 'crowd':
            if not isinstance(value, dict) or set(value) - {'colors'}:
                raise ArgumentError(f'expected the crowd as {{"colors": [...]}}, found {value!r}')
            colors = value.get('colors', result[key]['colors'])
            if not isinstance(colors, list) or not 1 <= len(colors) <= 4:
                raise ArgumentError('expected one to four colors of the fans')
            result[key] = {'colors': [color(entry, 'a color of the fans') for entry in colors]}
        else:
            raise ArgumentError(f'expected an entry of a style, found {key!r}')
    return result


def resize(style_, format):
    size = FORMATS[format]
    roster = style_['players'][:size]
    roster += [player_style(index) for index in range(len(roster), size)]
    return {**style_, 'players': roster}


def known(saved, base):
    return {key: value for key, value in saved.items() if key in base and key != 'players'}


def shipped_models():
    """Returns the description of every model in the web assets, each json with a glb beside it."""
    folder = Path(WEB_ROOT) / 'assets' / 'models'
    models = []
    for path in sorted(folder.glob('*.json')):
        with contextlib.suppress(OSError, ValueError):
            description = json.loads(path.read_text())
            if path.with_suffix('.glb').is_file():
                description['url'] = f'assets/models/{path.stem}.glb'
                models.append(description)
    return models


def model_names():
    return tuple(model['id'] for model in shipped_models())


def image_file(identifier, slot):
    """Returns the path of the file of an image slot of a team, whatever its extension, or None."""
    for path in FOLDERS.get(identifier, {}):
        name = path[len(TEAM_FOLDER) + 1 :]
        if (
            path.startswith(TEAM_FOLDER + '/')
            and '/' not in name
            and name.startswith(slot + '.')
            and PurePosixPath(name).suffix in ('.png', '.jpg', '.webp', '.json')
        ):
            return path
    return None


def sound_file(identifier, slot):
    """Returns the path of the file of a sound slot of a team, whatever its extension, or None."""
    for path in FOLDERS.get(identifier, {}):
        name = path[len(TEAM_FOLDER) + 1 :]
        if (
            path.startswith(TEAM_FOLDER + '/')
            and '/' not in name
            and name.startswith(f'sound-{slot}.')
            and PurePosixPath(name).suffix in SOUND_TYPES
        ):
            return path
    return None


def sound_suffix(data):
    if data.startswith(b'OggS'):
        return '.ogg'
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return '.wav'
    # MP3, a tag of ID3 or the sync bits of a first frame.
    if data.startswith(b'ID3') or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return '.mp3'
    raise ArgumentError('expected an Ogg, MP3 or WAV sound')


@locked
def set_sound(identifier, slot, data):
    entry(identifier)
    choice(slot, SOUND_SLOTS, 'the sound slot')
    if len(data) > MAXIMUM_SOUND:
        raise ArgumentError(f'expected a sound of at most 2 MB, found {len(data)} bytes')
    suffix = sound_suffix(data)
    old = sound_file(identifier, slot)
    files = FOLDERS.setdefault(identifier, {})
    files[f'{TEAM_FOLDER}/sound-{slot}{suffix}'] = data
    if old is not None and old != f'{TEAM_FOLDER}/sound-{slot}{suffix}':
        del files[old]
    return entry(identifier)


@locked
def sound_path(identifier, slot):
    if slot not in SOUND_SLOTS:
        return None
    if identifier != BUILTIN and not valid_identifier(identifier):
        return None
    return sound_file(identifier, slot)


@locked
def remove_sound(identifier, slot):
    path = sound_path(identifier, slot)
    if path is None:
        return False
    del FOLDERS[identifier][path]
    return True


def complete(entry, kit=None):
    """Returns a stored entry with its format, style, model and images filled and checked."""
    entry = dict(entry)
    if entry.get('format') not in FORMATS:
        entry['format'] = '11v11'
    base = default_style(entry['format'], kit)
    saved = entry.get('style') or {}
    try:
        entry['style'] = resize(
            style(saved, {**base, **known(saved, base)}, model_names()), entry['format']
        )
    except ArgumentError:
        entry['style'] = base
    entry.setdefault('model', None)
    entry['images'] = [slot for slot in IMAGE_SLOTS if image_file(entry['id'], slot) is not None]
    entry['sounds'] = [slot for slot in SOUND_SLOTS if sound_file(entry['id'], slot) is not None]
    return entry


def builtin():
    entry = dict(BUILTIN_ENTRY)
    with contextlib.suppress(OSError, ValueError):
        meta = FOLDERS.get(BUILTIN, {}).get(META)
        if meta is None:
            raise FileNotFoundError(META)
        saved = json.loads(meta)
        entry.update(
            {key: saved[key] for key in ('name', 'style', 'model', 'format') if key in saved}
        )
    return complete(entry, BUILTIN_KIT)


@locked
def teams():
    """Returns every team, the builtin team first, then the submissions from newest to oldest."""
    entries = []
    for identifier in sorted(FOLDERS):
        meta = FOLDERS[identifier].get(META)
        if identifier == BUILTIN or meta is None:
            continue
        try:
            entry = complete(json.loads(meta))
        except (OSError, ValueError, KeyError):
            continue
        entries.append(entry)
    entries.sort(key=lambda entry: entry.get('created') or '', reverse=True)
    return [builtin(), *entries]


@locked
def entry(identifier):
    for team in teams():
        if team['id'] == identifier:
            return team
    raise ArgumentError(f'expected a team of the registry, found {identifier!r}')


def write(identifier, current):
    stored = {key: value for key, value in current.items() if key not in ('images', 'sounds')}
    FOLDERS.setdefault(identifier, {})[META] = json.dumps(stored).encode()


@locked
def update(identifier, name=None, changes=None, format=None):
    current = entry(identifier)
    if name is not None:
        name = str(name).strip()[:80]
        if not name:
            raise ArgumentError('expected a team name with at least one character')
        current['name'] = name
    if format is not None:
        current['format'] = choice(format, tuple(FORMATS), 'the format')
        current['style'] = resize(current['style'], current['format'])
    current['style'] = style(changes, current['style'], model_names())
    roster = current['style']['players']
    chosen = {current['style']['model'], *(player['model'] for player in roster)}
    if 'custom' in chosen and model_path(identifier) is None:
        raise ArgumentError('expected an uploaded model before choosing the custom model')
    write(identifier, current)
    return current


def image_suffix(slot, data):
    if slot.endswith('-edit'):
        try:
            json.loads(data)
        except ValueError as error:
            raise ArgumentError(
                f'expected the JSON document of the editor, found {error}'
            ) from error
        return '.json'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if data.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    raise ArgumentError('expected a PNG, JPEG or WebP image')


@locked
def set_image(identifier, slot, data):
    entry(identifier)
    choice(slot, IMAGE_SLOTS, 'the image slot')
    if len(data) > MAXIMUM_IMAGE:
        raise ArgumentError(f'expected an image of at most 8 MB, found {len(data)} bytes')
    suffix = image_suffix(slot, data)
    old = image_file(identifier, slot)
    files = FOLDERS.setdefault(identifier, {})
    files[f'{TEAM_FOLDER}/{slot}{suffix}'] = data
    if old is not None and old != f'{TEAM_FOLDER}/{slot}{suffix}':
        del files[old]
    return entry(identifier)


@locked
def image_path(identifier, slot):
    if slot not in IMAGE_SLOTS:
        return None
    if identifier != BUILTIN and not valid_identifier(identifier):
        return None
    return image_file(identifier, slot)


@locked
def image(identifier, slot):
    """Returns the path and bytes of a team's image slot, or None."""
    path = image_path(identifier, slot)
    return None if path is None else (path, FOLDERS[identifier][path])


@locked
def remove_image(identifier, slot):
    path = image_path(identifier, slot)
    if path is None:
        return False
    del FOLDERS[identifier][path]
    return True


def skin_joints(data):
    """Returns the names of the joints of every skin of a binary glTF 2.0 file."""
    if len(data) < 20 or data[:4] != b'glTF':
        raise ArgumentError('expected a binary glTF file, .glb, found other bytes')
    version, length = struct.unpack_from('<II', data, 4)
    chunk_length, chunk_type = struct.unpack_from('<II', data, 12)
    if version != 2 or length > len(data) or chunk_type != 0x4E4F534A:
        raise ArgumentError('expected a binary glTF 2.0 file with its JSON chunk first')
    try:
        document = json.loads(data[20 : 20 + chunk_length])
    except ValueError as error:
        raise ArgumentError(f'expected the JSON chunk of a glTF file, found {error}') from error
    if not isinstance(document, dict):
        raise ArgumentError('expected the JSON chunk of a glTF file to hold an object')
    for kind in ('buffers', 'images'):
        for item in document.get(kind) or []:
            uri = item.get('uri') if isinstance(item, dict) else None
            if isinstance(uri, str) and not uri.startswith('data:'):
                raise ArgumentError(f'expected a self-contained model, found a link to {uri!r}')
    nodes = document.get('nodes') or []
    skins = document.get('skins') or []
    if not skins:
        raise ArgumentError('expected a rigged model, a glTF file with a skin, found none')
    names = set()
    for skin in skins:
        joints = skin.get('joints', []) if isinstance(skin, dict) else []
        for joint in joints:
            if (
                isinstance(joint, int)
                and 0 <= joint < len(nodes)
                and isinstance(nodes[joint], dict)
            ):
                names.add(str(nodes[joint].get('name', '')))
    return names


@locked
def set_model(identifier, data, joints):
    current = entry(identifier)
    if len(data) > MAXIMUM_MODEL:
        raise ArgumentError(f'expected a model of at most 30 MB, found {len(data)} bytes')
    bones = skin_joints(data)
    if not isinstance(joints, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in joints.items()
    ):
        raise ArgumentError(f'expected the joints as names of bones by joint, found {joints!r}')
    if 'body' not in joints:
        raise ArgumentError('expected a bone for the joint body, the pelvis, which every rig has')
    unknown = [joint for joint in joints if joint not in (*JOINTS, 'player')]
    if unknown:
        raise ArgumentError(f"expected joints of GRF's skeleton, found {', '.join(unknown)}")
    absent = [bone for bone in joints.values() if bone not in bones]
    if absent:
        raise ArgumentError(f'expected bones of the model, found {", ".join(map(str, absent))}')
    FOLDERS.setdefault(identifier, {})[f'{TEAM_FOLDER}/{MODEL}'] = data
    current['model'] = {
        'joints': {key: str(value) for key, value in joints.items()},
        'bytes': len(data),
    }
    current['style']['model'] = 'custom'
    write(identifier, current)
    return current


@locked
def remove_model(identifier):
    current = entry(identifier)
    had = FOLDERS.get(identifier, {}).pop(f'{TEAM_FOLDER}/{MODEL}', None) is not None
    current['model'] = None
    if current['style']['model'] == 'custom':
        current['style']['model'] = 'grf'
    write(identifier, current)
    return had


@locked
def model_path(identifier):
    if identifier != BUILTIN and not valid_identifier(identifier):
        return None
    return FOLDERS.get(identifier, {}).get(f'{TEAM_FOLDER}/{MODEL}')


def header_team(identifier):
    team = entry(identifier)
    kit = team['style']['kit']
    return {
        'id': team['id'],
        'name': team['name'],
        'format': team['format'],
        'color': kit['primary'],
        'secondary': kit['secondary'],
        'style': team['style'],
    }


@locked
def match_teams(left, right):
    """Returns the header teams of a match, the right one in its away kit on a clash of shirts."""
    sides = [header_team(left), header_team(right)]
    first, second = sides[0]['style']['kit'], sides[1]['style']['kit']
    distance = sum((a - b) ** 2 for a, b in zip(first['primary'], second['primary'], strict=True))
    if distance < CLASH**2:
        kit = dict(second)
        kit['primary'], kit['secondary'] = second['secondary'], second['primary']
        kit['shorts'], kit['socks'] = second['socks'], second['shorts']
        if sum((a - b) ** 2 for a, b in zip(first['primary'], kit['primary'], strict=True)) < (
            CLASH**2
        ):
            kit['primary'] = [255 - value for value in first['primary']]
        sides[1] = {
            **sides[1],
            'color': kit['primary'],
            'secondary': kit['secondary'],
            'style': {**sides[1]['style'], 'kit': kit},
            'away': True,
        }
    return sides


def check_member(name):
    """Returns an archive member's path, refusing one that could land outside the team's folder."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ('', '.', '..') or any(c in part for c in ':\\\0') for part in path.parts)
    ):
        raise ArgumentError(f'expected archive members inside the archive, found {name!r}')
    return path


def inside(path):
    """Returns the key of a member path, refusing one that resolves outside the folder."""
    destination = UNPACK_ROOT.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(UNPACK_ROOT.resolve()):
        raise ArgumentError(f'expected archive members inside the archive, found {str(path)!r}')
    return '/'.join(path.parts)


def check_unpacked(count, size):
    if count > MAXIMUM_FILES or size > MAXIMUM_UNPACKED:
        raise ArgumentError('expected an archive of at most 10,000 files and 500 MB unpacked')


def unbundle(data, files):
    """Reads the files of a folder bundle, dropping a single top folder above main.py."""
    if len(data) < len(FOLDER_MAGIC) + 4:
        raise ArgumentError('expected the files of a submission folder, found too few bytes')
    (length,) = struct.unpack_from('<I', data, len(FOLDER_MAGIC))
    start = len(FOLDER_MAGIC) + 4
    try:
        index = json.loads(data[start : start + length])
    except ValueError as error:
        raise ArgumentError(f'expected the index of a submission folder, found {error}') from error
    if not isinstance(index, list) or len(index) > MAXIMUM_FILES:
        raise ArgumentError('expected the index of a submission folder, a list of its files')
    paths = [check_member(str(item['path'])) for item in index]
    tops = {path.parts[0] for path in paths if len(path.parts) > 1}
    if PurePosixPath(MAIN) not in paths and len(tops) == 1 and all(len(p.parts) > 1 for p in paths):
        paths = [PurePosixPath(*path.parts[1:]) for path in paths]
    position = start + length
    for path, item in zip(paths, index, strict=True):
        size = int(item['size'])
        if size < 0 or position + size > len(data):
            raise ArgumentError('expected the files that the index of the folder lists')
        files[inside(path)] = data[position : position + size]
        position += size


def extract(data, filename, files):
    """Reads a submission into files, a folder bundle, a Python file or an archive."""
    lower = filename.lower()
    if data.startswith(FOLDER_MAGIC):
        unbundle(data, files)
    elif lower.endswith('.py'):
        files[MAIN] = data
    elif lower.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = archive.getmembers()
            check_unpacked(len(members), sum(member.size for member in members))
            for member in members:
                inside(check_member(member.name))
                if not (member.isfile() or member.isdir()):
                    raise ArgumentError(f'expected plain files in the archive, found {member.name}')
            for member in members:
                if member.isfile():
                    files[inside(check_member(member.name))] = archive.extractfile(member).read()
    elif lower.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            check_unpacked(len(members), sum(member.file_size for member in members))
            for member in members:
                inside(check_member(member.filename))
            for member in members:
                if not member.is_dir():
                    files[inside(check_member(member.filename))] = archive.read(member)
    else:
        raise ArgumentError(
            f'expected a submission folder, a .py file or a .tar.gz or .zip archive, found '
            f'{filename!r}'
        )
    if MAIN not in files:
        raise ArgumentError(f'expected {MAIN} at the top of the submission, as Kaggle requires')


def import_team(files, entry, name_given):
    """Takes the format, style, images and model of an exported team into entry."""
    prefix = TEAM_FOLDER + '/'
    if not any(path.startswith(prefix) for path in files):
        return
    saved = {}
    with contextlib.suppress(OSError, ValueError):
        saved = json.loads(files.get(META, b''))
    if not isinstance(saved, dict):
        saved = {}
    if saved.get('format') in FORMATS:
        entry['format'] = saved['format']
    with contextlib.suppress(ArgumentError, TypeError, KeyError, AttributeError):
        base = default_style(entry['format'])
        entry['style'] = resize(
            style(known(saved.get('style') or {}, base), base, model_names()), entry['format']
        )
        if isinstance((saved.get('style') or {}).get('players'), list):
            entry['style']['players'] = players(
                saved['style']['players'][: FORMATS[entry['format']]],
                entry['style']['players'],
                model_names(),
            )
    if not name_given and isinstance(saved.get('name'), str) and saved['name'].strip():
        entry['name'] = saved['name'].strip()[:80]
    children = {path[len(prefix) :].split('/')[0] for path in files if path.startswith(prefix)}
    for name in children:
        if name in ('team.json', MODEL):
            continue
        key = prefix + name
        child = PurePosixPath(name)
        data = files.get(key)
        valid = False
        with contextlib.suppress(ArgumentError):
            if child.stem.startswith('sound-'):
                valid = (
                    data is not None
                    and child.stem[len('sound-') :] in SOUND_SLOTS
                    and len(data) <= MAXIMUM_SOUND
                    and sound_suffix(data) == child.suffix
                )
            else:
                valid = (
                    data is not None
                    and child.stem in IMAGE_SLOTS
                    and len(data) <= MAXIMUM_IMAGE
                    and image_suffix(child.stem, data) == child.suffix
                )
        if not valid:
            for path in [path for path in files if path == key or path.startswith(key + '/')]:
                del files[path]
    joints = (
        (saved.get('model') or {}).get('joints') if isinstance(saved.get('model'), dict) else None
    )
    try:
        data = files.get(prefix + MODEL)
        if data is None:
            raise FileNotFoundError(MODEL)
        bones = skin_joints(data)
        if len(data) > MAXIMUM_MODEL or not isinstance(joints, dict) or 'body' not in joints:
            raise ArgumentError('expected a model of the limits with a joint for body')
        if not all(isinstance(bone, str) and bone in bones for bone in joints.values()):
            raise ArgumentError('expected bones of the model')
        entry['model'] = {
            'joints': {
                str(key): value for key, value in joints.items() if key in (*JOINTS, 'player')
            },
            'bytes': len(data),
        }
    except (OSError, ArgumentError, AttributeError):
        files.pop(prefix + MODEL, None)
        entry['model'] = None
        if entry['style'].get('model') == 'custom':
            entry['style']['model'] = 'grf'


@locked
def add(name, filename, data, format='11v11'):
    """Stores a Kaggle submission as a new team and returns its entry."""
    if len(data) > MAXIMUM_SUBMISSION:
        raise ArgumentError(f'expected a submission of at most 100 MB, found {len(data)} bytes')
    choice(format, tuple(FORMATS), 'the format')
    name_given = bool(name and name.strip())
    name = (name or Path(filename).stem or 'team').strip()[:80]
    identifier = new_identifier('t')
    files = {}
    extract(data, filename, files)
    FOLDERS[identifier] = files
    entry = {
        'id': identifier,
        'name': name,
        'kind': 'kaggle',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'format': format,
        'style': default_style(format),
        'model': None,
    }
    import_team(files, entry, name_given)
    completed = complete(entry)
    write(identifier, completed)
    return completed


@locked
def export(identifier):
    """Returns a file name and the bytes of a zip of a team's folder, its team.json first."""
    team = entry(identifier)
    stored = {key: value for key, value in team.items() if key not in ('images', 'sounds')}
    files = FOLDERS.get(identifier, {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(META, json.dumps(stored))
        for path in sorted(files, key=lambda path: Path(*path.split('/'))):
            if path != META and not path.endswith('.part') and '__pycache__' not in path.split('/'):
                archive.writestr(path, files[path])
    safe = ''.join(character if character.isalnum() else '-' for character in team['name'])
    return f'{safe.strip("-") or "team"}.zip', buffer.getvalue()


@locked
def remove(identifier):
    if identifier == BUILTIN:
        FOLDERS.pop(BUILTIN, None)
        return True
    if not valid_identifier(identifier) or identifier not in FOLDERS:
        return False
    del FOLDERS[identifier]
    return True


@locked
def save_match(header, frames):
    """Stores a recording in the library and returns its identifier."""
    identifier = new_identifier('m')
    data = gzip.compress(render_sample.encode_recording(header, frames), 6)
    score = list(struct.unpack_from('<2i', frames[-1], 8)) if frames else [0, 0]
    MATCHES[identifier] = {
        'summary': {
            'id': identifier,
            'title': header['title'],
            'scenario': header['scenario'],
            'teams': [team['name'] for team in header['teams']],
            'score': score,
            'steps': len(frames),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'bytes': len(data),
            'folder': '',
        },
        'data': data,
    }
    return identifier


@locked
def matches():
    entries = [dict(item['summary']) for item in MATCHES.values()]
    entries.sort(key=lambda entry: entry.get('created') or '', reverse=True)
    return entries


@locked
def recording(identifier):
    if not valid_identifier(identifier) or identifier not in MATCHES:
        return None
    return MATCHES[identifier]['data']


def folder_path(folder):
    if not isinstance(folder, str):
        raise ArgumentError(f'expected a folder as text, found {folder!r}')
    parts = [part.strip() for part in folder.replace('\\', '/').split('/') if part.strip()]
    if any(part in ('.', '..') or len(part) > 64 for part in parts) or len(parts) > 8:
        raise ArgumentError(
            f'expected a folder of at most 8 names of 64 characters, found {folder!r}'
        )
    return '/'.join(parts)


@locked
def update_match(identifier, title=None, folder=None):
    if recording(identifier) is None:
        raise ArgumentError(f'expected a recording of the library, found {identifier!r}')
    summary = dict(MATCHES[identifier]['summary'])
    if title is not None:
        title = str(title).strip()[:120]
        if not title:
            raise ArgumentError('expected a title with at least one character')
        summary['title'] = title
    if folder is not None:
        summary['folder'] = folder_path(folder)
    summary.setdefault('folder', '')
    MATCHES[identifier]['summary'] = summary
    return summary


@locked
def remove_match(identifier):
    if recording(identifier) is None:
        return False
    del MATCHES[identifier]
    return True


def scenario_size(scenario):
    for size in (5, 11):
        if scenario.startswith(f'{size}_vs_{size}'):
            return size
    return None


def check_scenario(sides, scenario):
    """Refuses a scenario whose size differs from the format of a submitted team."""
    size = scenario_size(scenario)
    for team in sides:
        if team['kind'] != 'builtin' and size != FORMATS[team['format']]:
            raise ArgumentError(
                f'expected a scenario of {team["format"]} for {team["name"]}, found {scenario}'
            )


class Requests:
    """The queue of requested matches, played one at a time as synthetic matches."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries_ = []
        self.wake = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def add(self, left, right, scenario, seed):
        if scenario not in SCENARIOS:
            raise ArgumentError(f'expected a shipped scenario, found {scenario!r}')
        check_scenario([entry(left), entry(right)], scenario)
        item = {
            'id': new_identifier('r'),
            'left': left,
            'right': right,
            'scenario': scenario,
            'seed': seed,
            'state': 'queued',
            'progress': 0.0,
            'match': None,
            'error': None,
        }
        with self.lock:
            self.entries_.append(item)
        self.wake.set()
        return dict(item)

    def entries(self):
        with self.lock:
            return [dict(item) for item in reversed(self.entries_)]

    def cancel(self, identifier):
        with self.lock:
            item = next((item for item in self.entries_ if item['id'] == identifier), None)
            if item is None or item['state'] not in ('queued', 'running'):
                return False
            item['state'], item['error'] = 'failed', 'cancelled'
        return True

    def run(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    queued = [item for item in self.entries_ if item['state'] == 'queued']
                if not queued:
                    break
                self.play(queued[0])

    def play(self, item):
        with self.lock:
            if item['state'] != 'queued':
                return
            item['state'] = 'running'
        try:
            teams_ = [entry(item['left']), entry(item['right'])]
            check_scenario(teams_, item['scenario'])
            names = [team['name'] for team in teams_]
            sides = match_teams(item['left'], item['right'])
        except ArgumentError as error:
            with self.lock:
                item['state'], item['error'] = 'failed', f'ArgumentError: {error}'
            return
        for k in range(10):
            time.sleep(0.3)
            with self.lock:
                if item['state'] != 'running':
                    return
                item['progress'] = (k + 1) / 12
        per_team = 5 if scenario_size(item['scenario']) == 5 else 11
        seed = 7 if item['seed'] is None else item['seed']
        header, frames = render_sample.generate_frames(
            per_team, MATCH_FRAMES, seed, tuple(names), sides
        )
        header['title'] = f'{names[0]} against {names[1]}'
        header['scenario'] = item['scenario']
        with self.lock:
            if item['state'] != 'running':
                return
        match = save_match(header, frames)
        with self.lock:
            item['progress'], item['match'], item['state'] = 1.0, match, 'done'


def with_flags(frame, flags):
    return frame[:28] + struct.pack('<i', flags) + frame[32:]


class LiveSource:
    """Replays synthetic episodes forever, each frame published at real time.

    A batch resets itself after an episode, so the step that ends one shows the next kick off with
    flags 2, and the frame of the reset before the first step is never shown.
    """

    def __init__(self, per_team):
        self.id = new_identifier('live')
        self.per_team = per_team
        self.scenario = '11_vs_11_stochastic' if per_team == 11 else '5_vs_5'
        self.title = f'{self.scenario}, environment 0, process {os.getpid()}'
        self.cond = threading.Condition()
        self.header = None
        self.frames = []
        self.serial = 0
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        seed = 10
        while True:
            header, frames = render_sample.generate_frames(
                self.per_team, 600, seed, ('Left', 'Right'), DEFAULT_TEAMS
            )
            header['title'] = self.title
            header['scenario'] = self.scenario
            header['frame_count'] = None
            for index, frame in enumerate(frames[:-1]):
                if index == 0:
                    if self.header is None:
                        continue
                    frame = with_flags(frame, 2)
                time.sleep(1.0 / render_sample.STEPS_PER_SECOND)
                with self.cond:
                    if self.header is None:
                        self.header = header
                    self.frames.append(frame)
                    self.frames = self.frames[-BACKLOG_SECONDS * render_sample.STEPS_PER_SECOND :]
                    self.serial += 1
                    self.cond.notify_all()
            seed += 1

    def backlog(self):
        """Returns the newest frames that follow each other step by step, under the lock."""
        frames = self.frames
        start = len(frames) - 1
        while start > 0:
            if (
                struct.unpack_from('<i', frames[start])[0]
                != struct.unpack_from('<i', frames[start - 1])[0] + 1
            ):
                break
            start -= 1
        return frames[max(0, start) :]

    def describe(self):
        return {'id': self.id, 'title': self.title, 'scenario': self.scenario, 'environment': 0}


def make_handler(requests):
    """Returns the request handler, the real server's checks and routes on the data above."""
    allowed_hosts = set(LOOPBACK)

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = 'cuju-render-mock'

        def _allowed(self):
            named = self.headers.get('Host') or ''
            if urllib.parse.urlsplit('//' + named).hostname not in allowed_hosts:
                self._error(403, 'expected a request to this server by its loopback address')
                return False
            origin = self.headers.get('Origin')
            if origin is not None and self.command != 'GET' and origin != f'http://{named}':
                self._error(403, 'expected a request from the page of this server')
                return False
            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                self._error(400, 'expected a Content-Length of digits')
                return False
            if length > MAXIMUM_BODY:
                self._error(413, 'expected a request body of at most 110 MB')
                return False
            return True

        def log_message(self, fmt, *arguments):
            sys.stderr.write(f'{self.address_string()} {fmt % arguments}\n')

        def _handle(self, method):
            if not self._allowed():
                return
            try:
                method()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (ArgumentError, ValueError, TypeError, KeyError, AttributeError) as error:
                self._error(400, f'{type(error).__name__}: {error}')
            except Exception as error:
                self._error(500, f'{type(error).__name__}: {error}')

        def _json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._json({'error': message}, status)

        def _bytes(self, data, kind, no_store=False):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            if no_store:
                self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get('Content-Length') or 0)
            return self.rfile.read(length) if length > 0 else b''

        def _document(self):
            document = json.loads(self._body() or b'{}')
            if not isinstance(document, dict):
                raise ArgumentError('expected a JSON object')
            return document

        def _route(self):
            parsed = urllib.parse.urlsplit(self.path)
            parts = [urllib.parse.unquote(part) for part in parsed.path.split('/') if part]
            query = dict(urllib.parse.parse_qsl(parsed.query))
            return parts, query

        def do_GET(self):
            self._handle(self._get)

        def _get(self):
            parts, _ = self._route()
            try:
                if parts[:1] != ['api']:
                    self._static(parts)
                elif parts == ['api', 'matches']:
                    self._json(matches())
                elif len(parts) == 3 and parts[1] == 'matches':
                    data = recording(parts[2])
                    if data is None:
                        self._error(404, 'no such match')
                    else:
                        self._bytes(data, 'application/octet-stream')
                elif parts == ['api', 'live']:
                    self._json([source.describe() for source in LIVE.values()])
                elif len(parts) == 4 and parts[1] == 'live' and parts[3] == 'stream':
                    self._relay(parts[2])
                elif parts == ['api', 'scenarios']:
                    self._json(SCENARIOS)
                elif parts == ['api', 'teams']:
                    self._json(teams())
                elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                    data = model_path(parts[2])
                    if data is None:
                        self._error(404, 'no such model')
                    else:
                        self._bytes(data, 'model/gltf-binary', no_store=True)
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                    self._image(parts[2], parts[4])
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                    path = sound_path(parts[2], parts[4])
                    if path is None:
                        self._error(404, 'no such sound')
                    else:
                        data = FOLDERS[parts[2]][path]
                        self._bytes(data, SOUND_TYPES[PurePosixPath(path).suffix], no_store=True)
                elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'export':
                    self._export(parts[2])
                elif parts == ['api', 'models']:
                    self._json(shipped_models())
                elif parts == ['api', 'requests']:
                    self._json(requests.entries())
                else:
                    self._error(404, 'no such resource')
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            self._handle(self._post)

        def _post(self):
            parts, query = self._route()
            try:
                if parts == ['api', 'teams']:
                    team = add(
                        query.get('name', ''),
                        query.get('filename', 'main.py'),
                        self._body(),
                        query.get('format', '11v11'),
                    )
                    self._json(team, 201)
                elif parts == ['api', 'requests']:
                    document = self._document()
                    seed = document.get('seed')
                    item = requests.add(
                        str(document.get('left', BUILTIN)),
                        str(document.get('right', BUILTIN)),
                        str(document.get('scenario', '11_vs_11_kaggle')),
                        None if seed in (None, '') else int(seed),
                    )
                    self._json(item, 201)
                else:
                    self._error(404, 'no such resource')
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_PUT(self):
            self._handle(self._put)

        def _put(self):
            parts, query = self._route()
            try:
                if len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                    joints = json.loads(query.get('joints') or '{}')
                    self._json(set_model(parts[2], self._body(), joints))
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                    self._json(set_image(parts[2], parts[4], self._body()))
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                    self._json(set_sound(parts[2], parts[4], self._body()))
                else:
                    self._error(404, 'no such resource')
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_PATCH(self):
            self._handle(self._patch)

        def _patch(self):
            parts, _ = self._route()
            try:
                if len(parts) == 3 and parts[1] == 'matches':
                    document = self._document()
                    self._json(
                        update_match(parts[2], document.get('title'), document.get('folder'))
                    )
                elif len(parts) == 3 and parts[1] == 'teams':
                    document = self._document()
                    self._json(
                        update(
                            parts[2],
                            document.get('name'),
                            document.get('style'),
                            document.get('format'),
                        )
                    )
                else:
                    self._error(404, 'no such resource')
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_DELETE(self):
            self._handle(self._delete)

        def _delete(self):
            parts, _ = self._route()
            if len(parts) == 3 and parts[1] == 'requests':
                removed = requests.cancel(parts[2])
            elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                try:
                    removed = remove_model(parts[2])
                except ArgumentError:
                    removed = False
            elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                removed = remove_image(parts[2], parts[4])
            elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                removed = remove_sound(parts[2], parts[4])
            elif len(parts) == 3 and parts[1] == 'matches':
                removed = remove_match(parts[2])
            elif len(parts) == 3 and parts[1] == 'teams':
                removed = remove(parts[2])
            else:
                self._error(404, 'no such resource')
                return
            if removed:
                self.send_response(204)
                self.end_headers()
            else:
                self._error(404, 'no such item')

        def _static(self, parts):
            path = os.path.realpath(os.path.join(WEB_ROOT, *(parts or ['index.html'])))
            try:
                inside_root = os.path.commonpath([WEB_ROOT, path]) == WEB_ROOT
            except ValueError:
                inside_root = False
            if not inside_root or path == WEB_ROOT or not os.path.isfile(path):
                self._error(404, 'no such file')
                return
            with open(path, 'rb') as f:
                data = f.read()
            kind = TYPES.get(os.path.splitext(path)[1]) or mimetypes.guess_type(path)[0]
            self._bytes(data, kind or 'application/octet-stream')

        def _image(self, identifier, slot):
            found = image(identifier, slot)
            if found is None:
                self._error(404, 'no such image')
                return
            path, data = found
            kind = IMAGE_TYPES.get(PurePosixPath(path).suffix, 'application/json')
            self._bytes(data, kind, no_store=True)

        def _export(self, identifier):
            try:
                filename, data = export(identifier)
            except ArgumentError as error:
                self._error(404, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            fallback = filename.encode('ascii', 'replace').decode().replace('?', '-')
            quoted = urllib.parse.quote(filename)
            self.send_header(
                'Content-Disposition',
                f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quoted}',
            )
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _relay(self, identifier):
            source = LIVE.get(identifier)
            if source is None:
                self._error(404, 'no such live source')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with source.cond:
                source.cond.wait_for(lambda: source.header is not None)
                head = render_sample.encode_header(source.header)
                seen = source.serial
                backlog = source.backlog()
            self.wfile.write(head + b''.join(backlog))
            self.wfile.flush()
            while True:
                with source.cond:
                    start = seen
                    source.cond.wait_for(lambda start=start: source.serial > start, timeout=5.0)
                    fresh = min(len(source.frames), source.serial - seen)
                    batch = source.frames[-fresh:] if fresh else []
                    seen = source.serial
                if batch:
                    self.wfile.write(b''.join(batch))
                    self.wfile.flush()

    return Handler


def load_recordings(folder):
    """Adds every .cjr file of a folder to the library, oldest first."""
    for path in sorted(Path(folder).glob('*.cjr')):
        data = gzip.decompress(path.read_bytes())
        if not data.startswith(render_sample.MAGIC):
            raise ValueError(f'expected a cuju recording, found {path}')
        start = len(render_sample.MAGIC) + 4
        (length,) = struct.unpack_from('<I', data, len(render_sample.MAGIC))
        header = json.loads(data[start : start + length])
        size = render_sample.frame_bytes(len(header['players']))
        body = data[start + length :]
        save_match(header, [body[i : i + size] for i in range(0, len(body) - size + 1, size)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--short', action='store_true', help='use short sample matches')
    parser.add_argument('--recordings', help='a folder of .cjr files to add to the library')
    args = parser.parse_args()
    print('generating synthetic recordings')
    reds = {'name': 'Reds', 'color': [200, 30, 40], 'secondary': [250, 250, 250]}
    builtin_name = entry(BUILTIN)['name']
    for per_team, steps, names, sides in (
        (11, 2400, ('Reds', builtin_name), [reds, header_team(BUILTIN)]),
        (5, 1800, ('Greens', 'Golds'), None),
    ):
        header, frames = render_sample.generate_frames(
            per_team, min(steps, 120) if args.short else steps, 1, names, sides
        )
        save_match(header, frames)
    if args.recordings:
        load_recordings(args.recordings)
    source = LiveSource(11)
    LIVE[source.id] = source
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Requests()))
    server.daemon_threads = True
    print(f'serving {WEB_ROOT} on http://127.0.0.1:{server.server_address[1]}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
