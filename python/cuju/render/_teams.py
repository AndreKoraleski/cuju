# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The team registry of the renderer, Kaggle submissions kept in the renderer's folder.

A team is a submission of Kaggle's GRF competition, a folder whose `main.py` defines the agent, as
Kaggle receives it, also accepted as one Python file or an archive. The built-in AI of GRF is always
a team as well. A submission is code, which runs with the permissions of the process that plays
it.

Each team carries a format, `5v5` or `11v11`, and a style, its kit, its goalkeeper's kit, its
players' names, numbers, skin and hair, its shirt's painted front and back, its sponsors and its
player model, GRF's, a shipped one or its own rigged binary glTF file. Only the renderer draws them,
and none reaches a match. They live in the folder `cuju-team` inside the submission, so a team
exported from the page carries them along, and a submission folder that holds one brings them back.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import threading
import time
import types
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..errors import ArgumentError
from . import _home

BUILTIN = 'builtin'
# The start of a submission folder as the page uploads it, an index and the files' bytes.
FOLDER_MAGIC = b'CUJDIR1\n'
_BUILTIN_ENTRY = {'id': BUILTIN, 'name': "GRF's built-in AI", 'kind': 'builtin', 'created': None}
# Kaggle: the file that an archive's agent lives in.
_MAIN = 'main.py'
# Submissions larger than this are refused, as Kaggle limits them too.
_MAXIMUM_BYTES = 100 * 1024 * 1024
# An archive may unpack to this much and this many files, so a small upload cannot fill the disk.
_MAXIMUM_UNPACKED = 500 * 1024 * 1024
_MAXIMUM_FILES = 10000
# One lock serializes every change of the registry, which the server makes from several threads.
_LOCK = threading.RLock()


_PATTERNS = ('plain', 'stripes', 'hoops', 'halves', 'sash', 'checkers')
# `grf` keeps the skin or hair that GRF gives the player's index.
_SKINS = ('grf', 'light', 'tan', 'medium', 'dark')
_HAIRS = ('grf', 'bald', 'short01', 'short02', 'medium01', 'long01', 'long02')
_HAIR_COLORS = ('grf', 'black', 'brown', 'darkblonde', 'blonde', 'red')
_NUMBERS = ('white', 'black', 'secondary')
_PARTS = ('primary', 'secondary', 'shorts', 'socks')
# The sizes of a team for competition, the players a side of `5_vs_5` and of the full matches.
FORMATS = {'5v5': 5, '11v11': 11}
ROLES = ('GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'LM', 'RM', 'AM', 'CF')
# The players that a submission of each format may control, as agents of its side. GRF: the
# goalkeeper of `5_vs_5` is not controllable, every player of the eleven a side scenarios is.
CONTROLLABLE = {'5v5': 4, '11v11': 11}
# A submission controls the active player alone by default, as on Kaggle.
DEFAULT_CONTROL = {
    'players': 1,
    'assignment': 'dynamic',
    'selection': [],
    'goalkeeper': 'scenario',
    'action_set': 'default',
}
# The images that a team may hold, the flat front and back of its shirt, the editor's documents
# that made them, and the sponsors that the adboards show while the team plays.
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
# The folder of a submission that holds the renderer's files of its team.
TEAM_FOLDER = 'cuju-team'
_MAXIMUM_IMAGE_BYTES = 8 * 1024 * 1024
# The sounds a team may hold, each played when one of its players does that, or its fans sing.
SOUND_SLOTS = ('goal', 'shot', 'pass', 'tackle', 'save', 'kick', 'celebration', 'chant')
_MAXIMUM_SOUND_BYTES = 2 * 1024 * 1024
_SOUND_SUFFIXES = ('.ogg', '.mp3', '.wav')
_KIT = {
    'pattern': 'plain',
    'primary': [214, 48, 49],
    'secondary': [255, 255, 255],
    'shorts': [255, 255, 255],
    'socks': [214, 48, 49],
}
_GOALKEEPER = {
    'primary': [46, 204, 113],
    'secondary': [20, 20, 20],
    'shorts': [20, 20, 20],
    'socks': [46, 204, 113],
}


def _player_style(index: int) -> dict[str, Any]:
    """Returns the default style of the player at `index` of a team, GRF's own look."""
    return {
        'name': '',
        'number': index + 1,
        'skin': 'grf',
        'hair': 'grf',
        'hair_color': 'grf',
        'model': 'team',
        'celebration': 'team',
    }


def default_style(format: str = '11v11', kit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns the style of a new team of `format`, GRF's model in a plain kit."""
    colors = kit or _KIT
    return {
        'model': 'grf',
        'kit': dict(colors),
        'goalkeeper': dict(_GOALKEEPER),
        'badge': '',
        'number': 'white',
        'colors': {},
        # The fans wear the kit's colors unless the team picks colors of their own.
        'crowd': {'colors': [list(colors['primary']), list(colors['secondary'])]},
        # The celebrations the page picks from after a goal, empty for every one it knows.
        'celebrations': [],
        'players': [_player_style(index) for index in range(FORMATS[format])],
    }


_BUILTIN_KIT = {
    'pattern': 'plain',
    'primary': [9, 132, 227],
    'secondary': [255, 255, 255],
    'shorts': [9, 132, 227],
    'socks': [255, 255, 255],
}


def _color(value: Any, name: str) -> list[int]:
    """Returns an RGB color as three integers from 0 to 255, or raises `ArgumentError`."""
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


def _printable(text: str) -> str:
    """Returns `text` with its control characters, line breaks among them, turned into spaces."""
    return ''.join(character if character.isprintable() else ' ' for character in text)


def _choice(value: Any, choices: tuple[str, ...], name: str) -> str:
    """Returns `value` if it is one of `choices`, or raises `ArgumentError`."""
    if value not in choices:
        raise ArgumentError(f'expected {name} as one of {", ".join(choices)}, found {value!r}')
    return value


def _players(
    value: Any, current: list[dict[str, Any]], models: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Returns the roster with the changes of `value`, a list of per-player dictionaries."""
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
                roster[index][key] = _printable(entry)
            elif key == 'number':
                if isinstance(entry, bool) or not isinstance(entry, int) or not 0 <= entry <= 99:
                    raise ArgumentError(f'expected a shirt number from 0 to 99, found {entry!r}')
                roster[index][key] = entry
            elif key == 'skin':
                roster[index][key] = _choice(entry, _SKINS, 'the skin')
            elif key == 'hair':
                roster[index][key] = _choice(entry, _HAIRS, 'the hair')
            elif key == 'hair_color':
                roster[index][key] = _choice(entry, _HAIR_COLORS, 'the hair color')
            elif key == 'celebration':
                # `team` picks among the team's celebrations, any other value is the player's own.
                roster[index][key] = _celebration(entry)
            elif key == 'model':
                # `team` follows the team's model, any other value gives the player one of its own.
                roster[index][key] = _choice(
                    entry, ('team', 'grf', 'custom', *models), 'the player model'
                )
            else:
                raise ArgumentError(f'expected an entry of a player, found {key!r}')
    return roster


_HEX = re.compile(r'#[0-9a-f]{6}')
# A celebration is named by the page, such as `knee_slide`, or `clip:` and an animation of a model.
_CELEBRATION = re.compile(r'[a-z0-9_]{1,40}(:[A-Za-z0-9_. -]{1,60})?')


def _celebration(value: Any) -> str:
    """Returns the name of a celebration, which the page knows or finds among a model's clips."""
    if not isinstance(value, str) or not _CELEBRATION.fullmatch(value):
        raise ArgumentError(f'expected the name of a celebration, found {value!r}')
    return value


# A model's colors to replace, at most this many per model.
_MAXIMUM_RECOLORS = 32


def _colors(value: Any, models: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Returns the colors that replace others on each model, as `#rrggbb` pairs by model.

    The page lists the colors present in a model's textures and materials, and each pair maps one
    of them to the color the team wants instead. GRF's own model takes the kit's colors instead.
    """
    if not isinstance(value, dict):
        raise ArgumentError(f'expected the colors as a dictionary by model, found {value!r}')
    result: dict[str, dict[str, str]] = {}
    for model, pairs in value.items():
        _choice(model, models, 'the model of the colors')
        if not isinstance(pairs, dict) or len(pairs) > _MAXIMUM_RECOLORS:
            raise ArgumentError(f'expected at most {_MAXIMUM_RECOLORS} colors for {model!r}')
        result[model] = {}
        for old, new in pairs.items():
            if not (
                isinstance(old, str)
                and isinstance(new, str)
                and _HEX.fullmatch(old.lower())
                and _HEX.fullmatch(new.lower())
            ):
                raise ArgumentError(f'expected colors as #rrggbb, found {old!r} and {new!r}')
            result[model][old.lower()] = new.lower()
    return result


def style(
    changes: dict[str, Any] | None, base: dict[str, Any], models: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Returns the complete style `base` with `changes` applied and checked.

    Args:
        changes: Entries of the style to change, or `None`.
        base: The style to change, whose roster sets the team's size.
        models: The shipped models a team may choose besides `grf` and `custom`.

    Raises:
        ArgumentError: If a change names an unknown entry or holds a value outside its choices.
    """
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
                    result[key][part] = _choice(entry, _PATTERNS, 'the pattern')
                elif part in _PARTS:
                    result[key][part] = _color(entry, f'{key} {part}')
                else:
                    raise ArgumentError(f'expected a part of {key}, found {part!r}')
        elif key == 'model':
            result[key] = _choice(value, ('grf', 'custom', *models), 'the model')
        elif key == 'number':
            result[key] = _choice(value, _NUMBERS, 'the numbers')
        elif key == 'badge':
            if not isinstance(value, str) or len(value) > 3:
                raise ArgumentError(
                    f'expected a badge of at most three characters, found {value!r}'
                )
            result[key] = value
        elif key == 'players':
            result[key] = _players(value, result[key], models)
        elif key == 'colors':
            result[key] = _colors(value, ('custom', *models))
        elif key == 'celebrations':
            if not isinstance(value, list) or len(value) > 32:
                raise ArgumentError('expected a list of at most 32 celebrations')
            result[key] = [_celebration(item) for item in value]
        elif key == 'crowd':
            if not isinstance(value, dict) or set(value) - {'colors'}:
                raise ArgumentError(f'expected the crowd as {{"colors": [...]}}, found {value!r}')
            colors = value.get('colors', result[key]['colors'])
            if not isinstance(colors, list) or not 1 <= len(colors) <= 4:
                raise ArgumentError('expected one to four colors of the fans')
            result[key] = {'colors': [_color(color, 'a color of the fans') for color in colors]}
        else:
            raise ArgumentError(f'expected an entry of a style, found {key!r}')
    return result


def _resize(style_: dict[str, Any], format: str) -> dict[str, Any]:
    """Returns the style with its roster cut or extended to the size of `format`."""
    size = FORMATS[format]
    roster = style_['players'][:size]
    roster += [_player_style(index) for index in range(len(roster), size)]
    return {**style_, 'players': roster}


def shipped_models() -> list[dict[str, Any]]:
    """Returns the description of every player model that the package ships."""
    folder = Path(__file__).parent / 'web' / 'assets' / 'models'
    models = []
    for path in sorted(folder.glob('*.json')):
        with contextlib.suppress(OSError, ValueError):
            description = json.loads(path.read_text())
            if path.with_suffix('.glb').is_file():
                description['url'] = f'assets/models/{path.stem}.glb'
                models.append(description)
    return models


def _model_names() -> tuple[str, ...]:
    """Returns the identifiers of the shipped models."""
    return tuple(model['id'] for model in shipped_models())


def _complete(entry: dict[str, Any], kit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns a stored entry with its format, style, model and images filled and checked."""
    entry = dict(entry)
    if entry.get('format') not in FORMATS:
        entry['format'] = '11v11'
    base = default_style(entry['format'], kit)
    saved = entry.get('style') or {}
    try:
        entry['style'] = _resize(
            style(saved, {**base, **_known(saved, base)}, _model_names()), entry['format']
        )
    except ArgumentError:
        entry['style'] = base
    entry.setdefault('model', None)
    try:
        entry['control'] = check_control(entry.get('control'), entry['format'])
    except ArgumentError:
        entry['control'] = dict(DEFAULT_CONTROL)
    folder = _meta_path(entry['id']).parent
    entry['images'] = [slot for slot in IMAGE_SLOTS if _image_file(folder, slot) is not None]
    entry['sounds'] = [slot for slot in SOUND_SLOTS if _sound_file(folder, slot) is not None]
    return entry


def _known(saved: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Returns the entries of a saved style that the current style still has, for old files."""
    return {key: value for key, value in saved.items() if key in base and key != 'players'}


def _meta_path(identifier: str) -> Path:
    """Returns the file of a team's entry, the built-in AI's included."""
    return _home.folder('teams') / identifier / TEAM_FOLDER / 'team.json'


def _builtin() -> dict[str, Any]:
    """Returns the entry of the built-in AI, with the name and style its file keeps, if any."""
    entry = dict(_BUILTIN_ENTRY)
    with contextlib.suppress(OSError, ValueError):
        saved = json.loads(_meta_path(BUILTIN).read_text())
        entry.update(
            {key: saved[key] for key in ('name', 'style', 'model', 'format') if key in saved}
        )
    return _complete(entry, _BUILTIN_KIT)


def teams() -> list[dict[str, Any]]:
    """Returns every team, the built-in AI first, then the submissions from newest to oldest."""
    entries = []
    for folder in _home.folder('teams').iterdir():
        meta = folder / TEAM_FOLDER / 'team.json'
        if folder.name == BUILTIN or not (folder.is_dir() and meta.is_file()):
            continue
        try:
            entry = _complete(json.loads(meta.read_text()))
        except (OSError, ValueError, KeyError):
            continue
        entries.append(entry)
    entries.sort(key=lambda entry: entry.get('created') or '', reverse=True)
    return [_builtin(), *entries]


def check_control(control: Any, format: str) -> dict[str, Any]:
    """Returns normalized team control and migrates earlier settings."""
    control = dict(DEFAULT_CONTROL) if control is None else control
    allowed = {'players', 'goalkeeper', 'assignment', 'selection', 'action_set'}
    if not isinstance(control, dict) or set(control) - allowed:
        raise ArgumentError('expected players, assignment, selection, goalkeeper and action_set')
    players = control.get('players', 1)
    most = FORMATS[format]
    if isinstance(players, bool) or not isinstance(players, int) or not 1 <= players <= most:
        raise ArgumentError(f'expected from 1 to {most} controlled players')
    legacy = 'assignment' not in control
    keeper = _choice(
        control.get('goalkeeper', 'scenario'), ('agent', 'builtin', 'scenario'), 'goalkeeper'
    )
    assignment = control.get('assignment', 'fixed' if keeper == 'builtin' else 'dynamic')
    _choice(assignment, ('dynamic', 'fixed'), 'assignment')
    if legacy and keeper == 'builtin' and (format != '11v11' or players != 10):
        raise ArgumentError('expected 10 players for the earlier goalkeeper setting')
    if legacy and keeper == 'agent':
        keeper = 'scenario'
    if assignment == 'dynamic' and keeper == 'builtin':
        raise ArgumentError('expected fixed assignment for a goalkeeper handed to the AI')
    selection = control.get('selection', [])
    if legacy and assignment == 'fixed':
        selection = list(range(1, players + 1))
    if not isinstance(selection, list):
        raise ArgumentError('expected a list of player indices or roles')
    if assignment == 'fixed' and len(selection) != players:
        raise ArgumentError('expected one selection for each controlled player')
    if assignment == 'dynamic' and selection:
        raise ArgumentError('expected an empty selection for dynamic assignment')
    if selection:
        indices = all(type(item) is int and 0 <= item <= 10 for item in selection)
        roles = all(isinstance(item, str) and item in ROLES for item in selection)
        if not (indices or roles) or (indices and len(set(selection)) != len(selection)):
            raise ArgumentError('expected distinct player indices or GRF role names')
    action_set = _choice(control.get('action_set', 'default'), ('default', 'v2'), 'action set')
    return dict(
        players=players,
        assignment=assignment,
        selection=list(selection),
        goalkeeper=keeper,
        action_set=action_set,
    )


def scenario_path(scenario: str) -> Path:
    """Returns the source of a shipped or renderer scenario."""
    local = Path(__file__).parent / 'scenarios' / f'{scenario}.py'
    if local.is_file():
        return local
    return Path(__file__).parents[1] / 'data' / 'scenarios' / f'{scenario}.py'


def renderer_scenarios() -> list[str]:
    """Returns the additional training scenarios."""
    return sorted(path.stem for path in (Path(__file__).parent / 'scenarios').glob('*.py'))


def scenario_players(scenario: str, side: int) -> list[Any]:
    """Returns the initial formation of a side."""
    from ..scenarios import Builder, load_scenario

    builder = Builder(0)
    load_scenario(scenario_path(scenario))(builder)
    config = builder.config()
    return config.left_team if side == 0 else config.right_team


def agent_count(team: dict[str, Any], scenario: str | None = None, side: int = 0) -> int:
    """Returns the engine agents required by a side."""
    if team['kind'] == 'builtin':
        return 0
    control = check_control(team.get('control'), team['format'])
    if control['assignment'] == 'dynamic':
        return control['players']
    if scenario is None:
        return CONTROLLABLE[team['format']]
    return sum(player.controllable for player in scenario_players(scenario, side))


def selected_players(control: dict[str, Any], formation: list[Any]) -> list[int]:
    """Resolves selections in agent order, with repeated roles in lineup order."""
    available = [index for index, player in enumerate(formation) if player.controllable]
    if control['players'] > len(available):
        raise ArgumentError(f'scenario has only {len(available)} controllable players')
    keeper = [index for index, player in enumerate(formation) if player.role == 0]
    if control['assignment'] == 'dynamic':
        if control['goalkeeper'] == 'agent' and not any(i in available for i in keeper):
            raise ArgumentError('scenario makes the goalkeeper uncontrollable')
        return available
    selected = []
    for item in control['selection']:
        candidates = (
            [item]
            if type(item) is int
            else [index for index in available if formation[index].role == ROLES.index(item)]
        )
        found = next((i for i in candidates if i in available and i not in selected), None)
        if found is None:
            raise ArgumentError(f'scenario cannot select player {item}')
        selected.append(found)
    if control['goalkeeper'] == 'builtin' and not any(i in available for i in keeper):
        raise ArgumentError('scenario makes the goalkeeper uncontrollable, choose scenario')
    owns_keeper = any(i in keeper for i in selected)
    if control['goalkeeper'] == 'agent' and not owns_keeper:
        raise ArgumentError('expected the goalkeeper in the fixed selection')
    if control['goalkeeper'] in ('builtin', 'scenario') and owns_keeper:
        raise ArgumentError('expected the goalkeeper outside the fixed selection')
    if control['goalkeeper'] == 'scenario' and any(i in available for i in keeper):
        raise ArgumentError('scenario has a controllable goalkeeper, choose agent or builtin')
    return selected


def update(
    identifier: str,
    name: str | None = None,
    changes: dict[str, Any] | None = None,
    format: str | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Renames a team, changes its format, its control or its style, and returns its new entry.

    Args:
        identifier: The team, the built-in AI included.
        name: The new name, or `None` to keep it.
        changes: Entries of the style to change, or `None`.
        format: `5v5` or `11v11`, the size of the team for competition, which sets the length of
            its roster, or `None` to keep it.
        control: The players that the submission controls, as `check_control` takes them, or
            `None` to keep them, or the default where the new format refuses them.

    Raises:
        ArgumentError: If the team is unknown or a change is invalid.
    """
    current = entry(identifier)
    if name is not None:
        name = _printable(str(name)).strip()[:80]
        if not name:
            raise ArgumentError('expected a team name with at least one character')
        current['name'] = name
    if format is not None:
        current['format'] = _choice(format, tuple(FORMATS), 'the format')
        current['style'] = _resize(current['style'], current['format'])
    if control is not None:
        current['control'] = check_control(control, current['format'])
    else:
        try:
            current['control'] = check_control(current.get('control'), current['format'])
        except ArgumentError:
            current['control'] = dict(DEFAULT_CONTROL)
    current['style'] = style(changes, current['style'], _model_names())
    chosen = {
        current['style']['model'],
        *(player['model'] for player in current['style']['players']),
    }
    if 'custom' in chosen and model_path(identifier) is None:
        raise ArgumentError('expected an uploaded model before choosing the custom model')
    _write(identifier, current)
    return current


def _image_file(folder: Path, slot: str) -> Path | None:
    """Returns the file of an image slot of a team's folder, whatever its extension, or `None`."""
    for path in folder.glob(f'{slot}.*'):
        if path.suffix in ('.png', '.jpg', '.webp', '.json'):
            return path
    return None


def _image_suffix(slot: str, data: bytes) -> str:
    """Returns the extension of an image's bytes, or raises `ArgumentError` for another kind."""
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


def set_image(identifier: str, slot: str, data: bytes) -> dict[str, Any]:
    """Stores an image of a team, a shirt side, its editor document or a sponsor.

    Args:
        identifier: The team, the built-in AI included.
        slot: One of `IMAGE_SLOTS`.
        data: The image, PNG, JPEG or WebP, or for an `-edit` slot the editor's JSON document.

    Returns:
        The team's new entry.

    Raises:
        ArgumentError: If the team or slot is unknown, or the data is too large or of another kind.
    """
    entry(identifier)
    _choice(slot, IMAGE_SLOTS, 'the image slot')
    if len(data) > _MAXIMUM_IMAGE_BYTES:
        raise ArgumentError(f'expected an image of at most 8 MB, found {len(data)} bytes')
    suffix = _image_suffix(slot, data)
    folder = _meta_path(identifier).parent
    with _LOCK:
        old = _image_file(folder, slot)
        _replace(folder / f'{slot}{suffix}', data)
        if old is not None and old.name != f'{slot}{suffix}':
            old.unlink()
    return entry(identifier)


def image_path(identifier: str, slot: str) -> Path | None:
    """Returns the file of a team's image slot, or `None`."""
    if slot not in IMAGE_SLOTS:
        return None
    if identifier != BUILTIN and not _home.valid_identifier(identifier):
        return None
    return _image_file(_meta_path(identifier).parent, slot)


def remove_image(identifier: str, slot: str) -> bool:
    """Deletes a team's image slot and returns whether it held an image."""
    path = image_path(identifier, slot)
    if path is None:
        return False
    path.unlink()
    return True


def _sound_file(folder: Path, slot: str) -> Path | None:
    """Returns the file of a sound slot of a team's folder, whatever its extension, or `None`."""
    for path in folder.glob(f'sound-{slot}.*'):
        if path.suffix in _SOUND_SUFFIXES:
            return path
    return None


def _sound_suffix(data: bytes) -> str:
    """Returns the extension of a sound's bytes, or raises `ArgumentError` for another kind."""
    if data.startswith(b'OggS'):
        return '.ogg'
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return '.wav'
    # MP3: a tag of ID3 or the sync bits of a first frame.
    if data.startswith(b'ID3') or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return '.mp3'
    raise ArgumentError('expected an Ogg, MP3 or WAV sound')


def set_sound(identifier: str, slot: str, data: bytes) -> dict[str, Any]:
    """Stores a sound of a team, which the page plays when one of its players does that.

    Args:
        identifier: The team, the built-in AI included.
        slot: One of `SOUND_SLOTS`.
        data: The sound, Ogg, MP3 or WAV, of at most 2 MB.

    Returns:
        The team's new entry.

    Raises:
        ArgumentError: If the team or slot is unknown, or the data is too large or of another kind.
    """
    entry(identifier)
    _choice(slot, SOUND_SLOTS, 'the sound slot')
    if len(data) > _MAXIMUM_SOUND_BYTES:
        raise ArgumentError(f'expected a sound of at most 2 MB, found {len(data)} bytes')
    suffix = _sound_suffix(data)
    folder = _meta_path(identifier).parent
    with _LOCK:
        old = _sound_file(folder, slot)
        _replace(folder / f'sound-{slot}{suffix}', data)
        if old is not None and old.name != f'sound-{slot}{suffix}':
            old.unlink()
    return entry(identifier)


def sound_path(identifier: str, slot: str) -> Path | None:
    """Returns the file of a team's sound slot, or `None`."""
    if slot not in SOUND_SLOTS:
        return None
    if identifier != BUILTIN and not _home.valid_identifier(identifier):
        return None
    return _sound_file(_meta_path(identifier).parent, slot)


def remove_sound(identifier: str, slot: str) -> bool:
    """Deletes a team's sound slot and returns whether it held a sound."""
    path = sound_path(identifier, slot)
    if path is None:
        return False
    path.unlink()
    return True


# The joints of GRF's skeleton that a model may map. Only `body`, the pelvis, is required, so a
# simple rig without elbows or knees maps what it has, and its bones follow their parents.
JOINTS = (
    'body',
    'middle',
    'neck',
    'left_shoulder',
    'left_elbow',
    'right_shoulder',
    'right_elbow',
    'left_thigh',
    'left_knee',
    'left_ankle',
    'right_thigh',
    'right_knee',
    'right_ankle',
)
_MODEL = 'model.glb'
_MAXIMUM_MODEL_BYTES = 30 * 1024 * 1024


def _skin_joints(data: bytes) -> set[str]:
    """Returns the names of the joints of every skin of a binary glTF 2.0 file.

    Raises:
        ArgumentError: If the bytes are no binary glTF 2.0 file or hold no skin.
    """
    # glTF: a binary file opens with `glTF`, the version, the length and a JSON chunk.
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


def set_model(identifier: str, data: bytes, joints: dict[str, str]) -> dict[str, Any]:
    """Gives a team a player model of its own and returns the team's new entry.

    Args:
        identifier: The team, the built-in AI included.
        data: The binary glTF 2.0 file, with a skinned mesh.
        joints: The bone of the model that each joint of GRF's skeleton drives, `body` always and
            any other joint of `JOINTS` or `player` that the rig has.

    Raises:
        ArgumentError: If the file is too large, is no rigged binary glTF file, or the mapping
            lacks `body` or names a bone outside the model's skins.
    """
    current = entry(identifier)
    if len(data) > _MAXIMUM_MODEL_BYTES:
        raise ArgumentError(f'expected a model of at most 30 MB, found {len(data)} bytes')
    bones = _skin_joints(data)
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
    folder = _meta_path(identifier).parent
    _replace(folder / _MODEL, data)
    current['model'] = {
        'joints': {key: str(value) for key, value in joints.items()},
        'bytes': len(data),
    }
    current['style']['model'] = 'custom'
    _write(identifier, current)
    return current


def remove_model(identifier: str) -> bool:
    """Takes a team's own player model away and returns whether it had one."""
    current = entry(identifier)
    path = _meta_path(identifier).parent / _MODEL
    if not path.is_file():
        return False
    path.unlink()
    had = True
    current['model'] = None
    if current['style']['model'] == 'custom':
        current['style']['model'] = 'grf'
    _write(identifier, current)
    return had


def model_path(identifier: str) -> Path | None:
    """Returns the file of a team's player model, or `None`."""
    if identifier != BUILTIN and not _home.valid_identifier(identifier):
        return None
    path = _meta_path(identifier).parent / _MODEL
    return path if path.is_file() else None


def _replace(path: Path, data: bytes) -> None:
    """Writes `path` through a temporary file of its own, so no reader sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix='.part')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _write(identifier: str, current: dict[str, Any]) -> None:
    """Writes a team's entry, without the fields the folder implies."""
    stored = {key: value for key, value in current.items() if key not in ('images', 'sounds')}
    _replace(_meta_path(identifier), json.dumps(stored).encode())


# Kits whose shirts lie closer than this in RGB look alike on the pitch.
_CLASH = 120


def scenario_size(scenario: str) -> int | None:
    """Returns the players a side of a full-match scenario, 5 or 11, or `None` for the others."""
    for size in (5, 11):
        if scenario.startswith(f'{size}_vs_{size}'):
            return size
    return None


def check_ability(ability: Any) -> tuple[float | None, float | None]:
    """Returns a pair of abilities, each a number from 0 to 1 or `None` for the scenario's.

    Raises:
        ArgumentError: If `ability` is no pair, or a value is no number from 0 to 1.
    """
    if not isinstance(ability, (list, tuple)) or len(ability) != 2:
        raise ArgumentError(f'expected a pair of abilities, left and right, found {ability!r}')
    pair = []
    for value in ability:
        if value is None:
            pair.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ArgumentError(f'expected an ability from 0 to 1, found {value!r}')
        pair.append(float(value))
    return (pair[0], pair[1])


def check_scenario(teams_: list[dict[str, Any]], scenario: str) -> None:
    """Checks the format, count, selected players and goalkeeper on each scenario side."""
    size = scenario_size(scenario.removeprefix('benchmark_full_game_'))
    for side, team in enumerate(teams_):
        if team['kind'] == 'builtin':
            continue
        if size is not None and size != FORMATS[team['format']]:
            raise ArgumentError(
                f'expected a scenario of {team["format"]} for {team["name"]}, found {scenario}'
            )
        control = check_control(team.get('control'), team['format'])
        try:
            selected_players(control, scenario_players(scenario, side))
        except ArgumentError as error:
            raise ArgumentError(f'{team["name"]} cannot play {scenario}, {error}') from error


def match_teams(left: str, right: str) -> list[dict[str, Any]]:
    """Returns the two teams of a match as a recording's header holds them.

    The right team plays in its away kit, its shirt and shorts colors swapped with its trim, when
    its shirt would look like the left team's, as a team of the same registry entry always does.
    """
    teams = [header_team(left), header_team(right)]
    first, second = teams[0]['style']['kit'], teams[1]['style']['kit']
    distance = sum((a - b) ** 2 for a, b in zip(first['primary'], second['primary'], strict=True))
    if distance < _CLASH**2:
        kit = dict(second)
        kit['primary'], kit['secondary'] = second['secondary'], second['primary']
        kit['shorts'], kit['socks'] = second['socks'], second['shorts']
        if sum((a - b) ** 2 for a, b in zip(first['primary'], kit['primary'], strict=True)) < (
            _CLASH**2
        ):
            kit['primary'] = [255 - value for value in first['primary']]
        teams[1] = {
            **teams[1],
            'color': kit['primary'],
            'secondary': kit['secondary'],
            'style': {**teams[1]['style'], 'kit': kit},
            'away': True,
        }
    return teams


def header_team(identifier: str) -> dict[str, Any]:
    """Returns a team as a recording's header holds it, with its identifier, name and style."""
    team = entry(identifier)
    kit = team['style']['kit']
    return {
        'id': team['id'],
        'name': team['name'],
        'format': team['format'],
        'control': team.get('control', dict(DEFAULT_CONTROL)),
        'color': kit['primary'],
        'secondary': kit['secondary'],
        'style': team['style'],
    }


def folder_bundle(files: list[tuple[str, bytes]]) -> bytes:
    """Returns the bytes of a submission folder as the page uploads it.

    Args:
        files: The path of each file relative to the folder, with `/` between parts, and its bytes.
    """
    index = json.dumps([{'path': path, 'size': len(data)} for path, data in files]).encode()
    return (
        FOLDER_MAGIC + struct.pack('<I', len(index)) + index + b''.join(data for _, data in files)
    )


def _unbundle(data: bytes, target: Path) -> None:
    """Writes the files of a folder bundle into `target`, without the folder's own name.

    A browser names each file of a chosen folder with the folder in front, such as
    `submission/main.py`, so a single top folder above `main.py` is dropped.
    """
    if len(data) < len(FOLDER_MAGIC) + 4:
        raise ArgumentError('expected the files of a submission folder, found too few bytes')
    (length,) = struct.unpack_from('<I', data, len(FOLDER_MAGIC))
    start = len(FOLDER_MAGIC) + 4
    try:
        index = json.loads(data[start : start + length])
    except ValueError as error:
        raise ArgumentError(f'expected the index of a submission folder, found {error}') from error
    if not isinstance(index, list) or len(index) > _MAXIMUM_FILES:
        raise ArgumentError('expected the index of a submission folder, a list of its files')
    paths = [_check_member(str(entry['path'])) for entry in index]
    tops = {path.parts[0] for path in paths if len(path.parts) > 1}
    if (
        PurePosixPath(_MAIN) not in paths
        and len(tops) == 1
        and all(len(p.parts) > 1 for p in paths)
    ):
        paths = [PurePosixPath(*path.parts[1:]) for path in paths]
    position = start + length
    for path, entry in zip(paths, index, strict=True):
        size = int(entry['size'])
        if size < 0 or position + size > len(data):
            raise ArgumentError('expected the files that the index of the folder lists')
        destination = _inside(target, path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data[position : position + size])
        position += size


def _extract(data: bytes, filename: str, target: Path) -> None:
    """Writes a submission into `target`, a folder bundle, a Python file or an archive."""
    lower = filename.lower()
    if data.startswith(FOLDER_MAGIC):
        _unbundle(data, target)
    elif lower.endswith('.py'):
        (target / _MAIN).write_bytes(data)
    elif lower.endswith(('.tar.gz', '.tgz', '.tar')):
        try:
            archive = tarfile.open(fileobj=io.BytesIO(data))
        except (tarfile.TarError, EOFError, OSError) as error:
            raise ArgumentError(f'expected a readable .tar.gz archive, found {error}') from error
        with archive:
            members = archive.getmembers()
            _check_unpacked(len(members), sum(member.size for member in members))
            for member in members:
                _inside(target, _check_member(member.name))
                if not (member.isfile() or member.isdir()):
                    raise ArgumentError(f'expected plain files in the archive, found {member.name}')
            # Python 3.12 brought the data filter, and the checks above cover older versions.
            if hasattr(tarfile, 'data_filter'):
                archive.extractall(target, filter='data')
            else:
                archive.extractall(target)
    elif lower.endswith('.zip'):
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, OSError) as error:
            raise ArgumentError(f'expected a readable .zip archive, found {error}') from error
        with archive:
            members = archive.infolist()
            _check_unpacked(len(members), sum(member.file_size for member in members))
            for member in members:
                _inside(target, _check_member(member.filename))
            archive.extractall(target)
    else:
        raise ArgumentError(
            f'expected a submission folder, a .py file or a .tar.gz or .zip archive, found '
            f'{filename!r}'
        )
    if not (target / _MAIN).is_file():
        raise ArgumentError(f'expected {_MAIN} at the top of the submission, as Kaggle requires')


def add_folder(
    folder: str | Path, name: str | None = None, format: str = '11v11'
) -> dict[str, Any]:
    """Stores a submission folder or archive as a new team and returns its entry.

    Args:
        folder: A submission folder, submission file or exported archive.
        name: The name of the team, by default the folder's name.
        format: `5v5` or `11v11`, the size of the team for competition.
    """
    folder = Path(folder)
    if folder.is_file():
        return add(name or '', folder.name, folder.read_bytes(), format)
    if not (folder / _MAIN).is_file():
        raise ArgumentError(f'expected {_MAIN} in the folder {str(folder)!r}, as Kaggle requires')
    files = [
        (path.relative_to(folder).as_posix(), path.read_bytes())
        for path in sorted(folder.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts
    ]
    return add(name or folder.resolve().name, folder.name, folder_bundle(files), format)


def _check_member(name: str) -> PurePosixPath:
    """Returns an archive member's path, refusing one that could land outside the team's folder.

    Windows reads `D:name` as a path on another drive and a backslash as a separator, so a part
    may hold neither a colon, a backslash nor a null, whatever system unpacks it.
    """
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or any(part in ('', '.', '..') or any(c in part for c in ':\\\0') for part in path.parts)
    ):
        raise ArgumentError(f'expected archive members inside the archive, found {name!r}')
    return path


def _inside(target: Path, path: PurePosixPath) -> Path:
    """Returns `target / path`, refusing it unless it resolves inside `target`."""
    destination = target.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(target.resolve()):
        raise ArgumentError(f'expected archive members inside the archive, found {str(path)!r}')
    return destination


def _check_unpacked(count: int, size: int) -> None:
    """Refuses an archive of too many files or too large once unpacked."""
    if count > _MAXIMUM_FILES or size > _MAXIMUM_UNPACKED:
        raise ArgumentError('expected an archive of at most 10,000 files and 500 MB unpacked')


def add(
    name: str,
    filename: str,
    data: bytes,
    format: str = '11v11',
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stores a Kaggle submission as a new team and returns its entry.

    Args:
        name: The name that the page and recordings show.
        filename: The name of the uploaded file, whose extension says how to read it.
        data: The file's bytes.
        format: `5v5` or `11v11`, the size of the team for competition, which decides the
            scenarios it plays and the length of its roster.
        control: The players that the submission controls, as `check_control` takes them, or
            `None` for the active player alone, or for the setting of an exported folder.

    Raises:
        ArgumentError: If the submission is too large, of another kind, holds no `main.py` or
            the control does not fit the format.
    """
    if len(data) > _MAXIMUM_BYTES:
        raise ArgumentError(f'expected a submission of at most 100 MB, found {len(data)} bytes')
    _choice(format, tuple(FORMATS), 'the format')
    if control is not None:
        check_control(control, format)
    name_given = bool(name and name.strip())
    name = _printable((name or '').strip() or Path(filename).stem.strip() or 'team').strip()[:80]
    identifier = _home.new_identifier('t')
    folder = _home.folder('teams') / identifier
    folder.mkdir()
    try:
        _extract(data, filename, folder)
    except OSError as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise ArgumentError(f'expected a submission whose files unpack, found {error}') from error
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    entry = {
        'id': identifier,
        'name': name,
        'kind': 'kaggle',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'format': format,
        'style': default_style(format),
        'model': None,
    }
    # A folder exported from the page brings its format, style, images and model along, each
    # checked as an upload is, and whatever fails a check is left out.
    _import(folder / TEAM_FOLDER, entry, name_given)
    if control is not None:
        entry['control'] = check_control(control, entry['format'])
    completed = _complete(entry)
    _write(identifier, completed)
    return completed


def _import(team_folder: Path, entry: dict[str, Any], name_given: bool) -> None:
    """Takes the format, style, images and model of an exported team into `entry`."""
    if not team_folder.is_dir():
        return
    saved: Any = {}
    with contextlib.suppress(OSError, ValueError):
        saved = json.loads((team_folder / 'team.json').read_text())
    if not isinstance(saved, dict):
        saved = {}
    if saved.get('format') in FORMATS:
        entry['format'] = saved['format']
    with contextlib.suppress(ArgumentError):
        entry['control'] = check_control(saved.get('control'), entry['format'])
    with contextlib.suppress(ArgumentError, TypeError, KeyError, AttributeError):
        base = default_style(entry['format'])
        entry['style'] = _resize(
            style(_known(saved.get('style') or {}, base), base, _model_names()), entry['format']
        )
        if isinstance((saved.get('style') or {}).get('players'), list):
            entry['style']['players'] = _players(
                saved['style']['players'][: FORMATS[entry['format']]],
                entry['style']['players'],
                _model_names(),
            )
    if not name_given and isinstance(saved.get('name'), str) and saved['name'].strip():
        entry['name'] = saved['name'].strip()[:80]
    for path in list(team_folder.iterdir()):
        slot, suffix = path.stem, path.suffix
        if path.name in ('team.json', _MODEL):
            continue
        try:
            data = path.read_bytes()
            if slot.startswith('sound-'):
                valid = (
                    slot[len('sound-') :] in SOUND_SLOTS
                    and len(data) <= _MAXIMUM_SOUND_BYTES
                    and _sound_suffix(data) == suffix
                )
            else:
                valid = (
                    slot in IMAGE_SLOTS
                    and len(data) <= _MAXIMUM_IMAGE_BYTES
                    and _image_suffix(slot, data) == suffix
                )
        except (OSError, ArgumentError):
            valid = False
        if not valid:
            path.unlink() if path.is_file() else shutil.rmtree(path, ignore_errors=True)
    model = team_folder / _MODEL
    joints = (
        (saved.get('model') or {}).get('joints') if isinstance(saved.get('model'), dict) else None
    )
    try:
        data = model.read_bytes()
        bones = _skin_joints(data)
        if len(data) > _MAXIMUM_MODEL_BYTES or not isinstance(joints, dict) or 'body' not in joints:
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
        model.unlink(missing_ok=True)
        entry['model'] = None
        if entry['style'].get('model') == 'custom':
            entry['style']['model'] = 'grf'


def export(identifier: str) -> tuple[str, bytes]:
    """Returns a team as a zip of its submission folder, with its `cuju-team` folder inside.

    The archive holds `main.py` and the submission's files at its top, as Kaggle reads them, and
    `cuju-team` with the team's format, style, images and model, which the page reads back when
    the folder is uploaded again. The built-in AI exports only its `cuju-team` folder.

    Returns:
        A file name for the archive and its bytes.
    """
    team = entry(identifier)
    root = _meta_path(identifier).parent.parent
    meta = _meta_path(identifier)
    stored = {key: value for key, value in team.items() if key not in ('images', 'sounds')}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f'{TEAM_FOLDER}/team.json', json.dumps(stored))
        for path in sorted(root.rglob('*')) if root.is_dir() else []:
            if (
                path.is_file()
                and path != meta
                and not path.name.endswith('.part')
                and '__pycache__' not in path.parts
            ):
                archive.write(path, path.relative_to(root).as_posix())
    safe = ''.join(character if character.isalnum() else '-' for character in team['name'])
    return f'{safe.strip("-") or "team"}.zip', buffer.getvalue()


def remove(identifier: str) -> bool:
    """Deletes a team's folder and returns whether it existed.

    The built-in AI stays, and removing it only resets its name, style, model and images.
    """
    if identifier == BUILTIN:
        shutil.rmtree(_meta_path(BUILTIN).parent.parent, ignore_errors=True)
        return True
    if not _home.valid_identifier(identifier):
        return False
    folder = _home.folder('teams') / identifier
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True


def entry(identifier: str) -> dict[str, Any]:
    """Returns the entry of a team, or raises `ArgumentError` for an unknown one."""
    for team in teams():
        if team['id'] == identifier:
            return team
    raise ArgumentError(f'expected a team of the registry, found {identifier!r}')


def load_agent(identifier: str) -> Any:
    """Loads a submission's agent function as Kaggle does, the last callable its `main.py` defines.

    The agent runs from its folder, which comes first on `sys.path`, as on Kaggle, so its own
    modules import and its working files resolve. The modules it imports from its folder leave
    `sys.modules` afterwards, so a second submission with modules of the same names gets its own.
    """
    folder = _home.folder('teams') / identifier
    main = folder / _MAIN
    if not _home.valid_identifier(identifier) or not main.is_file():
        raise ArgumentError(f'expected a Kaggle team of the registry, found {identifier!r}')
    module = types.ModuleType(f'cuju_team_{identifier.replace("-", "_")}')
    module.__file__ = str(main)
    with _from_folder(folder):
        before = set(sys.modules)
        # Kaggle: `get_last_callable` runs the file and takes the last callable it defined.
        exec(compile(main.read_text(), str(main), 'exec'), module.__dict__)
        resolved = folder.resolve()
        for name in set(sys.modules) - before:
            origin = getattr(sys.modules[name], '__file__', None)
            if origin and Path(origin).resolve().is_relative_to(resolved):
                del sys.modules[name]
    callables = [value for value in module.__dict__.values() if callable(value)]
    if not callables:
        raise ArgumentError(f'expected an agent function in the submission {identifier!r}')
    agent = callables[-1]

    @functools.wraps(agent)
    def run(*arguments: Any) -> Any:
        with _from_folder(folder):
            return agent(*arguments)

    return run


@contextlib.contextmanager
def _from_folder(folder: Path) -> Any:
    """Runs a block from a submission's folder, first on `sys.path`, as Kaggle runs an agent."""
    previous = os.getcwd()
    sys.path.insert(0, str(folder))
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(folder))
