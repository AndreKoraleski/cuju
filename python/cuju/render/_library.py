# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The library of saved matches, each a `.cjr` file with a small summary beside it.

The files lie side by side in one folder, and each summary names the folder of the page's explorer
that its match sits in, a path such as `League/Week 1`, so moving a match rewrites one small file.
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path
from typing import Any

from ..errors import ArgumentError
from . import _home, _recording

_SCORE = struct.Struct('<2i')
# The sample matches that ship with the package, which the library receives once each.
_SAMPLES = Path(__file__).parent / 'samples'
# The file of the home folder that names the samples the library has received.
_SEEDED = 'samples.json'


def save(head: bytes, frames: list[bytes], identifier: str | None = None) -> Path:
    """Writes a recording and its summary into the library and returns the recording's path.

    Args:
        head: The magic, length and header, as `_recording.header` returns them.
        frames: The frames.
        identifier: The name of the files, by default a new one.
    """
    identifier = identifier or _home.new_identifier('m')
    path = _home.folder('matches') / f'{identifier}.cjr'
    _recording.write(path, head, frames)
    document = json.loads(head[len(_recording.MAGIC) + 4 :])
    _write_summary(path, document, frames[-1] if frames else None, len(frames))
    return path


def _write_summary(
    path: Path, document: dict[str, Any], last: bytes | None, steps: int
) -> dict[str, Any]:
    """Writes the summary of the recording at `path` beside it and returns the summary."""
    summary = {
        'id': path.stem,
        'title': document['title'],
        'scenario': document['scenario'],
        'teams': [team['name'] for team in document['teams']],
        'score': list(_SCORE.unpack_from(last, 8)) if last else [0, 0],
        'steps': steps,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'bytes': path.stat().st_size,
        'folder': '',
    }
    path.with_suffix('.json').write_text(json.dumps(summary))
    return summary


def samples() -> list[str]:
    """Returns the names of the sample matches that ship with the package, such as `sample-5v5`."""
    return sorted(path.stem for path in _SAMPLES.glob('*.cjr'))


def _seeded() -> set[str]:
    """Returns the names of the samples that the library has received."""
    try:
        return set(json.loads((_home.home() / _SEEDED).read_text())['seeded'])
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def _add_sample(name: str) -> Path | None:
    """Copies a sample into the library and returns its path, or `None` when the name is taken.

    The library keeps the sample as it keeps any recording, and a recording or summary of the same
    name stays as it is.
    """
    path = _home.folder('matches') / f'{name}.cjr'
    if path.exists() or path.with_suffix('.json').exists():
        return None
    temporary = path.with_name(path.name + '.part')
    shutil.copyfile(_SAMPLES / f'{name}.cjr', temporary)
    temporary.replace(path)
    last = None
    for frame in _recording.frames(path):
        last = frame
    document = _recording.read_header(path)
    _write_summary(path, document, last, document['frame_count'])
    return path


def add_samples(names: list[str] | None = None, again: bool = True) -> list[Path]:
    """Copies sample matches into the library and returns the paths of those copied.

    Each sample is marked as received in the home folder, so a deleted sample stays deleted until
    it is requested again.

    Args:
        names: The samples, by default every one.
        again: Whether to copy samples that the library has received before, as a request does,
            or only new ones, as the first start of the server does.

    Raises:
        ArgumentError: If a name is not a shipped sample.
    """
    shipped = samples()
    names = shipped if names is None else names
    unknown = [name for name in names if name not in shipped]
    if unknown:
        raise ArgumentError(f'expected a sample from {shipped}, found {unknown}')
    seeded = _seeded()
    wanted = [name for name in names if again or name not in seeded]
    if not wanted:
        return []
    added = [path for path in map(_add_sample, wanted) if path is not None]
    marker = _home.home() / _SEEDED
    temporary = marker.with_name(marker.name + '.part')
    temporary.write_text(json.dumps({'seeded': sorted(seeded | set(wanted))}))
    temporary.replace(marker)
    return added


def matches() -> list[dict[str, Any]]:
    """Returns the summary of every recording, newest first."""
    entries = []
    for path in _home.folder('matches').glob('*.cjr'):
        summary = path.with_suffix('.json')
        try:
            entry = json.loads(summary.read_text())
        except (OSError, ValueError):
            entry = _summarize(path)
        entry.setdefault('folder', '')
        entries.append(entry)
    entries.sort(key=lambda entry: entry.get('created') or '', reverse=True)
    return entries


def _summarize(path: Path) -> dict[str, Any]:
    """Returns the summary of a recording that has none, such as one copied into the folder."""
    try:
        document = _recording.read_header(path)
        teams = [team['name'] for team in document['teams']]
        title, scenario = document['title'], document['scenario']
    except Exception:
        teams, title, scenario = ['', ''], path.stem, ''
    created = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(path.stat().st_mtime))
    return {
        'id': path.stem,
        'title': title,
        'scenario': scenario,
        'teams': teams,
        'score': None,
        'steps': None,
        'created': created,
        'bytes': path.stat().st_size,
    }


def _folder_path(folder: Any) -> str:
    """Returns a folder of the explorer in canonical form, or raises `ArgumentError`."""
    if not isinstance(folder, str):
        raise ArgumentError(f'expected a folder as text, found {folder!r}')
    parts = [part.strip() for part in folder.replace('\\', '/').split('/') if part.strip()]
    if any(part in ('.', '..') or len(part) > 64 for part in parts) or len(parts) > 8:
        raise ArgumentError(
            f'expected a folder of at most 8 names of 64 characters, found {folder!r}'
        )
    return '/'.join(parts)


def update(identifier: str, title: str | None = None, folder: str | None = None) -> dict[str, Any]:
    """Renames a recording or moves it to a folder of the explorer, and returns its summary.

    Args:
        identifier: The recording.
        title: The new title, or `None` to keep it.
        folder: The folder, such as `League/Week 1`, the empty text for the top, or `None`.

    Raises:
        ArgumentError: If the recording is unknown, the title empty or the folder invalid.
    """
    path = path_of(identifier)
    if path is None:
        raise ArgumentError(f'expected a recording of the library, found {identifier!r}')
    summary_path = path.with_suffix('.json')
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        summary = _summarize(path)
    if title is not None:
        title = str(title).strip()[:120]
        if not title:
            raise ArgumentError('expected a title with at least one character')
        summary['title'] = title
    if folder is not None:
        summary['folder'] = _folder_path(folder)
    summary.setdefault('folder', '')
    temporary = summary_path.with_name(summary_path.name + '.part')
    temporary.write_text(json.dumps(summary))
    temporary.replace(summary_path)
    return summary


def path_of(identifier: str) -> Path | None:
    """Returns the path of a recording, or `None` for an unknown or unsafe identifier."""
    if not _home.valid_identifier(identifier):
        return None
    path = _home.folder('matches') / f'{identifier}.cjr'
    return path if path.is_file() else None


def remove(identifier: str) -> bool:
    """Deletes a recording and its summary and returns whether it existed."""
    path = path_of(identifier)
    if path is None:
        return False
    path.unlink()
    path.with_suffix('.json').unlink(missing_ok=True)
    return True
