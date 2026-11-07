# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds the renderer's sound files from CC0 and public domain sources and lists them for the page.

`RECORDINGS` names the cut of a downloaded recording behind each other file, with its
source and license, and `sounds.json` records the source and license of every file beside the list
of files. A sound that no file supplies stays synthesized on the page.

The recordings are read from a folder that holds each download in a subfolder of its own, under the
names that `RECORDINGS` gives, as the downloads were made for the page. The Freesound sounds are
the high quality Ogg previews that Freesound's API serves. Each cut is decoded with
ffmpeg, faded in and out, or crossfaded end to start for a loop, brought to a peak of half full
scale, since the page levels every file again on loading, and written as Ogg Vorbis.

    python ci/render_sounds.py --recordings DOWNLOADS
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent.parent / 'python' / 'cuju' / 'render' / 'web' / 'assets'
_OUT = _OUT / 'sounds'
_RATE = 44100


_COMMONS = 'https://commons.wikimedia.org/wiki/File:'
_FREESOUND = 'https://freesound.org/s/'
_SARDIN = 'by Joseph Sardin, BigSoundBank, https://bigsoundbank.com/'
_SCIFI = 'Sci-Fi Sounds 1.0 by Kenney, www.kenney.nl, CC0 1.0'


@dataclass(frozen=True)
class Cut:
    """A stretch of one or more sources, played one after the other with crossfades."""

    sources: tuple[str, ...]
    start: float
    length: float
    loop: bool
    credit: str


RECORDINGS = {
    'crowd-loop': Cut(
        ('fs-ambience/829454.ogg',),
        2.0,
        42.0,
        True,
        '"CRWDReac-SOCCER_Millerntor Stadium Crowd Reaction General Ambience 01" '
        'by itmightgetloud, '
        f'{_FREESOUND}829454/, CC0 1.0, from 2 seconds for 42 seconds',
    ),
    'crowd-cheer': Cut(
        ('fs-goal/829455.ogg',),
        0.0,
        13.8,
        False,
        '"CRWDReac-SOCCER_Millerntor Stadium Crowd Reaction Goal 01" by itmightgetloud, '
        f'{_FREESOUND}829455/, CC0 1.0, the first goal, the first 13.8 seconds',
    ),
    'crowd-ooh': Cut(
        ('fs-miss/829452.ogg',),
        45.4,
        4.2,
        False,
        '"CRWDReac-SOCCER_Millerntor Stadium Crowd Reaction Chance Missed 01" by itmightgetloud, '
        f'{_FREESOUND}829452/, CC0 1.0, from 45.4 seconds for 4.2 seconds',
    ),
    'applause': Cut(
        ('applause/applause.ogg',),
        1.5,
        22.0,
        False,
        f'"Applause, 600 People" {_SARDIN} sound 0021, CC0 1.0, from 1.5 seconds for 22 seconds',
    ),
    'whistle-short': Cut(
        ('whistle/whistle.ogg',),
        5.55,
        0.6,
        False,
        f'"Whistle, plastic #2" {_SARDIN} sound 1105, CC0 1.0, the blast at 5.6 seconds',
    ),
    'whistle-long': Cut(
        ('whistle/whistle.ogg',),
        18.32,
        1.3,
        False,
        f'"Whistle, plastic #2" {_SARDIN} sound 1105, CC0 1.0, the blast at 18.4 seconds',
    ),
    **{
        f'beach-wave-{n}': Cut(
            (f'waves/wave_0{n}.flac',),
            0.0,
            5.0,
            False,
            f'wave_0{n} of "Beach Ocean Waves" by qubodup, '
            'https://opengameart.org/content/beach-ocean-waves, CC0 1.0, '
            'from Freesound 18363 by jasinski',
        )
        for n in (1, 2, 3, 4)
    },
    'gulls': Cut(
        ('gull/gull.ogg',),
        0.0,
        4.6,
        False,
        '"Western Gull, Golden Gate National Recreation Area" by the National Park Service, '
        f'{_COMMONS}Western_Gull_Golden_Gate_National_Recreation_Area.ogg, public domain',
    ),
    'city-loop': Cut(
        ('traffic/traffic.ogg',),
        0.0,
        58.0,
        True,
        '"High traffic road sounds" by IgnasD, '
        'https://opengameart.org/content/high-traffic-road-sounds, '
        'CC0 1.0',
    ),
    'city-siren': Cut(
        ('siren/siren.ogg',),
        19.5,
        6.3,
        False,
        f'"American police siren i" by lezer, {_COMMONS}American_police_siren_i.ogg, '
        'public domain, '
        'from 19.5 seconds for 6.3 seconds',
    ),
    'city-horn': Cut(
        ('horn/horn.wav',),
        0.0,
        3.5,
        False,
        f'"Car Horn" by 15HPanska_Ruttner_Jan, {_COMMONS}Car_Horn.wav, CC0 1.0',
    ),
    'space-radio': Cut(
        ('chorus/chorus.mp3',),
        9.0,
        6.0,
        False,
        '"Chorus Radio Waves within Earth\'s Atmosphere" by NASA, '
        f"{_COMMONS}Chorus_Radio_Waves_within_Earth's_Atmosphere.mp3, public domain, "
        'from 9 seconds for 6 seconds',
    ),
    'space-beep': Cut(
        ('sputnik/sputnik.mp3',),
        0.0,
        2.6,
        False,
        f'"Sputnik - Beep" by NASA, {_COMMONS}Sputnik_-_Beep.mp3, public domain, '
        'the first 2.6 seconds',
    ),
    'space-hum': Cut(
        (
            'kenney-sci-fi-sounds/spaceEngineLow_000.ogg',
            'kenney-sci-fi-sounds/spaceEngineLow_002.ogg',
            'kenney-sci-fi-sounds/spaceEngineLow_003.ogg',
        ),
        0.0,
        10.5,
        True,
        f'spaceEngineLow_000, 002 and 003 joined, {_SCIFI}',
    ),
    'neon-sting': Cut(
        ('kenney-digital-audio/powerUp1.ogg',),
        0.0,
        1.2,
        False,
        'powerUp1, Digital Audio by Kenney, www.kenney.nl, CC0 1.0',
    ),
    'rain-loop': Cut(
        ('rain/rain.mp3',),
        0.0,
        60.0,
        True,
        '"Rain in the Gutter Loop" by Ogrebane, '
        'https://opengameart.org/content/rain-gutter-loop, CC0 1.0',
    ),
    'wind-loop': Cut(
        ('wind/wind.ogg',),
        0.0,
        66.0,
        True,
        '"Killiney Hill Storm Floris 20250804 1537" by Karlunun, '
        f'{_COMMONS}Killiney_Hill_Storm_Floris_20250804_1537.ogg, CC0 1.0',
    ),
}
# The members of the Kenney packs that `RECORDINGS` reads, unpacked into the folders it names.
KENNEY_PACKS = {
    'kenney-sci-fi-sounds': (
        'sci-fi-sounds.zip',
        ['spaceEngineLow_000', 'spaceEngineLow_002', 'spaceEngineLow_003'],
    ),
    'kenney-digital-audio': ('digital-audio.zip', ['powerUp1']),
}
_FADE = 0.03
_CROSSFADE = 1.5


def _manifest(out: Path) -> dict:
    """Returns the current list of files and their credits, or an empty one."""
    path = out / 'sounds.json'
    if path.is_file():
        document = json.loads(path.read_text(encoding='utf-8'))
        document.setdefault('credits', {})
        return document
    return {'files': [], 'credits': {}}


def _record(document: dict, file: str, credit: str) -> None:
    """Lists `file` for the page and records where it comes from."""
    document['files'] = sorted({*document['files'], file})
    document['credits'][file] = credit


def _decode(path: Path) -> np.ndarray:
    """Returns a sound as mono floats at the page's rate."""
    raw = subprocess.run(
        [
            'ffmpeg',
            '-v',
            'error',
            '-i',
            str(path),
            '-f',
            'f32le',
            '-ac',
            '1',
            '-ar',
            str(_RATE),
            '-',
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).copy()


def _join(parts: list[np.ndarray], overlap: int) -> np.ndarray:
    """Joins sounds one after the other, each fading into the next over `overlap` samples."""
    out = parts[0]
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    for part in parts[1:]:
        mixed = out[-overlap:] * ramp[::-1] + part[:overlap] * ramp
        out = np.concatenate([out[:-overlap], mixed, part[overlap:]])
    return out


def _cut(folder: Path, cut: Cut) -> np.ndarray:
    """Returns the samples of one cut, faded, or crossfaded end to start for a loop."""
    overlap = int(_CROSSFADE * _RATE)
    whole = _join([_decode(folder / source) for source in cut.sources], overlap)
    start = int(cut.start * _RATE)
    length = int(cut.length * _RATE)
    # A recording that sits off zero is centered, so its fades and loops start and end at rest.
    whole = whole - whole.mean()
    if cut.loop:
        # The stretch after the loop's end fades into its start, so the last sample leads into the
        # first without a seam.
        data = whole[start : start + length + overlap]
        if len(data) < length + overlap:
            raise SystemExit(f'{cut.sources[0]} is too short for a loop of {cut.length} seconds')
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        head = data[:overlap] * ramp + data[length : length + overlap] * ramp[::-1]
        data = np.concatenate([head, data[overlap:length]])
    else:
        data = whole[start : start + length].copy()
        fade = min(len(data) // 2, int(_FADE * _RATE))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        data[:fade] *= ramp
        data[len(data) - fade :] *= ramp[::-1]
    peak = float(np.abs(data).max()) or 1.0
    return data * (0.5 / peak)


def _encode(data: np.ndarray, target: Path) -> None:
    """Writes mono floats as Ogg Vorbis."""
    subprocess.run(
        [
            *(
                'ffmpeg',
                '-v',
                'error',
                '-y',
                '-f',
                'f32le',
                '-ac',
                '1',
                '-ar',
                str(_RATE),
                '-i',
                '-',
            ),
            *('-c:a', 'libvorbis', '-q:a', '4', str(target)),
        ],
        input=data.astype(np.float32).tobytes(),
        check=True,
    )


def _unpack(folder: Path) -> None:
    """Unpacks the Kenney members that the cuts read beside their archives."""
    for sub, (archive, stems) in KENNEY_PACKS.items():
        with zipfile.ZipFile(folder / sub / archive) as pack:
            license_text = pack.read('License.txt').decode(errors='replace')
            if 'Creative Commons Zero, CC0' not in license_text:
                raise SystemExit(f'expected the CC0 license in {archive}')
            for name in pack.namelist():
                stem = Path(name).stem
                if name.endswith('.ogg') and stem in stems:
                    (folder / sub / f'{stem}.ogg').write_bytes(pack.read(name))


def main() -> None:
    """Copies the sounds of the ball and cuts the recordings that the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings', type=Path, help='the folder of the downloaded recordings')
    parser.add_argument('--out', type=Path, default=_OUT, help='the folder of the sounds')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    document = _manifest(options.out)
    if options.recordings:
        _unpack(options.recordings)
        for name, cut in RECORDINGS.items():
            file = f'{name}.ogg'
            _encode(_cut(options.recordings, cut), options.out / file)
            _record(document, file, cut.credit)
            print(file, (options.out / file).stat().st_size)
    text = json.dumps(document, indent=4, ensure_ascii=False) + '\n'
    (options.out / 'sounds.json').write_bytes(text.encode())


if __name__ == '__main__':
    main()
