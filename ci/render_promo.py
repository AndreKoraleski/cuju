# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Builds the promotional montage of the renderer from the storyboard in render_promo.json.

The montage is a fixed asset. This script rebuilds it only when run by hand, and CI never runs it.

It serves the web folder with render_mock_server.py and a folder of real recordings, opens the
page in headless Chrome with the capture parameter, which steps the page by one frame at a time,
captures each scene of the storyboard losslessly frame by frame at 60 frames per second, composes
the grid of matches from tile clips, joins the scenes with cuts and dissolves on the beats of the
music into a near lossless master in the work folder, and encodes docs/assets/cuju-montage.mp4,
the poster, the teaser and the smaller copy that the Home page shows from that master, with the
music. The option --tempo prints the beat that the storyboard's lengths follow.

Run it as python ci/render_promo.py --recordings DIR --music FILE, where DIR holds full matches
of 11_vs_11_easy_stochastic_l0_r0 saved by cuju.render.Recorder and FILE is the song. The option
--only captures some scenes again, and --assemble builds the outputs from scenes already captured.
"""

import argparse
import base64
import itertools
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORYBOARD = ROOT / 'ci' / 'render_promo.json'
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
FPS = 60
# The quality of the scenes and the master in the work folder, near lossless in full chroma, since
# lossless files of every scene outgrow an ordinary disk.
STORE_CRF = 4
# The length of the page's logo wipe in seconds, as WIPE in transition.js.
WIPE = 0.72
# The viewport in CSS pixels and its scale, which give frames of 1920 by 1080.
VIEW = (1280, 720, 1.5)


class CDP:
    """A connection to one page of Chrome over its debugging protocol, on the standard library."""

    def __init__(self, url):
        from urllib.parse import urlsplit

        address = urlsplit(url)
        self.socket = socket.create_connection((address.hostname, address.port), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall(
            (
                f'GET {address.path} HTTP/1.1\r\nHost: {address.netloc}\r\n'
                f'Upgrade: websocket\r\nConnection: Upgrade\r\n'
                f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
            ).encode()
        )
        header = b''
        while not header.endswith(b'\r\n\r\n'):
            header += self.socket.recv(1)
        if b' 101 ' not in header.split(b'\r\n')[0]:
            raise RuntimeError(header.decode(errors='replace'))
        self.sequence = 0
        self.errors = []

    def read(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise RuntimeError('the connection to Chrome closed')
            data += chunk
        return bytes(data)

    def call(self, method, params=None):
        self.sequence += 1
        data = json.dumps({'id': self.sequence, 'method': method, 'params': params or {}})
        data = data.encode()
        mask = os.urandom(4)
        size = len(data)
        if size < 126:
            header = bytes([129, 128 | size])
        elif size < 65536:
            header = bytes([129, 254]) + struct.pack('!H', size)
        else:
            header = bytes([129, 255]) + struct.pack('!Q', size)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        self.socket.sendall(header + mask + masked)
        while True:
            first, second = self.read(2)
            size = second & 127
            if size == 126:
                size = struct.unpack('!H', self.read(2))[0]
            elif size == 127:
                size = struct.unpack('!Q', self.read(8))[0]
            payload = self.read(size)
            if first & 15 != 1:
                continue
            message = json.loads(payload)
            if message.get('method') == 'Runtime.exceptionThrown':
                details = message['params']['exceptionDetails']
                self.errors.append(details.get('exception', {}).get('description', details))
            if message.get('id') == self.sequence:
                if 'error' in message:
                    raise RuntimeError(f'{method} failed, {message["error"]}')
                return message.get('result', {})

    def evaluate(self, expression):
        result = self.call(
            'Runtime.evaluate',
            {'expression': expression, 'returnByValue': True, 'awaitPromise': True},
        )
        if 'exceptionDetails' in result:
            details = result['exceptionDetails']
            raise RuntimeError(details.get('exception', {}).get('description', details))
        return result.get('result', {}).get('value')


def wait_json(url, tries=300):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.2)
    raise RuntimeError(f'no answer at {url}')


# The graphics of the montage and the helpers that open its scenes, in the page's own style.
HELPERS = (ROOT / 'ci' / 'render_promo.js').read_text(encoding='utf-8') + '\ntrue\n'


def chrome_command(port, profile, gpu):
    flags = [
        CHROME,
        '--headless=new',
        '--mute-audio',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-background-networking',
        '--disable-component-update',
        '--disable-sync',
        '--disable-default-apps',
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--autoplay-policy=user-gesture-required',
        '--proxy-server=http://127.0.0.1:9',
        '--proxy-bypass-list=127.0.0.1',
        f'--remote-debugging-port={port}',
        f'--user-data-dir={profile}',
        f'--window-size={VIEW[0]},{VIEW[1]}',
        '--hide-scrollbars',
    ]
    if gpu:
        flags += ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist']
    else:
        flags += ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    return [*flags, 'about:blank']


PRELOAD = """
localStorage.setItem('cuju.muted', 'true');
localStorage.setItem('cuju.music', 'false');
localStorage.setItem('cuju.volume', '0');
localStorage.setItem('cuju.quality', '"high"');
localStorage.setItem('cuju.theme', '"dark"');
localStorage.setItem('cuju.tvMode', 'true');
localStorage.setItem('cuju.idleReturn', 'false');
localStorage.setItem('cuju.pointerLock', 'false');
localStorage.setItem('cuju.camera', '"broadcast"');
// Controllers plugged into this machine stay out of the picture.
Object.defineProperty(navigator, 'getGamepads', { configurable: true, value: () => [] });
Element.prototype.requestPointerLock = function () {
    throw new Error('Pointer lock is off while capturing.');
};
"""


class Browser:
    """Headless Chrome with one page, the mock server behind it, and the capture of scenes."""

    def __init__(self, args, work):
        self.args = args
        self.work = work
        self.processes = []
        self.base = f'http://127.0.0.1:{args.port}/'
        self.client = None
        self.loads = 0

    def start(self):
        server = [
            sys.executable,
            str(ROOT / 'ci' / 'render_mock_server.py'),
            '--port',
            str(self.args.port),
            '--recordings',
            self.args.recordings,
        ]
        log = open(self.work / 'server.log', 'w')
        self.processes.append(subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT))
        wait_json(self.base + 'api/scenarios')
        for gpu in (not self.args.software, False):
            self.profile = tempfile.mkdtemp(prefix='cuju-promo-')
            chrome = subprocess.Popen(
                chrome_command(self.args.port + 1, self.profile, gpu),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(chrome)
            targets = wait_json(f'http://127.0.0.1:{self.args.port + 1}/json')
            target = next(item for item in targets if item['type'] == 'page')
            self.client = CDP(target['webSocketDebuggerUrl'])
            self.client.call('Runtime.enable')
            self.client.call('Page.enable')
            self.client.call('Page.addScriptToEvaluateOnNewDocument', {'source': PRELOAD})
            width, height, scale = VIEW
            self.client.call(
                'Emulation.setDeviceMetricsOverride',
                {'width': width, 'height': height, 'deviceScaleFactor': scale, 'mobile': False},
            )
            self.client.call(
                'Emulation.setEmulatedMedia',
                {'features': [{'name': 'prefers-color-scheme', 'value': 'dark'}]},
            )
            self.load()
            renderer = self.client.evaluate("""(() => {
                const gl = cuju.view.renderer.getContext();
                const info = gl.getExtension('WEBGL_debug_renderer_info');
                return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unknown';
            })()""")
            print(f'WebGL renderer, {renderer}', flush=True)
            if not gpu or 'swiftshader' not in renderer.lower():
                return
            print('The GPU path failed, falling back to software.', flush=True)
            self.close_chrome()

    def close_chrome(self):
        try:
            self.client.call('Browser.close')
        except (OSError, RuntimeError):
            pass
        chrome = self.processes.pop()
        try:
            chrome.wait(10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

    def stop(self):
        if self.client:
            self.close_chrome()
        for process in self.processes:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()

    def load(self, page='home'):
        # A new query each time, so the page loads afresh rather than moving to an anchor.
        # A busy machine sometimes stalls a load, which a second navigation clears.
        for _ in range(3):
            self.loads += 1
            url = f'{self.base}?capture={self.loads}#page={page}'
            self.client.call('Page.navigate', {'url': url})
            if self.started():
                break
        else:
            raise RuntimeError(f'the page did not start, {self.client.errors[:3]}')
        # The shipped assets load after the page starts.
        time.sleep(2.5)
        self.client.evaluate(HELPERS)

    def started(self):
        for _ in range(300):
            time.sleep(0.1)
            try:
                if self.client.evaluate('!!(window.cuju && cuju.menus && window.cujuCapture)'):
                    return True
            except RuntimeError:
                pass
        return False

    def step(self):
        self.client.evaluate(f'cujuCapture.step({1000 / FPS})')

    def frame(self):
        result = self.client.call(
            'Page.captureScreenshot',
            {'format': 'png', 'optimizeForSpeed': True},
        )
        return base64.b64decode(result['data'])

    def probe(self):
        titles = sorted(
            {
                m['title']
                for m in wait_json(self.base + 'api/matches')
                if not m['title'].startswith('Promo')
            }
        )
        for title in titles:
            self.load()
            self.client.evaluate(f'promo.open({json.dumps(title)}, 0).then(() => true)')
            events = self.client.evaluate('promo.events()')
            print(title, json.dumps(events), flush=True)

    def still(self, js, path):
        """Draws a graphic alone on a clear page and saves it as a picture with transparency."""
        self.load()
        c = self.client
        c.call(
            'Emulation.setDefaultBackgroundColorOverride',
            {'color': {'r': 0, 'g': 0, 'b': 0, 'a': 0}},
        )
        c.evaluate(
            "document.getElementById('app').style.visibility = 'hidden';"
            "document.documentElement.style.background = 'transparent';"
            "document.body.style.background = 'transparent';"
            f'{js}; for (const a of document.getAnimations()) '
            '{ try { a.finish(); } catch { continue; } } true'
        )
        time.sleep(0.5)
        data = c.call('Page.captureScreenshot', {'format': 'png'})['data']
        path.write_bytes(base64.b64decode(data))
        c.call('Emulation.setDefaultBackgroundColorOverride', {})

    def prepare(self, scene, steps=None):
        """Opens the scene and runs it to its first frame, returns the frames run before it."""
        self.load(scene.get('page', 'home'))
        c = self.client
        c.errors.clear()
        if 'field' in scene:
            field = scene['field']
            c.evaluate(f'promo.field({json.dumps(field.get("preset"))}, {json.dumps(field)})')
        if scene.get('setup'):
            c.evaluate(f'(async () => {{ {scene["setup"]} }})().then(() => true)')
        time.sleep(scene.get('settle', 0.5))
        c.evaluate('cujuCapture.start(); promo.go()')
        until = None if steps is not None else scene.get('until')
        if steps is None:
            steps = round(scene.get('pre', 0) * FPS)
        count = 0
        while count < steps or (until and not c.evaluate(until) and count < 40 * FPS):
            self.step()
            count += 1
        return count

    def scene(self, scene, path):
        """Captures one scene of the storyboard into a video file."""
        c = self.client
        count = self.prepare(scene)
        if scene.get('lead'):
            # The scene starts this long before the condition, or after it when negative, so it
            # runs again to that point.
            c.evaluate('cujuCapture.stop()')
            self.prepare(scene, max(0, count - round(scene['lead'] * FPS)))
        if scene.get('sting_out'):
            c.evaluate('promo.stingCovered(); promo.stingOut()')
        total = round(scene['seconds'] * FPS)
        label = json.dumps(scene.get('wipe_label', ''))
        events = sorted(scene.get('events', []), key=lambda e: e['at'])
        if scene.get('sting_in'):
            events.append({'at': scene['seconds'] - 0.45, 'js': 'promo.stingIn()'})
        encoder = subprocess.Popen(
            [
                'ffmpeg',
                '-v',
                'error',
                '-y',
                '-f',
                'image2pipe',
                '-framerate',
                str(FPS),
                '-c:v',
                'png',
                '-i',
                '-',
                '-c:v',
                'libx264',
                '-preset',
                'veryfast',
                '-crf',
                str(STORE_CRF),
                '-pix_fmt',
                'yuv444p',
                '-r',
                str(FPS),
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        try:
            for index in range(total):
                t = index / FPS
                while events and events[0]['at'] <= t + 1e-6:
                    c.evaluate(f'(async () => {{ {events.pop(0)["js"]} }})().then(() => true)')
                self.step()
                # The page's logo wipe, revealing the scene at its start or covering its end.
                if scene.get('wipe_in') and t < WIPE / 2:
                    c.evaluate(f'promo.wipe({0.5 + t / WIPE}, {label})')
                elif scene.get('wipe_in') and t < WIPE / 2 + 2 / FPS:
                    c.evaluate('promo.wipe(null)')
                if scene.get('wipe_out') and t >= scene['seconds'] - WIPE / 2:
                    p = (t - scene['seconds'] + WIPE / 2) / WIPE
                    c.evaluate(f'promo.wipe({p}, {json.dumps(scene.get("wipe_out_label", ""))})')
                encoder.stdin.write(self.frame())
        finally:
            encoder.stdin.close()
            encoder.wait()
        c.evaluate('cujuCapture.stop()')
        if scene.get('after'):
            c.evaluate(f'(async () => {{ {scene["after"]} }})().then(() => true)')
        if c.errors:
            print(f'  page errors in {scene["name"]}, {c.errors[:3]}', flush=True)


def ffmpeg(*arguments):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *arguments], check=True)


def peak(path, filters):
    out = subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-i',
            str(path),
            '-af',
            f'{filters},volumedetect',
            '-f',
            'null',
            '-',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    return float(re.search(r'max_volume: (-?[\d.]+) dB', out).group(1))


def tempo(music):
    """Returns the beat of a song in seconds and the time of its first beat, from its onsets."""
    import numpy as np

    rate, hop, size = 22050, 256, 1024
    data = subprocess.run(
        [
            'ffmpeg',
            '-v',
            'error',
            '-i',
            str(music),
            '-ac',
            '1',
            '-ar',
            str(rate),
            '-f',
            'f32le',
            '-',
        ],
        capture_output=True,
        check=True,
    ).stdout
    x = np.frombuffer(data, np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, size)[::hop] * np.hanning(size)
    flux = np.maximum(0, np.diff(np.log1p(np.abs(np.fft.rfft(frames, axis=1))), axis=0)).sum(1)
    flux = np.maximum(0, flux - np.convolve(flux, np.ones(16) / 16, 'same'))
    per = rate / hop
    best = (-1.0, 0.0, 0.0)
    # Every beat from 70 to 180 a minute and every phase, scored by the onsets on its grid.
    for beat in np.linspace(60 / 180, 60 / 70, 1200):
        for phase in np.linspace(0, beat, 48, endpoint=False):
            index = np.round(np.arange(phase, len(flux) / per, beat) * per).astype(int)
            score = flux[index[index < len(flux)]].mean()
            if score > best[0]:
                best = (score, beat, phase)
    return best[1], best[2]


def eased(keys, t):
    """The tiles per side at time t, eased between keyframes in powers of two."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, n0), (t1, n1) in itertools.pairwise(keys):
        if t <= t1:
            x = (t - t0) / (t1 - t0)
            x = x * x * (3 - 2 * x)
            return 2 ** (math.log2(n0) + (math.log2(n1) - math.log2(n0)) * x)
    return keys[-1][1]


def grid(scene, work):
    """Composes the grid of matches from the tile clips, more and smaller tiles as it runs."""
    import numpy as np
    from PIL import Image

    width, height = 1920, 1080
    clips = scene['clips']
    # One stream in full size for the first tile, the others smaller, each with its own start.
    variants = [(clips[0], 0.0, width, height)]
    variants += [(c, o, 960, 540) for c in clips[1:] for o in scene['offsets']]
    variants += [(clips[0], o, 960, 540) for o in scene['offsets'][1:]]
    readers = [
        subprocess.Popen(
            [
                'ffmpeg',
                '-v',
                'error',
                '-stream_loop',
                '-1',
                '-ss',
                f'{offset:.2f}',
                '-i',
                str(work / f'{clip}.mkv'),
                '-vf',
                f'scale={w}:{h},fps={FPS}',
                '-f',
                'rawvideo',
                '-pix_fmt',
                'rgb24',
                '-',
            ],
            stdout=subprocess.PIPE,
        )
        for clip, offset, w, h in variants
    ]
    total = round(scene['seconds'] * FPS)
    writer = subprocess.Popen(
        [
            'ffmpeg',
            '-v',
            'error',
            '-y',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgb24',
            '-s',
            f'{width}x{height}',
            '-framerate',
            str(FPS),
            '-i',
            '-',
            '-loop',
            '1',
            '-i',
            str(work / 'grid-line.png'),
            '-filter_complex',
            f'[1:v]format=rgba,fade=t=in:st={scene["line_at"]}:d=0.8:alpha=1[l];'
            '[0:v][l]overlay=0:0:shortest=1,format=yuv444p[v]',
            '-map',
            '[v]',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            str(STORE_CRF),
            '-r',
            str(FPS),
            str(work / f'{scene["name"]}.mkv'),
        ],
        stdin=subprocess.PIPE,
    )
    rng = np.random.default_rng(7)
    picks = rng.integers(1, len(variants), size=(512, 512))
    picks[0, 0] = 0
    gap = np.array([32, 33, 36], np.uint8)
    for index in range(total):
        frames = []
        for (_, _, w, h), reader in zip(variants, readers, strict=True):
            data = reader.stdout.read(w * h * 3)
            frames.append(np.frombuffer(data, np.uint8).reshape(h, w, 3))
        n = eased(scene['keys'], index / FPS)
        tw, th = max(2, round(width / n)), max(2, round(height / n))
        atlas = np.stack(
            [np.asarray(Image.fromarray(f).resize((tw, th), Image.BILINEAR)) for f in frames]
        )
        line = 2 if tw > 400 else (1 if tw > 6 else 0)
        if line:
            atlas[:, -line:, :] = gap
            atlas[:, :, -line:] = gap
        cols, rows = -(-width // tw), -(-height // th)
        mosaic = atlas[picks[:rows, :cols]].transpose(0, 2, 1, 3, 4)
        mosaic = mosaic.reshape(rows * th, cols * tw, 3)[:height, :width]
        writer.stdin.write(np.ascontiguousarray(mosaic).tobytes())
    writer.stdin.close()
    writer.wait()
    for reader in readers:
        reader.kill()
        reader.wait()


def assemble(board, work, music):
    scenes = [s for s in board['scenes'] if not s.get('tile')]
    for s in scenes:
        if s.get('kind') == 'grid':
            grid(s, work)
    # Scenes join with a cut or, where a scene asks for one, a dissolve of that many seconds.
    inputs, chain = [], []
    starts = {}
    t = 0.0
    last = '[s0]'
    for index, s in enumerate(scenes):
        inputs += ['-i', str(work / f'{s["name"]}.mkv')]
        chain.append(
            f'[{index}:v]trim=0:{s["seconds"]},setpts=PTS-STARTPTS,fps={FPS},'
            f'format=yuv444p,setsar=1,settb=AVTB[s{index}]'
        )
        starts[s['name']] = t
        if index:
            mix = scenes[index - 1].get('transition', 0)
            name = f'[m{index}]'
            if mix:
                chain.append(
                    f'{last}[s{index}]xfade=transition=fade:duration={mix}:offset={t:.4f}{name}'
                )
            else:
                chain.append(f'{last}[s{index}]concat=n=2:v=1:a=0{name}')
            last = name
        t += s['seconds'] - s.get('transition', 0)
    length = t + scenes[-1].get('transition', 0)
    # The master keeps the quality of the scenes, and every file that ships is encoded from it.
    master = work / 'master.mkv'
    ffmpeg(
        *inputs,
        '-filter_complex',
        ';'.join(chain),
        '-map',
        last,
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        str(STORE_CRF),
        '-pix_fmt',
        'yuv444p',
        str(master),
    )
    audio = work / 'music.wav'
    fade = board['music_fade']
    # The song loops, so after one pass it runs on into its own opening, joined on the beat.
    loop = board['music_loop']
    splice = (
        f'[0:a]atrim=0:{loop + 0.015},asetpts=PTS-STARTPTS[a];'
        f'[0:a]atrim=0:{max(0.1, length - loop + 0.1)},asetpts=PTS-STARTPTS[b];'
        f'[a][b]acrossfade=d=0.03:c1=tri:c2=tri,atrim=0:{length},'
        f'afade=t=in:d=0.05,afade=t=out:st={length - fade}:d={fade}'
    )
    ffmpeg(
        '-i',
        str(music),
        '-filter_complex',
        splice + '[m]',
        '-map',
        '[m]',
        '-ar',
        '48000',
        '-ac',
        '2',
        str(work / 'music-raw.wav'),
    )
    gain = board['music_peak'] - peak(work / 'music-raw.wav', 'anull')
    ffmpeg('-i', str(work / 'music-raw.wav'), '-af', f'volume={gain:.2f}dB', str(audio))
    out = ROOT / 'docs' / 'assets' / 'cuju-montage.mp4'
    web = ROOT / 'python' / 'cuju' / 'render' / 'web' / 'assets'
    for target, size, rate, video, sound in (
        (out, '1920:1080', FPS, board['video_kbps'], 192),
        (web / 'promo.mp4', '1280:720', board['web_fps'], board['web_kbps'], 128),
    ):
        common = [
            '-vf',
            f'scale={size}:flags=lanczos:out_color_matrix=bt709:out_range=tv,fps={rate},'
            f'fade=t=out:st={length - 0.8}:d=0.8,format=yuv420p',
            '-c:v',
            'libx264',
            '-preset',
            'slower',
            '-b:v',
            f'{video}k',
            '-maxrate',
            f'{video * 2}k',
            '-bufsize',
            f'{video * 4}k',
            '-profile:v',
            'high',
            '-colorspace',
            'bt709',
            '-color_primaries',
            'bt709',
            '-color_trc',
            'bt709',
            '-color_range',
            'tv',
        ]
        log = str(work / f'pass-{target.stem}')
        ffmpeg(
            '-i',
            str(master),
            *common,
            '-pass',
            '1',
            '-passlogfile',
            log,
            '-an',
            '-f',
            'mp4',
            os.devnull,
        )
        ffmpeg(
            '-i',
            str(master),
            '-i',
            str(audio),
            '-map',
            '0:v',
            '-map',
            '1:a',
            *common,
            '-pass',
            '2',
            '-passlogfile',
            log,
            '-c:a',
            'aac',
            '-b:a',
            f'{sound}k',
            '-movflags',
            '+faststart',
            '-shortest',
            str(target),
        )
    poster = board['poster']
    at = starts[poster['scene']] + poster['at']
    ffmpeg(
        '-ss',
        f'{at:.3f}',
        '-i',
        str(master),
        '-frames:v',
        '1',
        '-q:v',
        '2',
        str(ROOT / 'docs' / 'assets' / 'cuju-montage-poster.jpg'),
    )
    shutil.copyfile(ROOT / 'docs' / 'assets' / 'cuju-montage-poster.jpg', web / 'promo-poster.jpg')
    teaser = board['teaser']
    parts = []
    for index, (name, offset, seconds) in enumerate(teaser['shots']):
        begin = starts[name] + offset
        parts.append(
            f'[0:v]trim={begin:.3f}:{begin + seconds:.3f},setpts=PTS-STARTPTS,'
            f'fps={teaser["fps"]},scale={teaser["width"]}:-2:flags=lanczos[v{index}]'
        )
    joined = ''.join(f'[v{i}]' for i in range(len(parts)))
    graph = ';'.join(parts) + f';{joined}concat=n={len(parts)}:v=1:a=0[out]'
    ffmpeg(
        '-i',
        str(master),
        '-filter_complex',
        graph,
        '-map',
        '[out]',
        '-c:v',
        'libwebp',
        '-lossless',
        '0',
        '-q:v',
        str(teaser['quality']),
        '-compression_level',
        '6',
        '-loop',
        '0',
        '-an',
        str(ROOT / 'docs' / 'assets' / 'cuju-teaser.webp'),
    )
    for path in (out, web / 'promo.mp4', ROOT / 'docs' / 'assets' / 'cuju-teaser.webp'):
        print(f'{path.relative_to(ROOT)}, {path.stat().st_size / 1e6:.2f} MB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings', help='a folder of full matches as .cjr files')
    parser.add_argument('--music', help='the song, looped and faded to the length of the video')
    parser.add_argument('--work', default=str(Path(tempfile.gettempdir()) / 'cuju-promo'))
    parser.add_argument('--port', type=int, default=8790, help='the server, Chrome takes the next')
    parser.add_argument('--only', help='names of scenes to capture again, joined by commas')
    parser.add_argument('--assemble', action='store_true', help='build from captured scenes')
    parser.add_argument('--probe', action='store_true', help='list the goals and decisions')
    parser.add_argument('--software', action='store_true', help='draw without the GPU')
    parser.add_argument('--tempo', action='store_true', help='print the beat of the music')
    args = parser.parse_args()
    if args.tempo:
        beat, phase = tempo(args.music)
        print(f'{60 / beat:.2f} beats a minute, a beat of {beat:.5f} s, the first at {phase:.3f} s')
        return
    board = json.loads(STORYBOARD.read_text(encoding='utf-8'))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if not args.assemble:
        if not args.recordings:
            parser.error('--recordings is required to capture')
        browser = Browser(args, work)
        try:
            browser.start()
            if args.probe:
                browser.probe()
                return
            only = set(args.only.split(',')) if args.only else None
            for scene in board['scenes']:
                if only and scene['name'] not in only:
                    continue
                started = time.time()
                if scene.get('kind') == 'grid':
                    browser.still(scene['line'], work / 'grid-line.png')
                else:
                    browser.scene(scene, work / f'{scene["name"]}.mkv')
                print(f'{scene["name"]}, {time.time() - started:.0f} s', flush=True)
        finally:
            browser.stop()
    if args.music and (args.assemble or not args.only):
        assemble(board, work, args.music)


if __name__ == '__main__':
    main()
