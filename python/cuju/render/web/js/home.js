// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The Home page, the first page of the viewer, laid out as a landing page. A wide opening band
// carries the logo as the name, a line on what the viewer does and the ways into its pages. Below
// it the page runs down in bands, the live sources and the newest recordings with a way into each,
// or a first run guide while the library is empty, then what the viewer does, the keys of the
// Watch page, a note on what cuju is with links to the project, and the version. Every text from
// the server goes into the page as text, never as markup.

import { api, formatDate } from './api.js';
import { el as h } from './menu.js';
import { scenarioName, titleText } from './scenarios.js';

const REFRESH_MS = 8000;
const LATEST = 5;
const SVG_NS = 'http://www.w3.org/2000/svg';

const LINKS = [
    ['Repository', 'The source code on GitHub', 'https://github.com/AndreKoraleski/cuju'],
    ['Documentation', 'Guides and the reference of the package', 'https://github.com/AndreKoraleski/cuju/tree/main/docs'],
    ['Rendering guide', 'This viewer, its server and its formats', 'https://github.com/AndreKoraleski/cuju/blob/main/docs/rendering.md'],
    ['Issues', 'Report a problem or ask for a feature', 'https://github.com/AndreKoraleski/cuju/issues'],
    ['Changelog', 'What changed in each release', 'https://github.com/AndreKoraleski/cuju/blob/main/CHANGELOG.md'],
    ['PyPI', 'The package on the Python Package Index', 'https://pypi.org/project/cuju/'],
];
const NOTICE_URL = 'https://github.com/AndreKoraleski/cuju/blob/main/NOTICE';

// What the viewer does, each with an icon drawn in lines. A part given as an array is code.
const FEATURES = [
    ['camera', 'Broadcast cameras', ['A director cuts between the broadcast, tactical and behind the goal cameras, or one camera follows a player, the ball or flies free.']],
    ['replay', 'Goal and decision replays', ['Every goal replays from several angles in slow motion, and fouls, cards and offsides get a replay of their own.']],
    ['shirt', 'Team customization', ['Each team wears its own kit and brings its own players, model, fans, celebrations, sounds and sponsors.']],
    ['live', 'Live from a running batch', ['A training script started with ', ['CUJU_RENDER=on'], ' streams its environments here while it learns.']],
    ['podium', 'A show at full time', ['Lineups before kick-off, then the facts, the highlights, the player of the match and a podium after the whistle.']],
    ['field', 'The field and the weather', ['Grass, lines, nets, boards, the crowd, the time of day and the weather, all set on the Field page.']],
];

// The icons, in a box of 24 by 24, as paths stroked in the color of the text.
const ICONS = {
    camera: ['M3 7.5h12.5v10H3z', 'M15.5 11l5.5-3.5v10l-5.5-3.5', 'M6 4.5h4'],
    replay: ['M6.3 6.3A8 8 0 1 1 4 12', 'M6.3 2.3v4h4', 'M10.2 9v6l5-3z'],
    shirt: ['M8.5 3.5L3 6.5l2 4 2.5-1.2V20.5h9V9.3l2.5 1.2 2-4-5.5-3c-.6 1.6-1.9 2.5-3.5 2.5s-2.9-.9-3.5-2.5z'],
    live: ['M8.5 8.5a5 5 0 0 0 0 7', 'M15.5 8.5a5 5 0 0 1 0 7', 'M5.6 5.6a9 9 0 0 0 0 12.8', 'M18.4 5.6a9 9 0 0 1 0 12.8', 'M12 10.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z'],
    podium: ['M3 20.5v-7h6v7', 'M9 20.5v-11h6v11', 'M15 20.5v-5h6v5', 'M2 20.5h20', 'M12 3.5l1 2 2 .3-1.5 1.4.4 2-1.9-1-1.9 1 .4-2L9 5.8l2-.3z'],
    field: ['M2.5 5.5h19v13h-19z', 'M12 5.5v13', 'M12 9.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5z', 'M2.5 9h3v6h-3', 'M21.5 9h-3v6h3'],
};

// Each key group lists keys, where an array of keys reads as one chord or one row of keys.
const KEYS = [
    ['Cameras', [
        [['V'], 'Broadcast, tactical, behind the goal, follow player and free in turn'],
        [['B'], 'Follow the ball'],
        [['P'], 'Follow the player on the ball'],
        [['1', 'to', '0'], 'Player cameras, with Shift the right team'],
        [['G'], 'The goalkeeper, with Shift the right one'],
        [['W', 'A', 'S', 'D'], 'Fly the free camera'],
        [['Q', 'E'], 'Lower and raise it'],
    ]],
    ['Playback', [
        [['Space'], 'Play or pause'],
        [['J', 'K', 'L'], 'Backwards, pause, forwards'],
        [['←', '→'], 'Step a frame or skip 5 seconds'],
        [['N'], 'Next goal'],
    ]],
    ['Replays', [
        [['R'], 'Replay the last goal'],
        [['Shift', 'R'], 'More angles of it'],
        [['Enter'], 'Skip a part'],
        [['Esc'], 'End the replays or the show'],
    ]],
    ['Screen', [
        [['T'], 'Broadcast mode, the picture alone as on television'],
        [['F'], 'Fullscreen'],
        [['M'], 'Mute'],
        [['H'], 'All shortcuts'],
    ]],
];

function createdMs(m) {
    if (typeof m.created === 'number') {
        return m.created < 1e12 ? m.created * 1000 : m.created;
    }
    return Date.parse(m.created) || 0;
}

function plural(n, one, many) {
    return `${n} ${n === 1 ? one : many}`;
}

// An element of an SVG drawing, with its attributes and children.
function svg(tag, attrs = {}, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
        node.setAttribute(k, String(v));
    }
    node.append(...children);
    return node;
}

function icon(name) {
    return svg('svg', { class: 'home-icon', viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
        ...(ICONS[name] || []).map((d) => svg('path', { d })));
}

// The markings of a football pitch of 105 by 68 meters in tenths of a meter, drawn behind the
// opening band as a quiet backdrop.
function pitchArt() {
    const line = (d) => svg('path', { d });
    const spot = (cx) => svg('circle', { class: 'spot', cx, cy: 340, r: 4 });
    const drawing = svg('svg', { class: 'home-pitch', viewBox: '-40 -20 1130 720', 'aria-hidden': 'true', focusable: 'false' },
        line('M0 0H1050V680H0Z'),
        line('M525 0V680'),
        svg('circle', { cx: 525, cy: 340, r: 91.5 }),
        line('M0 138.5H165V541.5H0'),
        line('M1050 138.5H885V541.5H1050'),
        line('M0 248.4H55V431.6H0'),
        line('M1050 248.4H995V431.6H1050'),
        line('M165 266.9A91.5 91.5 0 0 1 165 413.1'),
        line('M885 266.9A91.5 91.5 0 0 0 885 413.1'),
        line('M10 0A10 10 0 0 1 0 10'),
        line('M1040 0A10 10 0 0 0 1050 10'),
        line('M0 670A10 10 0 0 1 10 680'),
        line('M1050 670A10 10 0 0 0 1040 680'),
        line('M0 303.4H-20V376.6H0'),
        line('M1050 303.4H1070V376.6H1050'),
        spot(110), spot(525), spot(940),
        svg('circle', { class: 'ball', cx: 760, cy: 250, r: 7 }));
    return h('div', { class: 'home-pitch-frame', 'aria-hidden': 'true' }, drawing);
}

// The head of a band, a heading with an optional line above it and an optional control at its right.
function bandHead(id, title, { eyebrow, aside, level = 'h2' } = {}) {
    return h('div', { class: 'home-head' },
        h('div', { class: 'home-head-text' },
            eyebrow ? h('p', { class: 'home-eyebrow' }, eyebrow) : null,
            h(level, { id, class: 'home-h2' }, title)),
        aside || null);
}

// A command to copy, with a button that puts it on the clipboard.
function command(text) {
    const button = h('button', { type: 'button', class: 'btn small ghost home-copy', 'aria-label': `Copy ${text}` }, 'Copy');
    button.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(text);
            button.textContent = 'Copied';
        } catch {
            // Without the clipboard the text is selected, so a key press copies it.
            const range = document.createRange();
            range.selectNodeContents(code);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            button.textContent = 'Selected';
        }
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 1600);
    });
    const code = h('code', {}, text);
    return h('div', { class: 'home-command' }, code, button);
}

function keyChord(keys) {
    return keys.flatMap((k, i) => {
        const node = k === 'to' ? h('span', { class: 'home-key-word' }, 'to') : h('kbd', {}, k);
        return i ? [' ', node] : [node];
    });
}

function newTab(label) {
    return [label, h('span', { class: 'sr-only' }, ', opens in a new tab')];
}

export class HomePage {
    constructor(app) {
        this.app = app;
        this.root = document.getElementById('page-home');
        this.timer = null;
        this.token = 0;
        this.shown = false;
        this.signature = null;
        this.build();
        // A file chosen from this page plays on the Watch page.
        const input = document.getElementById('file-input');
        if (input) {
            input.addEventListener('change', () => {
                if (this.app.page === 'home' && this.pickedFile) {
                    this.app.showPage('watch');
                }
                this.pickedFile = false;
            });
        }
        document.addEventListener('visibilitychange', () => {
            if (this.shown && !document.hidden) {
                this.refresh();
            }
        });
        this.loadVersion();
    }

    show() {
        this.shown = true;
        clearInterval(this.timer);
        this.timer = setInterval(() => {
            if (!document.hidden) {
                this.refresh();
            }
        }, REFRESH_MS);
        this.refresh();
    }

    hide() {
        this.shown = false;
        clearInterval(this.timer);
        this.timer = null;
        // An answer that arrives after the page is left is dropped.
        this.token++;
    }

    build() {
        // The live sources and the recordings, or the guide or the offline note in their place.
        this.status = h('div', { class: 'home-status', 'aria-live': 'polite' },
            h('p', { class: 'muted home-checking' }, 'Asking the server for live sources and recordings.'));
        this.root.replaceChildren(h('div', { class: 'home' },
            this.hero(),
            h('div', { class: 'home-band home-now' }, h('div', { class: 'home-wrap' }, this.status)),
            this.reelBand(),
            this.featuresBand(),
            this.keysBand(),
            this.footBand()));
    }

    // The opening band, the logo as the name, the line beneath it and the ways into the pages.
    hero() {
        const go = (page) => () => this.app.showPage(page);
        this.glanceLive = h('span', { class: 'home-glance-item live' });
        this.glanceRecordings = h('button', { type: 'button', class: 'home-glance-item link', onclick: go('recordings') });
        // A glance at what the server holds, shown once both counts are known.
        this.glance = h('p', { class: 'home-glance', hidden: true }, this.glanceLive, this.glanceRecordings);
        return h('header', { class: 'home-hero' },
            pitchArt(),
            h('div', { class: 'home-wrap home-hero-inner' },
                h('p', { class: 'home-eyebrow' }, 'Match viewer'),
                h('h1', { class: 'home-title' },
                    h('img', { class: 'home-wordmark', src: 'assets/cuju-logo-384.png', alt: 'cuju', width: '384', height: '384' })),
                h('p', { class: 'home-tagline' }, 'Matchday for your agents.'),
                h('p', { class: 'home-lede' }, 'Watch matches, play a game or build a team.'),
                h('nav', { class: 'home-actions', 'aria-label': 'Pages of the viewer' },
                    h('button', { type: 'button', class: 'btn primary home-cta', onclick: go('watch') }, 'Watch'),
                    h('button', { type: 'button', class: 'btn', onclick: go('play') }, 'Play a game'),
                    h('button', { type: 'button', class: 'btn', onclick: go('matches') }, 'Request a match'),
                    h('button', { type: 'button', class: 'btn', onclick: go('teams') }, 'Teams'),
                    h('button', { type: 'button', class: 'btn', onclick: go('field') }, 'Field')),
                this.glance));
    }

    // A montage of the viewer, muted until the viewer turns the sound on, and played from its controls.
    reelBand() {
        const video = h('video', {
            class: 'home-reel-video', src: 'assets/promo.mp4', poster: 'assets/promo-poster.jpg',
            controls: true, muted: true, playsinline: true, preload: 'none', width: '1280', height: '720',
            'aria-label': 'A montage of the viewer, from a floodlit goal to the arenas, the team editor, a game, the show at full time and a grid of many matches',
        });
        video.muted = true;
        return h('section', { class: 'home-band home-reel', 'aria-labelledby': 'home-reel-title' },
            h('div', { class: 'home-wrap' },
                bandHead('home-reel-title', 'The viewer at a glance', { eyebrow: 'Montage' }),
                video));
    }

    featuresBand() {
        const items = FEATURES.map(([name, title, parts]) => h('li', { class: 'home-feature' },
            icon(name),
            h('h3', {}, title),
            h('p', {}, ...parts.map((p) => (Array.isArray(p) ? h('code', {}, p[0]) : p)))));
        return h('section', { class: 'home-band home-features', 'aria-labelledby': 'home-features-title' },
            h('div', { class: 'home-wrap' },
                bandHead('home-features-title', 'A broadcast of every match', { eyebrow: 'What it does' }),
                h('ul', { class: 'home-feature-list' }, ...items)));
    }

    keysBand() {
        const groups = KEYS.map(([name, rows]) => h('div', { class: 'home-keys-group' },
            h('h3', {}, name),
            h('dl', {}, ...rows.flatMap(([keys, text]) => [h('dt', {}, ...keyChord(keys)), h('dd', {}, text)]))));
        const all = h('button', { type: 'button', class: 'btn ghost', onclick: () => {
            const btn = document.getElementById('open-help');
            if (btn) {
                btn.click();
            }
        } }, 'All shortcuts ', h('kbd', { 'aria-hidden': 'true' }, 'H'));
        return h('section', { class: 'home-band home-keys', 'aria-labelledby': 'home-keys-title' },
            h('div', { class: 'home-wrap' },
                bandHead('home-keys-title', 'Keys', { eyebrow: 'On the Watch page', aside: all }),
                h('div', { class: 'home-keys-grid' }, ...groups)));
    }

    // The foot of the page, what cuju is with its credits, the links and the version.
    footBand() {
        this.version = h('span', { class: 'home-version', hidden: true });
        const links = LINKS.map(([label, hint, href]) => h('li', {},
            h('a', { href, target: '_blank', rel: 'noopener noreferrer', class: 'home-link' },
                h('span', { class: 'home-link-label' }, ...newTab(label)),
                h('span', { class: 'home-link-hint' }, hint))));
        const about = h('section', { class: 'home-about', 'aria-labelledby': 'home-about-title' },
            bandHead('home-about-title', 'About'),
            h('p', {}, 'cuju runs Google Research Football matches on the CPU and CUDA.'),
            h('p', {}, 'Watch saved and live matches without changing their results.'),
            h('p', {}, h('a', { href: 'https://en.wikipedia.org/wiki/Cuju', target: '_blank', rel: 'noopener noreferrer' },
                ...newTab('On the name cuju'))),
            h('p', { class: 'home-credits' }, 'The page ships three.js and Fabric.js, assets converted from the media of Google Research Football, the KayKit character packs by Kay Lousberg, sounds by Kenney and the montage’s music, Upbeat loop by mistermender. ',
                h('a', { href: NOTICE_URL, target: '_blank', rel: 'noopener noreferrer' }, ...newTab('NOTICE')),
                ' lists each one with its license.'));
        const linkList = h('section', { class: 'home-links', 'aria-labelledby': 'home-links-title' },
            bandHead('home-links-title', 'Links'),
            h('ul', { class: 'home-link-list' }, ...links));
        return h('footer', { class: 'home-band home-foot' },
            h('div', { class: 'home-wrap' },
                h('div', { class: 'home-foot-grid' }, about, linkList),
                h('div', { class: 'home-colophon' },
                    h('img', { class: 'home-colophon-logo', src: 'assets/cuju-logo.png', alt: '', width: '24', height: '24' }),
                    h('span', {}, 'Released under the Apache License 2.0'),
                    this.version)));
    }

    // The installed version, shown only when the client of the server offers it.
    async loadVersion() {
        if (typeof api.info !== 'function') {
            return;
        }
        try {
            const info = await api.info();
            if (info && typeof info.version === 'string' && info.version) {
                this.version.textContent = `Version ${info.version}`;
                this.version.hidden = false;
            }
        } catch {
            // An older server has no version to show, the line stays hidden.
        }
    }

    async refresh() {
        const token = ++this.token;
        const [live, matches] = await Promise.allSettled([api.live(), api.matches()]);
        if (token !== this.token) {
            return;
        }
        const failed = [live, matches].filter((r) => r.status === 'rejected').map((r) => r.reason);
        const offline = failed.some((err) => !err || !err.status);
        if (this.app.library && typeof this.app.library.setOffline === 'function') {
            this.app.library.setOffline(offline);
        }
        const state = {
            offline,
            live: live.status === 'fulfilled' && Array.isArray(live.value) ? live.value : null,
            liveError: live.status === 'rejected' ? String(live.reason && live.reason.message || live.reason) : null,
            matches: matches.status === 'fulfilled' && Array.isArray(matches.value) ? matches.value : null,
            matchesError: matches.status === 'rejected' ? String(matches.reason && matches.reason.message || matches.reason) : null,
        };
        // The same answer leaves the page as it is, so the focus and the hover stay put.
        const signature = JSON.stringify(state);
        if (signature === this.signature) {
            return;
        }
        this.signature = signature;
        this.renderStatus(state);
    }

    renderStatus(state) {
        const known = !state.offline && state.live && state.matches;
        this.glance.hidden = !known;
        if (known) {
            const on = state.live.length > 0;
            this.glanceLive.classList.toggle('on', on);
            this.glanceLive.replaceChildren(
                h('i', { class: 'home-live-dot', 'aria-hidden': 'true' }),
                on ? `${plural(state.live.length, 'source', 'sources')} live now` : 'Nothing live right now');
            this.glanceRecordings.textContent = plural(state.matches.length, 'recording', 'recordings');
        }
        // The focus returns to the same control after a redraw.
        const active = document.activeElement;
        const focusKey = active && this.status.contains(active) ? active.dataset.focusKey : null;
        let parts;
        if (state.offline) {
            parts = [this.offlineBlock()];
        } else if (state.live && state.matches && !state.live.length && !state.matches.length) {
            parts = [this.firstRunBlock()];
        } else {
            parts = [this.liveBlock(state), this.recordingsBlock(state)];
        }
        this.status.replaceChildren(...parts);
        if (focusKey) {
            const again = this.status.querySelector(`[data-focus-key="${CSS.escape(focusKey)}"]`);
            if (again) {
                again.focus();
            }
        }
    }

    openFile() {
        const input = document.getElementById('file-input');
        if (input) {
            this.pickedFile = true;
            input.click();
        }
    }

    offlineBlock() {
        return h('section', { class: 'home-block home-wide home-offline', 'aria-labelledby': 'home-offline-title' },
            bandHead('home-offline-title', 'The server is not answering'),
            h('div', { class: 'home-prose' },
                h('p', {}, 'Start the server to browse teams and matches, or open a local file.'),
                command('cuju render serve --open'),
                h('div', { class: 'home-row' },
                    h('button', { type: 'button', class: 'btn primary', 'data-focus-key': 'open-file', onclick: () => this.openFile() }, 'Open a file'),
                    h('button', { type: 'button', class: 'btn', 'data-focus-key': 'retry', onclick: () => this.refresh() }, 'Try again'))));
    }

    firstRunBlock() {
        const step = (n, title, text, ...extra) => h('li', { class: 'home-step' },
            h('span', { class: 'home-step-n', 'aria-hidden': 'true' }, String(n)),
            h('h3', {}, title),
            h('p', {}, text),
            ...extra);
        return h('section', { class: 'home-block home-wide home-first', 'aria-labelledby': 'home-first-title' },
            bandHead('home-first-title', 'Getting started', { eyebrow: 'Nothing to watch yet' }),
            h('p', { class: 'home-intro' }, 'Choose how to bring a match here.'),
            h('ol', { class: 'home-steps' },
                step(1, 'Run the page', 'The server of the viewer serves this page and the library on this computer alone. This command starts it and opens the page.',
                    command('cuju render serve --open')),
                step(2, 'Stream a running batch', 'With the variable set, a training script streams its environments here while it learns, and each one shows on this page as it starts.',
                    command('CUJU_RENDER=on python train.py')),
                step(3, 'Make a recording', [h('code', {}, 'cuju.render.Recorder'), ' writes the matches of a batch into the library, and the Matches page plays a match between two teams and keeps it.'],
                    h('div', { class: 'home-row' },
                        h('button', { type: 'button', class: 'btn', 'data-focus-key': 'first-matches', onclick: () => this.app.showPage('matches') }, 'Request a match'),
                        h('button', { type: 'button', class: 'btn ghost', 'data-focus-key': 'first-file', onclick: () => this.openFile() }, 'Open a file')))));
    }

    liveBlock(state) {
        const list = state.live || [];
        const count = list.length
            ? h('span', { class: 'home-live-count' }, h('i', { class: 'home-live-dot', 'aria-hidden': 'true' }), plural(list.length, 'source', 'sources'))
            : null;
        let body;
        if (state.liveError) {
            body = h('p', { class: 'home-empty error-text' }, `The live sources could not be read. ${state.liveError}`);
        } else if (!list.length) {
            body = h('p', { class: 'home-empty muted' }, 'No live source right now. A batch started with ', h('code', {}, 'CUJU_RENDER=on'), ' streams here, and this list refreshes on its own.');
        } else {
            body = h('ul', { class: 'home-list' }, ...list.map((s) => {
                const meta = [scenarioName(s.scenario), Number.isFinite(s.environment) ? `Environment ${s.environment}` : null].filter(Boolean);
                return h('li', { class: 'home-item live' },
                    h('div', { class: 'home-item-main' },
                        h('p', { class: 'home-item-title' }, titleText(s.title, s.scenario) || s.id),
                        h('p', { class: 'home-item-meta' }, ...meta.map((m) => h('span', {}, m)))),
                    h('button', {
                        type: 'button',
                        class: 'btn small primary',
                        'data-focus-key': `live:${s.id}`,
                        'aria-label': `Watch ${titleText(s.title, s.scenario) || s.id}`,
                        onclick: () => this.app.openLive(s.id),
                    }, 'Watch'));
            }));
        }
        return h('section', { class: 'home-block home-live', 'aria-labelledby': 'home-live-title' },
            bandHead('home-live-title', 'Live now', { aside: count }), body);
    }

    recordingsBlock(state) {
        const all = (state.matches || []).slice().sort((a, b) => createdMs(b) - createdMs(a));
        const latest = all.slice(0, LATEST);
        const more = h('button', {
            type: 'button',
            class: 'link home-more',
            'data-focus-key': 'all-recordings',
            onclick: () => this.app.showPage('recordings'),
        }, all.length > LATEST ? `All ${all.length} in Recordings` : 'Open Recordings');
        let body;
        if (state.matchesError) {
            body = h('p', { class: 'home-empty error-text' }, `The library could not be read. ${state.matchesError}`);
        } else if (!latest.length) {
            body = h('div', { class: 'home-empty' },
                h('p', { class: 'muted' }, 'No recordings yet. ', h('code', {}, 'cuju.render.Recorder'), ' writes them from a batch, and a match requested on the Matches page lands here when it ends.'),
                h('div', { class: 'home-row' },
                    h('button', { type: 'button', class: 'btn', 'data-focus-key': 'rec-matches', onclick: () => this.app.showPage('matches') }, 'Request a match')));
        } else {
            body = h('ul', { class: 'home-list' }, ...latest.map((m) => this.recordingItem(m)));
        }
        return h('section', { class: 'home-block home-recordings', 'aria-labelledby': 'home-rec-title' },
            bandHead('home-rec-title', 'Latest recordings', { aside: more }), body);
    }

    recordingItem(m) {
        const teams = Array.isArray(m.teams) ? m.teams : [];
        const score = Array.isArray(m.score) ? m.score : null;
        const board = teams.length === 2
            ? h('p', { class: 'home-board' },
                h('span', { class: 'home-board-team left' }, teams[0]),
                h('span', { class: 'home-score' },
                    h('span', { 'aria-hidden': 'true' }, score ? String(score[0]) : '-'),
                    h('span', { 'aria-hidden': 'true' }, score ? String(score[1]) : '-'),
                    h('span', { class: 'sr-only' }, score ? ` ${score[0]} to ${score[1]} ` : ' no score ')),
                h('span', { class: 'home-board-team right' }, teams[1]))
            : null;
        const date = formatDate(m.created);
        const meta = [scenarioName(m.scenario), date].filter(Boolean);
        const title = titleText(m.title, m.scenario) || m.id;
        // A title that only names the two teams says what the scoreboard below it says.
        const plain = board && title === `${teams[0]} against ${teams[1]}`;
        return h('li', { class: 'home-item' },
            h('div', { class: 'home-item-main' },
                plain ? null : h('p', { class: 'home-item-title' }, title),
                board,
                h('p', { class: 'home-item-meta' }, ...meta.map((x) => h('span', {}, x)))),
            h('button', {
                type: 'button',
                class: 'btn small',
                'data-focus-key': `rec:${m.id}`,
                'aria-label': `Play ${title}`,
                onclick: () => this.app.openMatchInWatch(m.id),
            }, 'Play'));
    }
}
