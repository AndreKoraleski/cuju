// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The Play a game page, where a person controls one player of a side with the keyboard or a
// controller, Xbox, Nintendo or PlayStation, against GRF's built-in AI, which plays every other player. The
// server plays the game in a process of its own at ten steps a second and streams each frame at
// once, and the page sends the held keys and buttons of each person's side on every change. The
// keys and buttons follow Gameplay Football's layout, and the server turns them into GRF's 19
// actions with GRF's own controller logic.

import * as THREE from 'three';
import { api } from './api.js';
import { LiveSource } from './live.js';
import { el as h } from './menu.js';
import { PLAYER_FLOATS } from './format.js';
import { listedScenarios, scenarioName, scenarioText } from './scenarios.js';

const DEFAULT_SCENARIO = '11_vs_11_kaggle';
// The view keeps this many frames behind the newest one, enough to blend between two frames.
const PLAY_DELAY_FRAMES = 1.1;
// The page repeats its input this often, so the server knows it is still there.
const HEARTBEAT_MS = 500;
// A stick counts as pushed past half way along an axis, as GRF's gamepad player reads it.
const STICK = 0.5;
// The settings of the viewer that would hold play back, off while a game shows.
const HELD_BACK = { celebrations: false, replays: false, transitions: false, decisionReplays: false, podium: false, lineups: false };

// Keyboard, by the physical key, so every layout plays the same. Each key presses the buttons of
// Gameplay Football, the ones outside GRF's 19 actions left out.
const KEYS = {
    ArrowLeft: ['left'], ArrowRight: ['right'], ArrowUp: ['up'], ArrowDown: ['down'],
    KeyW: ['long_pass'], KeyA: ['high_pass', 'sliding'], KeyS: ['short_pass'], KeyD: ['shot'],
    KeyE: ['sprint'], KeyC: ['dribble'],
};
// Controller actions bind to the position of a button, GRF's own Xbox layout, so the same hands
// play the same on every style of controller. South is the lower face button, east the right one,
// west the left one and north the upper one.
const PAD_BUTTONS = {
    south: ['short_pass'], east: ['shot'], west: ['high_pass', 'sliding'], north: ['long_pass'],
    rt: ['sprint'], rb: ['dribble'], up: ['up'], down: ['down'], left: ['left'], right: ['right'],
};
const POSITIONS = ['south', 'east', 'west', 'north', 'lb', 'rb', 'lt', 'rt', 'back', 'start', 'up', 'down', 'left', 'right'];
// The standard mapping of the Gamepad API, a button index for each position and the axes of the
// left stick. A source is a button, b, or an axis, a, with its sign, s, and threshold, t.
const STANDARD = {
    south: { b: 0 }, east: { b: 1 }, west: { b: 2 }, north: { b: 3 }, lb: { b: 4 }, rb: { b: 5 },
    lt: { b: 6 }, rt: { b: 7 }, back: { b: 8 }, start: { b: 9 },
    up: { b: 12 }, down: { b: 13 }, left: { b: 14 }, right: { b: 15 },
    stickX: { a: 0, s: 1, t: STICK }, stickY: { a: 1, s: 1, t: STICK },
};
// A hat axis gives the eight directions of a D-pad as values from -1 to 1 in steps of two
// sevenths, up first and then clockwise. A direction counts as held at its own value and at the
// two diagonals beside it. A value off that ring counts only when it matches the recorded one.
const HAT_STEP = 2 / 7;
function hatIndex(value) {
    const k = Math.round((value + 1) / HAT_STEP);
    return k >= 0 && k <= 7 && Math.abs(value - (-1 + k * HAT_STEP)) < 0.1 ? k : -1;
}
function hatOn(value, recorded) {
    if (value === undefined || Math.abs(value) > 1.05) {
        return false;
    }
    const k = hatIndex(value);
    const r = hatIndex(recorded);
    if (k < 0 || r < 0) {
        return Math.abs(value - recorded) < 0.1;
    }
    const d = Math.abs(k - r) % 8;
    return d === 0 || d === 1 || d === 7;
}

// A trigger or button that reports as an axis counts as pressed past this value.
const AXIS_ON = 0.5;
// The setup asks for each position in this order, the buttons, the left stick and the D-pad.
const SETUP_STEPS = ['south', 'east', 'west', 'north', 'lb', 'rb', 'lt', 'rt', 'back', 'start',
    'stickX', 'stickY', 'up', 'down', 'left', 'right'];
// The layouts set up by hand, one for each controller id.
const LAYOUTS_KEY = 'cuju.padLayouts';
const DIRECTIONS = new Set(['left', 'right', 'up', 'down']);
// A controller works the menus of the page by the positions of its buttons, as consoles do. A held
// direction moves once, then again after the delay and at the repeat rate while it stays held. A
// stick direction holds until the stick falls back under the release value, so one push is one move.
const NAV_DELAY_MS = 400;
const NAV_REPEAT_MS = 120;
const NAV_RELEASE = 0.3;
const FOCUSABLE = 'button, select, input, a[href], [tabindex]:not([tabindex="-1"])';

// Focus that a controller moves shows the ring of keyboard focus.
function padFocus(el) {
    el.focus({ focusVisible: true });
}

export const KEYBOARD_CONTROLS = [
    ['Arrow keys', 'Move, sticky while held'],
    ['S', 'Short pass'],
    ['W', 'Long pass'],
    ['A', 'High pass, or a sliding tackle when defending'],
    ['D', 'Shot'],
    ['E', 'Sprint while held'],
    ['C', 'Dribble while held'],
    ['Esc', 'Pause menu'],
    ['H or ?', 'Help'],
];
// The controller rows name positions, drawn in the icons of the controller's style.
export const PAD_CONTROLS = [
    ['stick or dpad', 'Move, sticky while held'],
    ['south', 'Short pass'],
    ['east', 'Shot'],
    ['west', 'High pass, or a sliding tackle when defending'],
    ['north', 'Long pass'],
    ['rt', 'Sprint while held'],
    ['rb', 'Dribble while held'],
    ['start', 'Pause menu'],
    ['back', 'This card'],
];

// The buttons of each style of controller by position, each with its classes, spoken name, text
// on the icon and short name, drawn without images so they stay crisp at any size.
const PAD_STYLES = {
    xbox: {
        name: 'Xbox',
        south: ['pad-face pad-a', 'A button', 'A', 'A'],
        east: ['pad-face pad-b', 'B button', 'B', 'B'],
        west: ['pad-face pad-x', 'X button', 'X', 'X'],
        north: ['pad-face pad-y', 'Y button', 'Y', 'Y'],
        lb: ['pad-shoulder', 'Left bumper', 'LB', 'LB'],
        rb: ['pad-shoulder', 'Right bumper', 'RB', 'RB'],
        lt: ['pad-trigger', 'Left trigger', 'LT', 'LT'],
        rt: ['pad-trigger', 'Right trigger', 'RT', 'RT'],
        back: ['pad-view', 'View button, also called Back', '', 'View'],
        start: ['pad-menu', 'Menu button, also called Start', '', 'Menu'],
    },
    nintendo: {
        name: 'Nintendo',
        south: ['pad-face', 'B button', 'B', 'B'],
        east: ['pad-face', 'A button', 'A', 'A'],
        west: ['pad-face', 'Y button', 'Y', 'Y'],
        north: ['pad-face', 'X button', 'X', 'X'],
        lb: ['pad-shoulder', 'L button', 'L', 'L'],
        rb: ['pad-shoulder', 'R button', 'R', 'R'],
        lt: ['pad-trigger', 'ZL button', 'ZL', 'ZL'],
        rt: ['pad-trigger', 'ZR button', 'ZR', 'ZR'],
        back: ['pad-sys', 'Minus button', '−', 'minus'],
        start: ['pad-sys', 'Plus button', '+', 'plus'],
    },
    playstation: {
        name: 'PlayStation',
        south: ['pad-face pad-ps pad-cross', 'Cross button', '✕', 'cross'],
        east: ['pad-face pad-ps pad-circle', 'Circle button', '○', 'circle'],
        west: ['pad-face pad-ps pad-square', 'Square button', '□', 'square'],
        north: ['pad-face pad-ps pad-triangle', 'Triangle button', '△', 'triangle'],
        lb: ['pad-shoulder', 'L1 button', 'L1', 'L1'],
        rb: ['pad-shoulder', 'R1 button', 'R1', 'R1'],
        lt: ['pad-trigger', 'L2 button', 'L2', 'L2'],
        rt: ['pad-trigger', 'R2 button', 'R2', 'R2'],
        back: ['pad-sys', 'Create button, also called Share', '', 'Create'],
        start: ['pad-menu', 'Options button', '', 'Options'],
    },
};
const PAD_COMMON = { stick: ['pad-stick', 'Left stick', 'L', 'left stick'], dpad: ['pad-dpad', 'Directional pad', '', 'D-pad'] };
const ARROWS = [['←', 'Left arrow'], ['↑', 'Up arrow'], ['↓', 'Down arrow'], ['→', 'Right arrow']];

function padInfo(style, pos) {
    return PAD_COMMON[pos] || (PAD_STYLES[style] || PAD_STYLES.xbox)[pos];
}

export function padIcon(style, pos) {
    const [kind, label, text] = padInfo(style, pos);
    return h('span', { class: `padicon ${kind}`, role: 'img', 'aria-label': label, title: label }, text);
}

// The likely style of a controller from the id the browser reports, Xbox when nothing says.
function detectStyle(id) {
    if (/xbox|xinput|microsoft|045e/i.test(id)) {
        return 'xbox';
    }
    if (/dualsense|dualshock|playstation|wireless controller|054c/i.test(id)) {
        return 'playstation';
    }
    if (/pro controller|joy-con|nintendo|057e/i.test(id)) {
        return 'nintendo';
    }
    return 'xbox';
}

function loadLayouts() {
    try {
        return JSON.parse(localStorage.getItem(LAYOUTS_KEY) || '{}') || {};
    } catch {
        return {};
    }
}

function saveLayout(id, entry) {
    const all = loadLayouts();
    all[id] = entry;
    try {
        localStorage.setItem(LAYOUTS_KEY, JSON.stringify(all));
    } catch {
        // The layout holds for this visit.
    }
}

// The keys of a row as keycaps or controller icons, alternatives joined by or.
function controlKeys(key, style) {
    if (key === 'Arrow keys') {
        return h('span', { class: 'keys' }, ARROWS.map(([text, label]) => h('kbd', { 'aria-label': label, title: label }, text)));
    }
    const parts = key.split(' or ');
    return h('span', { class: 'keys' }, parts.flatMap((part, i) => {
        const info = style && padInfo(style, part);
        return [
            i ? h('span', { class: 'keys-or' }, 'or') : null,
            info ? padIcon(style, part) : h('kbd', {}, part),
            info && !info[2] && (part === 'back' || part === 'start') ? h('span', { class: 'pad-name' }, info[3]) : null,
        ];
    }).filter(Boolean));
}

// A table of controls, the keyboard's or, given a style, the controller's.
function controlsTable(rows, style = null) {
    return h('table', { class: 'controls-table' },
        h('tbody', {}, rows.map(([key, what]) => h('tr', {}, h('th', {}, controlKeys(key, style)), h('td', {}, what)))));
}

// The controller line of the Help page in a style's icons.
function padHelp(style) {
    const i = (pos) => padIcon(style, pos);
    return ['Controller, ', i('stick'), ' left stick or ', i('dpad'), ' D pad moves. ',
        i('south'), ' passes short, ', i('east'), ' shoots, ', i('west'), ' passes high or tackles, ',
        i('north'), ' passes long. ', i('rt'), ' sprints, ', i('rb'), ' dribbles. ',
        i('start'), ' pauses, ', i('back'), ' shows controls. The controls card in Play sets up other controllers.'];
}

// A small sketch of a controller with one position lit, the face buttons in the style's labels.
function padSketch(style, on) {
    const ps = style === 'playstation';
    const [sx, sy] = ps ? [80, 86] : [58, 58];
    const [dx, dy] = ps ? [58, 58] : [80, 86];
    const lit = (pos) => (pos === on || (pos === 'stick' && (on === 'stickX' || on === 'stickY')) ? 'part on' : 'part');
    const text = (x, y, t, size = 9) => `<text x="${x}" y="${y}" font-size="${size}" text-anchor="middle" dominant-baseline="central">${t}</text>`;
    const info = (pos) => padInfo(style, pos)[2];
    const face = [['north', 146, 44], ['south', 146, 80], ['west', 128, 62], ['east', 164, 62]];
    const pad = [['up', dx - 5, dy - 16, 10, 11], ['down', dx - 5, dy + 5, 10, 11], ['left', dx - 16, dy - 5, 11, 10], ['right', dx + 5, dy - 5, 11, 10]];
    return `<svg class="pad-sketch" viewBox="0 0 200 124" aria-hidden="true">
<rect class="${lit('lt')}" x="26" y="2" width="40" height="13" rx="5"/>${text(46, 9, info('lt'), 7)}
<rect class="${lit('rt')}" x="134" y="2" width="40" height="13" rx="5"/>${text(154, 9, info('rt'), 7)}
<rect class="${lit('lb')}" x="22" y="17" width="48" height="9" rx="4"/>${text(46, 21.5, info('lb'), 7)}
<rect class="${lit('rb')}" x="130" y="17" width="48" height="9" rx="4"/>${text(154, 21.5, info('rb'), 7)}
<rect class="body" x="10" y="28" width="180" height="92" rx="44"/>
<circle class="${lit('stick')}" cx="${sx}" cy="${sy}" r="12"/>
${pad.map(([pos, x, y, w, hh]) => `<rect class="${lit(pos)}" x="${x}" y="${y}" width="${w}" height="${hh}" rx="2"/>`).join('')}
<circle class="${lit('back')}" cx="88" cy="48" r="5"/><circle class="${lit('start')}" cx="112" cy="48" r="5"/>
${face.map(([pos, x, y]) => `<circle class="${lit(pos)}" cx="${x}" cy="${y}" r="9"/>${text(x, y, info(pos))}`).join('')}
</svg>`;
}

// One input device, its held inputs and the buttons pressed since the last send.
class Device {
    constructor(id, label) {
        this.id = id;
        this.label = label;
        this.held = new Set();
        this.pressed = new Set();
        this.usedAt = 0;
    }

    set(names, down) {
        for (const name of names) {
            if (down && !this.held.has(name) && !DIRECTIONS.has(name)) {
                this.pressed.add(name);
            }
            if (down) {
                this.held.add(name);
            } else {
                this.held.delete(name);
            }
        }
        this.usedAt = performance.now();
    }
}

export class PlayPage {
    constructor(app) {
        this.app = app;
        this.root = document.getElementById('page-play');
        this.setup = { scenario: DEFAULT_SCENARIO, side: 'left', ability: [null, null], record: true };
        this.game = null;
        this.keyboard = new Device('keyboard', 'Keyboard');
        this.pads = new Map();
        this.built = false;
        this.saved = null;
        window.addEventListener('keydown', (e) => this.onKey(e, true), true);
        window.addEventListener('keyup', (e) => this.onKey(e, false), true);
        window.addEventListener('blur', () => this.releaseKeys());
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.playing) {
                this.pause(true);
            }
        });
        window.addEventListener('pagehide', () => {
            if (this.game && !this.game.ended) {
                fetch(`/api/game/${encodeURIComponent(this.game.id)}/pause`, {
                    method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: '{"paused": true}',
                }).catch(() => {});
            }
        });
        const loop = () => {
            requestAnimationFrame(loop);
            this.pollPads();
        };
        requestAnimationFrame(loop);
        setInterval(() => this.heartbeat(), HEARTBEAT_MS);
        this.marker = this.makeMarker();
        app.poseHooks.push((pose, s) => this.placeMarker(pose, s));
    }

    get visible() {
        return !this.root.hidden;
    }

    get playing() {
        return !!(this.game && !this.game.ended && !this.game.paused && this.game.started);
    }

    // The page.

    build() {
        this.built = true;
        this.scenarioSelect = h('select', { 'aria-label': 'Scenario' });
        this.scenarioSelect.addEventListener('change', () => {
            this.setup.scenario = this.scenarioSelect.value;
            this.scenarioText.textContent = scenarioText(this.setup.scenario);
        });
        this.scenarioText = h('p', { class: 'scenario-text muted small', 'aria-live': 'polite' });
        const sides = [['left', 'Left, at home'], ['right', 'Right, away'], ['both', 'Both, two players']];
        this.sideSelect = h('select', { 'aria-label': 'Your side' }, sides.map(([v, t]) => h('option', { value: v }, t)));
        this.sideSelect.addEventListener('change', () => {
            this.setup.side = this.sideSelect.value;
        });
        const ability = (side) => {
            const own = h('input', { type: 'checkbox', 'aria-label': `Set the ability of the built-in AI on the ${side ? 'right' : 'left'}` });
            const range = h('input', { type: 'range', min: '0', max: '1', step: '0.05', value: '0.5', disabled: true, 'aria-label': `${side ? 'Right' : 'Left'} ability, from 0 to 1` });
            const shown = h('output', { class: 'ability-value' }, "Scenario's");
            const sync = () => {
                this.setup.ability[side] = own.checked ? Number(range.value) : null;
                range.disabled = !own.checked;
                shown.textContent = own.checked ? Number(range.value).toFixed(2) : "Scenario's";
            };
            own.addEventListener('change', sync);
            range.addEventListener('input', sync);
            return h('div', { class: 'ability' }, h('label', { class: 'check' }, own, side ? 'Right ability' : 'Left ability'), range, shown);
        };
        this.recordBox = h('input', { type: 'checkbox', checked: true });
        this.recordBox.addEventListener('change', () => {
            this.setup.record = this.recordBox.checked;
        });
        this.startButton = h('button', { type: 'button', class: 'btn primary', onclick: () => this.start() }, 'Play');
        this.padNote = h('div', { class: 'muted small play-pads' });
        this.padTable = h('div', {}, h('h3', {}, 'Controller'), controlsTable(PAD_CONTROLS, 'xbox'));
        this.setupView = h('section', { class: 'play-setup', 'aria-labelledby': 'play-title' },
            h('header', {}, h('h2', { id: 'play-title' }, 'Play a game'),
                h('p', { class: 'muted' }, 'Use a keyboard or controller to play.')),
            h('div', { class: 'setup-row' },
                h('label', { class: 'field' }, 'Scenario', this.scenarioSelect),
                h('label', { class: 'field' }, 'Your side', this.sideSelect)),
            this.scenarioText,
            h('div', { class: 'setup-row', title: 'How well the built-in AI plays the players it controls, from 0 to 1' }, ability(0), ability(1)),
            h('label', { class: 'check' }, this.recordBox, 'Save to Recordings'),
            h('div', { class: 'play-tables' },
                h('div', {}, h('h3', {}, 'Keyboard'), controlsTable(KEYBOARD_CONTROLS)),
                this.padTable),
            this.padNote,
            h('div', { class: 'play-row' }, this.startButton));
        this.view = h('div', { class: 'play-view', tabindex: '0', 'aria-label': 'Game, Escape opens the pause menu' });
        this.view.addEventListener('focusout', () => this.releaseKeys());
        this.status = h('div', { class: 'play-status', 'aria-live': 'polite' });
        this.card = h('div', { class: 'play-card', hidden: true });
        this.resumeButton = h('button', { type: 'button', class: 'btn primary', onclick: () => this.pause(false) }, 'Resume');
        this.controlsButton = h('button', { type: 'button', class: 'btn', onclick: () => this.toggleCard(true) }, 'Controls');
        this.menu = h('div', { class: 'play-menu', role: 'dialog', 'aria-label': 'Paused', hidden: true },
            h('div', { class: 'play-panel' }, h('h2', {}, 'Paused'),
                this.resumeButton,
                this.controlsButton,
                h('button', { type: 'button', class: 'btn', onclick: () => this.openPadSetup() }, 'Set up controller'),
                h('button', { type: 'button', class: 'btn danger', onclick: () => this.quit() }, 'Quit'),
                h('button', { type: 'button', class: 'btn danger', onclick: () => this.goHome(true) }, 'Quit to Home'),
                h('button', { type: 'button', class: 'btn', onclick: () => this.goHome() }, 'Home'),
                this.menuHint = h('p', { class: 'muted small keys pad-hint', hidden: true })));
        this.endPanel = h('div', { class: 'play-panel' });
        this.end = h('div', { class: 'play-menu', role: 'dialog', 'aria-label': 'Full time', hidden: true }, this.endPanel);
        this.end.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                this.showSetup();
                this.startButton.focus();
            }
        });
        this.wait = h('div', { class: 'play-wait', hidden: true }, h('div', { class: 'spinner' }), h('span', {}, 'Starting the game'));
        this.stage = h('div', { class: 'play-stage', hidden: true }, this.view, this.status, this.card, this.wait, this.menu, this.end);
        this.offer = h('div', { class: 'pad-offer', role: 'status', hidden: true },
            h('span', {}, 'This controller reports its own layout. Set it up so each button does what the card shows.'),
            h('button', { type: 'button', class: 'btn primary', onclick: () => this.openPadSetup() }, 'Set up controller'),
            h('button', { type: 'button', class: 'btn', onclick: () => { this.offer.hidden = true; } }, 'Not now'));
        this.setupBox = h('div', { class: 'play-panel pad-setup-panel' });
        this.setupPanel = h('div', { class: 'pad-setup', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Set up controller', hidden: true }, this.setupBox);
        this.root.replaceChildren(this.setupView, this.stage, this.offer, this.setupPanel);
        this.shownStyle = null;
        this.refreshStyle();
    }

    async show() {
        if (!this.built) {
            this.build();
        }
        try {
            const scenarios = listedScenarios(await api.scenarios());
            this.scenarioSelect.replaceChildren(...scenarios.map((s) => h('option', { value: s }, scenarioName(s))));
            this.scenarioSelect.value = scenarios.includes(this.setup.scenario) ? this.setup.scenario : scenarios[0];
            this.setup.scenario = this.scenarioSelect.value;
            this.scenarioText.textContent = scenarioText(this.setup.scenario);
        } catch {
            // The server is away, the start says so.
        }
        this.updatePadNote();
        if (this.game) {
            this.enterStage();
            if (!this.game.ended && this.app.source !== this.game.source) {
                this.connect();
            }
        }
    }

    hide() {
        if (this.game && !this.game.ended && this.game.started && !this.game.paused) {
            this.pause(true);
        }
        this.leaveStage();
    }

    // The stage borrows the viewer's picture and its scoreboard while it shows.
    enterStage() {
        this.setupView.hidden = true;
        this.stage.hidden = false;
        this.app.view.attach(this.view);
        this.updateStatus();
        const hud = document.getElementById('hud');
        if (hud.parentNode !== this.stage) {
            this.stage.insertBefore(hud, this.status);
        }
        if (!this.saved) {
            this.saved = this.app.settings;
            this.app.settings = { ...this.saved, ...HELD_BACK };
        }
        this.app.cameras.setMode('broadcast');
    }

    leaveStage() {
        if (!this.built) {
            return;
        }
        const hud = document.getElementById('hud');
        if (hud.parentNode === this.stage) {
            document.getElementById('watch').insertBefore(hud, document.getElementById('stinger'));
        }
        this.app.view.attach(document.getElementById('stage'));
        if (this.saved) {
            // Changes made in the settings panel meanwhile stay, the held back ones return.
            const now = this.app.settings;
            this.app.settings = this.saved;
            for (const key of Object.keys(now)) {
                if (!(key in HELD_BACK)) {
                    this.saved[key] = now[key];
                }
            }
            this.saved = null;
        }
        this.marker.visible = false;
    }

    showSetup() {
        this.leaveStage();
        this.game = null;
        this.setupView.hidden = false;
        this.stage.hidden = true;
        this.startButton.focus();
    }

    // A game.

    async start() {
        const s = this.setup;
        const humans = [s.side !== 'right', s.side !== 'left'];
        this.startButton.disabled = true;
        let entry;
        try {
            entry = await api.startGame({ scenario: s.scenario, humans, ability: s.ability.slice(), record: s.record });
        } catch (err) {
            this.app.hud.toast(err.status === 409 ? 'A game is already playing. Quit it first.' : err.message, 'error', 7000);
            this.startButton.disabled = false;
            return;
        }
        this.startButton.disabled = false;
        this.game = { id: entry.id, humans, entry, started: false, paused: false, ended: false, source: null, sent: {}, busy: false };
        this.menu.hidden = true;
        this.end.hidden = true;
        this.enterStage();
        this.view.focus();
        this.connect();
    }

    connect() {
        const game = this.game;
        this.wait.hidden = false;
        const live = new LiveSource(game.id, {
            open: (signal) => api.gameStream(game.id, signal),
            onHeader: async (header, store) => {
                if (this.game !== game) {
                    live.stop();
                    return;
                }
                const source = { kind: 'live', id: game.id, header, store, live, game: true };
                game.source = source;
                await this.app.setSource(source);
                const tl = this.app.timeline;
                tl.delay = PLAY_DELAY_FRAMES;
                tl.lean = 0.5;
                tl.jump = 8;
                tl.goLive();
                this.app.cameras.setMode('broadcast');
                this.wait.hidden = true;
                if (!game.started) {
                    game.started = true;
                    this.toggleCard(true, 8000);
                    await this.sendPause(false);
                } else if (game.paused) {
                    this.hush();
                }
            },
            onFrame: () => {},
            onEnd: () => this.refresh(),
            onError: (err) => {
                this.wait.hidden = true;
                if (this.game === game && !game.ended) {
                    this.app.hud.toast(`The game stream stopped, ${err.message}`, 'error', 7000);
                }
            },
        });
        live.start();
    }

    async sendPause(paused) {
        const game = this.game;
        if (!game || game.ended) {
            return;
        }
        game.paused = paused;
        try {
            this.apply(await api.pauseGame(game.id, paused));
        } catch (err) {
            this.app.hud.toast(err.message, 'error');
        }
    }

    pause(on) {
        const game = this.game;
        if (!game || game.ended || !game.started) {
            return;
        }
        this.menu.hidden = !on;
        if (on) {
            this.releaseKeys();
            this.resumeButton.focus();
            this.hush();
        } else if (this.app.timeline) {
            this.view.focus();
            this.app.timeline.goLive();
        }
        this.sendPause(on);
    }

    // A paused or ended game holds its picture, and the sound, which plays only while the
    // timeline plays, falls silent. Resume plays the timeline again.
    hush() {
        const tl = this.app.timeline;
        if (tl && this.game && this.app.source === this.game.source) {
            tl.pause();
        }
    }

    async quit() {
        const game = this.game;
        if (!game) {
            return;
        }
        this.menu.hidden = true;
        try {
            this.apply(await api.quitGame(game.id));
        } catch {
            // The poll below reads the end.
        }
        this.refresh();
    }

    async goHome(quit = false) {
        if (!quit && this.game && !this.game.ended) {
            this.pause(true);
            this.app.showPage('home');
            document.getElementById('home-btn').focus();
            return;
        }
        const game = this.game;
        if (game && !game.ended) {
            try {
                this.apply(await api.quitGame(game.id));
                await this.refresh();
            } catch (err) {
                this.app.hud.toast(err.message, 'error');
                return;
            }
            if (!game.ended) {
                this.app.hud.toast('The game is still ending. Try Home again.', 'error');
                return;
            }
        }
        if (game && game.source) {
            game.source.live.stop();
        }
        this.showSetup();
        this.app.showPage('home');
        document.getElementById('home-btn').focus();
    }

    // The entry the server answers with, its state and score.
    apply(entry) {
        const game = this.game;
        if (!game || !entry || entry.id !== game.id) {
            return;
        }
        game.entry = entry;
        if (entry.state === 'paused' && entry.reason === 'input' && !game.waiting) {
            game.waiting = true;
            this.notice('Waiting for the connection to the game.');
        } else if (entry.state === 'playing' && game.waiting) {
            game.waiting = false;
        }
        if ((entry.state === 'ended' || entry.state === 'failed') && !game.ended) {
            this.finish(entry);
        }
    }

    async refresh() {
        if (!this.game || this.game.ended) {
            return;
        }
        const game = this.game;
        for (let tries = 0; tries < 40 && this.game === game && !game.ended; tries++) {
            try {
                this.apply(await api.game());
            } catch {
                return;
            }
            await new Promise((resolve) => setTimeout(resolve, 250));
        }
    }

    finish(entry) {
        const game = this.game;
        game.ended = true;
        this.menu.hidden = true;
        this.marker.visible = false;
        this.hush();
        const score = entry.score || [0, 0];
        const names = entry.teams || ['Left', 'Right'];
        const header = game.source && game.source.header;
        const store = game.source && game.source.store;
        const frame = store && store.get(store.last);
        const full = entry.ending === 'full time' && header && header.full_match && frame &&
            !!(frame.flags & 1) &&
            frame.step >= header.duration && frame.timeMs >= header.duration * 1800;
        const played = entry.ending === 'full time' || entry.ending === 'episode';
        const title = entry.state === 'failed' ? 'The game stopped' :
            full ? 'Full time' : played ? 'End of episode' : 'Game over';
        this.end.setAttribute('aria-label', title);
        const lines = [h('h2', {}, title),
            h('p', { class: 'play-score' }, `${names[0]} ${score[0]}, ${names[1]} ${score[1]}`)];
        if (entry.state === 'failed' && entry.error) {
            lines.push(h('p', { class: 'error-text' }, entry.error));
        }
        if (entry.match) {
            lines.push(h('p', { class: 'muted' }, 'Saved to Recordings.'),
                h('button', { type: 'button', class: 'btn primary', onclick: () => this.app.openMatchInWatch(entry.match) }, 'Watch it'));
        }
        lines.push(h('button', { type: 'button', class: 'btn', onclick: () => this.start() }, 'Play again'),
            h('button', { type: 'button', class: 'btn', onclick: () => this.showSetup() }, 'New game'),
            h('button', { type: 'button', class: 'btn', onclick: () => this.goHome() }, 'Home'));
        this.endPanel.replaceChildren(...lines);
        this.end.hidden = false;
    }

    // Input.

    onKey(e, down) {
        if (!down && KEYS[e.code] && this.keyboard.held.size) {
            this.keyboard.set(KEYS[e.code], false);
            this.flush();
        }
        if (this.padSetup) {
            // The setup panel takes the keyboard, Escape cancels it and Tab and Enter work its buttons.
            if (down && e.code === 'Escape' && this.app.page === 'play') {
                e.preventDefault();
                e.stopImmediatePropagation();
                this.closePadSetup(false);
            }
            return;
        }
        if (this.app.page !== 'play' || !this.game || this.game.ended || !this.game.started) {
            return;
        }
        if (e.defaultPrevented || e.target.closest('input, select, textarea, [contenteditable], .dd-panel, dialog') || e.ctrlKey || e.metaKey || e.altKey) {
            return;
        }
        if (e.code === 'Escape' && down) {
            e.preventDefault();
            e.stopImmediatePropagation();
            this.pause(this.menu.hidden);
            return;
        }
        if (e.target.closest('button, a') || !this.menu.hidden) {
            return;
        }
        if (down && e.code === 'KeyH') {
            this.app.toggleHelp();
        } else if (down && e.code === 'KeyM') {
            this.app.setMuted(!this.app.settings.muted);
        } else if (down && e.code === 'KeyF') {
            this.app.toggleFullscreen();
        }
        const names = KEYS[e.code];
        if (!names) {
            return;
        }
        e.preventDefault();
        e.stopImmediatePropagation();
        if (!this.menu.hidden || e.repeat) {
            return;
        }
        this.use(this.keyboard);
        this.keyboard.set(names, down);
        this.flush();
    }

    releaseKeys() {
        if (this.keyboard.held.size) {
            this.keyboard.held.clear();
            this.flush();
        }
    }

    pollPads() {
        const list = navigator.getGamepads ? Array.from(navigator.getGamepads() || []) : [];
        const seen = new Set();
        for (const pad of list) {
            if (!pad || !pad.connected) {
                continue;
            }
            seen.add(pad.index);
            let device = this.pads.get(pad.index);
            if (!device || device.gpId !== pad.id) {
                device = new Device(`pad${pad.index}`, 'Controller');
                device.index = pad.index;
                device.gpId = pad.id;
                device.mapping = pad.mapping;
                device.buttons = {};
                const stored = loadLayouts()[pad.id];
                device.custom = !!(stored && stored.layout);
                device.entry = device.custom ? stored : { style: detectStyle(pad.id), layout: STANDARD };
                this.pads.set(pad.index, device);
                this.notice(`A controller is connected${this.pads.size > 1 ? `, controller ${this.pads.size}` : ''}.`);
                if (pad.mapping !== 'standard' && !device.custom && this.built) {
                    this.offer.hidden = false;
                }
            }
            if (this.padSetup) {
                // While the setup shows, no controller plays. The controller being set up works the
                // style choice, then records its presses. Another controller works the buttons of
                // the panel while presses are recorded, and before the style is chosen a press on it
                // sets that one up instead, so each person sets up their own.
                const { now, x, y } = this.padHeld(pad, device.entry.layout);
                const before = device.buttons;
                const pressed = POSITIONS.some((pos) => now[pos] && !before[pos]);
                device.buttons = now;
                const s = this.padSetup;
                const mine = s.index === pad.index;
                const works = mine ? s.step < 0 : s.step >= 0;
                const move = this.navStep(device, this.navDirection(device, now, x, y), works);
                if (mine && s.step >= 0) {
                    this.recordPad(pad);
                } else if (works) {
                    this.navigate([this.setupBox], (pos) => now[pos] && !before[pos], move);
                } else if (pressed) {
                    this.lastPad = device;
                    this.padSetup = { ...this.padSetup, index: pad.index, id: pad.id, style: device.entry.style, layout: { ...device.entry.layout } };
                    this.renderSetup();
                }
                continue;
            }
            this.readPad(pad, device);
        }
        for (const [index, device] of this.pads) {
            if (!seen.has(index)) {
                this.pads.delete(index);
                device.held.clear();
                if (this.padSetup && this.padSetup.index === index) {
                    this.closePadSetup(false);
                }
                this.notice('A controller is disconnected.');
                this.flush();
            }
        }
    }

    // The positions a controller holds in a layout, and the actions they press.
    padHeld(pad, layout) {
        const button = (i) => !!(pad.buttons[i] && (pad.buttons[i].pressed || pad.buttons[i].value > AXIS_ON));
        const axis = (src) => (src && src.a !== undefined ? src.s * (pad.axes[src.a] || 0) : 0);
        const on = (src) => {
            if (!src) {
                return false;
            }
            if (src.b !== undefined) {
                return button(src.b);
            }
            if (src.v !== undefined) {
                return hatOn(pad.axes[src.a], src.v);
            }
            return axis(src) > (src.t ?? AXIS_ON);
        };
        const now = {};
        for (const pos of POSITIONS) {
            now[pos] = on(layout[pos]);
        }
        const held = new Set();
        for (const [pos, names] of Object.entries(PAD_BUTTONS)) {
            if (now[pos]) {
                names.forEach((n) => held.add(n));
            }
        }
        const x = axis(layout.stickX);
        const y = axis(layout.stickY);
        if (x < -STICK) {
            held.add('left');
        } else if (x > STICK) {
            held.add('right');
        }
        if (y < -STICK) {
            held.add('up');
        } else if (y > STICK) {
            held.add('down');
        }
        return { now, held, x, y };
    }

    readPad(pad, device) {
        const { now, held, x, y } = this.padHeld(pad, device.entry.layout);
        const before = device.buttons;
        device.buttons = now;
        const edge = (pos) => now[pos] && !before[pos];
        const scopes = this.navScopes();
        const move = this.navStep(device, this.navDirection(device, now, x, y), !!scopes);
        const live = this.game && !this.game.ended && this.game.started && this.app.page === 'play';
        if (live && edge('start')) {
            // The controller that opens the menu is the one its setup button sets up.
            this.lastPad = device;
            this.pause(this.menu.hidden);
            return;
        }
        if (live && edge('back')) {
            this.lastPad = device;
            this.toggleCard(this.card.hidden);
        }
        if (scopes) {
            // A button pressed in a menu stays out of play until it is released, so the press that
            // picks Resume or Play again passes no ball.
            device.navHeld = new Set([...held].filter((n) => !DIRECTIONS.has(n) && !device.held.has(n)));
            this.navigate(scopes, edge, move);
            return;
        }
        if (!live) {
            return;
        }
        if (device.navHeld) {
            for (const n of device.navHeld) {
                if (held.has(n)) {
                    held.delete(n);
                } else {
                    device.navHeld.delete(n);
                }
            }
        }
        const same = held.size === device.held.size && [...held].every((n) => device.held.has(n));
        if (same) {
            return;
        }
        for (const n of held) {
            if (!device.held.has(n) && !DIRECTIONS.has(n)) {
                device.pressed.add(n);
            }
        }
        device.held = held;
        device.usedAt = performance.now();
        this.use(device);
        this.flush();
    }

    // Menu navigation by controller.

    // The parts of the page that a controller works as a menu, or null while the buttons play.
    navScopes() {
        if (!this.built || this.app.page !== 'play') {
            return null;
        }
        if (this.padSetup) {
            return [this.setupBox];
        }
        const offer = this.offer.hidden ? [] : [this.offer];
        if (!this.setupView.hidden) {
            return [this.setupView, ...offer];
        }
        if (!this.end.hidden) {
            return [this.end];
        }
        if (!this.menu.hidden) {
            return [this.menu, ...(this.card.hidden ? [] : [this.card]), ...offer];
        }
        return null;
    }

    // The direction a controller points, the D-pad first, then the left stick past half way.
    navDirection(device, now, x, y) {
        const pull = { left: -x, right: x, up: -y, down: y };
        const dpad = [...DIRECTIONS].find((d) => now[d]);
        if (device.navDir === 'held') {
            return dpad || Object.values(pull).some((v) => v > NAV_RELEASE) ? 'held' : null;
        }
        if (dpad) {
            return dpad;
        }
        if (device.navDir && pull[device.navDir] > NAV_RELEASE) {
            return device.navDir;
        }
        const best = Object.keys(pull).reduce((a, b) => (pull[b] > pull[a] ? b : a));
        return pull[best] > STICK ? best : null;
    }

    // The move a direction makes this frame, once on its press and then repeated while held. The
    // direction is followed during play too, so a menu that opens under a held stick waits for a
    // fresh push.
    navStep(device, dir, active) {
        const time = performance.now();
        let move = null;
        if (dir !== device.navDir) {
            move = dir;
            device.navAt = time + NAV_DELAY_MS;
        } else if (dir && device.navAt && time >= device.navAt) {
            move = dir;
            device.navAt = time + NAV_REPEAT_MS;
        }
        device.navDir = dir;
        if (!active || dir === 'held') {
            device.navAt = 0;
            return null;
        }
        return move;
    }

    navTargets(scopes) {
        return scopes.flatMap((scope) => [...scope.querySelectorAll(FOCUSABLE)])
            .filter((el) => !el.disabled && !el.closest('[hidden]') && el.getClientRects().length);
    }

    // South works the focused control, east goes back, and a direction moves the focus in reading
    // order, or changes a select or a slider with left and right. The first press focuses the main
    // action of the menu.
    navigate(scopes, edge, move) {
        if (edge('east')) {
            this.navBack();
            return;
        }
        const south = edge('south');
        if (!south && !move) {
            return;
        }
        const targets = this.navTargets(scopes);
        const el = document.activeElement;
        const at = targets.indexOf(el);
        if (at < 0) {
            const first = targets.find((t) => t.classList.contains('primary')) || targets[0];
            if (first) {
                padFocus(first);
            }
            return;
        }
        if (south) {
            this.navActivate(el);
            return;
        }
        if ((move === 'left' || move === 'right') && this.navAdjust(el, move === 'right' ? 1 : -1)) {
            return;
        }
        const next = targets[at + (move === 'up' || move === 'left' ? -1 : 1)];
        if (next) {
            padFocus(next);
        }
    }

    // A select steps to its next option and wraps, a slider waits for left and right, and any
    // other control takes a click, which toggles a checkbox.
    navActivate(el) {
        if (el.tagName === 'SELECT') {
            if (el.options.length) {
                el.selectedIndex = (el.selectedIndex + 1) % el.options.length;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        } else if (el.type !== 'range') {
            el.click();
        }
    }

    // Left and right change a select or a slider directly, and report whether they did.
    navAdjust(el, step) {
        if (el.tagName === 'SELECT') {
            const i = el.selectedIndex + step;
            if (i >= 0 && i < el.options.length) {
                el.selectedIndex = i;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
            return true;
        }
        if (el.type === 'range') {
            if (step > 0) {
                el.stepUp();
            } else {
                el.stepDown();
            }
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }
        return false;
    }

    // East closes the setup panel or the controls card, resumes from the pause menu, and leaves
    // the end card for the setup, as Esc does.
    navBack() {
        if (this.padSetup) {
            this.closePadSetup(false);
        } else if (!this.menu.hidden && !this.card.hidden) {
            const inside = this.card.contains(document.activeElement);
            this.toggleCard(false);
            if (inside) {
                padFocus(this.controlsButton);
            }
        } else if (!this.menu.hidden) {
            this.pause(false);
        } else if (!this.end.hidden) {
            this.showSetup();
        }
    }

    // The devices of each person's side. Two people take the keyboard and the first controller,
    // or two controllers, and one person takes every device.
    devicesOf(side) {
        const pads = [...this.pads.keys()].sort((a, b) => a - b).map((i) => this.pads.get(i));
        if (!this.game || !this.game.humans[0] || !this.game.humans[1]) {
            return [this.keyboard, ...pads];
        }
        if (pads.length >= 2) {
            return side === 0 ? [this.keyboard, pads[0]] : [pads[1]];
        }
        return side === 0 ? [this.keyboard] : pads.slice(0, 1);
    }

    activeOf(side) {
        const devices = this.devicesOf(side);
        return devices.reduce((best, d) => (!best || d.usedAt > best.usedAt ? d : best), null);
    }

    // Whichever device was used last drives its side, and a change shows its controls.
    use(device) {
        const pad = device !== this.keyboard;
        const changed = this.lastDevice !== device.label || (pad && this.lastPad !== device);
        if (pad) {
            this.lastPad = device;
            this.refreshStyle();
        }
        if (changed) {
            this.lastDevice = device.label;
            if (this.game && this.game.started) {
                this.toggleCard(true, 5000);
            }
        }
        this.updateStatus();
    }

    flush() {
        const game = this.game;
        if (!game || game.ended || !game.started) {
            return;
        }
        game.humans.forEach((human, side) => {
            if (human) {
                this.send(side, false);
            }
        });
    }

    async send(side, always) {
        const game = this.game;
        const device = this.activeOf(side);
        const held = device ? [...device.held].sort() : [];
        const pressed = device ? [...device.pressed] : [];
        const key = held.join(',');
        if (!always && !pressed.length && game.sent[side] === key) {
            return;
        }
        if (game.busy) {
            game.dirty = true;
            return;
        }
        if (device) {
            device.pressed.clear();
        }
        game.sent[side] = key;
        game.busy = true;
        try {
            this.apply(await api.gameInput(game.id, { side, held, pressed }));
        } catch {
            // The heartbeat sends again.
        } finally {
            game.busy = false;
        }
        if (game.dirty) {
            game.dirty = false;
            this.flush();
        }
    }

    heartbeat() {
        const game = this.game;
        if (!game || game.ended || !game.started || game.busy) {
            return;
        }
        const side = game.humans[0] ? 0 : 1;
        this.send(side, true);
    }

    // Notices and cards.

    notice(text) {
        if (this.app.page === 'play') {
            this.app.hud.toast(text, 'info', 3000);
        }
        this.refreshStyle();
        this.updatePadNote();
        this.updateStatus();
    }

    // The controller in use, the last one used or else the first.
    activePad() {
        if (this.lastPad && this.pads.get(this.lastPad.index) === this.lastPad) {
            return this.lastPad;
        }
        const first = [...this.pads.keys()].sort((a, b) => a - b)[0];
        return first === undefined ? null : this.pads.get(first);
    }

    // The number of a controller, by the order the browser lists them.
    padNumber(index) {
        return [...this.pads.keys()].sort((a, b) => a - b).indexOf(index) + 1;
    }

    padStyle() {
        const pad = this.activePad();
        return pad ? pad.entry.style : 'xbox';
    }

    // What the browser reports of the controller in use, to tell a standard one from a custom one.
    padReport(pad) {
        return `Controller: ${pad.gpId}, mapping ${pad.mapping === 'standard' ? 'standard' : 'custom'}${pad.custom ? ', set up here' : ''}`;
    }

    // The tables, the pause menu and the Help page draw the style of the controller in use.
    refreshStyle() {
        const style = this.padStyle();
        if (!this.built || style === this.shownStyle) {
            return;
        }
        this.shownStyle = style;
        const name = PAD_STYLES[style].name;
        this.padTable.replaceChildren(h('h3', {}, `Controller, ${name}`), controlsTable(PAD_CONTROLS, style));
        this.menuHint.replaceChildren(padIcon(style, 'south'), ' selects, ', padIcon(style, 'east'), ' resumes');
        const help = document.querySelector('.pad-help');
        if (help) {
            help.replaceChildren(...padHelp(style));
        }
    }

    // The report line of a controller and the action that sets it up.
    padTools(pad) {
        return pad ? [h('p', { class: 'muted small pad-report' }, this.padReport(pad)),
            h('button', { type: 'button', class: 'btn', onclick: () => this.openPadSetup() }, 'Set up controller')] : [];
    }

    updatePadNote() {
        if (this.padNote) {
            const n = this.pads.size;
            const pad = this.activePad();
            this.padNote.replaceChildren(
                h('p', {}, n ? `${n} ${n === 1 ? 'controller' : 'controllers'} connected. Whichever device you used last drives your player.`
                    : 'No controller found yet. Connect one and press a button, the page picks it up at once.'),
                ...this.padTools(pad));
            this.menuHint.hidden = !n;
        }
    }

    updateStatus() {
        if (!this.status) {
            return;
        }
        const pads = this.pads.size;
        const parts = [`${this.lastDevice || 'Keyboard'}`];
        if (pads && this.lastDevice !== 'Controller') {
            parts.push(`${pads} ${pads === 1 ? 'controller' : 'controllers'} ready`);
        }
        this.status.textContent = parts.join(', ');
    }

    toggleCard(on, ms = 0) {
        clearTimeout(this.cardTimer);
        const pad = this.activePad();
        if (pad && this.lastDevice === 'Controller') {
            const style = pad.entry.style;
            this.card.replaceChildren(h('h3', {}, `Controller, ${PAD_STYLES[style].name}`), controlsTable(PAD_CONTROLS, style), ...this.padTools(pad));
        } else {
            this.card.replaceChildren(h('h3', {}, 'Keyboard'), controlsTable(KEYBOARD_CONTROLS), ...this.padTools(pad));
        }
        this.card.hidden = !on;
        if (on && ms) {
            this.cardTimer = setTimeout(() => {
                this.card.hidden = true;
            }, ms);
        }
    }

    // The controller setup, which asks the style of the controller, then each button in turn by
    // its position, and keeps the layout for the controller's id.

    openPadSetup() {
        const pad = this.activePad();
        if (!this.built || !pad) {
            this.app.hud.toast('Connect a controller and press a button first.', 'info', 4000);
            return;
        }
        if (this.playing) {
            this.pause(true);
        }
        this.offer.hidden = true;
        this.card.hidden = true;
        this.padSetup = {
            index: pad.index, id: pad.gpId, style: pad.entry.style,
            layout: { ...pad.entry.layout }, step: -1, armed: false, still: null, stillAt: 0, rest: [],
            back: document.activeElement,
        };
        this.setupPanel.hidden = false;
        this.renderSetup();
    }

    closePadSetup(save) {
        const s = this.padSetup;
        if (!s) {
            return;
        }
        this.padSetup = null;
        this.setupPanel.hidden = true;
        const device = this.pads.get(s.index);
        if (save) {
            const entry = { style: s.style, layout: s.layout };
            saveLayout(s.id, entry);
            if (device) {
                device.entry = entry;
                device.custom = true;
                // The new layout reads the buttons still held differently, so each waits for its release.
                device.buttons = Object.fromEntries(POSITIONS.map((pos) => [pos, true]));
                device.navDir = 'held';
                device.navAt = 0;
            }
            this.shownStyle = null;
            this.app.hud.toast('The controller is set up, its layout is kept for next time.', 'info', 4000);
        }
        this.refreshStyle();
        this.updatePadNote();
        if (!this.menu.hidden) {
            this.resumeButton.focus();
        } else if (s.back && s.back.isConnected) {
            s.back.focus();
        }
    }

    setupStep(step) {
        const s = this.padSetup;
        s.step = step;
        s.armed = false;
        if (step >= SETUP_STEPS.length) {
            this.closePadSetup(true);
            return;
        }
        this.renderSetup();
    }

    renderSetup() {
        const s = this.padSetup;
        const button = (text, onclick, cls = 'btn') => h('button', { type: 'button', class: cls, onclick }, text);
        const cancel = button('Cancel', () => this.closePadSetup(false));
        if (s.step < 0) {
            const choices = Object.entries(PAD_STYLES).map(([style, info]) => {
                const b = button(info.name, () => {
                    s.style = style;
                    this.setupStep(0);
                }, style === s.style ? 'btn primary' : 'btn');
                b.setAttribute('aria-pressed', String(style === s.style));
                return b;
            });
            this.setupBox.replaceChildren(h('h2', {}, 'Set up controller'),
                h('p', {}, 'Which style is your controller? The buttons then show in its own labels.'),
                h('div', { class: 'pad-setup-row' }, choices),
                h('p', { class: 'muted small' }, `Controller ${this.padNumber(s.index)}: ${s.id}`),
                ...(this.pads.size > 1 ? [h('p', { class: 'muted small' }, 'To set up another controller, press a button on it.')] : []),
                h('div', { class: 'pad-setup-row' }, cancel));
            choices[Object.keys(PAD_STYLES).indexOf(s.style)].focus();
            return;
        }
        const pos = SETUP_STEPS[s.step];
        let prompt;
        if (pos === 'stickX') {
            prompt = 'Push the left stick to the right';
        } else if (pos === 'stickY') {
            prompt = 'Push the left stick down';
        } else if (DIRECTIONS.has(pos)) {
            prompt = `Press ${pos} on the D-pad`;
        } else {
            prompt = `Press ${padInfo(s.style, pos)[3]}`;
        }
        const icon = pos.startsWith('stick') ? 'stick' : DIRECTIONS.has(pos) ? 'dpad' : pos;
        const sketch = h('div', { class: 'pad-sketch-box' });
        sketch.innerHTML = padSketch(s.style, pos);
        const skip = button('Skip', () => this.setupStep(s.step + 1));
        this.setupBox.replaceChildren(h('h2', {}, 'Set up controller'),
            h('p', { class: 'muted small' }, `Step ${s.step + 1} of ${SETUP_STEPS.length}, ${PAD_STYLES[s.style].name}`),
            h('p', { class: 'pad-prompt keys', 'aria-live': 'polite' }, padIcon(s.style, icon), prompt),
            sketch,
            h('p', { class: 'muted small' }, 'Release every button, then press. Skip keeps what this position had.'),
            h('div', { class: 'pad-setup-row' }, button('Back', () => this.setupStep(s.step - 1)), skip, cancel));
        skip.focus();
    }

    // One frame of the controller being set up. It waits until every button is up and the axes
    // rest for a moment, then takes the first button pressed, or the first axis that moves well
    // away from its rest past the threshold, with the sign of the move.
    recordPad(pad) {
        const s = this.padSetup;
        if (s.step < 0) {
            return;
        }
        const time = performance.now();
        const axes = Array.from(pad.axes);
        if (!s.still || axes.some((v, i) => Math.abs(v - (s.still[i] ?? 0)) > 0.1)) {
            s.still = axes;
            s.stillAt = time;
        }
        const pressed = pad.buttons.map((b) => !!(b && (b.pressed || b.value > AXIS_ON)));
        if (!s.armed) {
            if (!pressed.some(Boolean) && time - s.stillAt > 250) {
                s.armed = true;
                s.rest = s.still;
            }
            return;
        }
        const pos = SETUP_STEPS[s.step];
        let src = null;
        const button = pressed.indexOf(true);
        if (button >= 0 && !pos.startsWith('stick')) {
            src = { b: button };
        }
        for (let i = 0; !src && i < axes.length; i++) {
            const rest = s.rest[i] ?? 0;
            const move = axes[i] - rest;
            const sign = move > 0 ? 1 : -1;
            if (Math.abs(move) <= 0.6) {
                continue;
            }
            // A D-pad that reports as one hat axis rests outside the range of a stick and gives
            // each direction a value of its own, which is recorded as it is.
            if (Math.abs(rest) > 1.05 || (DIRECTIONS.has(pos) && Math.abs(axes[i]) < 0.9)) {
                src = { a: i, v: axes[i] };
            } else if (sign * axes[i] > AXIS_ON) {
                src = { a: i, s: sign, t: AXIS_ON };
            }
        }
        if (src) {
            s.layout[pos] = src;
            this.setupStep(s.step + 1);
        }
    }

    // The marker, an arrow in the team's color over each person's player.

    makeMarker() {
        const group = new THREE.Group();
        group.visible = false;
        this.arrows = [0, 1].map(() => {
            const geometry = new THREE.ConeGeometry(0.28, 0.6, 16);
            geometry.rotateX(Math.PI);
            const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.95 }));
            mesh.renderOrder = 10;
            group.add(mesh);
            return mesh;
        });
        this.app.view.scene.add(group);
        return group;
    }

    placeMarker(pose, s) {
        const game = this.game;
        const on = !!(game && game.started && !game.ended && this.app.source === game.source && this.app.page === 'play' && s.a);
        this.marker.visible = on;
        if (!on) {
            return;
        }
        const header = game.source.header;
        const bob = Math.sin(performance.now() / 180) * 0.08;
        for (let side = 0; side < 2; side++) {
            const arrow = this.arrows[side];
            const k = game.humans[side] ? header.players.findIndex((p, i) => p.team === side && (s.a.playerFlags[i] & 8)) : -1;
            arrow.visible = k >= 0;
            if (k < 0) {
                continue;
            }
            const style = this.app.styles && this.app.styles[side];
            if (style) {
                const c = style.kit.primary;
                arrow.material.color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
            }
            const o = k * PLAYER_FLOATS;
            arrow.position.set(pose.nodes[o], 2.55 + bob, -pose.nodes[o + 1]);
        }
    }
}
