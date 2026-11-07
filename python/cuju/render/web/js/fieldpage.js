// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The Field page, for grass, lines, nets, boards, crowd, time of day and weather, with the match
// view borrowed as a live preview. The preview flies the free camera, and editing a control
// flies it to the thing being changed, from where the user keeps free control.

import * as THREE from 'three';
import * as Cameras from './cameras.js';
import {
    BOARDS, CROWDS, MAX_BOARDS, MOWING, PRESETS, TIMES, TIME_HOURS, WEATHERS, cleanField, exportField, importField,
    presetField, saveField,
} from './field.js';
import { h } from './library.js';
import { menuHeading, menuItem } from './menu.js';
import { Painter } from './painter.js';
import { ARENAS, ARENA_LABELS, ARENA_NOTES, arenaThumbnail } from './arenas.js';
import { formatHour } from './sky.js';

// Board pictures are read whole before they are scaled down, so a very large file is refused first.
const BOARD_FILE_BYTES = 32 * 1024 * 1024;

const sameList = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

const LABELS = {
    stripes: 'Stripes', checkers: 'Checkers', circles: 'Circles', none: 'None',
    sponsors: 'Team sponsors', shipped: 'Shipped boards', custom: 'My own boards',
    grf: 'GRF crowd', dawn: 'Dawn', day: 'Day', dusk: 'Dusk', night: 'Night',
};
const CROWD_LABELS = { teams: 'Team fans', custom: 'Painted crowd in chosen colors', grf: 'GRF crowd' };
export const WEATHER_LABELS = {
    clear: 'Clear', overcast: 'Overcast', rain: 'Rain', storm: 'Heavy rain', snow: 'Snow', fog: 'Fog',
};

// Where the preview looks for each part of the field, used when cameras.js has no table of its
// own. Positions and targets are in three's frame, the main stand on positive z.
const VIEWS = {
    overview: { pos: [0, 30, 53], target: [0, 0, -4], fov: 55 },
    grass: { pos: [-28, 2.2, 22], target: [8, 0, -4], fov: 60 },
    lines: { pos: [-61, 4.5, 40], target: [10, 0, 34], fov: 55 },
    nets: { pos: [64, 3.4, 7], target: [55, 1.1, 0], fov: 55 },
    boards: { pos: [-18, 1.8, 33], target: [30, 0.6, 41.5], fov: 50 },
    crowd: { pos: [0, 8, 12], target: [0, 12, -70], fov: 55 },
    sky: { pos: [-42, 18, 49], target: [10, 16, -40], fov: 65 },
    arena: { pos: [-48, 46, 34], target: [20, 4, -24], fov: 62 },
};
// Each part of the form and the view it flies to.
const FOCUS = {
    grass: 'grass', mowing: 'grass', lines: 'lines', nets: 'nets', boards: 'boards', crowd: 'crowd',
    time: 'sky', weather: 'sky', arena: 'arena', presets: 'overview',
};
const FLY_SECONDS = 1.4;

const vec = (v) => {
    if (!v) {
        return null;
    }
    if (v.isVector3) {
        return v.clone();
    }
    if (Array.isArray(v)) {
        return new THREE.Vector3(v[0], v[1], v[2]);
    }
    return new THREE.Vector3(v.x, v.y, v.z);
};

export class FieldPage {
    constructor(app) {
        this.app = app;
        this.root = document.getElementById('page-field');
        this.painter = new Painter();
        this.focused = null;
        this.tween = null;
        this.bindQuick();
    }

    get field() {
        return this.app.field;
    }

    change(patch, rerender = false) {
        const next = cleanField({ ...this.field, ...patch });
        // The match view reloads the board pictures when the list changes, so an unchanged list
        // keeps its old array.
        if (sameList(next.customBoards, this.field.customBoards)) {
            next.customBoards = this.field.customBoards;
        }
        this.app.applyField(next);
        const err = saveField(next);
        if (err) {
            this.app.hud.toast(err, 'error', 8000);
        }
        this.syncQuick();
        if (rerender) {
            const selector = 'button, input:not(.color-value), select';
            const active = document.activeElement;
            const textColor = active.matches('.color-value');
            const controls = [...this.root.querySelectorAll(selector)];
            const index = controls.indexOf(textColor ? active.previousElementSibling : active);
            this.render();
            if (index >= 0) {
                const next = this.root.querySelectorAll(selector)[index];
                next?.focus();
                if (textColor) {
                    queueMicrotask(() => next?.nextElementSibling?.focus());
                }
            }
        }
    }

    // While a slider moves only the light follows, the whole field is stored when it stops.
    preview(patch) {
        if (this.app.environment) {
            this.app.environment.setField({ ...this.field, ...patch });
        }
    }

    render() {
        const f = this.field;
        const chips = (list, current, key, labels = LABELS) => h('div', { class: 'chips' }, list.map((v) => h('button', {
            type: 'button', class: `chip-btn${v === current ? ' active' : ''}`, 'aria-pressed': v === current ? 'true' : 'false',
            onclick: () => this.change({ [key]: v }, true),
        }, labels[v] || v)));
        const color = (label, key) => {
            const input = h('input', { type: 'color', value: f[key] });
            input.addEventListener('input', () => this.change({ [key]: input.value }));
            return h('label', { class: 'swatch' }, input, h('span', {}, label));
        };
        // A slider that shows its value, previews while it moves and stores when it stops.
        const slider = (key, min, max, step, show, rerender = false) => {
            const input = h('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(f[key]) });
            const label = h('span', { class: 'value' }, show(f[key]));
            input.addEventListener('input', () => {
                label.textContent = show(Number(input.value));
                this.preview({ [key]: Number(input.value) });
            });
            input.addEventListener('change', () => this.change({ [key]: Number(input.value) }, rerender));
            return [input, label];
        };
        const percent = (v) => `${Math.round(v * 100)} %`;
        const section = (focus, ...children) => h('section', { class: 'field-section', 'data-focus': focus }, ...children);
        const [width, widthLabel] = slider('stripeWidth', 2, 16, 0.5, (v) => `${v} m`);
        const density = h('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(f.crowdDensity) });
        density.addEventListener('change', () => this.change({ crowdDensity: Number(density.value) }, true));
        const crowdColors = h('div', { class: 'swatches small' }, f.crowdColors.map((c, i) => {
            const input = h('input', { type: 'color', value: c });
            input.addEventListener('change', () => {
                const list = f.crowdColors.slice();
                list[i] = input.value;
                this.change({ crowdColors: list, crowd: 'custom' }, true);
            });
            return h('label', { class: 'swatch' }, input);
        }), f.crowdColors.length < 12 ? h('button', {
            type: 'button', class: 'btn small', onclick: () => this.change({ crowdColors: [...f.crowdColors, '#888888'], crowd: 'custom' }, true),
        }, 'Add color') : null, f.crowdColors.length > 1 ? h('button', {
            type: 'button', class: 'btn small ghost', onclick: () => this.change({ crowdColors: f.crowdColors.slice(0, -1), crowd: 'custom' }, true),
        }, 'Remove last') : null);
        const presets = h('select', { 'aria-label': 'Presets' }, h('option', { value: '' }, 'Choose a preset'),
            Object.keys(PRESETS).map((n) => h('option', { value: n }, n)));
        presets.addEventListener('change', () => {
            if (presets.value) {
                const p = presetField(presets.value);
                // Presets keep the boards you drew, and the arena unless they name one.
                p.customBoards = f.customBoards;
                if (!('arena' in PRESETS[presets.value])) {
                    p.arena = f.arena;
                }
                this.change(p, true);
            }
        });
        const importInput = h('input', { type: 'file', accept: 'application/json,.json', hidden: true });
        importInput.addEventListener('change', async () => {
            const file = importInput.files[0];
            importInput.value = '';
            if (!file) {
                return;
            }
            try {
                const { field, notes } = await importField(file, this.field);
                this.change(field, true);
                this.app.hud.toast(['Field imported.', ...notes].join(' '), notes.length ? 'info' : undefined, notes.length ? 9000 : undefined);
            } catch (err) {
                this.app.hud.toast(err.message, 'error', 9000);
            }
        });
        const boardFile = h('input', { type: 'file', accept: 'image/*', hidden: true });
        boardFile.addEventListener('change', async () => {
            const file = boardFile.files[0];
            boardFile.value = '';
            if (!file) {
                return;
            }
            if (file.size > BOARD_FILE_BYTES) {
                this.app.hud.toast(`This picture is ${Math.ceil(file.size / 1024 / 1024)} MB, the limit is 32 MB.`, 'error');
                return;
            }
            try {
                this.addBoard(await this.boardDataUrl(file));
            } catch (err) {
                this.app.hud.toast(`This picture could not be read, ${err.message || err}.`, 'error');
            }
        });
        const full = f.customBoards.length >= MAX_BOARDS;
        const boards = h('div', { class: 'board-list' },
            f.customBoards.map((url, i) => h('div', { class: 'board-item' },
                h('img', { src: url, alt: `Board ${i + 1}` }),
                h('button', {
                    type: 'button', class: 'btn small ghost danger',
                    onclick: () => this.change({ customBoards: f.customBoards.filter((_, k) => k !== i) }, true),
                }, 'Remove'))),
            h('div', { class: 'row' },
                h('button', { type: 'button', class: 'btn small', disabled: full, onclick: () => boardFile.click() }, 'Upload an image'),
                h('button', { type: 'button', class: 'btn small', disabled: full, onclick: () => this.drawBoard() }, 'Draw a board'),
                boardFile),
            full ? h('p', { class: 'muted small' }, `This is the limit of ${MAX_BOARDS} boards. Remove one to add another.`) : null);
        const [hour, hourLabel] = slider('time', 0, 24, 0.25, formatHour);
        const named = TIMES.find((t) => Math.abs(TIME_HOURS[t] - f.time) < 0.01);
        const timeChips = h('div', { class: 'chips' }, TIMES.map((t) => h('button', {
            type: 'button', class: `chip-btn${t === named ? ' active' : ''}`, 'aria-pressed': t === named ? 'true' : 'false',
            title: formatHour(TIME_HOURS[t]), onclick: () => this.change({ time: TIME_HOURS[t] }, true),
        }, LABELS[t])));
        const [amount, amountLabel] = slider('weatherAmount', 0.1, 1, 0.05, percent);
        const [wind, windLabel] = slider('wind', 0, 1, 0.05, (v) => (v < 0.05 ? 'Calm' : percent(v)));
        const arenas = h('div', { class: 'arena-list' }, ARENAS.map((name) => h('button', {
            type: 'button', class: `arena-card${name === f.arena ? ' active' : ''}`, 'aria-pressed': name === f.arena ? 'true' : 'false',
            title: ARENA_NOTES[name], onclick: () => this.change({ arena: name }, true),
        }, h('img', { src: this.thumb(name), alt: '', width: '160', height: '90' }), h('span', {}, ARENA_LABELS[name]))));
        const form = h('div', { class: 'field-form' },
            section('presets', h('div', { class: 'row' }, presets,
                h('span', { class: 'spacer' }),
                h('button', { type: 'button', class: 'btn', onclick: () => exportField(this.field) }, 'Export'),
                h('button', { type: 'button', class: 'btn', onclick: () => importInput.click() }, 'Import'),
                h('button', {
                    type: 'button', class: 'btn ghost',
                    onclick: () => {
                        if (confirm('Reset the field to the default look? Your boards are removed too. Export the field first to keep a copy.')) {
                            this.change({ ...presetField('Summer afternoon'), arena: 'stadium' }, true);
                        }
                    },
                }, 'Reset'),
                importInput)),
            section('arena', h('h4', {}, 'Arena'),
                arenas,
                h('p', { class: 'muted small' }, `${ARENA_NOTES[f.arena]} The pitch, its lines and its goals stay as in GRF.`)),
            section('grass', h('h4', {}, 'Grass'),
                h('div', { class: 'swatches' }, color('Grass', 'grass'),
                    h('span', { 'data-focus': 'lines' }, color('Lines', 'lines')),
                    h('span', { 'data-focus': 'nets' }, color('Goal nets', 'net'))),
                h('div', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Mowing'), chips(MOWING, f.mowing, 'mowing')),
                h('label', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Band width'), width, widthLabel)),
            section('time', h('h4', {}, 'Time of day'),
                h('label', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Hour'), hour, hourLabel),
                timeChips),
            section('weather', h('h4', {}, 'Weather'),
                chips(WEATHERS, f.weather, 'weather', WEATHER_LABELS),
                f.weather !== 'clear' ? h('label', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Strength'), amount, amountLabel) : null,
                h('label', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Wind'), wind, windLabel),
                f.weather === 'snow' ? h('p', { class: 'muted small' }, 'Snow settles on the pitch over a minute or so, the lines stay clear.') : null),
            section('boards', h('h4', {}, 'Boards'),
                chips(BOARDS, f.boards, 'boards'),
                f.boards === 'custom' ? boards : h('p', { class: 'muted small' }, f.boards === 'sponsors'
                    ? 'The sponsors of both teams in turn, the shipped boards when they have none.'
                    : 'The boards that ship with cuju and the team names.')),
            section('crowd', h('h4', {}, 'Crowd'),
                chips(CROWDS, f.crowd, 'crowd', CROWD_LABELS),
                f.crowd !== 'grf' ? h('div', {},
                    h('label', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Density'), density),
                    f.crowd === 'custom' ? h('div', { class: 'field-row' }, h('span', { class: 'field-label' }, 'Shirts'), crowdColors)
                        : h('p', { class: 'muted small' }, 'Fans wear their team colors.')) : null),
            h('p', { class: 'muted small' }, 'Export settings to use them elsewhere.'),
            h('p', { class: 'muted small' }, 'Drag to look. WASD moves, Q and E change height.'));
        // Touching a control flies the preview to what it changes.
        const focusFrom = (e) => {
            const el = e.target && e.target.closest && e.target.closest('[data-focus]');
            if (el) {
                this.focus(FOCUS[el.dataset.focus] || 'overview');
            }
        };
        form.addEventListener('pointerdown', focusFrom);
        form.addEventListener('focusin', focusFrom);
        form.addEventListener('input', focusFrom);
        const preview = h('div', { class: 'field-preview', id: 'field-preview' });
        // A choice redraws the form, which keeps its place in the list.
        const old = this.root.querySelector('.field-form');
        const scroll = old ? old.scrollTop : 0;
        this.root.replaceChildren(h('div', { class: 'field-layout' }, form, preview));
        form.scrollTop = scroll;
        if (!this.root.hidden) {
            this.app.view.attach(preview);
        }
    }

    // Thumbnails are painted once.
    thumb(name) {
        this.thumbs = this.thumbs || {};
        if (!this.thumbs[name]) {
            this.thumbs[name] = arenaThumbnail(name);
        }
        return this.thumbs[name];
    }

    // The quick controls in the View menu of the Watch page, the hour and the weather, built
    // into the container the menu keeps for them.
    bindQuick() {
        const host = document.getElementById('view-environment');
        if (!host) {
            return;
        }
        const hour = h('input', {
            id: 'env-hour', type: 'range', min: '0', max: '24', step: '0.25', 'aria-label': 'Time of day',
        });
        const value = h('span', { id: 'env-hour-value', class: 'env-hour-value' });
        hour.addEventListener('input', () => {
            value.textContent = formatHour(Number(hour.value));
            this.preview({ time: Number(hour.value) });
        });
        hour.addEventListener('change', () => this.change({ time: Number(hour.value) }, !this.root.hidden));
        this.quickWeather = WEATHERS.map((w) => {
            const item = menuItem(WEATHER_LABELS[w] || w, {
                checked: false, keepOpen: true, onSelect: () => this.change({ weather: w }, !this.root.hidden),
            });
            item.dataset.weather = w;
            return item;
        });
        this.quickArena = ARENAS.map((name) => {
            const item = menuItem(ARENA_LABELS[name], {
                checked: false, keepOpen: true, onSelect: () => this.change({ arena: name }, !this.root.hidden),
            });
            item.dataset.arena = name;
            return item;
        });
        host.replaceChildren(
            h('label', { class: 'env-hour-row', role: 'presentation' }, h('span', { class: 'env-hour-label' }, 'Hour'), hour, value),
            ...this.quickWeather,
            menuHeading('Arena'),
            ...this.quickArena);
        this.syncQuick();
    }

    syncQuick() {
        const f = this.field;
        const hour = document.getElementById('env-hour');
        if (!f || !hour) {
            return;
        }
        if (document.activeElement !== hour) {
            hour.value = String(f.time);
        }
        document.getElementById('env-hour-value').textContent = formatHour(f.time);
        for (const item of this.quickWeather || []) {
            item.setAttribute('aria-checked', item.dataset.weather === f.weather ? 'true' : 'false');
        }
        for (const item of this.quickArena || []) {
            item.setAttribute('aria-checked', item.dataset.arena === f.arena ? 'true' : 'false');
        }
    }

    // Flies the preview to a part of the field, once per part, so moving the camera by hand and
    // then changing the same thing again leaves the camera where the user put it.
    focus(name, snap = false) {
        if (this.root.hidden || (name === this.focused && !snap)) {
            return;
        }
        this.focused = name;
        const table = Cameras.FIELD_VIEWS || VIEWS;
        const view = table[name] || VIEWS[name] || table.overview || VIEWS.overview;
        const pos = vec(view.pos || view.position);
        const target = vec(view.target);
        if (!pos || !target) {
            return;
        }
        const fov = view.fov || 50;
        const rig = this.app.cameras;
        if (typeof rig.flyTo === 'function') {
            rig.flyTo(pos, target, fov, snap ? 0 : (view.seconds || FLY_SECONDS));
            return;
        }
        this.fly(pos, target, snap ? 0 : FLY_SECONDS);
    }

    // The same flight on the free camera, when the rig has no flyTo of its own.
    fly(pos, target, seconds) {
        const rig = this.app.cameras;
        if (rig.mode !== 'free') {
            rig.setMode('free');
        }
        const cam = rig.camera;
        const d = new THREE.Vector3().subVectors(target, pos).normalize();
        const yaw = Math.atan2(-d.x, -d.z);
        const pitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
        const from = { pos: cam.position.clone(), yaw: rig.yaw, pitch: rig.pitch };
        let dyaw = yaw - from.yaw;
        dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
        const token = {};
        this.tween = token;
        const start = performance.now();
        const step = (now) => {
            if (this.tween !== token || this.root.hidden || rig.mode !== 'free') {
                return;
            }
            const t = seconds > 0 ? Math.min(1, (now - start) / (seconds * 1000)) : 1;
            const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            cam.position.lerpVectors(from.pos, pos, e);
            rig.yaw = from.yaw + dyaw * e;
            rig.pitch = from.pitch + (pitch - from.pitch) * e;
            if (t < 1) {
                requestAnimationFrame(step);
            } else {
                this.tween = null;
            }
        };
        step(start);
    }

    // Any key or drag on the preview hands the camera back to the user.
    stopFlight() {
        this.tween = null;
        const rig = this.app.cameras;
        if (typeof rig.stopFlight === 'function') {
            rig.stopFlight();
        }
    }

    // Says so when the list of boards is full.
    boardsFull() {
        if (this.field.customBoards.length < MAX_BOARDS) {
            return false;
        }
        this.app.hud.toast(`The field holds at most ${MAX_BOARDS} boards. Remove one to add another.`, 'error');
        return true;
    }

    addBoard(url) {
        if (this.boardsFull()) {
            return false;
        }
        this.change({ customBoards: [...this.field.customBoards, url], boards: 'custom' }, true);
        return true;
    }

    async boardDataUrl(file) {
        // Boards are stored small, as 1024 by 128 PNGs, so the settings fit in the browser.
        const bitmap = await createImageBitmap(file);
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        const scale = Math.max(1024 / bitmap.width, 128 / bitmap.height);
        const w = bitmap.width * scale;
        const hgt = bitmap.height * scale;
        ctx.drawImage(bitmap, (1024 - w) / 2, (128 - hgt) / 2, w, hgt);
        bitmap.close();
        return canvas.toDataURL('image/png');
    }

    drawBoard() {
        if (this.boardsFull()) {
            return;
        }
        this.painter.open({
            title: 'Board',
            width: 1024,
            height: 128,
            note: 'A wide board, 1024 by 128 pixels.',
            onSave: async (png) => {
                const url = await new Promise((resolve, reject) => {
                    const r = new FileReader();
                    r.onload = () => resolve(r.result);
                    r.onerror = () => reject(r.error || new Error('the drawing could not be read'));
                    r.readAsDataURL(png);
                });
                if (!this.addBoard(url)) {
                    throw new Error(`the field already holds ${MAX_BOARDS} boards`);
                }
            },
        }).catch((err) => this.app.hud.toast(`The painter could not open, ${err.message || err}.`, 'error'));
    }

    // The preview borrows the match view's canvas while the page is showing, with the free
    // camera on a wide view that shows the sky and the floodlight towers.
    show() {
        this.camera = this.app.cameras.mode;
        this.app.cameras.setMode('free');
        this.render();
        this.focused = null;
        // The sky view shows the towers and the sky, so the time of day reads at once.
        this.focus('sky', true);
        if (!this.stopper) {
            this.stopper = () => {
                if (!this.root.hidden) {
                    this.stopFlight();
                }
            };
            this.app.view.renderer.domElement.addEventListener('pointerdown', this.stopper);
            window.addEventListener('keydown', (e) => {
                if (!e.target.closest('input, textarea, select, button, [contenteditable]') &&
                    e.key.length === 1 && 'wasdqec'.includes(e.key.toLowerCase())) {
                    this.stopper();
                }
            });
        }
    }

    hide() {
        this.tween = null;
        this.app.view.attach(document.getElementById('stage'));
        if (this.camera) {
            this.app.cameras.setMode(this.camera);
            this.camera = null;
        }
    }
}
