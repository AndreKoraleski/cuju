// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The client of the local server, for recordings in the format of the section Recording format of
// docs/rendering.md, live sources, teams with their styles, models and images, and folder uploads.

export class ApiError extends Error {
    constructor(message, status = 0) {
        super(message);
        this.status = status;
    }
}

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
export const MAX_MODEL_BYTES = 30 * 1024 * 1024;

async function request(method, path, { body, json, headers, timeout } = {}) {
    let res;
    const init = { method, headers: { ...(headers || {}) }, cache: 'no-store' };
    if (timeout) {
        init.signal = AbortSignal.timeout(timeout);
    }
    if (json !== undefined) {
        init.body = JSON.stringify(json);
        init.headers['Content-Type'] = 'application/json';
    } else if (body !== undefined) {
        init.body = body;
        init.headers['Content-Type'] = init.headers['Content-Type'] || 'application/octet-stream';
    }
    try {
        res = await fetch(path, init);
    } catch {
        throw new ApiError(`Cannot reach the cuju server at ${location.origin}. Is it still running?`, 0);
    }
    if (!res.ok) {
        let detail = '';
        try {
            const text = await res.text();
            try {
                const parsed = JSON.parse(text);
                detail = parsed.error || parsed.message || text;
            } catch {
                detail = text;
            }
        } catch {
            detail = '';
        }
        throw new ApiError(detail ? `${detail}` : `The server answered ${res.status} ${res.statusText}.`, res.status);
    }
    return res;
}

async function getJson(path) {
    return (await request('GET', path)).json();
}

const enc = encodeURIComponent;

export const api = {
    matches: () => getJson('/api/matches'),
    matchBytes: async (id, onProgress) => {
        const res = await request('GET', `/api/matches/${enc(id)}`);
        const total = Number(res.headers.get('Content-Length')) || 0;
        if (!res.body || !onProgress) {
            return new Uint8Array(await res.arrayBuffer());
        }
        const reader = res.body.getReader();
        const chunks = [];
        let got = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            chunks.push(value);
            got += value.length;
            onProgress(got, total);
        }
        const out = new Uint8Array(got);
        let o = 0;
        for (const c of chunks) {
            out.set(c, o);
            o += c.length;
        }
        return out;
    },
    deleteMatch: (id) => request('DELETE', `/api/matches/${enc(id)}`),
    patchMatch: async (id, changes) => (await request('PATCH', `/api/matches/${enc(id)}`, { json: changes })).json().catch(() => null),
    live: () => getJson('/api/live'),
    liveStream: (id, signal) => fetch(`/api/live/${enc(id)}/stream`, { signal, cache: 'no-store' }),
    info: () => getJson('/api/info'),
    scenarios: () => getJson('/api/scenarios'),
    teams: () => getJson('/api/teams'),
    uploadTeam: async (name, filename, body, format, control = null) => {
        let path = `/api/teams?name=${enc(name)}&filename=${enc(filename)}`;
        if (format) path += `&format=${enc(format)}`;
        if (control) path += `&control=${enc(JSON.stringify(control))}`;
        return (await request('POST', path, { body })).json();
    },
    putImage: (id, slot, body, type) =>
        request('PUT', `/api/teams/${enc(id)}/images/${enc(slot)}`, { body, headers: { 'Content-Type': type } }),
    image: async (id, slot) => (await request('GET', `/api/teams/${enc(id)}/images/${enc(slot)}`)).blob(),
    deleteImage: (id, slot) => request('DELETE', `/api/teams/${enc(id)}/images/${enc(slot)}`),
    exportUrl: (id) => `/api/teams/${enc(id)}/export`,
    updateTeam: async (id, changes) => (await request('PATCH', `/api/teams/${enc(id)}`, { json: changes })).json(),
    deleteTeam: (id) => request('DELETE', `/api/teams/${enc(id)}`),
    putModel: (id, joints, bytes) =>
        request('PUT', `/api/teams/${enc(id)}/model?joints=${enc(JSON.stringify(joints))}`, { body: bytes }),
    modelBytes: async (id) => (await request('GET', `/api/teams/${enc(id)}/model`)).arrayBuffer(),
    deleteModel: (id) => request('DELETE', `/api/teams/${enc(id)}/model`),
    models: () => getJson('/api/models'),
    requests: () => getJson('/api/requests'),
    requestMatch: async (spec) => (await request('POST', '/api/requests', { json: spec })).json(),
    cancelRequest: (id) => request('DELETE', `/api/requests/${enc(id)}`),
    // The game that a person plays, one at a time, its stream, its input, pause and quit.
    game: () => getJson('/api/game'),
    startGame: async (spec) => (await request('POST', '/api/game', { json: spec })).json(),
    gameStream: (id, signal) => fetch(`/api/game/${enc(id)}/stream`, { signal, cache: 'no-store' }),
    gameInput: async (id, input) => (await request('POST', `/api/game/${enc(id)}/input`, { json: input, timeout: 1500 })).json(),
    pauseGame: async (id, paused) => (await request('POST', `/api/game/${enc(id)}/pause`, { json: { paused } })).json(),
    quitGame: async (id) => (await request('DELETE', `/api/game/${enc(id)}`)).json(),
    // Team sounds, Ogg, MP3 or WAV of at most MAX_SOUND_BYTES, one per slot of SOUND_SLOTS.
    putSound: async (id, slot, body, type) =>
        (await request('PUT', `/api/teams/${enc(id)}/sounds/${enc(slot)}`, { body, headers: { 'Content-Type': type || 'application/octet-stream' } })).json(),
    soundBytes: async (id, slot) => (await request('GET', `/api/teams/${enc(id)}/sounds/${enc(slot)}`)).arrayBuffer(),
    deleteSound: (id, slot) => request('DELETE', `/api/teams/${enc(id)}/sounds/${enc(slot)}`),
};

// The team setting travels as JSON through both forms.
export const CONTROLLABLE = { '5v5': 5, '11v11': 11 };

export function controlFromValue(value) {
    if (value && typeof value === 'object') return structuredClone(value);
    if (String(value).startsWith('{')) return JSON.parse(value);
    const players = parseInt(value, 10) || 1;
    const fixed = /-keeper$/.test(value);
    return {
        players, assignment: fixed ? 'fixed' : 'dynamic',
        selection: fixed ? Array.from({ length: players }, (_, i) => i + 1) : [],
        goalkeeper: fixed ? 'builtin' : 'scenario', action_set: 'default',
    };
}

export function controlValue(control) {
    if (!control?.assignment) {
        return JSON.stringify(controlFromValue(control?.goalkeeper === 'builtin'
            ? `${control.players}-keeper` : String(control?.players || 1)));
    }
    return JSON.stringify(control);
}

export function controlLabel(control) {
    const c = controlFromValue(controlValue(control));
    const selection = c.assignment === 'fixed' ? `, players ${c.selection.join(', ')}` : '';
    return `${c.players} ${c.assignment}${selection}, goalkeeper ${c.goalkeeper}, ${c.action_set}`;
}

export function controlForm(format, initial = null) {
    const root = document.createElement('div');
    let current = controlFromValue(controlValue(initial));
    function field(label, choices = null) {
        const row = document.createElement('label');
        row.className = 'field';
        row.append(document.createTextNode(label));
        const input = document.createElement(choices ? 'select' : 'input');
        for (const [value, text] of choices || []) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = text;
            input.append(option);
        }
        row.append(input);
        root.append(row);
        return input;
    }
    const count = field('Agent count');
    count.type = 'number';
    count.min = '1';
    count.value = current.players;
    const assignment = field('Assignment', [
        ['dynamic', 'Dynamic, GRF switches players'], ['fixed', 'Fixed, keep selected players'],
    ]);
    assignment.value = current.assignment;
    const by = field('Select fixed players by', [['index', 'Index'], ['role', 'GRF role']]);
    by.value = typeof current.selection[0] === 'string' ? 'role' : 'index';
    const selection = field('Players, in agent order');
    selection.value = current.selection.join(', ');
    const goalkeeper = field('Goalkeeper', [
        ['agent', 'In team control'], ['builtin', 'Handed to the AI, fixed assignment'],
        ['scenario', 'Scenario decides, fixed requires an uncontrollable keeper'],
    ]);
    goalkeeper.value = current.goalkeeper;
    const actionSet = field('Action set', [
        ['default', 'Default, 19 actions'], ['v2', 'v2, 20 actions, includes builtin AI'],
    ]);
    actionSet.value = current.action_set;
    const lineup = document.createElement('div');
    lineup.className = 'lineup-picker';
    lineup.setAttribute('role', 'group');
    lineup.setAttribute('aria-label', 'Fixed players');
    root.append(lineup);
    function update() {
        const fixed = assignment.value === 'fixed';
        by.disabled = selection.disabled = !fixed;
        current = {
            players: Number(count.value), assignment: assignment.value,
            selection: fixed && selection.value.trim() ? selection.value.split(',').map(
                (item) => by.value === 'role' ? item.trim().toUpperCase() : Number(item.trim())
            ) : [],
            goalkeeper: goalkeeper.value, action_set: actionSet.value,
        };
        root.value = JSON.stringify(current);
        const remainingRoles = [...current.selection];
        for (const button of lineup.querySelectorAll('button')) {
            const chosen = fixed && (by.value === 'index'
                ? current.selection.includes(Number(button.dataset.index))
                : remainingRoles.includes(button.dataset.role));
            if (chosen && by.value === 'role') {
                remainingRoles.splice(remainingRoles.indexOf(button.dataset.role), 1);
            }
            button.setAttribute('aria-pressed', String(chosen));
        }
    }
    root.addEventListener('input', update);
    root.addEventListener('change', update);
    root.setFormat = (next) => {
        count.max = CONTROLLABLE[next] || 11;
        const roles = next === '5v5' ? ['GK', 'RM', 'CF', 'LB', 'CB']
            : ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CM', 'LM', 'CF', 'RM'];
        lineup.replaceChildren(...roles.map((role, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'lineup-btn';
            const number = document.createElement('span');
            number.className = 'lineup-index';
            number.textContent = String(index);
            const name = document.createElement('span');
            name.className = 'lineup-role';
            name.textContent = role;
            button.append(number, name);
            button.setAttribute('aria-label', `Player ${index}, ${role}`);
            button.setAttribute('aria-pressed', 'false');
            button.dataset.index = String(index);
            button.dataset.role = role;
            button.disabled = next === '5v5' && index === 0;
            button.addEventListener('click', () => {
                const chosen = assignment.value !== 'fixed' ? [] : by.value === 'role'
                    ? [...lineup.querySelectorAll('button[aria-pressed="true"]')].map(
                        (item) => item.dataset.index
                    ) : selection.value.split(',').map((v) => v.trim()).filter(Boolean);
                assignment.value = 'fixed';
                by.value = 'index';
                const value = String(index);
                selection.value = chosen.includes(value)
                    ? chosen.filter((v) => v !== value).join(', ') : [...chosen, value].join(', ');
                count.value = selection.value.split(',').filter(Boolean).length || 1;
                if (!selection.value) {
                    assignment.value = 'dynamic';
                }
                update();
                root.dispatchEvent(new Event('change', { bubbles: true }));
            });
            return button;
        }));
        update();
    };
    root.setFormat(format);
    return root;
}

export const SOUND_SLOTS = ['goal', 'shot', 'pass', 'tackle', 'save', 'kick', 'celebration', 'chant'];
export const MAX_SOUND_BYTES = 2 * 1024 * 1024;

// Files of a Kaggle submission folder without caches and dot files, paths with forward slashes.
export function keepFile(path) {
    return !path.split('/').some((part) => part === '__pycache__' || part.startsWith('.'));
}

// Packs files as CUJDIR1, the magic, a uint32 length, a JSON index, then the bytes in order.
export async function packFolder(files) {
    const index = files.map((f) => ({ path: f.path, size: f.file.size }));
    const json = new TextEncoder().encode(JSON.stringify(index));
    const head = new Uint8Array(12);
    head.set(new TextEncoder().encode('CUJDIR1\n'), 0);
    new DataView(head.buffer).setUint32(8, json.length, true);
    const parts = [head, json];
    for (const f of files) {
        parts.push(new Uint8Array(await f.file.arrayBuffer()));
    }
    return new Blob(parts);
}

// Files from a folder input, each with its path and file.
export function filesFromInput(fileList) {
    return Array.from(fileList)
        .map((file) => ({ path: (file.webkitRelativePath || file.name).replace(/\\/g, '/'), file }))
        .filter((f) => keepFile(f.path));
}

// Files from a dropped folder, walking entries recursively.
export async function filesFromDrop(items) {
    const entries = [];
    for (const item of Array.from(items)) {
        const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
        if (entry) {
            entries.push(entry);
        }
    }
    const out = [];
    const readAll = (dir) => new Promise((resolve, reject) => {
        const reader = dir.createReader();
        const list = [];
        const next = () => reader.readEntries((batch) => {
            if (batch.length === 0) {
                resolve(list);
            } else {
                list.push(...batch);
                next();
            }
        }, reject);
        next();
    });
    const walk = async (entry) => {
        if (entry.isFile) {
            const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
            out.push({ path: entry.fullPath.replace(/^\/+/, ''), file });
        } else if (entry.isDirectory) {
            for (const child of await readAll(entry)) {
                await walk(child);
            }
        }
    };
    for (const e of entries) {
        await walk(e);
    }
    return { files: out.filter((f) => keepFile(f.path)), folder: entries.length === 1 && entries[0].isDirectory ? entries[0].name : null };
}

export function formatBytes(n) {
    if (!Number.isFinite(n)) {
        return '';
    }
    if (n < 1024) {
        return `${n} B`;
    }
    if (n < 1024 * 1024) {
        return `${(n / 1024).toFixed(1)} KB`;
    }
    if (n < 1024 * 1024 * 1024) {
        return `${(n / 1024 / 1024).toFixed(1)} MB`;
    }
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// Accepts an ISO string, or seconds or milliseconds since the epoch.
export function formatDate(value) {
    if (value == null || value === '') {
        return '';
    }
    let date;
    if (typeof value === 'number') {
        date = new Date(value < 1e12 ? value * 1000 : value);
    } else {
        date = new Date(value);
    }
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }
    return new Intl.DateTimeFormat(uiLocale(), { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

// Dates and numbers follow the language of the page, not the browser's locale.
export function uiLocale() {
    return document.documentElement.lang || 'en-US';
}

// A date as a calendar day alone, in the language of the page.
export function formatDay(value) {
    const date = typeof value === 'number' ? new Date(value < 1e12 ? value * 1000 : value) : new Date(value);
    if (Number.isNaN(date.getTime())) {
        return '';
    }
    return new Intl.DateTimeFormat(uiLocale(), { dateStyle: 'medium' }).format(date);
}
