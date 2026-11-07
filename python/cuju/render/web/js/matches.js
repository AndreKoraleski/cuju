// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The Matches page, where two teams are picked for a match that plays on this computer, and the
// queue of requested matches with their progress, cancelling and watching.

import { api, controlLabel, uiLocale } from './api.js';
import { css, normalizeStyle } from './kits.js';
import { Dropdown, el as h, enhanceSelect, icon, menuItem } from './menu.js';
import { listedScenarios, scenarioName, scenarioText } from './scenarios.js';

const DEFAULT_SCENARIO = '11_vs_11_kaggle';
const FORMAT_NAMES = { '5v5': '5 against 5', '11v11': '11 against 11' };

// Scenarios a team may play, by its format. The built-in AI plays any.
export function fits(team, scenario) {
    if (!team || team.kind === 'builtin' || !team.format) {
        return true;
    }
    if (scenario.startsWith('academy_')) return true;
    const name = scenario.replace(/^benchmark_full_game_/, '');
    return team.format === '5v5' ? name.startsWith('5_vs_5') : name.startsWith('11_vs_11');
}

// A small shirt in a team's kit, its pattern drawn with the same classes as the editor's.
export function kitShirt(team, side = 0, size = 'small') {
    const style = normalizeStyle(team, side);
    const k = style.kit;
    // The shirt is clipped to its outline, the wrapper draws an edge around it that shows a
    // white shirt on a white page.
    return h('span', { class: `kit-shirt ${size}`, 'aria-hidden': 'true' },
        h('span', {
            class: `kit-shirt-body p-${k.pattern}`,
            style: `--a:${css(k.primary)};--b:${css(k.secondary)};--c:${css(k.shorts)}`,
        }, style.badge ? h('b', { class: 'kit-badge' }, style.badge) : null));
}

function stateOf(r) {
    if (r.state === 'failed' && r.error === 'cancelled') {
        return 'cancelled';
    }
    return r.state;
}

const STATE_LABELS = { queued: 'Waiting', running: 'Playing', done: 'Finished', failed: 'Failed', cancelled: 'Canceled' };

// The time a request was made, read from its identifier, r-YYYYMMDD-HHMMSS-xxxx.
function requestTime(id) {
    const m = /^r-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(id || '');
    return m ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) : null;
}

export class MatchesPage {
    constructor(app) {
        this.app = app;
        this.root = document.getElementById('page-matches');
        this.teams = [];
        this.scenarios = [];
        this.requests = [];
        this.seen = new Set();
        this.primed = false;
        this.state = { left: null, right: 'builtin', scenario: null, seed: '', ability: [null, null] };
        this.filters = { status: 'all', team: 'all', scenario: 'all', date: 'any', sort: 'newest' };
        this.error = null;
        this.token = 0;
        this.built = false;
        setInterval(() => this.poll(), 1500);
    }

    get visible() {
        return !this.root.hidden;
    }

    async refresh() {
        const token = ++this.token;
        try {
            const [teams, scenarios, requests] = await Promise.all([
                api.teams(), api.scenarios().catch(() => []), api.requests().catch(() => []),
            ]);
            if (token !== this.token) {
                return;
            }
            this.teams = teams;
            const listed = listedScenarios(scenarios);
            this.scenarios = listed.length ? listed : [DEFAULT_SCENARIO];
            this.requests = requests;
            this.error = null;
        } catch (err) {
            if (token !== this.token) {
                return;
            }
            this.error = err;
        }
        this.render();
    }

    team(id) {
        return this.teams.find((t) => t.id === id) || null;
    }

    teamName(id) {
        const t = this.team(id);
        return t ? t.name : id;
    }

    render() {
        if (this.error) {
            this.built = false;
            this.root.replaceChildren(h('div', { class: 'page-pad' }, h('div', { class: 'notice error' },
                h('p', {}, this.error.message),
                h('p', { class: 'muted' }, 'Start the server with cuju render serve, then try again.'),
                h('button', { class: 'btn', type: 'button', onclick: () => this.refresh() }, 'Try again'))));
            return;
        }
        const s = this.state;
        if (!this.team(s.left)) {
            s.left = (this.teams.find((t) => t.kind !== 'builtin') || this.teams[0] || {}).id || null;
        }
        if (!this.team(s.right)) {
            s.right = (this.teams.find((t) => t.kind === 'builtin') || this.teams[0] || {}).id || null;
        }
        if (!this.built) {
            this.build();
        }
        this.updateSetup();
        this.renderQueue();
    }

    // The page is built once, later refreshes update its parts in place.
    build() {
        this.built = true;
        this.sides = [0, 1].map((side) => this.sidePicker(side));
        this.scenarioSelect = h('select', { 'aria-label': 'Scenario' });
        this.scenarioSelect.addEventListener('change', () => {
            this.state.scenario = this.scenarioSelect.value;
            this.describeScenario();
        });
        this.scenarioText = h('p', { class: 'scenario-text muted small', 'aria-live': 'polite' });
        this.seedInput = h('input', { type: 'number', min: '0', step: '1', placeholder: 'Random', value: this.state.seed, 'aria-label': 'Seed' });
        this.seedInput.addEventListener('input', () => {
            this.state.seed = this.seedInput.value;
        });
        this.clash = h('div', { class: 'clash', role: 'alert', hidden: true });
        this.playButton = h('button', { type: 'button', class: 'btn primary play-btn', onclick: () => this.play() },
            icon('play'), 'Play match');
        this.queueHost = h('div', { class: 'queue-list' });
        this.filterBar = this.buildFilters();
        const setup = h('section', { class: 'match-setup', 'aria-labelledby': 'setup-title' },
            h('header', { class: 'setup-head' },
                h('h2', { id: 'setup-title' }, 'New match'),
                h('p', { class: 'muted' }, 'Pick two teams and a scenario. '
                    + 'The match uses each team setting and lands in Recordings.')),
            h('div', { class: 'versus' }, this.sides[0].root, h('div', { class: 'vs', 'aria-hidden': 'true' }, 'vs'), this.sides[1].root),
            h('div', { class: 'setup-row' },
                h('label', { class: 'field' }, 'Scenario', this.scenarioSelect),
                h('label', { class: 'field seed-field' }, 'Seed, empty for a random one',
                    h('span', { class: 'row' }, this.seedInput,
                        h('button', { type: 'button', class: 'btn ghost', title: 'Draw a seed now', onclick: () => this.drawSeed() }, 'Draw')))),
            this.scenarioText,
            this.clash,
            h('div', { class: 'play-row' }, this.playButton));
        const queue = h('section', { class: 'match-queue', 'aria-labelledby': 'queue-title' },
            h('header', { class: 'queue-head' }, h('h2', { id: 'queue-title' }, 'Requested matches'), this.filterBar),
            this.queueHost);
        this.root.replaceChildren(h('div', { class: 'matches-layout' }, setup, queue));
        enhanceSelect(this.scenarioSelect, { label: 'Scenario', className: 'wide' });
    }

    // The short text of the chosen scenario under the list.
    describeScenario() {
        this.scenarioText.textContent = scenarioText(this.state.scenario || '');
    }

    // One side of the match, a large card of the chosen team that opens a list of every team.
    sidePicker(side) {
        const button = h('button', { type: 'button', class: 'team-card', 'aria-label': side ? 'Right team' : 'Left team' });
        const panel = h('div', { role: 'menu', class: 'team-menu', 'aria-label': side ? 'Right team' : 'Left team' });
        document.getElementById('app').appendChild(panel);
        const fill = () => {
            const current = side ? this.state.right : this.state.left;
            panel.replaceChildren(...this.teams.map((t, i) => menuItem(t.name, {
                checked: t.id === current,
                hint: [t.kind === 'builtin' ? 'Built-in AI, plays any scenario' : FORMAT_NAMES[t.format] || '', t.kind !== 'builtin' ? 'submission' : null].filter(Boolean).join(', '),
                icon: kitShirt(t, i % 2),
                onSelect: () => {
                    this.state[side ? 'right' : 'left'] = t.id;
                    this.updateSetup();
                },
            })));
        };
        const dropdown = new Dropdown(button, panel, { onOpen: fill, align: side ? 'right' : 'left' });
        // The ability of the built-in AI on this side, GRF's team difficulty, which steers every
        // player the built-in AI controls. Left alone it is the scenario's own.
        const name = side ? 'Right' : 'Left';
        const own = h('input', { type: 'checkbox', 'aria-label': `Set the ability of the ${name.toLowerCase()} side` });
        const range = h('input', {
            type: 'range', min: '0', max: '1', step: '0.05', value: '0.5', disabled: true,
            'aria-label': `${name} ability, from 0 to 1`,
        });
        const shown = h('output', { class: 'ability-value' }, "Scenario's");
        const sync = () => {
            const value = own.checked ? Number(range.value) : null;
            this.state.ability[side] = value;
            range.disabled = !own.checked;
            shown.textContent = value === null ? "Scenario's" : value.toFixed(2);
        };
        own.addEventListener('change', sync);
        range.addEventListener('input', sync);
        const ability = h('div', { class: 'ability', title: 'How well the built-in AI plays the players it controls, from 0 to 1' },
            h('label', { class: 'check' }, own, 'Ability'), range, shown);
        const root = h('div', { class: `side side-${side ? 'right' : 'left'}` },
            h('div', { class: 'side-label' }, side ? 'Right, plays away' : 'Left, plays at home'), button, ability);
        return { root, button, dropdown };
    }

    updateSetup() {
        if (!this.built) {
            return;
        }
        const s = this.state;
        [s.left, s.right].forEach((id, side) => {
            const t = this.team(id);
            const card = this.sides[side].button;
            if (!t) {
                card.replaceChildren(h('span', { class: 'team-card-empty' }, 'No teams yet'));
                return;
            }
            card.replaceChildren(
                kitShirt(t, side, 'large'),
                h('span', { class: 'team-card-text' },
                    h('span', { class: 'team-card-name' }, t.name),
                    h('span', { class: 'team-card-meta' }, t.kind === 'builtin' ? 'Built-in AI, any scenario' : `${FORMAT_NAMES[t.format] || t.format}, submission, controls ${controlLabel(t.control, t.format)}`)),
                h('span', { class: 'dd-caret', 'aria-hidden': 'true' }));
        });
        const l = this.team(s.left);
        const r = this.team(s.right);
        const clash = l && r && l.kind !== 'builtin' && r.kind !== 'builtin' && l.format && r.format && l.format !== r.format;
        let ok = this.scenarios.filter((sc) => fits(l, sc) && fits(r, sc));
        if (clash) {
            ok = ok.filter((sc) => sc.startsWith('academy_'));
            this.clash.replaceChildren(
                'These teams have different formats. Academy scenarios can accept both, '
                + 'when each control setting fits the scenario.');
        } else if (!l || !r) {
            this.clash.replaceChildren('Add a team in Teams to play a match.');
        }
        this.clash.hidden = !clash && !!(l && r);
        const keep = ok.includes(s.scenario) ? s.scenario
            : ok.includes(DEFAULT_SCENARIO) ? DEFAULT_SCENARIO
                : ok.includes('5_vs_5') ? '5_vs_5' : ok[0];
        this.scenarioSelect.replaceChildren(...(ok.length ? ok : ['No scenario fits both teams']).map((sc) =>
            h('option', { value: ok.length ? sc : '' }, ok.length ? scenarioName(sc) : sc)));
        this.scenarioSelect.value = keep || '';
        this.scenarioSelect.disabled = !ok.length;
        s.scenario = keep || null;
        this.describeScenario();
        this.playButton.disabled = !ok.length || !l || !r;
    }

    drawSeed() {
        const v = crypto.getRandomValues(new Uint32Array(1))[0];
        this.seedInput.value = String(v);
        this.state.seed = this.seedInput.value;
    }

    async play() {
        const s = this.state;
        const text = this.seedInput.value.trim();
        if (text !== '' && !/^\d+$/.test(text)) {
            this.app.hud.toast('The seed is a whole number from 0, or empty for a random one.', 'error');
            return;
        }
        this.playButton.disabled = true;
        try {
            await api.requestMatch({ left: s.left, right: s.right, scenario: s.scenario, seed: text === '' ? null : text, ability: s.ability.slice() });
            this.app.hud.toast(`${this.teamName(s.left)} against ${this.teamName(s.right)} is in the queue.`);
            this.playButton.classList.remove('kicked');
            void this.playButton.offsetWidth;
            this.playButton.classList.add('kicked');
            this.requests = await api.requests();
            this.renderQueue();
        } catch (err) {
            this.app.hud.toast(err.message, 'error', 7000);
        } finally {
            this.updateSetup();
        }
    }

    buildFilters() {
        const select = (key, label, options) => {
            const el = h('select', { 'aria-label': label }, options.map(([v, t]) => h('option', { value: v }, t)));
            el.value = this.filters[key];
            el.addEventListener('change', () => {
                this.filters[key] = el.value;
                this.renderQueue();
            });
            return el;
        };
        this.statusFilter = select('status', 'Status', [['all', 'Every state'], ['active', 'Waiting or playing'], ['done', 'Finished'], ['failed', 'Failed'], ['cancelled', 'Canceled']]);
        this.teamFilter = select('team', 'Team', [['all', 'Every team']]);
        this.scenarioFilter = select('scenario', 'Scenario', [['all', 'Every scenario']]);
        this.dateFilter = select('date', 'Date', [['any', 'Any date'], ['hour', 'Last hour'], ['today', 'Today'], ['week', 'Last 7 days']]);
        this.sortSelect = select('sort', 'Sort', [['newest', 'Newest first'], ['oldest', 'Oldest first']]);
        const bar = h('div', { class: 'filter-bar' }, this.statusFilter, this.teamFilter, this.scenarioFilter, this.dateFilter, this.sortSelect);
        queueMicrotask(() => {
            enhanceSelect(this.statusFilter, { label: 'Status', prefix: 'Show' });
            enhanceSelect(this.teamFilter, { label: 'Team', prefix: 'Team' });
            enhanceSelect(this.scenarioFilter, { label: 'Scenario', prefix: 'Scenario' });
            enhanceSelect(this.dateFilter, { label: 'Date', prefix: 'Date' });
            enhanceSelect(this.sortSelect, { label: 'Sort', prefix: 'Sort', align: 'right' });
        });
        return bar;
    }

    // The team and scenario filters list what the requests hold.
    refillFilters() {
        const ids = [...new Set(this.requests.flatMap((r) => [r.left, r.right]))];
        const scenarios = [...new Set(this.requests.map((r) => r.scenario))].sort();
        const fill = (select, key, all, entries) => {
            const keep = this.filters[key];
            const want = [['all', all], ...entries];
            const now = [...select.options].map((o) => `${o.value}|${o.textContent}`).join('\n');
            if (now !== want.map(([v, t]) => `${v}|${t}`).join('\n')) {
                select.replaceChildren(...want.map(([v, t]) => h('option', { value: v }, t)));
            }
            this.filters[key] = want.some(([v]) => v === keep) ? keep : 'all';
            select.value = this.filters[key];
        };
        fill(this.teamFilter, 'team', 'Every team', ids.map((id) => [id, this.teamName(id)]));
        fill(this.scenarioFilter, 'scenario', 'Every scenario', scenarios.map((s) => [s, scenarioName(s)]));
    }

    renderQueue() {
        if (!this.built) {
            return;
        }
        this.refillFilters();
        const f = this.filters;
        let rows = this.requests.filter((r) => {
            const st = stateOf(r);
            if (f.status === 'active' && st !== 'queued' && st !== 'running') {
                return false;
            }
            if (f.status !== 'all' && f.status !== 'active' && st !== f.status) {
                return false;
            }
            if (f.team !== 'all' && r.left !== f.team && r.right !== f.team) {
                return false;
            }
            if (f.date !== 'any') {
                const when = requestTime(r.id);
                const now = new Date();
                const from = f.date === 'hour' ? now.getTime() - 36e5
                    : f.date === 'today' ? new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
                        : now.getTime() - 7 * 864e5;
                if (!when || when.getTime() < from) {
                    return false;
                }
            }
            return f.scenario === 'all' || r.scenario === f.scenario;
        });
        // The server lists the requests newest first.
        if (f.sort === 'oldest') {
            rows = rows.slice().reverse();
        }
        if (!this.requests.length) {
            this.queueHost.replaceChildren(h('div', { class: 'empty-state' },
                h('p', { class: 'empty-title' }, 'No matches requested yet'),
                h('p', { class: 'muted' }, 'Choose teams and Play match. Watch opens the result.')));
            return;
        }
        if (!rows.length) {
            this.queueHost.replaceChildren(h('p', { class: 'muted pad' }, 'No requested match fits these filters.'));
            return;
        }
        // Rows are kept by request, so a progress bar moves smoothly instead of being redrawn.
        const old = new Map([...this.queueHost.querySelectorAll('.req-card')].map((el) => [el.dataset.id, el]));
        const list = this.queueHost.querySelector('.req-cards') || h('ul', { class: 'req-cards' });
        const items = rows.map((r) => {
            const key = `${r.id}|${stateOf(r)}|${r.match || ''}|${r.error || ''}`;
            let li = old.get(r.id);
            if (!li || li.dataset.key !== key) {
                li = this.requestCard(r);
                li.dataset.key = key;
            }
            const bar = li.querySelector('.progress .bar');
            if (bar) {
                const pct = Math.round((r.progress || 0) * 100);
                bar.style.width = `${pct}%`;
                const label = li.querySelector('.pct');
                if (label) {
                    label.textContent = `${pct} percent`;
                }
            }
            return li;
        });
        list.replaceChildren(...items);
        if (list.parentNode !== this.queueHost) {
            this.queueHost.replaceChildren(list);
        }
    }

    requestCard(r) {
        const st = stateOf(r);
        const when = requestTime(r.id);
        const side = (id, i) => {
            const t = this.team(id);
            return h('span', { class: 'req-team' }, t ? kitShirt(t, i) : null, h('span', { class: 'req-team-name' }, this.teamName(id)));
        };
        const actions = [];
        if (st === 'queued' || st === 'running') {
            actions.push(h('button', { type: 'button', class: 'btn small ghost danger', onclick: () => this.cancel(r) }, 'Cancel'));
        }
        if (st === 'done' && r.match) {
            actions.push(h('button', { type: 'button', class: 'btn small primary', onclick: () => this.app.openMatchInWatch(r.match) },
                icon('play'), 'Watch'));
        }
        const li = h('li', { class: `req-card state-${st}`, 'data-id': r.id },
            h('div', { class: 'req-main' },
                h('div', { class: 'req-teams' }, side(r.left, 0), h('span', { class: 'req-vs' }, 'vs'), side(r.right, 1)),
                h('div', { class: 'req-meta' },
                    h('span', { class: `state-pill ${st}` }, STATE_LABELS[st] || st),
                    h('span', {}, scenarioName(r.scenario)),
                    r.seed != null ? h('span', {}, `seed ${r.seed}`) : null,
                    Array.isArray(r.ability) && r.ability.some((v) => v !== null)
                        ? h('span', {}, `ability ${r.ability.map((v) => (v === null ? "scenario's" : Number(v).toFixed(2))).join(' and ')}`)
                        : null,
                    when ? h('span', {}, new Intl.DateTimeFormat(uiLocale(), { timeStyle: 'short', dateStyle: 'medium' }).format(when)) : null),
                st === 'queued' || st === 'running'
                    ? h('div', { class: 'req-progress' },
                        h('div', { class: `progress${st === 'queued' ? ' waiting' : ''}` }, h('div', { class: 'bar' })),
                        h('span', { class: 'pct muted small' }, st === 'queued' ? 'In the queue' : ''))
                    : null,
                st === 'failed' && r.error ? h('div', { class: 'error-text' }, r.error) : null),
            h('div', { class: 'req-actions' }, actions));
        if (st === 'queued') {
            li.querySelector('.pct').classList.remove('pct');
        }
        return li;
    }

    async cancel(r) {
        try {
            await api.cancelRequest(r.id);
            this.seen.add(r.id);
            this.app.hud.toast('Match canceled.');
        } catch (err) {
            this.app.hud.toast(err.status === 404 ? 'This match has already ended.' : err.message, 'error');
        }
        try {
            this.requests = await api.requests();
        } catch {
            // The next poll tries again.
        }
        this.renderQueue();
    }

    // Watches the queue in the background and tells when a match is ready or failed.
    async poll() {
        if (document.hidden || document.body.classList.contains('offline')) {
            return;
        }
        const active = this.requests.some((r) => r.state === 'queued' || r.state === 'running');
        if (!active && this.primed && !this.visible) {
            // Nothing runs, a slower check is enough while the page is elsewhere.
            this.idleTicks = (this.idleTicks || 0) + 1;
            if (this.idleTicks % 6) {
                return;
            }
        }
        let list;
        try {
            list = await api.requests();
        } catch {
            return;
        }
        for (const r of list) {
            const st = stateOf(r);
            if ((st === 'done' || st === 'failed' || st === 'cancelled') && !this.seen.has(r.id)) {
                this.seen.add(r.id);
                if (this.primed && st === 'done') {
                    this.app.hud.toast(`${this.teamName(r.left)} against ${this.teamName(r.right)} is ready to watch${this.visible ? '.' : ', open it from Matches.'}`, 'info', 6000);
                } else if (this.primed && st === 'failed') {
                    this.app.hud.toast(`A requested match failed. ${r.error || ''}`, 'error', 7000);
                }
            }
        }
        this.primed = true;
        const changed = JSON.stringify(list) !== JSON.stringify(this.requests);
        this.requests = list;
        if (changed && this.visible) {
            this.renderQueue();
        }
    }
}
