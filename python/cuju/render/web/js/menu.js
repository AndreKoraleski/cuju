// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Dropdown menus in the manner of a game menu. A button opens a panel, the arrow keys move between
// its items, Enter or Space picks one, Escape closes the panel from anywhere inside it and gives
// the focus back to the button. A panel is open while it carries the class open, so code that
// toggles the class directly stays in step with the menu.

import { formatDay } from './api.js';
import { scenarioName, titleText } from './scenarios.js';

const ITEM = '[role^="menuitem"]:not([disabled]):not([aria-disabled="true"])';
const open = new Set();

function isItem(el) {
    return !!(el && el.matches && el.matches(ITEM));
}

// Closes every open menu but the one given, and the menus inside it.
function closeOthers(keep) {
    for (const d of [...open]) {
        if (d !== keep && !(keep && d.panel.contains(keep.button))) {
            d.close(false);
        }
    }
}

document.addEventListener('pointerdown', (e) => {
    for (const d of [...open]) {
        if (!d.panel.contains(e.target) && !d.button.contains(e.target)) {
            d.close(false);
        }
    }
}, true);

window.addEventListener('resize', () => {
    for (const d of open) {
        d.place();
    }
});

export class Dropdown {
    // The options are align, left or right, the edge of the button the panel lines up with,
    // onOpen, called before the panel shows so it can fill its items, and focusFirst, whether
    // opening moves the focus into the panel at once.
    constructor(button, panel, options = {}) {
        this.button = button;
        this.panel = panel;
        this.options = { align: 'left', focusFirst: true, ...options };
        if (!panel.id) {
            panel.id = `dd-${Math.random().toString(36).slice(2, 9)}`;
        }
        panel.classList.add('dd-panel');
        button.classList.add('dd-btn');
        button.setAttribute('aria-haspopup', panel.getAttribute('role') === 'dialog' ? 'dialog' : 'menu');
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-controls', panel.id);
        button.addEventListener('click', (e) => {
            e.preventDefault();
            this.toggle(undefined, e.detail === 0);
        });
        button.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || (this.isOpen && (e.key === 'Home' || e.key === 'End'))) {
                e.preventDefault();
                e.stopPropagation();
                this.open(true, e.key === 'ArrowUp' || e.key === 'End' ? -1 : 0);
            } else if (e.key === 'Escape' && this.isOpen) {
                e.preventDefault();
                e.stopPropagation();
                this.close(true);
            } else if (e.key === 'Enter' || e.key === ' ') {
                // The click that follows does the work, the page's own keys must not see it.
                e.stopPropagation();
            }
        });
        panel.addEventListener('keydown', (e) => this.onKey(e));
        // A picked item closes its menu, unless it asks to stay, as a check box does.
        panel.addEventListener('click', (e) => {
            const item = e.target.closest && e.target.closest('[role^="menuitem"]');
            if (item && panel.contains(item) && !item.hasAttribute('data-keep-open') && item.getAttribute('aria-disabled') !== 'true') {
                this.close(e.detail === 0);
            }
        });
        // The class may change from elsewhere, the button and the listeners follow it.
        new MutationObserver(() => this.sync()).observe(panel, { attributes: true, attributeFilter: ['class'] });
        this.sync();
    }

    get isOpen() {
        return this.panel.classList.contains('open');
    }

    sync() {
        const on = this.isOpen;
        this.button.setAttribute('aria-expanded', on ? 'true' : 'false');
        this.button.classList.toggle('open', on);
        if (on) {
            open.add(this);
            // A menu in use needs the mouse and keeps the controls of the page awake, also when
            // code opened it by its class.
            if (document.pointerLockElement) {
                document.exitPointerLock();
            }
            const app = document.getElementById('app');
            if (app) {
                app.classList.remove('idle');
            }
            this.place();
        } else {
            open.delete(this);
        }
    }

    toggle(force, keyboard = false) {
        const want = force === undefined ? !this.isOpen : force;
        if (want) {
            this.open(keyboard);
        } else {
            this.close(keyboard);
        }
    }

    // Opens the panel, and with focus moves into it, to the checked item or the first.
    open(focus = false, from = 0) {
        // A menu needs the mouse, the free camera's pointer lock lets it go.
        if (document.pointerLockElement) {
            document.exitPointerLock();
        }
        closeOthers(this);
        if (this.options.onOpen) {
            this.options.onOpen(this);
        }
        this.panel.classList.add('open');
        this.sync();
        // Opening from the keyboard moves the focus in, to the chosen item or the first one.
        if (focus) {
            const items = this.items();
            const checked = items.find((i) => i.getAttribute('aria-checked') === 'true');
            const target = from < 0 ? items[items.length - 1] : (checked || items[0]);
            if (target) {
                target.focus();
            } else {
                const first = this.panel.querySelector('input, select, button, [tabindex]');
                if (first) {
                    first.focus();
                }
            }
        }
    }

    close(returnFocus = false) {
        if (!this.isOpen) {
            return;
        }
        this.panel.classList.remove('open');
        this.sync();
        if (returnFocus) {
            this.button.focus();
        }
        if (this.options.onClose) {
            this.options.onClose(this);
        }
    }

    items() {
        return [...this.panel.querySelectorAll(ITEM)].filter((el) => el.offsetParent !== null);
    }

    onKey(e) {
        if (e.key === 'Escape') {
            // Escape closes the panel even from a field inside it, a select keeps its own list.
            e.preventDefault();
            e.stopPropagation();
            this.close(true);
            return;
        }
        if (e.key === 'Tab') {
            // Tab leaves a menu of items, a panel of fields keeps the focus moving inside it.
            if (this.panel.getAttribute('role') !== 'dialog') {
                this.close(false);
            }
            return;
        }
        const target = e.target;
        if (!isItem(target) && target !== this.panel) {
            // Fields in a panel keep their own keys, the page's keys do not see them.
            e.stopPropagation();
            return;
        }
        const items = this.items();
        const i = items.indexOf(target);
        let next = null;
        if (e.key === 'ArrowDown') {
            next = items[(i + 1) % items.length];
        } else if (e.key === 'ArrowUp') {
            next = items[(i - 1 + items.length) % items.length];
        } else if (e.key === 'Home' || e.key === 'PageUp') {
            next = items[0];
        } else if (e.key === 'End' || e.key === 'PageDown') {
            next = items[items.length - 1];
        } else if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.stopPropagation();
            if (isItem(target)) {
                target.click();
            }
            return;
        } else if (e.key.length === 1 && /\S/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
            // Typing a letter jumps to the next item that starts with it.
            const k = e.key.toLowerCase();
            const order = [...items.slice(i + 1), ...items.slice(0, i + 1)];
            next = order.find((el) => (el.dataset.label || el.textContent).trim().toLowerCase().startsWith(k)) || null;
        }
        e.stopPropagation();
        if (next) {
            e.preventDefault();
            next.focus();
        }
    }

    // Places the panel under the button, or over it when there is more room above, kept inside
    // the window so a narrow screen never scrolls sideways.
    place() {
        const p = this.panel;
        if (!p.isConnected) {
            return;
        }
        const r = this.button.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        p.style.maxWidth = `${vw - 16}px`;
        const w = p.offsetWidth;
        const h = p.scrollHeight;
        let left = this.options.align === 'right' ? r.right - w : r.left;
        left = Math.max(8, Math.min(left, vw - w - 8));
        const below = vh - r.bottom - 12;
        const above = r.top - 12;
        const up = h > below && above > below;
        const room = up ? above : below;
        p.style.left = `${Math.round(left)}px`;
        p.style.maxHeight = `${Math.max(120, Math.round(room))}px`;
        if (up) {
            p.style.top = 'auto';
            p.style.bottom = `${Math.round(vh - r.top + 4)}px`;
        } else {
            p.style.bottom = 'auto';
            p.style.top = `${Math.round(r.bottom + 4)}px`;
        }
        p.classList.toggle('up', up);
    }
}

// An element with attributes, event handlers and children, the shared helper of the menus.
function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') {
            node.className = v;
        } else if (k.startsWith('on') && typeof v === 'function') {
            node.addEventListener(k.slice(2), v);
        } else if (v === true) {
            node.setAttribute(k, '');
        } else if (v !== false && v != null) {
            node.setAttribute(k, v);
        }
    }
    for (const c of children.flat()) {
        if (c != null && c !== false) {
            node.append(c instanceof Node ? c : document.createTextNode(String(c)));
        }
    }
    return node;
}

// An item of a menu. The options are kbd, a key shown at its right, hint, a second line,
// checked, true or false for an item of a choice, onSelect, keepOpen, and disabled.
export function menuItem(label, options = {}) {
    const { kbd, hint, checked, onSelect, keepOpen, disabled, icon, className } = options;
    const role = checked === undefined ? 'menuitem' : (options.check ? 'menuitemcheckbox' : 'menuitemradio');
    const item = el('button', {
        type: 'button',
        role,
        class: `dd-item${className ? ` ${className}` : ''}`,
        tabindex: '-1',
        'aria-checked': checked === undefined ? null : (checked ? 'true' : 'false'),
        'aria-disabled': disabled ? 'true' : null,
        'data-keep-open': keepOpen ? true : null,
        'data-label': typeof label === 'string' ? label : null,
    },
    icon || null,
    el('span', { class: 'dd-text' }, el('span', { class: 'dd-label' }, label), hint ? el('span', { class: 'dd-hint' }, hint) : null),
    kbd ? el('kbd', {}, kbd) : null);
    if (onSelect) {
        item.addEventListener('click', (e) => {
            if (!disabled) {
                onSelect(e);
            }
        });
    }
    return item;
}

export function menuHeading(text, extra = null) {
    return el('div', { class: 'dd-heading', role: 'presentation' }, el('span', {}, text), extra);
}

export function menuNote(text) {
    return el('div', { class: 'dd-note', role: 'presentation' }, text);
}

// Turns a select into a dropdown of the same look as the menus. The select stays in the page,
// hidden, and remains the source of the value, so code that reads or sets its value or listens
// for its change event goes on working, and the button follows every change.
export function enhanceSelect(select, options = {}) {
    if (select.dataset.enhanced) {
        return select.enhanced;
    }
    select.dataset.enhanced = '1';
    const label = options.label || select.getAttribute('aria-label') || '';
    const button = el('button', {
        type: 'button',
        class: `dd-select${options.className ? ` ${options.className}` : ''}`,
        title: select.title || options.title || null,
    }, options.prefix ? el('span', { class: 'dd-prefix' }, options.prefix) : null,
    el('span', { class: 'dd-value' }), el('span', { class: 'dd-caret', 'aria-hidden': 'true' }));
    const panel = el('div', { class: 'dd-list', role: 'menu', 'aria-label': label || null });
    const wrap = el('span', { class: 'dd-wrap' }, button);
    select.after(wrap);
    // The panel is placed with fixed coordinates, so no scrolling box clips it, and it goes
    // with the select when a page rebuilds its form. A box that moves passes a host instead.
    (options.host || wrap).appendChild(panel);
    select.hidden = true;
    select.tabIndex = -1;
    const fill = () => {
        panel.replaceChildren(...[...select.children].flatMap((child) => {
            if (child.tagName === 'OPTGROUP') {
                return [menuHeading(child.label), ...[...child.children].map(optionItem)];
            }
            return [optionItem(child)];
        }));
    };
    const optionItem = (o) => menuItem(o.textContent, {
        checked: o.value === select.value,
        disabled: o.disabled,
        kbd: o.dataset.kbd || null,
        onSelect: () => {
            if (select.value !== o.value) {
                select.value = o.value;
                select.dispatchEvent(new Event('change', { bubbles: true }));
            }
        },
    });
    const sync = () => {
        const o = select.options[select.selectedIndex];
        button.querySelector('.dd-value').textContent = o ? o.textContent : '';
        button.setAttribute('aria-label', label ? `${label}, ${o ? o.textContent : ''}` : (o ? o.textContent : ''));
        button.disabled = select.disabled;
    };
    // The value setter of this select also updates the button, as code sets it directly.
    const proto = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
    Object.defineProperty(select, 'value', {
        configurable: true,
        get() {
            return proto.get.call(this);
        },
        set(v) {
            proto.set.call(this, v);
            sync();
        },
    });
    select.addEventListener('change', sync);
    new MutationObserver(sync).observe(select, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled'] });
    const dropdown = new Dropdown(button, panel, { onOpen: fill, align: options.align || 'left' });
    sync();
    select.enhanced = { button, panel, dropdown, sync, wrap };
    return select.enhanced;
}

// An icon of the page's sprite, parsed as SVG so its use element draws.
export function icon(name, className = 'icon') {
    const t = document.createElement('span');
    t.innerHTML = `<svg class="${className}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
    return t.firstChild;
}

export { el };

// The menus of the top bar and the Watch page, the pages on a narrow screen, settings, help, the
// View and Match menus and the speed. Returns the dropdowns by name.
export function bindTopMenus(app, { CAMERA_MODES, api, scorerAt }) {
    const $ = (id) => document.getElementById(id);
    const root = $('app');
    const menus = {};
    // Panels sit at the top of the application, out of every box that moves or clips.
    for (const id of ['page-menu', 'settings', 'view-menu', 'match-menu']) {
        root.appendChild($(id));
    }
    menus.pages = new Dropdown($('page-menu-btn'), $('page-menu'));
    menus.settings = new Dropdown($('open-settings'), $('settings'), { align: 'right' });
    menus.help = { close() {}, toggle() { app.showPage('help'); } };
    for (const id of ['theme', 'slowmo']) {
        if ($(id)) {
            enhanceSelect($(id), { className: 'in-panel', align: 'right' });
        }
    }

    const radio = (host, entries, current, pick) => {
        host.replaceChildren(...entries.map(([value, label, kbd]) => menuItem(label, {
            checked: value === current, kbd, onSelect: () => pick(value),
        })));
    };
    const fillView = () => {
        radio($('view-cameras'), CAMERA_MODES.map((m) => [m.id, m.label, m.id === 'ball' ? 'B' : m.id === 'owner' ? 'P' : null]),
            app.cameras.mode, (id) => app.setCamera(id));
        // The picture alone, as on television, and fullscreen, each a toggle of the view.
        if ($('view-screen')) {
            $('view-screen').replaceChildren(
                menuItem('Broadcast mode', {
                    hint: 'Only the picture and its graphics, as on television',
                    checked: !!app.tvMode, check: true, kbd: 'T', onSelect: () => app.setTvMode(!app.tvMode),
                }),
                menuItem('Fullscreen', { checked: !!document.fullscreenElement, check: true, kbd: 'F', onSelect: () => app.toggleFullscreen() }));
        }
        radio($('view-quality'), [['low', 'Low, for older laptops'], ['medium', 'Medium'], ['high', 'High, shadows and effects']],
            app.settings.quality, (q) => app.setQuality(q));
        radio($('view-theme'), [['system', 'Follow the system'], ['dark', 'Dark'], ['light', 'Light']],
            app.settings.theme, (t) => {
                const sel = $('theme');
                sel.value = t;
                sel.dispatchEvent(new Event('change', { bubbles: true }));
            });
    };
    menus.view = new Dropdown($('view-menu-btn'), $('view-menu'), { onOpen: fillView, align: 'right' });

    const matchMenu = $('match-menu');
    const section = (title) => ({ head: menuHeading(title), body: el('div', { class: 'dd-group' }) });
    const playerName = (k) => {
        const header = app.source.header;
        const p = header.players[k];
        const roster = app.styles && app.styles[p.team] && app.styles[p.team].players ? app.styles[p.team].players[p.index] : null;
        const name = roster && roster.name ? roster.name : (p.goalkeeper ? 'Goalkeeper' : `Player ${p.index + 1}`);
        const number = roster ? roster.number : (p.goalkeeper ? 1 : p.index + 1);
        return { name, number };
    };
    const goalItems = (body) => {
        const tl = app.timeline;
        const store = tl.store;
        if (!store.goals.length) {
            body.append(menuNote(tl.live ? 'No goals yet.' : 'No goals in this match.'));
        }
        for (const g of store.goals) {
            const f = store.get(g.index);
            const minute = f ? Math.floor(f.timeMs / 60000) + 1 : null;
            const team = app.source.header.teams[g.team] || {};
            let who = null;
            let own = false;
            try {
                const s = typeof scorerAt === 'function' ? scorerAt(store, g.index, g.team, app.source.header) : null;
                if (s && s.scorer >= 0) {
                    who = playerName(s.scorer);
                    own = s.ownGoal;
                }
            } catch {
                who = null;
            }
            body.append(menuItem(`${minute != null ? `${minute}'  ` : ''}${who ? who.name : 'Goal'}`, {
                hint: [team.name, who ? `number ${who.number}` : null, own ? 'own goal' : null].filter(Boolean).join(', ') || null,
                icon: el('span', { class: `goal-mark team-${g.team}`, 'aria-hidden': 'true' }),
                onSelect: () => {
                    if (app.timeline !== tl) {
                        return;
                    }
                    app.director.stop(false);
                    tl.seek(Math.max(tl.first, g.index - 6 * tl.sps));
                    app.lastGoal = g;
                    app.afterSeek();
                    app.updatePlayUi();
                    app.updateHash(true);
                },
            }));
        }
    };
    const fillMatch = () => {
        const parts = [];
        const open = section('Open');
        open.body.append(
            menuItem('Open a recording file', { hint: 'A .cjr file from this computer, or drop one on the page', onSelect: () => $('file-input').click() }),
            menuItem('Type an address', { hint: 'A live source or recording id', onSelect: () => setTimeout(() => $('address-input').focus(), 0) }),
            menuItem('Browse the recordings', { onSelect: () => app.showPage('recordings') }),
            menuItem('Play a new match', { hint: 'Two teams, played on this computer', onSelect: () => app.showPage('matches') }));
        parts.push(open.head, open.body);
        if (app.timeline && app.source) {
            const goals = section('Goals');
            goalItems(goals.body);
            parts.push(goals.head, goals.body);
            if (app.source.kind !== 'live' && app.timeline.store.size > 1) {
                const stats = section('This match');
                stats.body.append(menuItem('Match statistics', {
                    hint: 'Awards, grades and charts, Close returns here',
                    onSelect: () => {
                        if (app.page !== 'watch') {
                            app.showPage('watch');
                        }
                        app.postMatch.openStats();
                    },
                }));
                parts.push(stats.head, stats.body);
            }
        }
        const live = section('Live now');
        live.body.append(menuNote('Looking for live sources'));
        const recent = section('Recent recordings');
        recent.body.append(menuNote('Loading'));
        parts.push(live.head, live.body, recent.head, recent.body);
        matchMenu.replaceChildren(...parts);
        api.live().then((list) => {
            live.body.replaceChildren(...(list.length ? list.map((s) => menuItem(titleText(s.title, s.scenario) || s.id, {
                hint: [scenarioName(s.scenario), s.environment != null ? `environment ${s.environment}` : null].filter(Boolean).join(', '),
                icon: el('span', { class: 'live-dot', 'aria-hidden': 'true' }),
                onSelect: () => {
                    $('address-input').value = s.id;
                    app.openLive(s.id);
                },
            })) : [menuNote('No live sources. A script run with CUJU_RENDER=on streams one.')]));
            menus.match.place();
        }).catch((err) => live.body.replaceChildren(menuNote(err.message)));
        api.matches().then((list) => {
            const when = (m) => (typeof m.created === 'number' ? m.created * 1000 : Date.parse(m.created) || 0);
            const top = list.slice().sort((a, b) => when(b) - when(a)).slice(0, 6);
            recent.body.replaceChildren(...(top.length ? top.map((m) => menuItem(titleText(m.title, m.scenario) || m.id, {
                // The title usually names the teams already, the hint adds what it does not say.
                hint: [(m.teams || []).every((t) => (m.title || '').includes(t)) ? null : (m.teams || []).join(' against '),
                    m.score ? `${m.score[0]} to ${m.score[1]}` : null, scenarioName(m.scenario), formatDay(m.created)].filter(Boolean).join(', '),
                checked: app.source && app.source.kind === 'match' ? app.source.id === m.id : undefined,
                onSelect: () => app.openMatchInWatch(m.id),
            })) : [menuNote('No recordings yet.')]));
            menus.match.place();
        }).catch((err) => recent.body.replaceChildren(menuNote(err.message)));
    };
    menus.match = new Dropdown($('match-menu-btn'), matchMenu, { onOpen: fillMatch });
    menus.speed = enhanceSelect($('speed'), { label: 'Speed', className: 'speed-select', host: root }).dropdown;
    return menus;
}

const controls = 'button, a[href], input, select, textarea, [tabindex]';
const visible = (node) => !node.disabled && node.getClientRects().length &&
    !node.closest('[hidden], [inert]') && getComputedStyle(node).visibility !== 'hidden';
const focusables = (root) => [...root.querySelectorAll(controls)].filter((node) =>
    node.tabIndex >= 0 && visible(node));
const panels = new Map();

function activePanel() {
    return [...panels.keys()].filter(visible).at(-1);
}

function update() {
    for (const node of document.querySelectorAll('[role="dialog"], dialog')) {
        const shown = visible(node) && (node.tagName === 'DIALOG' ? node.open :
            node.classList.contains('dd-panel') ? node.classList.contains('open') : !node.hidden);
        if (shown && !panels.has(node)) {
            panels.set(node, document.activeElement);
            node.setAttribute('aria-modal', 'true');
            node.tabIndex = -1;
            if (!node.contains(document.activeElement)) {
                (focusables(node)[0] || node).focus();
            }
        } else if (!shown && panels.has(node)) {
            const opener = panels.get(node);
            panels.delete(node);
            if (opener && opener.isConnected && visible(opener)) {
                opener.focus();
            }
        }
    }
    for (const node of document.querySelectorAll('.post-card, .post-skip, #lineups')) {
        if (!node.hasAttribute('data-keyboard-action')) {
            node.dataset.keyboardAction = 'true';
            node.tabIndex = 0;
            node.setAttribute('role', 'button');
            node.setAttribute('aria-label', 'Continue, Escape finishes');
            node.addEventListener('keydown', (e) => {
                if (e.target === node && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    e.stopPropagation();
                    node.click();
                }
            });
        }
    }
    for (const color of document.querySelectorAll('input[type="color"]:not([data-keyboard-color])')) {
        color.dataset.keyboardColor = 'true';
        const text = document.createElement('input');
        text.type = 'text';
        text.className = 'color-value';
        text.value = color.value;
        text.maxLength = 7;
        text.setAttribute('aria-label', `${color.getAttribute('aria-label') || color.title ||
            color.closest('label')?.textContent.trim() || 'Color'}, hex value`);
        text.addEventListener('change', () => {
            const valid = /^#[0-9a-f]{6}$/i.test(text.value);
            text.setCustomValidity(valid ? '' : 'Use a hash and six hex digits');
            if (valid) {
                color.value = text.value;
                color.dispatchEvent(new Event('input', { bubbles: true }));
                color.dispatchEvent(new Event('change', { bubbles: true }));
            } else {
                text.reportValidity();
            }
        });
        const value = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
        Object.defineProperty(color, 'value', {
            configurable: true,
            get() { return value.get.call(this); },
            set(next) {
                value.set.call(this, next);
                text.value = value.get.call(this);
            },
        });
        color.addEventListener('input', () => { text.value = color.value; });
        const syncColor = () => {
            text.disabled = color.disabled;
            text.hidden = color.hidden;
        };
        new MutationObserver(syncColor).observe(color, {
            attributes: true, attributeFilter: ['disabled', 'hidden'],
        });
        syncColor();
        color.after(text);
    }
    for (const node of document.querySelectorAll('#stage, #field-preview')) {
        node.tabIndex = 0;
        node.setAttribute('aria-label', 'Picture, camera and playback keyboard controls');
    }
}

document.addEventListener('focusin', (e) => {
    const panel = activePanel();
    if (!panel || panel.contains(e.target)) {
        return;
    }
    const menu = e.target.closest('.dd-panel.open');
    if (menu && panel.querySelector(`[aria-controls="${menu.id}"]`)) {
        return;
    }
    (focusables(panel)[0] || panel).focus();
});

document.addEventListener('keydown', (e) => {
    const panel = activePanel();
    if (panel && e.key === 'Tab') {
        const nested = [...document.querySelectorAll('.dd-panel.open')].filter((node) =>
            !panel.contains(node) && panel.querySelector(`[aria-controls="${node.id}"]`));
        const items = [...focusables(panel), ...nested.flatMap(focusables)];
        const first = items[0] || panel;
        const last = items.at(-1) || panel;
        if (e.shiftKey && (document.activeElement === first || !items.includes(document.activeElement))) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && (document.activeElement === last || !items.includes(document.activeElement))) {
            e.preventDefault();
            first.focus();
        }
    }
}, true);

document.addEventListener('keydown', (e) => {
    if (!e.defaultPrevented && e.key === 'Escape' && !activePanel() &&
        e.target.closest('#team-editor') && !e.target.matches('select')) {
        document.querySelector('#team-editor [aria-label="Close the editor"]')?.click();
        e.preventDefault();
    }
    if (e.defaultPrevented || !e.target.matches('.folder, .chip-btn, .arena-card')) {
        return;
    }
    const items = [...e.target.parentElement.children].filter((node) =>
        node.matches('.folder, .chip-btn, .arena-card') && visible(node));
    const i = items.indexOf(e.target);
    const next = e.key === 'Home' ? items[0] : e.key === 'End' ? items.at(-1) :
        ['ArrowRight', 'ArrowDown'].includes(e.key) ? items[(i + 1) % items.length] :
        ['ArrowLeft', 'ArrowUp'].includes(e.key) ? items[(i + items.length - 1) % items.length] : null;
    if (next) {
        e.preventDefault();
        e.stopPropagation();
        next.focus();
    }
});

new MutationObserver(update).observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'class', 'open'],
});
update();
