// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The Teams page list and team uploads, and the small DOM helper. Matches between teams are
// requested on the Matches page, see matches.js.

import {
    MAX_UPLOAD_BYTES, api, controlForm, controlFromValue, controlLabel, filesFromDrop,
    filesFromInput, formatBytes, formatDate, packFolder,
} from './api.js';
import { normalizeStyle } from './kits.js';
import { kitShirt } from './matches.js';

const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') {
            el.className = v;
        } else if (k.startsWith('on')) {
            el.addEventListener(k.slice(2), v);
        } else if (v === true) {
            el.setAttribute(k, '');
        } else if (v !== false && v != null) {
            el.setAttribute(k, v);
        }
    }
    for (const c of children.flat()) {
        if (c != null && c !== false) {
            el.append(c instanceof Node ? c : document.createTextNode(String(c)));
        }
    }
    return el;
}

export { h };

function message(kind, text, ...extra) {
    return h('div', { class: `notice ${kind}` }, h('p', {}, text), ...extra);
}

// Moves the focus through a list with the arrow keys, Home and End, the items being focusable.
export function listKeys(list, itemSelector) {
    list.addEventListener('keydown', (e) => {
        const items = [...list.querySelectorAll(itemSelector)];
        const i = items.indexOf(document.activeElement);
        if (i < 0) {
            return;
        }
        let next = null;
        if (e.key === 'ArrowDown') {
            next = items[Math.min(items.length - 1, i + 1)];
        } else if (e.key === 'ArrowUp') {
            next = items[Math.max(0, i - 1)];
        } else if (e.key === 'Home') {
            next = items[0];
        } else if (e.key === 'End') {
            next = items[items.length - 1];
        }
        if (next) {
            e.preventDefault();
            next.focus();
        }
    });
}

// The left column of the Teams page, with the teams and a form to add one.
export class TeamsPage {
    constructor(app) {
        this.app = app;
        this.root = $('teams-side');
        this.teams = [];
        this.offline = false;
        this.selectedId = null;
        this.upload = null;
        this.refreshToken = 0;
    }

    get visible() {
        return !$('page-teams').hidden;
    }

    async loadTeams() {
        this.teams = await api.teams();
        return this.teams;
    }

    setOffline(off) {
        this.offline = off;
        document.body.classList.toggle('offline', off);
    }

    async refresh() {
        // Only the newest refresh draws, an older one that answers late is dropped.
        const token = ++this.refreshToken;
        let teams;
        try {
            teams = await api.teams();
            if (token !== this.refreshToken) {
                return;
            }
            this.teams = teams;
            this.setOffline(false);
        } catch (err) {
            if (token !== this.refreshToken) {
                return;
            }
            if (err.status === 0) {
                this.setOffline(true);
            }
            this.root.replaceChildren(message('error', err.message,
                h('p', { class: 'muted' }, 'Start the server with cuju render serve, then try again.'),
                h('button', { class: 'btn', onclick: () => this.refresh() }, 'Try again')));
            return;
        }
        const focused = document.activeElement && document.activeElement.closest && document.activeElement.closest('.teams-list > li');
        const focusId = focused ? focused.dataset.id : null;
        const list = teams.length ? h('ul', { class: 'list teams-list', 'aria-label': 'Teams' }, teams.map((t, i) => this.teamItem(t, i)))
            : message('empty', 'No teams yet. Add one below.');
        if (teams.length) {
            listKeys(list, '.teams-list > li');
        }
        // The upload form is built once, so a file chosen and a name typed survive every refresh.
        if (!this.upload) {
            this.upload = this.uploadForm();
        }
        this.root.replaceChildren(
            h('section', { class: 'panel-section' }, h('h3', {}, 'Teams'), list),
            h('section', { class: 'panel-section' }, h('h3', {}, 'Add a team'), this.upload),
            h('p', { class: 'muted small side-note' }, 'To play a match between teams, open ',
                h('button', { type: 'button', class: 'link', onclick: () => this.app.showPage('matches') }, 'Matches'), '.'));
        if (focusId) {
            const again = this.root.querySelector(`.teams-list > li[data-id="${CSS.escape(focusId)}"]`);
            if (again) {
                again.focus();
            }
        }
    }

    teamItem(t, i) {
        const style = normalizeStyle(t, i % 2);
        const choose = () => {
            if (this.app.editor.team && this.app.editor.team.id === t.id) {
                return;
            }
            if (!this.app.editor.confirmLeave()) {
                return;
            }
            this.selectedId = t.id;
            for (const el of this.root.querySelectorAll('.teams-list > li')) {
                el.classList.toggle('selected', el === item);
                el.setAttribute('aria-selected', el === item ? 'true' : 'false');
            }
            this.app.editTeam(t);
        };
        const item = h('li', {
            class: `item selectable${t.id === this.selectedId ? ' selected' : ''}`,
            'data-id': t.id,
            title: t.kind === 'builtin' ? null : `Added ${formatDate(t.created)}`,
            tabindex: '0',
            role: 'option',
            'aria-selected': t.id === this.selectedId ? 'true' : 'false',
            onclick: choose,
            onkeydown: (e) => {
                if ((e.key === 'Enter' || e.key === ' ') && e.target === item) {
                    e.preventDefault();
                    choose();
                }
            },
        },
        kitShirt(t, i % 2),
        h('div', { class: 'item-main' },
            h('div', { class: 'item-title' }, h('span', { class: 'ellipsis' }, t.name)),
            h('div', { class: 'item-meta' }, [
                t.kind === 'builtin' ? 'Built-in AI' : (t.format === '5v5' ? '5 against 5' : '11 against 11'),
                t.kind === 'builtin' ? null : `controls ${controlLabel(t.control, t.format)}`,
                style.model !== 'grf' ? (style.model === 'custom' ? 'own model' : style.model.replace(/^kaykit-/, '').replace(/-/g, ' ')) : null,
                t.sounds && t.sounds.length ? `${t.sounds.length} ${t.sounds.length === 1 ? 'sound' : 'sounds'}` : null,
            ].filter(Boolean).join(', '))),
        t.kind === 'builtin' ? null : h('button', {
            class: 'btn small ghost danger',
            type: 'button',
            'aria-label': `Delete ${t.name}`,
            title: 'Delete this team',
            onclick: async (e) => {
                e.stopPropagation();
                if (!confirm(`Delete the team "${t.name}"? Recordings that used it stay.`)) {
                    return;
                }
                try {
                    await api.deleteTeam(t.id);
                    this.app.hud.toast('Team deleted.');
                    if (this.selectedId === t.id) {
                        this.selectedId = null;
                    }
                    if (this.app.editor.team && this.app.editor.team.id === t.id) {
                        this.app.editor.close(true);
                    }
                } catch (err) {
                    this.app.hud.toast(err.message, 'error');
                }
                this.refresh();
            },
        }, 'Delete'));
        return item;
    }

    uploadForm() {
        const name = h('input', { type: 'text', placeholder: 'Team name', required: true, maxlength: '60' });
        const format = h('select', {}, h('option', { value: '11v11' }, '11 against 11'), h('option', { value: '5v5' }, '5 against 5'));
        const control = controlForm(format.value);
        let controlChanged = false;
        control.addEventListener('change', () => { controlChanged = true; });
        control.addEventListener('input', () => { controlChanged = true; });
        format.addEventListener('change', () => control.setFormat(format.value));
        const folderInput = h('input', { type: 'file', webkitdirectory: true, multiple: true, hidden: true });
        const fileInput = h('input', { type: 'file', accept: '.py,.gz,.tgz,.zip', hidden: true });
        const summary = h('div', { class: 'upload-summary muted' }, 'No folder chosen.');
        const send = h('button', { class: 'btn primary', type: 'submit', disabled: true }, 'Upload team');
        let pending = null;
        const choose = (p) => {
            pending = p;
            if (!p) {
                summary.textContent = 'No folder chosen.';
                send.disabled = true;
                return;
            }
            const too = p.size > MAX_UPLOAD_BYTES;
            const hasMain = p.kind === 'file' || p.files.some((f) => /(^|\/)main\.py$/.test(f.path));
            // Only the parts that apply are added, an absent one would print as null.
            summary.replaceChildren(...[
                h('strong', {}, p.label), `, ${p.kind === 'folder' ? `${p.files.length} files, ` : ''}${formatBytes(p.size)}`,
                too ? h('div', { class: 'error-text' }, 'This is over the 100 MB limit.') : null,
                !hasMain ? h('div', { class: 'error-text' }, 'The folder has no main.py.') : null,
            ].filter(Boolean));
            send.disabled = too || !hasMain;
            if (!name.value.trim()) {
                name.value = p.label.replace(/\.(py|tar\.gz|tgz|zip)$/i, '');
            }
        };
        const setFolder = (files, folderName) => {
            if (!files.length) {
                choose(null);
                return;
            }
            const label = folderName || files[0].path.split('/')[0] || 'submission';
            choose({ kind: 'folder', files, label, size: files.reduce((s, f) => s + f.file.size, 0) });
        };
        folderInput.addEventListener('change', () => setFolder(filesFromInput(folderInput.files)));
        fileInput.addEventListener('change', () => {
            const f = fileInput.files[0];
            if (f) {
                choose({ kind: 'file', file: f, label: f.name, size: f.size });
            }
        });
        const dropZone = h('div', { class: 'dropzone' },
            h('p', {}, 'Drop a submission folder or archive.'),
            h('div', { class: 'row' },
                h('button', { class: 'btn', type: 'button', onclick: () => folderInput.click() }, 'Choose folder'),
                h('button', { class: 'btn ghost', type: 'button', onclick: () => fileInput.click() }, 'Choose file')),
            folderInput, fileInput);
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('over');
            const items = e.dataTransfer.items;
            const first = items && items[0] && items[0].webkitGetAsEntry ? items[0].webkitGetAsEntry() : null;
            if (first && first.isFile && items.length === 1) {
                const f = e.dataTransfer.files[0];
                choose({ kind: 'file', file: f, label: f.name, size: f.size });
                return;
            }
            try {
                const { files, folder } = await filesFromDrop(items);
                setFolder(files, folder);
            } catch (err) {
                this.app.hud.toast(`The folder could not be read, ${err.message}`, 'error');
            }
        });
        const form = h('form', { class: 'form' }, h('div', { class: 'grid2' }, h('label', {}, 'Name', name), h('label', {}, 'Format', format)),
            h('div', {}, control), dropZone, summary,
            h('div', { class: 'form-actions' }, send));
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!pending || !name.value.trim()) {
                this.app.hud.toast('Give the team a name and choose its submission.', 'error');
                return;
            }
            send.disabled = true;
            send.textContent = 'Uploading';
            try {
                if (pending.kind === 'folder') {
                    const blob = await packFolder(pending.files);
                    await api.uploadTeam(name.value.trim(), pending.label, blob, format.value, controlChanged ? controlFromValue(control.value) : null);
                } else {
                    await api.uploadTeam(name.value.trim(), pending.label, pending.file, format.value, controlChanged ? controlFromValue(control.value) : null);
                }
                this.app.hud.toast(`Team ${name.value.trim()} added.`);
                // The form outlives refreshes, so it is emptied here for the next team.
                controlChanged = false;
                name.value = '';
                folderInput.value = '';
                fileInput.value = '';
                choose(null);
                send.textContent = 'Upload team';
                this.refresh();
            } catch (err) {
                this.app.hud.toast(err.message, 'error', 7000);
                send.disabled = false;
                send.textContent = 'Upload team';
            }
        });
        return form;
    }

    teamName(id) {
        const t = this.teams.find((x) => x.id === id);
        return t ? t.name : id;
    }
}
