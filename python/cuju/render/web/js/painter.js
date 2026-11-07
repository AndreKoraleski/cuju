// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A small image editor on Fabric.js for shirt artwork and sponsor boards, with brush and eraser,
// shapes, text, pictures, move, scale and rotate, layers, undo and redo.

import {
    Canvas, Circle, FabricImage, IText, Line, PencilBrush, Rect, Triangle,
} from '../vendor/fabric/fabric.min.mjs';
import { h } from './library.js';

const HISTORY_LIMIT = 60;
// Every step of the history is a whole document with its pictures, so the steps kept are also
// bounded by their total length.
const HISTORY_CHARS = 48 * 1024 * 1024;
// Pictures are stored inside the document, so they are scaled down before they go in.
const PICTURE_SIDE = 1024;
const PICTURE_FILE_BYTES = 32 * 1024 * 1024;
// The server keeps an editor document of at most 8 MB.
const DOCUMENT_BYTES = 8 * 1024 * 1024;
// The object types this editor makes, as Fabric names them in a document.
const TYPES = new Set(['rect', 'circle', 'triangle', 'line', 'itext', 'i-text', 'image', 'path']);
// Fabric gradients carry a type of their own and no address.
const GRADIENTS = new Set(['linear', 'radial']);

const safeSource = (v) => typeof v === 'string' && (v.startsWith('data:image/') || v.startsWith('blob:'));

// Whether a value nested in an object holds only gradients and safe image addresses, so loading
// it fetches nothing from elsewhere.
function safeValue(v) {
    if (Array.isArray(v)) {
        return v.every(safeValue);
    }
    if (v && typeof v === 'object') {
        if (typeof v.type === 'string' && !GRADIENTS.has(v.type.toLowerCase())) {
            return false;
        }
        for (const [k, x] of Object.entries(v)) {
            if ((k === 'src' || k === 'source') && !safeSource(x)) {
                return false;
            }
            if (!safeValue(x)) {
                return false;
            }
        }
    }
    return true;
}

// Keeps from a saved document only the objects this editor makes, with pictures from data or blob
// addresses, and drops the rest. Returns the document and how many objects were dropped.
export function sanitizeDocument(text) {
    const raw = typeof text === 'string' ? JSON.parse(text) : text;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error('this is not a drawing of this editor');
    }
    const objects = Array.isArray(raw.objects) ? raw.objects : [];
    const kept = [];
    for (const o of objects) {
        if (!o || typeof o !== 'object' || typeof o.type !== 'string' || !TYPES.has(o.type.toLowerCase())) {
            continue;
        }
        const clean = { ...o };
        // Clipping, filters and nested objects are never made here and could load other things.
        delete clean.clipPath;
        delete clean.filters;
        delete clean.resizeFilter;
        delete clean.objects;
        if (clean.type.toLowerCase() === 'image' && !safeSource(clean.src)) {
            continue;
        }
        // The object's own type and picture were checked above, the values inside it are checked here.
        const rest = { ...clean };
        delete rest.src;
        delete rest.type;
        if (!safeValue(rest)) {
            continue;
        }
        kept.push(clean);
    }
    const doc = { version: typeof raw.version === 'string' ? raw.version : undefined, objects: kept };
    if (typeof raw.backgroundColor === 'string') {
        doc.backgroundColor = raw.backgroundColor;
    }
    return { doc, dropped: objects.length - kept.length };
}

// A picture file as a data address no longer than PICTURE_SIDE on its long side.
async function pictureUrl(file) {
    const bitmap = await createImageBitmap(file);
    try {
        const scale = Math.min(1, PICTURE_SIDE / Math.max(bitmap.width, bitmap.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        // WebP keeps transparency and stays small, browsers without the encoder give PNG.
        return canvas.toDataURL('image/webp', 0.92);
    } finally {
        bitmap.close();
    }
}

export class Painter {
    constructor() {
        this.dialog = document.createElement('dialog');
        this.dialog.className = 'painter';
        document.getElementById('app').appendChild(this.dialog);
        this.dialog.addEventListener('close', () => this.teardown());
        // Escape asks before it throws away unsaved work.
        this.dialog.addEventListener('cancel', (e) => {
            e.preventDefault();
            this.requestClose();
        });
    }

    // Opens the editor. The options are a title, a width and height, a guide image drawn under
    // the canvas and never saved, the json of a saved document to continue, and onSave, called
    // with the PNG blob and the json.
    async open(options) {
        if (this.dialog.open) {
            return;
        }
        this.options = options;
        this.mode = 'select';
        this.color = '#ffffff';
        this.size = 12;
        this.history = [];
        this.future = [];
        this.restoring = false;
        this.saving = false;
        this.queue = Promise.resolve();
        this.render();
        if (document.pointerLockElement) {
            document.exitPointerLock();
        }
        this.dialog.showModal();
        const el = this.dialog.querySelector('canvas');
        const canvas = new Canvas(el, {
            width: options.width, height: options.height, preserveObjectStacking: true,
            backgroundColor: '',
        });
        this.canvas = canvas;
        const surface = canvas.upperCanvasEl;
        surface.tabIndex = 0;
        surface.setAttribute('role', 'application');
        surface.setAttribute('aria-label', 'Drawing canvas. Arrows move. Space paints. Page Up and Page Down select objects. Plus and minus resize. Brackets rotate. Enter edits text.');
        this.cursor = { x: options.width / 2, y: options.height / 2, painting: false };
        surface.addEventListener('keydown', (e) => this.canvasKey(e));
        // Fabric types into a hidden textarea, which it puts in the body by default. The body is
        // inert behind a modal dialog, so the textarea goes inside the dialog instead.
        const textHost = this.dialog.querySelector('.painter-canvas');
        canvas.on('object:added', (e) => {
            if (e.target && 'hiddenTextareaContainer' in e.target) {
                e.target.hiddenTextareaContainer = textHost;
            }
        });
        this.onResize = () => this.fit();
        window.addEventListener('resize', this.onResize);
        canvas.on('path:created', (e) => {
            if (this.mode === 'eraser') {
                e.path.globalCompositeOperation = 'destination-out';
                e.path.selectable = false;
                canvas.requestRenderAll();
            }
            this.snapshot();
        });
        canvas.on('object:modified', () => this.snapshot());
        canvas.on('text:editing:exited', () => this.snapshot());
        if (options.json) {
            try {
                const { doc, dropped } = sanitizeDocument(options.json);
                await canvas.loadFromJSON(doc);
                if (this.canvas !== canvas) {
                    return;
                }
                canvas.renderAll();
                if (dropped) {
                    this.status(`${dropped} parts of the saved drawing were left out, they were not made by this editor.`, 'error');
                }
            } catch (err) {
                console.warn('saved drawing not loaded', err);
                this.status(`The saved drawing could not be loaded, ${err.message || err}. You start from an empty one.`, 'error');
            }
        }
        if (this.canvas !== canvas) {
            return;
        }
        this.snapshot();
        this.clean = this.history[this.history.length - 1];
        this.fit();
    }

    get dirty() {
        return !!this.canvas && this.history.length > 0 && this.history[this.history.length - 1] !== this.clean;
    }

    requestClose() {
        if (this.saving) {
            return;
        }
        if (this.dirty && !confirm('Close the drawing and lose the changes since it was last saved?')) {
            return;
        }
        this.dialog.close();
    }

    // A line under the canvas for errors and notes, the page's toasts sit behind the dialog.
    status(text, kind = 'info') {
        const el = this.dialog.querySelector('.painter-status');
        if (el) {
            el.textContent = text || '';
            el.className = `painter-status ${kind === 'error' ? 'error-text' : 'muted small'}`;
            el.hidden = !text;
        }
    }

    // Shows the canvas scaled down to the dialog when it is wider than the screen.
    fit() {
        const wrap = this.dialog.querySelector('.painter-stage');
        if (!wrap || !this.canvas) {
            return;
        }
        // The stage's padding is inside its client width, the canvas fits within it.
        const pad = parseFloat(getComputedStyle(wrap).paddingLeft) + parseFloat(getComputedStyle(wrap).paddingRight);
        const max = Math.min(Math.max(80, (wrap.clientWidth || this.options.width) - pad), this.options.width);
        const zoom = max / this.options.width;
        this.canvas.setDimensions({ width: `${this.options.width * zoom}px`, height: `${this.options.height * zoom}px` }, { cssOnly: true });
    }

    teardown() {
        window.removeEventListener('resize', this.onResize);
        if (this.canvas) {
            this.canvas.dispose();
            this.canvas = null;
        }
        this.history = [];
        this.future = [];
        this.dialog.replaceChildren();
    }

    render() {
        const o = this.options;
        const tool = (id, label, title) => h('button', {
            type: 'button', class: `tool${this.mode === id ? ' active' : ''}`, 'data-tool': id, title,
            'aria-pressed': this.mode === id ? 'true' : 'false',
            onclick: () => this.setMode(id),
        }, label);
        const action = (label, title, fn) => h('button', { type: 'button', class: 'tool', title, onclick: fn }, label);
        const color = h('input', { type: 'color', value: this.color, title: 'Color' });
        // Dragging in the color picker recolors live, the history takes one step when it closes.
        color.addEventListener('input', () => {
            this.color = color.value;
            this.applyBrush();
            const active = this.canvas && this.canvas.getActiveObjects();
            if (active && active.length) {
                for (const obj of active) {
                    if (obj.type === 'line' || (obj.type === 'path' && !obj.fill)) {
                        obj.set('stroke', this.color);
                    } else {
                        obj.set('fill', this.color);
                    }
                }
                this.canvas.requestRenderAll();
            }
        });
        color.addEventListener('change', () => this.snapshot());
        const size = h('input', { type: 'range', min: '1', max: '80', value: String(this.size), title: 'Brush size' });
        size.addEventListener('input', () => {
            this.size = Number(size.value);
            this.applyBrush();
        });
        const file = h('input', { type: 'file', accept: 'image/*', hidden: true });
        file.addEventListener('change', () => {
            const f = file.files[0];
            file.value = '';
            if (f) {
                this.addImage(f).catch((err) => this.status(`The picture could not be placed, ${err.message || err}.`, 'error'));
            }
        });
        const guide = o.guide ? `background-image:url(${o.guide});` : '';
        this.dialog.replaceChildren(h('div', { class: 'painter-box' },
            h('header', { class: 'editor-head' }, h('h2', {}, o.title),
                h('button', { type: 'button', class: 'icon-btn', 'aria-label': 'Close', onclick: () => this.requestClose() }, '✕')),
            h('div', { class: 'toolbar' },
                h('div', { class: 'tool-group' },
                    tool('select', 'Select', 'Move, scale and rotate'),
                    tool('brush', 'Brush', 'Paint freehand'),
                    tool('eraser', 'Eraser', 'Erase paint')),
                h('div', { class: 'tool-group' },
                    action('Rectangle', 'Add a rectangle', () => this.addShape('rect')),
                    action('Circle', 'Add a circle', () => this.addShape('circle')),
                    action('Triangle', 'Add a triangle', () => this.addShape('triangle')),
                    action('Line', 'Add a line', () => this.addShape('line')),
                    action('Text', 'Add text', () => this.addText()),
                    action('Picture', 'Place an image file', () => file.click()), file),
                h('div', { class: 'tool-group' }, h('label', { class: 'tool-field' }, 'Color', color), h('label', { class: 'tool-field' }, 'Size', size)),
                h('div', { class: 'tool-group' },
                    action('Forward', 'Bring the selection forward', () => this.layer(1)),
                    action('Backward', 'Send the selection backward', () => this.layer(-1)),
                    action('Delete', 'Delete the selection', () => this.removeSelection()),
                    action('Undo', 'Undo, Ctrl Z', () => this.undo()),
                    action('Redo', 'Redo, Ctrl Y', () => this.redo()),
                    action('Clear', 'Remove everything', () => this.clear()))),
            h('div', { class: 'painter-stage' }, h('div', { class: 'painter-canvas', style: guide }, h('canvas'))),
            o.note ? h('p', { class: 'muted small painter-note' }, o.note) : null,
            h('p', { class: 'muted small painter-note' },
                'Canvas keyboard. Arrows move, Space paints, Page Up and Page Down select objects. Plus and minus resize, brackets rotate, Enter edits text.'),
            h('p', { class: 'painter-status muted small', role: 'status', hidden: true }),
            h('div', { class: 'editor-actions' },
                h('span', { class: 'spacer' }),
                h('button', { type: 'button', class: 'btn', onclick: () => this.requestClose() }, 'Cancel'),
                h('button', { type: 'button', class: 'btn primary', onclick: () => this.save() }, 'Save'))));
        this.dialog.onkeydown = (e) => {
            const typing = e.target && (e.target.tagName === 'INPUT' || e.target.isContentEditable || e.target.tagName === 'TEXTAREA');
            if (e.key === 'Escape') {
                // Escape in the text being edited, Fabric's hidden textarea, only ends the editing.
                e.preventDefault();
                if (e.target.tagName !== 'TEXTAREA') {
                    this.requestClose();
                }
            } else if (!typing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    this.redo();
                } else {
                    this.undo();
                }
            } else if (!typing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
                e.preventDefault();
                this.redo();
            } else if ((e.key === 'Delete' || e.key === 'Backspace') && !typing && this.canvas) {
                const active = this.canvas.getActiveObject();
                if (active && !active.isEditing) {
                    e.preventDefault();
                    this.removeSelection();
                }
            }
        };
    }

    canvasKey(e) {
        const canvas = this.canvas;
        const cursor = this.cursor;
        if (!canvas || e.ctrlKey || e.metaKey || e.altKey) {
            return;
        }
        let obj = canvas.getActiveObject();
        const distance = e.shiftKey ? 20 : 2;
        const dx = e.key === 'ArrowRight' ? distance : e.key === 'ArrowLeft' ? -distance : 0;
        const dy = e.key === 'ArrowDown' ? distance : e.key === 'ArrowUp' ? -distance : 0;
        if (e.key === 'PageUp' || e.key === 'PageDown') {
            const objects = canvas.getObjects();
            const i = objects.indexOf(obj);
            const direction = e.key === 'PageDown' ? 1 : -1;
            obj = objects[(i + direction + objects.length) % objects.length];
            canvas.discardActiveObject();
            if (obj) {
                canvas.setActiveObject(obj);
            }
        } else if (e.key === ' ') {
            cursor.painting = !cursor.painting;
            canvas.discardActiveObject();
            this.status(`Paint ${cursor.painting ? 'started' : 'stopped'}, cursor ${Math.round(cursor.x)}, ${Math.round(cursor.y)}.`);
        } else if (dx || dy) {
            if (obj && this.mode === 'select') {
                obj.set({ left: obj.left + dx, top: obj.top + dy });
                obj.setCoords();
                this.snapshot();
            } else {
                const x = Math.max(0, Math.min(this.options.width, cursor.x + dx));
                const y = Math.max(0, Math.min(this.options.height, cursor.y + dy));
                if (cursor.painting) {
                    const line = new Line([cursor.x, cursor.y, x, y], {
                        stroke: this.color, strokeWidth: this.size, strokeLineCap: 'round',
                        selectable: this.mode !== 'eraser',
                        globalCompositeOperation: this.mode === 'eraser' ? 'destination-out' : 'source-over',
                    });
                    canvas.add(line);
                    this.snapshot();
                }
                cursor.x = x;
                cursor.y = y;
                this.status(`Cursor ${Math.round(x)}, ${Math.round(y)}. Paint ${cursor.painting ? 'on' : 'off'}.`);
            }
        } else if (obj && ['+', '=', '-', '[', ']'].includes(e.key)) {
            if (e.key === '[' || e.key === ']') {
                obj.rotate(obj.angle + (e.key === '[' ? -5 : 5));
            } else {
                const factor = e.key === '-' ? 0.95 : 1.05;
                obj.set({ scaleX: obj.scaleX * factor, scaleY: obj.scaleY * factor });
            }
            obj.setCoords();
            this.snapshot();
        } else if (obj && e.key === 'Enter' && typeof obj.enterEditing === 'function') {
            obj.enterEditing();
            obj.selectAll();
        } else {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        canvas.requestRenderAll();
        if (obj) {
            this.status(`Selected ${obj.type}.`);
        }
    }

    setMode(mode) {
        if (!this.canvas) {
            return;
        }
        this.mode = mode;
        for (const b of this.dialog.querySelectorAll('[data-tool]')) {
            b.classList.toggle('active', b.dataset.tool === mode);
            b.setAttribute('aria-pressed', b.dataset.tool === mode ? 'true' : 'false');
        }
        this.canvas.isDrawingMode = mode === 'brush' || mode === 'eraser';
        this.applyBrush();
    }

    applyBrush() {
        if (!this.canvas) {
            return;
        }
        if (!this.canvas.freeDrawingBrush) {
            this.canvas.freeDrawingBrush = new PencilBrush(this.canvas);
        }
        const brush = this.canvas.freeDrawingBrush;
        brush.width = this.size;
        // The eraser stroke is drawn as a cut out of everything under it, any color will do.
        brush.color = this.mode === 'eraser' ? 'rgba(0, 0, 0, 1)' : this.color;
    }

    center() {
        return { left: this.options.width / 2, top: this.options.height / 2 };
    }

    addShape(kind) {
        if (!this.canvas) {
            return;
        }
        const s = Math.min(this.options.width, this.options.height) * 0.4;
        const c = this.center();
        let obj;
        const base = { left: c.left, top: c.top, originX: 'center', originY: 'center', fill: this.color };
        if (kind === 'rect') {
            obj = new Rect({ ...base, width: s * 1.4, height: s });
        } else if (kind === 'circle') {
            obj = new Circle({ ...base, radius: s / 2 });
        } else if (kind === 'triangle') {
            obj = new Triangle({ ...base, width: s, height: s });
        } else {
            obj = new Line([c.left - s, c.top, c.left + s, c.top], { stroke: this.color, strokeWidth: Math.max(2, this.size / 2) });
        }
        this.setMode('select');
        this.canvas.add(obj);
        this.canvas.setActiveObject(obj);
        this.snapshot();
    }

    addText() {
        if (!this.canvas) {
            return;
        }
        const c = this.center();
        const text = new IText('Text', {
            left: c.left, top: c.top, originX: 'center', originY: 'center', fill: this.color,
            fontFamily: 'Georgia, serif', fontWeight: 'bold', fontSize: Math.round(this.options.height * 0.35),
            hiddenTextareaContainer: this.dialog.querySelector('.painter-canvas'),
        });
        this.setMode('select');
        this.canvas.add(text);
        this.canvas.setActiveObject(text);
        text.enterEditing();
        text.selectAll();
        this.snapshot();
    }

    async addImage(file) {
        if (file.size > PICTURE_FILE_BYTES) {
            throw new Error(`the file is ${Math.ceil(file.size / 1024 / 1024)} MB, the limit is ${PICTURE_FILE_BYTES / 1024 / 1024} MB`);
        }
        const canvas = this.canvas;
        const url = await pictureUrl(file);
        const img = await FabricImage.fromURL(url);
        if (!canvas || this.canvas !== canvas) {
            return;
        }
        const scale = Math.min(1, (this.options.width * 0.9) / img.width, (this.options.height * 0.9) / img.height);
        const c = this.center();
        img.set({ left: c.left, top: c.top, originX: 'center', originY: 'center', scaleX: scale, scaleY: scale });
        this.setMode('select');
        canvas.add(img);
        canvas.setActiveObject(img);
        this.snapshot();
    }

    layer(direction) {
        const obj = this.canvas && this.canvas.getActiveObject();
        if (!obj) {
            return;
        }
        if (direction > 0) {
            this.canvas.bringObjectForward(obj);
        } else {
            this.canvas.sendObjectBackwards(obj);
        }
        this.canvas.requestRenderAll();
        this.snapshot();
    }

    removeSelection() {
        const objs = this.canvas ? this.canvas.getActiveObjects() : [];
        if (!objs.length) {
            return;
        }
        this.canvas.discardActiveObject();
        for (const o of objs) {
            this.canvas.remove(o);
        }
        this.canvas.requestRenderAll();
        this.snapshot();
    }

    clear() {
        if (!this.canvas || !this.canvas.getObjects().length || !confirm('Remove everything from this drawing?')) {
            return;
        }
        this.canvas.remove(...this.canvas.getObjects());
        this.canvas.discardActiveObject();
        this.canvas.requestRenderAll();
        this.snapshot();
    }

    snapshot() {
        if (this.restoring || !this.canvas) {
            return;
        }
        const json = JSON.stringify(this.canvas.toJSON());
        if (this.history[this.history.length - 1] === json) {
            return;
        }
        this.history.push(json);
        this.future = [];
        this.trimHistory();
    }

    // Drops the oldest steps past the step count or the length budget, always keeping the latest.
    trimHistory() {
        let chars = this.history.reduce((n, s) => n + s.length, 0);
        while (this.history.length > 1 && (this.history.length > HISTORY_LIMIT || chars > HISTORY_CHARS)) {
            chars -= this.history.shift().length;
        }
    }

    async restore(json) {
        const canvas = this.canvas;
        if (!canvas) {
            return;
        }
        this.restoring = true;
        try {
            await canvas.loadFromJSON(json);
            canvas.renderAll();
        } finally {
            this.restoring = false;
        }
    }

    // Undo and redo run one after another, a second press waits for the first load to end.
    enqueue(step) {
        this.queue = this.queue.then(step).catch((err) => {
            console.warn('undo step failed', err);
            this.status(`That step could not be undone, ${err.message || err}.`, 'error');
        });
        return this.queue;
    }

    undo() {
        return this.enqueue(async () => {
            if (!this.canvas || this.history.length < 2) {
                return;
            }
            this.future.push(this.history.pop());
            await this.restore(this.history[this.history.length - 1]);
        });
    }

    redo() {
        return this.enqueue(async () => {
            if (!this.canvas || !this.future.length) {
                return;
            }
            const json = this.future.pop();
            this.history.push(json);
            await this.restore(json);
        });
    }

    async save() {
        if (!this.canvas || this.saving) {
            return;
        }
        // A pending undo finishes first, so the saved drawing is the one on screen.
        await this.queue;
        const canvas = this.canvas;
        if (!canvas) {
            return;
        }
        this.saving = true;
        try {
            canvas.discardActiveObject();
            canvas.renderAll();
            const url = canvas.toDataURL({ format: 'png', multiplier: 1 });
            const png = await (await fetch(url)).blob();
            const json = JSON.stringify(canvas.toJSON());
            if (new TextEncoder().encode(json).length > DOCUMENT_BYTES) {
                throw new Error('the drawing is over 8 MB, remove a picture or two and save again');
            }
            await this.options.onSave(png, json);
            this.clean = this.history[this.history.length - 1];
            this.dialog.close();
        } catch (err) {
            this.status(`The drawing could not be saved, ${err.message || err}.`, 'error');
        } finally {
            this.saving = false;
        }
    }
}
