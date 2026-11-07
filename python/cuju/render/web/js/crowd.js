// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The painted crowd. Every strip of the atlas is one row of seats seen from the pitch, a person
// to each seat, and a second atlas holds the same people on their feet for a goal. Shirts and
// scarves are painted grey and marked in a mask, so the stands take any colors without a repaint.

import * as THREE from 'three';

export const ATLAS = {
    // Only every fourth column holds a tall pose. Heights in meters above the tread that the calm
    // people of the other columns, the calm people of any column and GRF's people stay below.
    tallEvery: 4,
    seatedTop: 1.48,
    calmTop: 1.88,
    grfTop: 1.4,
    // Meters of one seat, of the painted part of a strip and of the whole strip with its margin.
    seat: 0.5,
    card: 112 / 48,
    strip: 128 / 48,
    seats: 64,
    strips: 8,
    // Pixels per meter at full scale.
    pxm: 48,
    // The grey of a shirt at its brightest fold, the shader divides by it to keep the shading.
    shirtGrey: 214,
};

const SKINS = ['#f4d0b4', '#eabd98', '#dba47e', '#c68c62', '#a8714a', '#8a5a3a', '#6a4129', '#4e2f1d'];
const HAIR = ['#141010', '#231812', '#3a2618', '#5a3c22', '#7c5530', '#a8834e', '#d2b47a', '#8c8c8c',
    '#c8c8c8', '#6e2c14'];
const LEGS = ['#1e2633', '#2b3a52', '#262626', '#3b3f45', '#4a5a78', '#2a2f26'];
const BACKGROUND = [28, 33, 41];

function rng(seed) {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function pick(rand, list) {
    return list[Math.floor(rand() * list.length)];
}

function weighted(rand, table) {
    let r = rand() * table.reduce((a, [, w]) => a + w, 0);
    for (const [name, w] of table) {
        r -= w;
        if (r <= 0) {
            return name;
        }
    }
    return table[0][0];
}

const CALM = [['sit', 58], ['lean', 10], ['clap', 9], ['crossed', 11], ['phone', 5]];
// Poses that rise above a seated head go only in every fourth column, so the shader skips the
// air above the heads of the other three without reading the atlas.
const CALM_TALL = [['stand', 50], ['armUp', 20], ['scarfUp', 22], ['sit', 8]];
const CHEER = [['armsUp', 42], ['fist', 18], ['scarfUp', 14], ['clapStand', 14], ['sitClap', 12]];

// One person for every seat of every strip, the same in both atlases.
function people() {
    const rand = rng(20261106);
    const out = [];
    for (let k = 0; k < ATLAS.strips * ATLAS.seats; k++) {
        const tall = (k % ATLAS.seats) % ATLAS.tallEvery === 0;
        const style = weighted(rand, [['short', 40], ['long', 14], ['bald', 8], ['cap', 14], ['beanie', 10],
            ['curly', 14]]);
        out.push({
            skin: pick(rand, SKINS),
            hair: rand() < 0.12 ? pick(rand, HAIR.slice(7, 9)) : pick(rand, HAIR),
            style,
            legs: pick(rand, LEGS),
            sleeves: rand() < 0.45,
            scarf: rand() < 0.32,
            build: 0.9 + rand() * 0.22,
            tall: 0.94 + rand() * 0.12,
            dx: (rand() - 0.5) * 0.06,
            calm: weighted(rand, tall ? CALM_TALL : CALM),
            cheer: weighted(rand, CHEER),
            lean: (rand() - 0.5) * 0.05,
            side: rand() < 0.5 ? -1 : 1,
        });
    }
    return out;
}

// Paints on three canvases at once, the colors, the coverage and the shirt and accent mask.
class Brush {
    constructor(width, height) {
        this.layers = [0, 1, 2].map(() => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            return canvas.getContext('2d', { willReadFrequently: true });
        });
        const [color, cover, mask] = this.layers;
        color.fillStyle = `rgb(${BACKGROUND.join(',')})`;
        color.fillRect(0, 0, width, height);
        cover.fillStyle = '#000';
        cover.fillRect(0, 0, width, height);
        mask.fillStyle = '#000';
        mask.fillRect(0, 0, width, height);
    }

    style(paint, role) {
        return [paint, '#fff', role === 'shirt' ? '#f00' : role === 'accent' ? '#0f0' : '#000'];
    }

    fill(path, paint, role = 'none') {
        const styles = this.style(paint, role);
        this.layers.forEach((ctx, i) => {
            ctx.fillStyle = styles[i];
            ctx.fill(path);
        });
    }

    stroke(path, width, paint, role = 'none') {
        const styles = this.style(paint, role);
        this.layers.forEach((ctx, i) => {
            ctx.lineWidth = width;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.strokeStyle = styles[i];
            ctx.stroke(path);
        });
    }

    // A tint on the colors alone, for shading that leaves the masks as they are.
    shade(path, paint) {
        this.layers[0].fillStyle = paint;
        this.layers[0].fill(path);
    }

    // RGBA bytes with the coverage as alpha, and RG bytes of the mask.
    read() {
        const [color, cover, mask] = this.layers;
        const { width, height } = color.canvas;
        const c = color.getImageData(0, 0, width, height).data;
        const a = cover.getImageData(0, 0, width, height).data;
        const m = mask.getImageData(0, 0, width, height).data;
        const rgba = new Uint8Array(width * height * 4);
        const rg = new Uint8Array(width * height * 2);
        for (let i = 0, j = 0, k = 0; i < c.length; i += 4, j += 4, k += 2) {
            rgba[j] = c[i];
            rgba[j + 1] = c[i + 1];
            rgba[j + 2] = c[i + 2];
            rgba[j + 3] = a[i];
            rg[k] = m[i];
            rg[k + 1] = m[i + 1];
        }
        return { rgba, rg };
    }
}

// Paints one person into its seat, in meters from the middle of the seat at the height of the
// tread, so the same code serves both scales.
function paintPerson(brush, p, pose, X, Y, pxm) {
    const g = (v) => `rgb(${v},${v},${v})`;
    const G = ATLAS.shirtGrey;
    const standing = !['sit', 'lean', 'clap', 'crossed', 'armUp', 'scarfUp', 'phone', 'sitClap'].includes(pose);
    const lift = standing ? 0.44 : pose === 'lean' ? -0.1 : 0;
    const t = p.tall;
    const bw = p.build;
    const hip = 0.5 + lift;
    const shoulder = (1.0 + lift) * (standing ? 1 : t) + (standing ? (t - 1) * 1.4 : 0);
    const head = shoulder + 0.17;
    const dx = p.dx + (pose === 'lean' ? p.lean : 0);
    const x = (m) => X + (dx + m) * pxm;
    const y = (m) => Y - m * pxm;
    const rect = (x0, h0, x1, h1, r) => {
        const path = new Path2D();
        path.roundRect(x(Math.min(x0, x1)), y(Math.max(h0, h1)), Math.abs(x1 - x0) * pxm, Math.abs(h1 - h0) * pxm, r * pxm);
        return path;
    };
    const ellipse = (cx, cy, rx, ry, a0 = 0, a1 = Math.PI * 2) => {
        const path = new Path2D();
        path.ellipse(x(cx), y(cy), rx * pxm, ry * pxm, 0, a0, a1);
        return path;
    };
    const line = (pts) => {
        const path = new Path2D();
        pts.forEach(([px, py], i) => (i ? path.lineTo(x(px), y(py)) : path.moveTo(x(px), y(py))));
        return path;
    };
    const sw = 0.19 * bw;
    const arm = 0.075 * bw;
    const sleeve = p.sleeves ? 'shirt' : 'none';
    const sleevePaint = p.sleeves ? g(Math.round(G * 0.8)) : p.skin;
    // Legs, mostly hidden behind the row in front but seen between the seats.
    if (standing) {
        brush.fill(rect(-0.15, 0.0, -0.02, hip + 0.04, 0.03), p.legs);
        brush.fill(rect(0.02, 0.0, 0.15, hip + 0.04, 0.03), p.legs);
    } else {
        brush.fill(rect(-0.17, 0.22, 0.17, hip + 0.06, 0.05), p.legs);
    }
    // Long hair falls behind the shoulders, so it goes first.
    if (p.style === 'long') {
        brush.fill(rect(-0.1, shoulder - 0.08, 0.1, head + 0.08, 0.06), p.hair);
    }
    // Arms hanging at the sides go behind the hands but in front of nothing else.
    const hang = ['sit', 'stand', 'lean', 'phone'].includes(pose);
    if (hang) {
        for (const s of [-1, 1]) {
            if (pose === 'phone' && s === p.side) {
                continue;
            }
            brush.stroke(line([[s * (sw - 0.01), shoulder - 0.06], [s * (sw + 0.035), hip + 0.12]]), arm * pxm, sleevePaint, sleeve);
            brush.fill(ellipse(s * (sw + 0.035), hip + 0.08, 0.04, 0.045), p.skin);
        }
    }
    // The shirt, grey with lit folds, which the shader colors.
    const torso = rect(-sw, hip, sw, shoulder + 0.02, 0.07);
    const grad = brush.layers[0].createLinearGradient(x(-sw), 0, x(sw), 0);
    grad.addColorStop(0, g(Math.round(G * 0.62)));
    grad.addColorStop(0.45, g(G));
    grad.addColorStop(1, g(Math.round(G * 0.7)));
    brush.fill(torso, grad, 'shirt');
    brush.shade(rect(-sw, hip, sw, hip + 0.1, 0.04), 'rgba(0,0,0,0.18)');
    // Neck and head.
    brush.fill(rect(-0.042, shoulder - 0.03, 0.042, shoulder + 0.08, 0.02), p.skin);
    brush.fill(ellipse(0, head, 0.082, 0.105), p.skin);
    brush.shade(ellipse(0, head - 0.06, 0.07, 0.04), 'rgba(0,0,0,0.12)');
    brush.shade(rect(-0.05, head + 0.005, 0.05, head + 0.022, 0.005), 'rgba(40,20,10,0.35)');
    // Hair or a hat, a hat counts as an accent in the team's colors.
    if (p.style === 'short' || p.style === 'long') {
        brush.fill(ellipse(0, head + 0.03, 0.088, 0.085, Math.PI, Math.PI * 2), p.hair);
    } else if (p.style === 'curly') {
        brush.fill(ellipse(0, head + 0.045, 0.097, 0.08, Math.PI * 0.95, Math.PI * 2.05), p.hair);
    } else if (p.style === 'cap') {
        brush.fill(ellipse(0, head + 0.035, 0.09, 0.08, Math.PI, Math.PI * 2), g(G), 'accent');
        brush.fill(rect(-0.1, head + 0.02, 0.1, head + 0.045, 0.01), g(Math.round(G * 0.75)), 'accent');
    } else if (p.style === 'beanie') {
        brush.fill(ellipse(0, head + 0.045, 0.09, 0.1, Math.PI, Math.PI * 2), g(G), 'accent');
        brush.fill(rect(-0.092, head + 0.025, 0.092, head + 0.06, 0.01), g(Math.round(G * 0.85)), 'shirt');
    }
    // A scarf around the neck, with an end over the chest.
    if (p.scarf && pose !== 'scarfUp') {
        brush.fill(rect(-0.12, shoulder - 0.05, 0.12, shoulder + 0.03, 0.03), g(G), 'accent');
        brush.fill(rect(0.02, shoulder - 0.34, 0.085, shoulder - 0.02, 0.01), g(Math.round(G * 0.9)), 'accent');
        brush.fill(rect(0.02, shoulder - 0.24, 0.085, shoulder - 0.19, 0), g(Math.round(G * 0.9)), 'shirt');
    }
    // Arms that move, drawn over the shirt.
    const sh = (s) => [s * (sw - 0.02), shoulder - 0.05];
    const hand = (hx, hy) => brush.fill(ellipse(hx, hy, 0.042, 0.048), p.skin);
    const raise = (s, hx, hy, ex, ey) => {
        brush.stroke(line([sh(s), [ex, ey], [hx, hy]]), arm * pxm, sleevePaint, sleeve);
        hand(hx, hy);
    };
    if (pose === 'clap' || pose === 'sitClap' || pose === 'clapStand') {
        const cy = shoulder - 0.2;
        raise(-1, -0.02, cy, -sw - 0.03, shoulder - 0.3);
        raise(1, 0.02, cy + 0.02, sw + 0.03, shoulder - 0.3);
    } else if (pose === 'crossed') {
        brush.fill(rect(-sw - 0.02, shoulder - 0.33, sw + 0.02, shoulder - 0.22, 0.05), sleevePaint, sleeve);
        brush.fill(ellipse(-sw + 0.02, shoulder - 0.275, 0.035, 0.04), p.skin);
        brush.fill(ellipse(sw - 0.02, shoulder - 0.275, 0.035, 0.04), p.skin);
    } else if (pose === 'armUp') {
        raise(p.side, p.side * 0.26, shoulder + 0.5, p.side * (sw + 0.07), shoulder + 0.2);
    } else if (pose === 'fist') {
        raise(p.side, p.side * 0.2, shoulder + 0.52, p.side * (sw + 0.1), shoulder + 0.18);
        raise(-p.side, -p.side * 0.05, shoulder - 0.2, -p.side * (sw + 0.03), shoulder - 0.3);
    } else if (pose === 'armsUp') {
        raise(-1, -0.27 - p.lean, shoulder + 0.52, -sw - 0.07, shoulder + 0.22);
        raise(1, 0.27 + p.lean, shoulder + 0.55, sw + 0.07, shoulder + 0.24);
    } else if (pose === 'phone') {
        raise(p.side, p.side * 0.06, head + 0.02, p.side * (sw + 0.05), shoulder - 0.2);
        brush.fill(rect(p.side * 0.06 - 0.03, head - 0.03, p.side * 0.06 + 0.03, head + 0.07, 0.01), '#0c0f14');
    } else if (pose === 'scarfUp') {
        const top = shoulder + 0.45;
        raise(-1, -0.2, top, -sw - 0.06, shoulder + 0.15);
        raise(1, 0.2, top, sw + 0.06, shoulder + 0.15);
        brush.fill(rect(-0.26, top - 0.06, 0.26, top + 0.06, 0.01), g(G), 'accent');
        for (const sx of [-0.18, -0.02, 0.14]) {
            brush.fill(rect(sx, top - 0.06, sx + 0.06, top + 0.06, 0), g(Math.round(G * 0.92)), 'shirt');
        }
        brush.fill(rect(-0.26, top - 0.06, -0.22, top - 0.2, 0.01), g(G), 'accent');
    }
}

const cache = new Map();

// The two atlases at a scale, painted once and kept for every stadium after.
export function paintCrowd(scale = 1) {
    if (cache.has(scale)) {
        return cache.get(scale);
    }
    const pxm = ATLAS.pxm * scale;
    const seatPx = ATLAS.seat * pxm;
    const stripPx = ATLAS.strip * pxm;
    const width = Math.round(ATLAS.seats * seatPx);
    const height = Math.round(ATLAS.strips * stripPx);
    const crowd = people();
    const atlas = { width, height };
    for (const variant of ['calm', 'cheer']) {
        const brush = new Brush(width, height);
        for (let s = 0; s < ATLAS.strips; s++) {
            const bottom = (s + 1) * stripPx;
            for (let c = 0; c < ATLAS.seats; c++) {
                const p = crowd[s * ATLAS.seats + c];
                paintPerson(brush, p, p[variant], (c + 0.5) * seatPx, bottom, pxm);
            }
        }
        atlas[variant] = brush.read();
    }
    cache.set(scale, atlas);
    return atlas;
}

// GRF's strip of people laid along every row of the atlas, with no mask.
export function grfCrowd(image, scale = 1) {
    const pxm = ATLAS.pxm * scale;
    const stripPx = ATLAS.strip * pxm;
    const width = Math.round(ATLAS.seats * ATLAS.seat * pxm);
    const height = Math.round(ATLAS.strips * stripPx);
    const brush = new Brush(width, height);
    // Six tiles of the strip across the atlas, each 1.35 meters high, which puts the heads where
    // seated heads are.
    const tileW = width / 6;
    const tileH = 1.35 * pxm;
    for (let s = 0; s < ATLAS.strips; s++) {
        const top = (s + 1) * stripPx - tileH;
        for (let k = 0; k < 6; k++) {
            brush.layers[0].drawImage(image, k * tileW, top, tileW, tileH);
        }
    }
    // The colors lie over the dark background, the coverage comes from a clean copy.
    const alpha = document.createElement('canvas');
    alpha.width = width;
    alpha.height = height;
    const actx = alpha.getContext('2d', { willReadFrequently: true });
    for (let s = 0; s < ATLAS.strips; s++) {
        const top = (s + 1) * stripPx - tileH;
        for (let k = 0; k < 6; k++) {
            actx.drawImage(image, k * tileW, top, tileW, tileH);
        }
    }
    const read = brush.read();
    const a = actx.getImageData(0, 0, width, height).data;
    for (let i = 3; i < read.rgba.length; i += 4) {
        read.rgba[i] = a[i];
    }
    return { width, height, calm: read, cheer: read };
}

export function atlasTextures(atlas, anisotropy = 4) {
    const make = (data, format, colorSpace) => {
        const tex = new THREE.DataTexture(data, atlas.width, atlas.height, format, THREE.UnsignedByteType);
        tex.colorSpace = colorSpace;
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.magFilter = THREE.LinearFilter;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.generateMipmaps = true;
        tex.anisotropy = anisotropy;
        tex.unpackAlignment = 1;
        tex.needsUpdate = true;
        return tex;
    };
    const calmColor = make(atlas.calm.rgba, THREE.RGBAFormat, THREE.SRGBColorSpace);
    const calmMask = make(atlas.calm.rg, THREE.RGFormat, THREE.NoColorSpace);
    const same = atlas.cheer === atlas.calm;
    return {
        calmColor,
        calmMask,
        cheerColor: same ? calmColor : make(atlas.cheer.rgba, THREE.RGBAFormat, THREE.SRGBColorSpace),
        cheerMask: same ? calmMask : make(atlas.cheer.rg, THREE.RGFormat, THREE.NoColorSpace),
        size: new THREE.Vector2(atlas.width, atlas.height),
    };
}
