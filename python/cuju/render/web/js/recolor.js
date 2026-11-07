// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The colors present in a model and their replacement. A model's colors come from the base colors
// of its materials, the texels of its textures that the mesh actually samples, and its vertex
// colors, each weighted by the surface area that shows it. Textures are cut into regions of
// smoothly varying texels first, so a gradient cell of an atlas such as KayKit's counts as one
// color. A replacement shifts every texel of a color by the same step in Lab, which keeps the
// lightness variation of gradients and shading painted into the texture. Each color also learns
// the part of the figure it covers from the names of meshes and materials, which gives it the kit
// color it follows, see assignRoles.

import * as THREE from 'three';
import { fromHex, hex, labToRgb, rgbToLab } from './kits.js';

// At most this many colors are listed, the rest stay as they are.
const MAX_COLORS = 12;
// Colors closer than this, Euclidean in Lab, are taken as one.
const MERGE = 10;
// A color the team names matches the listed color within this distance.
const MATCH = 14;
// Neighbor texels closer than this in every channel belong to one region.
const SMOOTH = 7;
// A region whose sampled colors spread wider than this is cut up texel by texel.
const SPREAD = 12;
// Textures larger than this on a side are read and recolored at this size.
const MAX_SIDE = 2048;
// Colors covering less than this share of the surface are not listed.
const MIN_SHARE = 0.004;

const analyses = new WeakMap();

function srgbOf(color) {
    const c = color.clone().convertLinearToSRGB();
    return [c.r * 255, c.g * 255, c.b * 255];
}

function dist2(p, q) {
    const a = p[0] - q[0];
    const b = p[1] - q[1];
    const c = p[2] - q[2];
    return a * a + b * b + c * c;
}

function imagePixels(image) {
    let w = image.width || image.videoWidth || 0;
    let h = image.height || image.videoHeight || 0;
    if (!w || !h) {
        return null;
    }
    const s = Math.min(1, MAX_SIDE / Math.max(w, h));
    w = Math.max(1, Math.round(w * s));
    h = Math.max(1, Math.round(h * s));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (image.data && image.width && !image.getContext) {
        // A DataTexture image holds raw RGBA bytes.
        const img = new ImageData(new Uint8ClampedArray(image.data.buffer.slice(0)), image.width, image.height);
        ctx.putImageData(img, 0, 0);
    } else {
        ctx.drawImage(image, 0, 0, w, h);
    }
    return ctx.getImageData(0, 0, w, h);
}

function find(parent, i) {
    while (parent[i] !== i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

// Regions of smoothly varying texels by union find over the four neighbors.
function regions(pixels) {
    const { width: w, height: h, data: d } = pixels;
    const n = w * h;
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) {
        parent[i] = i;
    }
    const near = (a, b) => Math.abs(d[a] - d[b]) <= SMOOTH && Math.abs(d[a + 1] - d[b + 1]) <= SMOOTH
        && Math.abs(d[a + 2] - d[b + 2]) <= SMOOTH;
    const join = (a, b) => {
        const ra = find(parent, a);
        const rb = find(parent, b);
        if (ra !== rb) {
            parent[Math.max(ra, rb)] = Math.min(ra, rb);
        }
    };
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = y * w + x;
            if (x + 1 < w && near(i * 4, i * 4 + 4)) {
                join(i, i + 1);
            }
            if (y + 1 < h && near(i * 4, (i + w) * 4)) {
                join(i, i + w);
            }
        }
    }
    const label = new Int32Array(n);
    const ids = new Map();
    for (let i = 0; i < n; i++) {
        const r = find(parent, i);
        let id = ids.get(r);
        if (id === undefined) {
            id = ids.size;
            ids.set(r, id);
        }
        label[i] = id;
    }
    return { label, count: ids.size };
}

// The parts of a figure that a color may cover. Legs split into shorts above and socks below.
const PARTS = ['shirt', 'legs', 'socks', 'trim', 'head', 'keep'];
// Names of meshes, materials and nodes that tell the part, the first match wins. KayKit names its
// meshes Body, ArmLeft, LegLeft, Cape, Head, Hat, Helmet, Eyes and so on, an uploaded model may name
// its materials shirt, pants or boots.
const PART_NAMES = [
    ['keep', /eye|glow|pupil|iris|teeth|tooth|mouth|tongue|skin|flesh|bone|metal|steel|iron|silver|gold|chrome|blade|sword|weapon/],
    ['head', /head|face|skull|jaw|hair|beard|nose/],
    ['trim', /hat|helmet|helm|hood|cap\b|crown|visor|mask|cape|cloak|scarf|mantle|quiver|collar|belt|trim/],
    ['socks', /boot|shoe|foot|feet|sock|toe|ankle|shin|calf/],
    ['legs', /pant|trouser|short|leg|thigh|hip|pelvis|skirt|kilt/],
    ['shirt', /shirt|jersey|body|torso|chest|spine|top|tunic|cloth|armou?r|coat|jacket|vest|robe|arm|sleeve|shoulder|hand/],
];

const LEGS = PARTS.indexOf('legs');
const SOCKS = PARTS.indexOf('socks');

function partByName(name) {
    const lower = String(name || '').toLowerCase();
    if (!lower) {
        return -1;
    }
    for (const [part, re] of PART_NAMES) {
        if (re.test(lower)) {
            return PARTS.indexOf(part);
        }
    }
    return -1;
}

// The part of a mesh drawn with a material, by the material's name, then the mesh's, then its
// parents', or -1 when no name tells.
function partOf(mesh, mat) {
    let p = partByName(mat && mat.name);
    for (let o = mesh; p < 0 && o; o = o.parent) {
        p = partByName(o.name);
    }
    return p;
}

// Visits every triangle of a mesh with its corner indices, its area in world units, its material
// and the height of its centroid.
function triangles(mesh, visit) {
    const g = mesh.geometry;
    const pos = g.attributes.position;
    if (!pos) {
        return;
    }
    const index = g.index;
    const count = index ? index.count : pos.count;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const groups = g.groups && g.groups.length && Array.isArray(mesh.material) ? g.groups
        : [{ start: 0, count, materialIndex: 0 }];
    const A = new THREE.Vector3();
    const B = new THREE.Vector3();
    const C = new THREE.Vector3();
    const m = mesh.matrixWorld;
    for (const group of groups) {
        const mat = mats[group.materialIndex || 0];
        if (!mat) {
            continue;
        }
        const end = Math.min(count, group.start + group.count);
        for (let t = group.start; t + 2 < end; t += 3) {
            const a = index ? index.getX(t) : t;
            const b = index ? index.getX(t + 1) : t + 1;
            const c = index ? index.getX(t + 2) : t + 2;
            A.fromBufferAttribute(pos, a).applyMatrix4(m);
            B.fromBufferAttribute(pos, b).applyMatrix4(m);
            C.fromBufferAttribute(pos, c).applyMatrix4(m);
            const y = (A.y + B.y + C.y) / 3;
            const area = B.sub(A).cross(C.sub(A)).length() / 2;
            if (area > 0) {
                visit(a, b, c, area, mat, y);
            }
        }
    }
}

// Builds the color items of a model, each a Lab color with a weight and where it lives.
function analyze(rig) {
    const scene = rig.scene;
    scene.updateMatrixWorld(true);
    const textures = new Map();
    const materials = new Map();
    const vertex = new Map();
    const items = [];
    scene.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry) {
            return;
        }
        const g = obj.geometry;
        const uv = g.attributes.uv;
        const colors = g.attributes.color;
        // Legs show shorts on their upper part and socks or boots below, split at 40 percent of
        // the height of the mesh.
        const box = new THREE.Box3().setFromBufferAttribute(g.attributes.position).applyMatrix4(obj.matrixWorld);
        const knee = box.min.y + (box.max.y - box.min.y) * 0.4;
        const parts = new Map();
        triangles(obj, (a, b, c, area, mat, y) => {
            let part = parts.get(mat);
            if (part === undefined) {
                part = partOf(obj, mat);
                parts.set(mat, part);
            }
            if (part === LEGS && y < knee) {
                part = SOCKS;
            }
            if (colors && mat.vertexColors) {
                let v = vertex.get(g);
                if (!v) {
                    v = { geometry: g, weights: new Float32Array(colors.count), part: new Int8Array(colors.count).fill(-1) };
                    vertex.set(g, v);
                }
                v.weights[a] += area / 3;
                v.weights[b] += area / 3;
                v.weights[c] += area / 3;
                v.part[a] = v.part[b] = v.part[c] = part;
            } else if (mat.map && mat.map.image && uv) {
                let t = textures.get(mat.map);
                if (!t) {
                    const pixels = imagePixels(mat.map.image);
                    if (!pixels) {
                        return;
                    }
                    t = { texture: mat.map, pixels, samples: [] };
                    textures.set(mat.map, t);
                }
                const u0 = uv.getX(a), v0 = uv.getY(a);
                const u1 = uv.getX(b), v1 = uv.getY(b);
                const u2 = uv.getX(c), v2 = uv.getY(c);
                // The corners, the edge midpoints and the centroid, so a thin strip of UVs and a
                // triangle squeezed to one point both sample what they show.
                const w = area / 7;
                const p = part;
                t.samples.push(u0, v0, w, p, u1, v1, w, p, u2, v2, w, p, (u0 + u1) / 2, (v0 + v1) / 2, w, p,
                    (u1 + u2) / 2, (v1 + v2) / 2, w, p, (u0 + u2) / 2, (v0 + v2) / 2, w, p,
                    (u0 + u1 + u2) / 3, (v0 + v1 + v2) / 3, w, p);
            } else if (mat.color) {
                let e = materials.get(mat);
                if (!e) {
                    e = { material: mat, weight: 0, parts: new Float64Array(PARTS.length) };
                    materials.set(mat, e);
                }
                e.weight += area;
                if (part >= 0) {
                    e.parts[part] += area;
                }
            }
        });
    });
    for (const e of materials.values()) {
        const lab = rgbToLab(srgbOf(e.material.color));
        items.push({ lab, weight: e.weight, kind: 'material', material: e.material, parts: e.parts });
    }
    for (const t of textures.values()) {
        analyzeTexture(t, items);
    }
    for (const v of vertex.values()) {
        analyzeVertex(v, items);
    }
    return { textures: [...textures.values()], materials: [...materials.values()], vertex: [...vertex.values()], items };
}

function analyzeTexture(t, items) {
    const { width: w, height: h, data: d } = t.pixels;
    const reg = regions(t.pixels);
    t.regions = reg;
    const sw = new Float64Array(reg.count);
    const sL = new Float64Array(reg.count * 3);
    const sQ = new Float64Array(reg.count);
    const NP = PARTS.length;
    const sP = new Float64Array(reg.count * NP);
    const lab = [0, 0, 0];
    const s = t.samples;
    const wrap = (x) => x - Math.floor(x);
    const texelLab = new Map();
    const labAt = (i) => {
        let c = texelLab.get(i);
        if (!c) {
            c = rgbToLab([d[i * 4], d[i * 4 + 1], d[i * 4 + 2]]);
            texelLab.set(i, c);
        }
        return c;
    };
    for (let k = 0; k < s.length; k += 4) {
        const x = Math.min(w - 1, Math.floor(wrap(s[k]) * w));
        const y = Math.min(h - 1, Math.floor(wrap(s[k + 1]) * h));
        const i = y * w + x;
        const r = reg.label[i];
        const c = labAt(i);
        const wt = s[k + 2];
        sw[r] += wt;
        if (s[k + 3] >= 0) {
            sP[r * NP + s[k + 3]] += wt;
        }
        sL[r * 3] += c[0] * wt;
        sL[r * 3 + 1] += c[1] * wt;
        sL[r * 3 + 2] += c[2] * wt;
        sQ[r] += (c[0] * c[0] * 0.25 + c[1] * c[1] + c[2] * c[2]) * wt;
    }
    t.samples = null;
    t.regionItem = new Int32Array(reg.count).fill(-1);
    t.mixed = new Uint8Array(reg.count);
    const bins = new Map();
    for (let k = 0; k < s.length; k += 4) {
        const x = Math.min(w - 1, Math.floor(wrap(s[k]) * w));
        const y = Math.min(h - 1, Math.floor(wrap(s[k + 1]) * h));
        const i = y * w + x;
        const r = reg.label[i];
        if (!sw[r]) {
            continue;
        }
        const mL = sL[r * 3] / sw[r];
        const ma = sL[r * 3 + 1] / sw[r];
        const mb = sL[r * 3 + 2] / sw[r];
        const variance = sQ[r] / sw[r] - (mL * mL * 0.25 + ma * ma + mb * mb);
        if (variance <= SPREAD * SPREAD) {
            continue;
        }
        // A region of many colors, a photograph or a painted texture, counts texel by texel in
        // bins of Lab.
        t.mixed[r] = 1;
        const c = labAt(i);
        const key = `${Math.round(c[0] / 6)},${Math.round(c[1] / 6)},${Math.round(c[2] / 6)}`;
        let b = bins.get(key);
        if (!b) {
            b = { lab: [0, 0, 0], weight: 0, kind: 'texels', texture: t, parts: new Float64Array(NP) };
            bins.set(key, b);
        }
        b.lab[0] += c[0] * s[k + 2];
        b.lab[1] += c[1] * s[k + 2];
        b.lab[2] += c[2] * s[k + 2];
        b.weight += s[k + 2];
        if (s[k + 3] >= 0) {
            b.parts[s[k + 3]] += s[k + 2];
        }
    }
    for (let r = 0; r < reg.count; r++) {
        if (!sw[r] || t.mixed[r]) {
            continue;
        }
        lab[0] = sL[r * 3] / sw[r];
        lab[1] = sL[r * 3 + 1] / sw[r];
        lab[2] = sL[r * 3 + 2] / sw[r];
        t.regionItem[r] = items.length;
        items.push({ lab: lab.slice(), weight: sw[r], kind: 'region', texture: t, region: r, parts: sP.slice(r * NP, r * NP + NP) });
    }
    for (const b of bins.values()) {
        b.lab = b.lab.map((v) => v / b.weight);
        items.push(b);
    }
}

function analyzeVertex(v, items) {
    const attr = v.geometry.attributes.color;
    const c = new THREE.Color();
    v.labs = new Float32Array(attr.count * 3);
    const bins = new Map();
    for (let i = 0; i < attr.count; i++) {
        c.setRGB(attr.getX(i), attr.getY(i), attr.getZ(i));
        const lab = rgbToLab(srgbOf(c));
        v.labs.set(lab, i * 3);
        const wt = v.weights[i];
        if (!wt) {
            continue;
        }
        const key = `${Math.round(lab[0] / 6)},${Math.round(lab[1] / 6)},${Math.round(lab[2] / 6)}`;
        let b = bins.get(key);
        if (!b) {
            b = { lab: [0, 0, 0], weight: 0, kind: 'vertex', parts: new Float64Array(PARTS.length) };
            bins.set(key, b);
        }
        b.lab[0] += lab[0] * wt;
        b.lab[1] += lab[1] * wt;
        b.lab[2] += lab[2] * wt;
        b.weight += wt;
        if (v.part[i] >= 0) {
            b.parts[v.part[i]] += wt;
        }
    }
    for (const b of bins.values()) {
        b.lab = b.lab.map((x) => x / b.weight);
        items.push(b);
    }
}

// Merges the items greedily, heaviest first, into colors.
function cluster(items) {
    const order = items.map((_, i) => i).sort((a, b) => items[b].weight - items[a].weight);
    const clusters = [];
    const of = new Int32Array(items.length).fill(-1);
    for (const i of order) {
        const it = items[i];
        let best = -1;
        let bestD = MERGE * MERGE;
        for (let c = 0; c < clusters.length; c++) {
            const d = dist2(clusters[c].lab, it.lab);
            if (d < bestD) {
                bestD = d;
                best = c;
            }
        }
        if (best < 0) {
            clusters.push({ lab: it.lab.slice(), weight: it.weight });
            of[i] = clusters.length - 1;
        } else {
            const cl = clusters[best];
            const total = cl.weight + it.weight;
            for (let k = 0; k < 3; k++) {
                cl.lab[k] = (cl.lab[k] * cl.weight + it.lab[k] * it.weight) / total;
            }
            cl.weight = total;
            of[i] = best;
        }
    }
    return { clusters, of };
}

// Skin tones, which a team rarely means to recolor, so they never lead as main colors.
export function skinLike(lab) {
    return lab[0] > 35 && lab[0] < 92 && lab[1] > 8 && lab[1] < 36 && lab[2] > 12 && lab[2] < 42 && lab[2] > lab[1] * 0.7;
}

// Gray of little chroma, as armor, helmets and blades are painted.
function metalLike(lab) {
    return Math.hypot(lab[1], lab[2]) < 7 && lab[0] > 30 && lab[0] < 80;
}

// The kit colors that roles follow.
export const ROLE_KIT = { shirt: 'primary', trim: 'secondary', shorts: 'shorts', socks: 'socks' };
const CLOTHING = { shirt: 'shirt', legs: 'shorts', socks: 'socks', trim: 'trim' };

// Gives each listed color the kit color it follows, shirt, trim, shorts or socks, by the part of
// the figure it covers most, or null for a color kept as it is, skin, bone, faces, eyes, hair,
// metal and small accents. The heaviest color above the legs takes the shirt, so the color that
// fills most of the figure shows the team, the other colors of the body, capes, hoods and hats
// the trim, and colors of the legs the shorts and socks. A model whose names tell nothing gives
// its colors the shirt, shorts, trim and socks by surface share.
function assignRoles(colors) {
    const byShare = ['shirt', 'shorts', 'trim', 'socks'];
    let unnamed = 0;
    let shirt = false;
    for (const c of [...colors].sort((x, y) => y.weight - x.weight)) {
        const p = c.parts;
        const head = p.head + p.keep;
        const chroma = Math.hypot(c.lab[1], c.lab[2]);
        c.role = null;
        // A strong color on the head is a hood or a hat, a soft one is skin, bone or hair.
        if (c.weight < 0.01 || p.keep >= 0.3 || (head >= 0.2 && (chroma < 30 || c.skin)) || (c.skin && head > 0.03)
            || metalLike(c.lab)) {
            continue;
        }
        let best = null;
        let bestShare = 0.05;
        for (const part of Object.keys(CLOTHING)) {
            if (p[part] > bestShare) {
                bestShare = p[part];
                best = part;
            }
        }
        if (!best) {
            c.role = unnamed < byShare.length ? byShare[unnamed++] : null;
        } else if (best === 'legs' || best === 'socks') {
            c.role = CLOTHING[best];
        } else {
            c.role = shirt ? 'trim' : 'shirt';
        }
        shirt = shirt || c.role === 'shirt';
    }
    if (!shirt) {
        // A figure dressed only in a cape or a hood shows the shirt color there.
        const first = colors.find((c) => c.role);
        if (first) {
            first.role = 'shirt';
        }
    }
}

// The color pairs that dress a model in a kit, each color with a role taking its kit color.
export function kitPairs(rig, kit) {
    if (!rig || !kit) {
        return {};
    }
    try {
        const out = {};
        for (const c of colorsOf(rig).colors) {
            const target = c.role && kit[ROLE_KIT[c.role]];
            if (target) {
                out[c.hex] = hex(target);
            }
        }
        return out;
    } catch (err) {
        console.warn('model colors not read', err);
        return {};
    }
}

// The analysis of a rig, made once and kept while the rig lives.
function colorsOf(rig) {
    let a = analyses.get(rig);
    if (a) {
        return a;
    }
    a = analyze(rig);
    const { clusters, of } = cluster(a.items);
    for (const c of clusters) {
        c.parts = new Float64Array(PARTS.length);
    }
    a.items.forEach((it, i) => {
        if (it.parts && of[i] >= 0) {
            for (let p = 0; p < PARTS.length; p++) {
                clusters[of[i]].parts[p] += it.parts[p];
            }
        }
    });
    const total = clusters.reduce((s, c) => s + c.weight, 0) || 1;
    const ranked = clusters.map((c, i) => ({ ...c, i })).sort((x, y) => y.weight - x.weight)
        .filter((c, n) => n < MAX_COLORS && c.weight / total >= MIN_SHARE);
    const slot = new Int32Array(clusters.length).fill(-1);
    ranked.forEach((c, n) => {
        slot[c.i] = n;
    });
    // The two heaviest colors that are not skin lead as the main colors, the rest follow by area.
    const main = ranked.filter((c) => !skinLike(c.lab)).slice(0, 2);
    const ordered = [...main, ...ranked.filter((c) => !main.includes(c))];
    a.colors = ordered.map((c) => {
        const rgb = labToRgb(c.lab).map(Math.round);
        const parts = {};
        PARTS.forEach((p, k) => {
            parts[p] = c.weight ? c.parts[k] / c.weight : 0;
        });
        return { hex: hex(rgb), weight: c.weight / total, lab: c.lab, main: main.includes(c), skin: skinLike(c.lab), parts };
    });
    assignRoles(a.colors);
    ordered.forEach((c, n) => {
        slot[c.i] = n;
    });
    a.items.forEach((it, i) => {
        it.slot = slot[of[i]];
    });
    for (const t of a.textures) {
        labelTexels(t, a);
    }
    for (const v of a.vertex) {
        v.slots = new Int16Array(v.labs.length / 3).fill(-1);
        for (let i = 0; i < v.slots.length; i++) {
            v.slots[i] = nearest(a.colors, [v.labs[i * 3], v.labs[i * 3 + 1], v.labs[i * 3 + 2]]);
        }
    }
    analyses.set(rig, a);
    return a;
}

function nearest(colors, lab, limit = MERGE * 1.5) {
    let best = -1;
    let bestD = limit * limit;
    for (let c = 0; c < colors.length; c++) {
        const d = dist2(colors[c].lab, lab);
        if (d < bestD) {
            bestD = d;
            best = c;
        }
    }
    return best;
}

// Gives every texel the color it belongs to and an index into a table of distinct pairs of texel
// color and color slot, so a recolor converts each distinct texel color once.
function labelTexels(t, a) {
    const { width: w, height: h, data: d } = t.pixels;
    const n = w * h;
    const regionSlot = new Int16Array(t.regions.count).fill(-1);
    for (let r = 0; r < t.regions.count; r++) {
        const item = t.regionItem[r];
        if (item >= 0) {
            regionSlot[r] = a.items[item].slot;
        }
    }
    const combo = new Int32Array(n).fill(-1);
    const keys = new Map();
    const comboSlot = [];
    const comboRgb = [];
    for (let i = 0; i < n; i++) {
        const r = t.regions.label[i];
        let s = regionSlot[r];
        const rgb = (d[i * 4] << 16) | (d[i * 4 + 1] << 8) | d[i * 4 + 2];
        if (t.mixed[r]) {
            s = -2;
        }
        if (s === -1) {
            continue;
        }
        const key = s * 16777216 + rgb;
        let id = keys.get(key);
        if (id === undefined) {
            id = comboSlot.length;
            keys.set(key, id);
            comboSlot.push(s);
            comboRgb.push(rgb);
        }
        combo[i] = id;
    }
    const m = comboSlot.length;
    // Only the labeled texels are visited when recoloring.
    let count = 0;
    for (let i = 0; i < n; i++) {
        if (combo[i] >= 0) {
            count++;
        }
    }
    t.texels = new Int32Array(count);
    t.texelCombo = new Int32Array(count);
    for (let i = 0, j = 0; i < n; i++) {
        if (combo[i] >= 0) {
            t.texels[j] = i;
            t.texelCombo[j++] = combo[i];
        }
    }
    t.comboSlot = new Int16Array(m);
    t.comboLab = new Float32Array(m * 3);
    for (let c = 0; c < m; c++) {
        const rgb = comboRgb[c];
        const lab = rgbToLab([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255]);
        t.comboLab.set(lab, c * 3);
        // Texels of a mixed region join the nearest listed color, if any is close.
        t.comboSlot[c] = comboSlot[c] === -2 ? nearest(a.colors, lab, MATCH) : comboSlot[c];
    }
    t.regions = null;
}

// The colors present in a model, each with its hex, its weight, the weights summing to about
// one, and whether it is a main color or a skin tone. The first two are the model's main colors,
// its two heaviest that are not skin, and the rest follow heaviest first.
export function modelColors(rig) {
    if (!rig) {
        return [];
    }
    try {
        return colorsOf(rig).colors.map((c) => ({ hex: c.hex, weight: c.weight, main: c.main, skin: c.skin, role: c.role, parts: { ...c.parts } }));
    } catch (err) {
        console.warn('model colors not read', err);
        return [];
    }
}

// The main colors of a model as hex strings, the heaviest first.
export function mainColors(rig, n = 2) {
    return modelColors(rig).slice(0, n).map((c) => c.hex);
}

// The Lab step of every listed color for the pairs given, null for colors left alone.
function steps(a, pairs) {
    const out = new Array(a.colors.length).fill(null);
    for (const [from, to] of Object.entries(pairs || {})) {
        const slot = nearest(a.colors, rgbToLab(fromHex(from)), MATCH);
        if (slot < 0) {
            continue;
        }
        const target = rgbToLab(fromHex(to));
        const c = a.colors[slot].lab;
        out[slot] = [target[0] - c[0], target[1] - c[1], target[2] - c[2]];
    }
    return out;
}

// The colors of a model as a team shows them, each listed color replaced by its pair.
export function shownColors(rig, pairs) {
    const a = colorsOf(rig);
    const st = steps(a, pairs);
    return a.colors.map((c, i) => {
        const s = st[i];
        const lab = s ? [c.lab[0] + s[0], c.lab[1] + s[1], c.lab[2] + s[2]] : c.lab;
        return { hex: hex(labToRgb(lab).map(Math.round)), weight: c.weight, lab, main: c.main, skin: c.skin };
    });
}

// Recolored materials and textures of one model for one team, shared by every instance of it.
// Changing the pairs rewrites them in place, so a preview follows a color picker quickly.
export class Recolor {
    constructor(rig, pairs = {}) {
        this.rig = rig;
        this.analysis = colorsOf(rig);
        this.copies = new Map();
        this.geometries = new Map();
        this.textures = new Map();
        this.instances = [];
        this.pairs = {};
        this.set(pairs);
    }

    // Puts the recolored materials on an instance of the rig and remembers it for later changes.
    apply(instance) {
        this.instances.push(instance);
        this.swap(instance.root);
    }

    forget(instance) {
        this.instances = this.instances.filter((i) => i !== instance);
    }

    swap(root) {
        root.traverse((obj) => {
            if (!obj.isMesh) {
                return;
            }
            const pick = (m) => this.copies.get(m) || m;
            obj.material = Array.isArray(obj.material) ? obj.material.map(pick) : pick(obj.material);
            const g = this.geometries.get(obj.geometry);
            if (g) {
                obj.geometry = g;
            }
        });
    }

    set(pairs) {
        this.pairs = { ...(pairs || {}) };
        const a = this.analysis;
        const st = steps(a, this.pairs);
        const any = st.some((s) => s);
        let added = false;
        const rgb = [0, 0, 0];
        const lab = [0, 0, 0];
        for (const t of a.textures) {
            const touched = st.some((s, i) => s && t.comboSlot.includes(i));
            let tex = this.textures.get(t);
            if (!touched && !tex) {
                continue;
            }
            const { width: w, height: h, data: d } = t.pixels;
            if (!tex) {
                tex = new THREE.DataTexture(new Uint8Array(d.length), w, h, THREE.RGBAFormat);
                const src = t.texture;
                tex.colorSpace = src.colorSpace;
                tex.wrapS = src.wrapS;
                tex.wrapT = src.wrapT;
                tex.magFilter = src.magFilter;
                tex.minFilter = src.minFilter;
                tex.anisotropy = src.anisotropy;
                tex.flipY = false;
                tex.generateMipmaps = true;
                tex.channel = src.channel;
                this.textures.set(t, tex);
                added = true;
            }
            const out = tex.image.data;
            out.set(d);
            const m = t.comboSlot.length;
            const packed = new Int32Array(m).fill(-1);
            for (let c = 0; c < m; c++) {
                const s = st[t.comboSlot[c]];
                if (!s) {
                    continue;
                }
                lab[0] = t.comboLab[c * 3] + s[0];
                lab[1] = t.comboLab[c * 3 + 1] + s[1];
                lab[2] = t.comboLab[c * 3 + 2] + s[2];
                labToRgb(lab, rgb);
                packed[c] = (Math.round(rgb[0]) << 16) | (Math.round(rgb[1]) << 8) | Math.round(rgb[2]);
            }
            const texels = t.texels;
            const combos = t.texelCombo;
            for (let j = 0, n = texels.length; j < n; j++) {
                const p = packed[combos[j]];
                if (p < 0) {
                    continue;
                }
                const o = texels[j] * 4;
                out[o] = p >> 16;
                out[o + 1] = (p >> 8) & 255;
                out[o + 2] = p & 255;
            }
            tex.needsUpdate = true;
        }
        for (const it of a.items) {
            if (it.kind !== 'material') {
                continue;
            }
            const s = it.slot >= 0 ? st[it.slot] : null;
            let copy = this.copies.get(it.material);
            if (!s && !copy) {
                continue;
            }
            if (!copy) {
                copy = it.material.clone();
                this.copies.set(it.material, copy);
                added = true;
            }
            const c = s ? labToRgb([it.lab[0] + s[0], it.lab[1] + s[1], it.lab[2] + s[2]]) : srgbOf(it.material.color);
            copy.color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
        }
        for (const v of a.vertex) {
            const touched = st.some((s, i) => s && v.slots.includes(i));
            let g = this.geometries.get(v.geometry);
            if (!touched && !g) {
                continue;
            }
            if (!g) {
                g = v.geometry.clone();
                this.geometries.set(v.geometry, g);
                added = true;
            }
            const attr = g.attributes.color;
            const c = new THREE.Color();
            for (let i = 0; i < v.slots.length; i++) {
                const s = v.slots[i] >= 0 ? st[v.slots[i]] : null;
                if (s) {
                    labToRgb([v.labs[i * 3] + s[0], v.labs[i * 3 + 1] + s[1], v.labs[i * 3 + 2] + s[2]], rgb);
                } else {
                    labToRgb([v.labs[i * 3], v.labs[i * 3 + 1], v.labs[i * 3 + 2]], rgb);
                }
                c.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
                attr.setXYZ(i, c.r, c.g, c.b);
            }
            attr.needsUpdate = true;
        }
        // Materials that sample a recolored texture take a copy that samples the new one.
        for (const t of a.textures) {
            const tex = this.textures.get(t);
            if (!tex) {
                continue;
            }
            this.rig.scene.traverse((obj) => {
                if (!obj.isMesh) {
                    return;
                }
                for (const m of [].concat(obj.material)) {
                    if (m.map === t.texture && !this.copies.has(m)) {
                        const copy = m.clone();
                        copy.map = tex;
                        this.copies.set(m, copy);
                        added = true;
                    } else if (m.map === t.texture && this.copies.get(m).map !== tex) {
                        this.copies.get(m).map = tex;
                    }
                }
            });
        }
        if (added) {
            for (const inst of this.instances) {
                this.swap(inst.root);
            }
        }
        this.active = any;
        return this;
    }

    dispose() {
        for (const m of this.copies.values()) {
            m.dispose();
        }
        for (const t of this.textures.values()) {
            t.dispose();
        }
        for (const g of this.geometries.values()) {
            g.dispose();
        }
        this.copies.clear();
        this.textures.clear();
        this.geometries.clear();
        this.instances = [];
    }
}

// Recolors one instance with pairs of its own, for a single figure. The recolor is kept on the
// instance as instance.recolor, and its dispose frees it.
export function recolorInstance(rig, instance, pairs) {
    if (instance.recolor) {
        instance.recolor.set(pairs);
        return instance.recolor;
    }
    instance.recolor = new Recolor(rig, pairs);
    instance.recolor.apply(instance);
    return instance.recolor;
}
