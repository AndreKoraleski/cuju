// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A two tier bowl around the pitch. Each row of seats is a step with a painted card of people on
// it, the tiers are split by a walkway and a ribbon board, gangways run up through the seats and a
// roof on rafters covers the stands. Four floodlight towers stand behind the corners.

import * as THREE from 'three';
import { ATLAS, atlasTextures, grfCrowd, paintCrowd } from './crowd.js';
import { FULL_HALF_H, FULL_HALF_W } from './pitch.js';
import { hourOf, lightingAt } from './sky.js';

const INNER_HALF_X = FULL_HALF_W + 6;
const INNER_HALF_Z = FULL_HALF_H + 6;
const CORNER = 12;
// The shape of the stands in meters, out from the inner edge and up from the grass.
const FRONT_WALL = 1.5;
const TREAD = 0.8;
const LOWER = { rows: 18, rise: 0.38 };
const WALKWAY = 3;
const UPPER_WALL = 3.2;
const RIBBON = 1;
const UPPER = { rows: 26, rise: 0.5 };
const LOWER_END = LOWER.rows * TREAD;
const WALK_Y = FRONT_WALL + LOWER.rows * LOWER.rise;
const UPPER_D = LOWER_END + WALKWAY;
const UPPER_Y = WALK_Y + UPPER_WALL;
const DEPTH = UPPER_D + UPPER.rows * TREAD;
const TOP_Y = UPPER_Y + UPPER.rows * UPPER.rise;
// The roof rises a little towards its front edge, which pulls back at the corners so the light
// of the towers reaches the pitch over it.
const ROOF_BACK = TOP_Y + 4.6;
const ROOF_FRONT = ROOF_BACK + 2;
const ROOF_REACH = 10;
const ROOF_CORNER = 9;
// People sit a little behind the front of their tread, on a card leaning back slightly.
const CARD_AT = 0.35;
const CARD_TILT = 0.2;
// Gangways on each long side, each end and each corner, spaced evenly, and half their width.
const AISLES = { side: 8, end: 5, corner: 1 };
const AISLE_HALF = 0.6;
const PER_CORNER = 12;
// Seats taken from one stretch of the atlas before the next stretch is picked.
const CHUNK = 12;

// The floodlight masts stand just behind the corners of the stands, tall enough that their light
// clears the roof on its way to the near corner of the pitch.
const MAST_HEIGHT = 62;
const MAST_OUT = CORNER + DEPTH + 3;
// Each tower lights the whole pitch. A wide cone with a long soft edge and the physical fall off
// with distance keep the four together within about 15 percent of even.
const FLOOD_CANDELA = 14000;
const FLOOD_ANGLE = 0.7;
const FLOOD_PENUMBRA = 0.85;
// How many towers cast shadows, each one a shadow pass of its own. While the sun still casts the
// main shadow, at dusk, two towers are enough.
const FLOOD_SHADOWS = {
    night: { low: 0, medium: 2, high: 4 },
    dusk: { low: 0, medium: 1, high: 2 },
};
const FLOOD_SHADOW_MAP = { low: 512, medium: 1024, high: 2048 };

// Shirts for fans of a team the page knows nothing about.
const MIXED = ['#d8d8d8', '#202428', '#b3262e', '#2a56b8', '#e0b33a', '#3c8a4a', '#7a2b8a', '#e86a2a',
    '#f2f2f2', '#9aa3ad', '#16365f', '#c43c3c'];
const MAX_COLORS = 12;
// How much of the light of the towers falls on the stands, at full strength.
const STAND_FLOOD = 0.3;

// Points around the bowl at a distance d out from its inner edge, counterclockwise seen from
// above, with the outward normal, the running length and the phase whose half integers are the
// middles of the gangways. Long sides and ends are cut at every whole phase, so the rafters sit
// on points.
function ring(d) {
    const pts = [];
    const hx = INNER_HALF_X - CORNER;
    const hz = INNER_HALF_Z - CORNER;
    const r = CORNER + d;
    const corners = [[hx, hz, 0], [-hx, hz, Math.PI / 2], [-hx, -hz, Math.PI], [hx, -hz, Math.PI * 1.5]];
    let phase = 0;
    for (let c = 0; c < 4; c++) {
        const [cx, cz, a0] = corners[c];
        const arc = (r * Math.PI) / 2 / AISLES.corner;
        for (let k = 0; k < PER_CORNER; k++) {
            const t = k / PER_CORNER;
            const a = a0 + (t * Math.PI) / 2;
            pts.push({
                x: cx + Math.cos(a) * r, z: cz + Math.sin(a) * r, nx: Math.cos(a), nz: Math.sin(a),
                phase: phase + t * AISLES.corner, mpp: arc, corner: t,
            });
        }
        phase += AISLES.corner;
        const a1 = a0 + Math.PI / 2;
        const nx = Math.cos(a1);
        const nz = Math.sin(a1);
        const [ncx, ncz] = corners[(c + 1) % 4];
        const sx = cx + nx * r;
        const sz = cz + nz * r;
        const ex = ncx + nx * r;
        const ez = ncz + nz * r;
        const n = c % 2 === 0 ? AISLES.side : AISLES.end;
        const mpp = Math.hypot(ex - sx, ez - sz) / n;
        for (let k = 0; k < n; k++) {
            const t = k / n;
            pts.push({ x: sx + (ex - sx) * t, z: sz + (ez - sz) * t, nx, nz, phase: phase + k, mpp, corner: -1 });
        }
        phase += n;
    }
    pts.push({ ...pts[0], phase });
    let run = 0;
    pts.forEach((p, i) => {
        if (i > 0) {
            run += Math.hypot(p.x - pts[i - 1].x, p.z - pts[i - 1].z);
        }
        p.s = run;
    });
    return pts;
}

const rings = new Map();
function ringAt(d) {
    const key = d.toFixed(3);
    if (!rings.has(key)) {
        rings.set(key, ring(d));
    }
    return rings.get(key);
}

// Vertices for the stands, each with its place along the row, its height or kind of surface,
// and the gangway phase with a third value that is a row seed or a wall height.
class Mesh {
    constructor() {
        this.pos = [];
        this.crowd = [];
        this.aisle = [];
        this.place = [];
        this.uv = [];
        this.index = [];
        this.count = 0;
    }

    // A band between two lines of the same number of vertices.
    band(a, b) {
        for (let i = 0; i < a.length; i++) {
            for (const p of [a[i], b[i]]) {
                this.pos.push(p.x, p.y, p.z);
                this.crowd.push(p.s || 0, p.v || 0);
                this.aisle.push(p.phase || 0, p.mpp || 1, p.w || 0);
                this.place.push(p.nx || 0, p.nz || 0, p.d || 0);
                this.uv.push(p.u || 0, p.uvv || 0);
            }
            if (i > 0) {
                const k = this.count + (i - 1) * 2;
                this.index.push(k, k + 2, k + 1, k + 1, k + 2, k + 3);
            }
        }
        this.count += a.length * 2;
    }

    quad(p0, p1, p2, p3) {
        this.band([p0, p1], [p3, p2]);
    }

    geometry(attrs = ['crowd', 'aisle']) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
        if (attrs.includes('crowd')) {
            geo.setAttribute('crowd', new THREE.Float32BufferAttribute(this.crowd, 2));
            geo.setAttribute('aisle', new THREE.Float32BufferAttribute(this.aisle, 3));
            geo.setAttribute('place', new THREE.Float32BufferAttribute(this.place, 3));
        }
        if (attrs.includes('uv')) {
            geo.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
        }
        geo.setIndex(this.index);
        geo.computeVertexNormals();
        return geo;
    }
}

// A line around the bowl at distance d and height y, with v naming the kind of surface.
function line(d, y, v, w = 0) {
    return ringAt(d).map((p) => ({
        x: p.x, y, z: p.z, s: p.s, v, phase: p.phase, mpp: p.mpp, w, nx: p.nx, nz: p.nz, d,
    }));
}

// Treads, risers and the cards of people for a tier, front row first so the nearer rows fill the
// depth buffer before the rows behind are shaded.
function tier(steps, cards, startD, startY, rows, rise, seed) {
    const lean = [Math.sin(CARD_TILT) * ATLAS.card, Math.cos(CARD_TILT) * ATLAS.card];
    for (let i = 0; i < rows; i++) {
        const d = startD + i * TREAD;
        const y = startY + i * rise;
        steps.band(line(d, y, 0), line(d + TREAD, y, 0.999));
        steps.band(line(d + TREAD, y, 1), line(d + TREAD, y + rise, 1.999));
        const bottom = line(d + CARD_AT, y, 0, seed + i * 1.618);
        const top = ringAt(d + CARD_AT + lean[0]).map((p, k) => ({
            ...bottom[k], x: p.x, y: y + lean[1], z: p.z, v: ATLAS.card,
        }));
        cards.band(bottom, top);
    }
}

// The stands of the arenas other than the stadium, one along each touch line, of the same steps
// and painted people. half is the half length of a stand, at its distance from the middle of the
// pitch, rows, tread and rise its steps, front the wall before the first row, and group the
// meters between gangways, or between the gaps that split a crowd on flat ground into groups.
// A stand without steps stands its people on the ground. density, seats and seat give the
// share of places taken, whether an empty place shows a seat and its color, neon draws figures of
// light, and the step colors and glow dress the steps.
const ARENA_STANDS = {
    beach: {
        half: 48, at: 44.5, rows: 3, tread: 1.7, rise: 0, front: 0, group: 6, steps: false,
        density: 0.6, seats: 1, seat: '#3f7fc6',
    },
    rooftop: {
        half: 34, at: 46, rows: 6, tread: TREAD, rise: 0.42, front: 0.9, group: 8.5, steps: true,
        concrete: '#7a7f86', wall: '#3b4048', facing: '#2c3138', nosing: '#d9b21e', seat: '#3a4250',
    },
    space: {
        half: 42, at: 47, rows: 5, tread: 0.9, rise: 0.5, front: 1, group: 10.5, steps: true, glass: true,
        concrete: '#5d6672', wall: '#2a3038', facing: '#1c232b', nosing: '#39d0ff', glow: 0.6, seat: '#d8dee6',
    },
    neon: {
        half: 42, at: 46, rows: 5, tread: TREAD, rise: 0.36, front: 0.5, group: 10.5, steps: true, neon: true,
        concrete: '#1a0f2a', wall: '#120a1e', facing: '#0d0716', nosing: '#ff2bd6', glow: 1.6, seats: 0,
    },
};

// The steps and the people of an arena's stands. The steps mark their surfaces as the bowl does,
// the light needs no roof, so every point gets the place of one far inside it.
function buildArenaStands(spec) {
    const steps = new Mesh();
    const cards = new Mesh();
    const lean = [Math.sin(CARD_TILT) * ATLAS.card, Math.cos(CARD_TILT) * ATLAS.card];
    const x0 = -spec.half;
    const x1 = spec.half;
    for (const side of [-1, 1]) {
        const at = (x, d, y, v, w = 0, open = true) => ({
            x, y, z: side * (spec.at + d), s: x - x0, v, w, nx: 0, nz: 0, d: -1000,
            phase: open ? (x - x0) / spec.group : 0, mpp: open ? spec.group : 1e4,
        });
        const row = (d, y, v, w = 0, open = true) => [at(x0, d, y, v, w, open), at(x1, d, y, v, w, open)];
        const top = spec.front + (spec.rows - 1) * spec.rise;
        const depth = spec.rows * spec.tread;
        if (spec.steps) {
            steps.band(row(0, 0, 4), row(0, spec.front, 4.999));
            for (let i = 0; i < spec.rows; i++) {
                const d = i * spec.tread;
                const y = spec.front + i * spec.rise;
                steps.band(row(d, y, 0), row(d + spec.tread, y, 0.999));
                if (i < spec.rows - 1) {
                    steps.band(row(d + spec.tread, y, 1), row(d + spec.tread, y + spec.rise, 1.999));
                }
                // The ends of the stand, a step at a time.
                for (const x of [x0, x1]) {
                    steps.quad(at(x, d, 0, 3, 0, false), at(x, d + spec.tread, 0, 3, 0, false),
                        at(x, d + spec.tread, y, 3.5, 0, false), at(x, d, y, 3.5, 0, false));
                }
            }
            steps.band(row(depth, 0, 5, 0, false), row(depth, top + 1.1, 5.999, 0, false));
        }
        for (let i = 0; i < spec.rows; i++) {
            const d = i * spec.tread + CARD_AT;
            const y = spec.front + i * spec.rise;
            const seed = side * 13.7 + i * 1.618 + 3.1;
            const bottom = row(d, y, 0, seed);
            const upper = row(d + lean[0], y + lean[1], ATLAS.card, seed);
            cards.band(bottom, upper.map((p, k) => ({ ...bottom[k], y: p.y, z: p.z, v: ATLAS.card })));
        }
    }
    return { steps: spec.steps ? steps.geometry() : null, cards: cards.geometry() };
}

function buildStands() {
    const steps = new Mesh();
    const cards = new Mesh();
    steps.band(line(0, 0, 4), line(0, FRONT_WALL, 4.999));
    tier(steps, cards, 0, FRONT_WALL, LOWER.rows, LOWER.rise, 3.1);
    steps.band(line(LOWER_END, WALK_Y, 3), line(UPPER_D, WALK_Y, 3.999));
    const wall = UPPER_WALL - RIBBON;
    steps.band(line(UPPER_D, WALK_Y, 2, wall), line(UPPER_D, WALK_Y + wall, 2.999, wall));
    tier(steps, cards, UPPER_D, UPPER_Y, UPPER.rows, UPPER.rise, 71.7);
    steps.band(line(DEPTH, 0, 5), line(DEPTH, ROOF_BACK + 0.6, 5.999));
    return { steps: steps.geometry(), cards: cards.geometry() };
}

// The ribbon board along the front of the upper tier, mapped like the boards around the pitch.
function buildRibbon(boardLength) {
    const m = new Mesh();
    const d = UPPER_D - 0.03;
    const at = (y, v) => ringAt(d).map((p) => ({ x: p.x, y, z: p.z, u: p.s / boardLength, uvv: v }));
    m.band(at(UPPER_Y - RIBBON, 0), at(UPPER_Y, 1));
    return m.geometry(['uv']);
}

// How far out the front edge of the roof lies at a point of the inner edge.
function roofReach(p) {
    return ROOF_REACH + (p.corner >= 0 ? ROOF_CORNER * Math.sin(Math.PI * p.corner) : 0);
}

function buildRoof() {
    const base = ringAt(0);
    const at = (p, d, y) => ({ x: p.x + p.nx * d, y, z: p.z + p.nz * d });
    const top = new Mesh();
    const under = new Mesh();
    const fascia = new Mesh();
    const strip = new Mesh();
    // The ribs of the sheeting run from back to front, a texture repeat every two meters of the
    // back edge.
    const backRun = ringAt(DEPTH);
    top.band(base.map((p, i) => ({ ...at(p, DEPTH + 0.4, ROOF_BACK + 0.6), u: backRun[i].s / 2, uvv: 0 })),
        base.map((p, i) => ({ ...at(p, roofReach(p), ROOF_FRONT + 0.6), u: backRun[i].s / 2, uvv: 1 })));
    under.band(base.map((p) => at(p, DEPTH, ROOF_BACK)), base.map((p) => at(p, roofReach(p), ROOF_FRONT)));
    fascia.band(base.map((p) => at(p, roofReach(p) - 0.05, ROOF_FRONT - 0.9)), base.map((p) => at(p, roofReach(p) - 0.05, ROOF_FRONT + 0.8)));
    strip.band(base.map((p) => at(p, roofReach(p) + 0.15, ROOF_FRONT - 0.92)), base.map((p) => at(p, roofReach(p) + 0.75, ROOF_FRONT - 0.88)));
    // A rafter under the roof at every whole phase and in the middle of each corner, deep at the
    // back wall and slim at the front edge.
    const rafters = new Mesh();
    for (const p of base.slice(0, -1)) {
        if (!(p.corner < 0 || Math.abs(p.corner - 0.5) < 1e-6)) {
            continue;
        }
        const tx = -p.nz * 0.22;
        const tz = p.nx * 0.22;
        const front = roofReach(p) + 0.3;
        const corners = [];
        for (const [d, y, depth] of [[DEPTH, ROOF_BACK, 2.6], [front, ROOF_FRONT, 0.5]]) {
            for (const s of [-1, 1]) {
                const x = p.x + p.nx * d + tx * s;
                const z = p.z + p.nz * d + tz * s;
                corners.push({ x, y: y - 0.02, z }, { x, y: y - depth, z });
            }
        }
        // Back left top, back left bottom, back right top and bottom, then the same at the front.
        const [blt, blb, brt, brb, flt, flb, frt, frb] = corners;
        rafters.quad(blt, flt, flb, blb);
        rafters.quad(brt, frt, frb, brb);
        rafters.quad(blb, flb, frb, brb);
        rafters.quad(flt, frt, frb, flb);
    }
    return {
        top: top.geometry(['uv']), under: under.geometry([]), fascia: fascia.geometry([]),
        strip: strip.geometry([]), rafters: rafters.geometry([]),
    };
}

// Light and dark ribs of the roof sheeting.
function makeRibTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 4;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#d4d8dc';
    ctx.fillRect(0, 0, 32, 4);
    ctx.fillStyle = '#a3a9b0';
    ctx.fillRect(0, 0, 6, 4);
    ctx.fillStyle = '#eef0f2';
    ctx.fillRect(6, 0, 2, 4);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.anisotropy = 4;
    return texture;
}

const GLSL_COMMON = `
float cjHash(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x * 43.17 + p.y);
}
`;

const VARYINGS = `
varying vec2 vCrowd;
varying vec3 vAisle;
varying float vCrowdX;
varying vec3 vPlace;
varying vec3 vStandN;
varying float vStandY;
`;

// The stands light themselves, from the sky, the sun and the floodlights, since each layer of
// people would otherwise pay for every light and shadow map of the scene. The roof shades a point
// when the way to the sun crosses the height of the roof inside its reach.
const STAND_LIGHT = `
uniform vec3 uSky;
uniform vec3 uGround;
uniform vec3 uSun;
uniform vec3 uSunDir;
uniform float uRoofY;
uniform float uRoofFront;
uniform float uFlood;
uniform float uLift;
vec3 standLight() {
    vec3 n = normalize(vStandN) * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 sky = mix(uGround, uSky, 0.5 + 0.5 * n.y);
    float ndl = max(dot(n, uSunDir), 0.0);
    float t = (uRoofY - vStandY) / max(uSunDir.y, 0.05);
    float reach = vPlace.z + t * dot(uSunDir.xz, vPlace.xy);
    float lit = 1.0 - smoothstep(uRoofFront - 1.0, uRoofFront + 1.0, reach);
    return (sky + uSun * ndl * lit) * RECIPROCAL_PI + uFlood + uLift;
}
`;

const VERTEX_CODE = `
vCrowd = crowd;
vAisle = aisle;
vCrowdX = position.x;
vPlace = place;
vStandN = normal;
vStandY = position.y;
`;

// A plain material with the stand attributes, whose color and light come from the code of the
// stands.
function standMaterial(params, uniforms, decl, body) {
    const material = new THREE.MeshBasicMaterial(params);
    material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', `#include <common>\nattribute vec2 crowd;\nattribute vec3 aisle;\nattribute vec3 place;\n${VARYINGS}`)
            .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERTEX_CODE}`);
        shader.fragmentShader = shader.fragmentShader
            .replace('#include <common>', `#include <common>\n${VARYINGS}\n${GLSL_COMMON}\n${STAND_LIGHT}\n${decl}`)
            .replace('#include <map_fragment>', `vec3 cjEmit = vec3(0.0);\n${body}\ndiffuseColor.rgb = diffuseColor.rgb * standLight() + cjEmit;`);
    };
    return material;
}

const f = (x) => x.toFixed(5);

const CARD_DECL = `
uniform sampler2D uFigA;
uniform sampler2D uMaskA;
uniform sampler2D uFigB;
uniform sampler2D uMaskB;
uniform vec2 uTexSize;
uniform float uTime;
uniform float uBob;
uniform float uJump;
uniform float uCheer;
uniform float uDensity;
uniform float uMosaic;
uniform float uSeats;
uniform vec3 uLeft[${MAX_COLORS}];
uniform vec3 uRight[${MAX_COLORS}];
uniform vec2 uCounts;
uniform vec2 uDominant;
uniform vec2 uNeutral;
uniform float uMix;
uniform vec3 uSeatColor;
uniform float uAisleHalf;
uniform float uShirt;
uniform float uHard;
uniform float uNeon;
uniform vec3 uNeonTint;
const float SEAT_M = ${f(ATLAS.seat)};
const float CARD_M = ${f(ATLAS.card)};
const float STRIP_M = ${f(ATLAS.strip)};
const float SEATS = ${f(ATLAS.seats)};
const float STRIPS = ${f(ATLAS.strips)};
const float CHUNK = ${f(CHUNK)};
const float TALL_EVERY = ${f(ATLAS.tallEvery)};
const float SEATED_TOP = ${f(ATLAS.seatedTop)};
const float CALM_TOP = ${f(ATLAS.calmTop)};
const float GRF_TOP = ${f(ATLAS.grfTop)};
`;

const CARD_BODY = `
{
    float open = smoothstep(uAisleHalf - 0.04, uAisleHalf + 0.04, abs(fract(vAisle.x) - 0.5) * vAisle.y);
    float seed = vAisle.z;
    float cs = vCrowd.x / SEAT_M;
    float h = vCrowd.y;
    float col = floor(cs);
    float fx = cs - col;
    float id = cjHash(vec2(col, seed));
    float strip;
    float tcol;
    // Stretches of seats come from random places of the atlas, so no pattern repeats along a row.
    if (uMosaic > 0.5) {
        float chunk = floor(col / CHUNK);
        strip = floor(cjHash(vec2(chunk, seed + 0.37)) * STRIPS);
        tcol = mod(col + floor(cjHash(vec2(chunk + 7.1, seed)) * SEATS), SEATS) + fx;
    } else {
        strip = floor(cjHash(vec2(seed, 2.9)) * STRIPS);
        tcol = cs + floor(cjHash(vec2(seed, 5.3)) * SEATS);
    }
    // After a goal each person gets up at a moment of their own and jumps.
    bool up = uCheer > 0.1 + 0.8 * cjHash(vec2(id, 4.4));
    float bob = uBob * sin(uTime * (1.2 + 1.8 * cjHash(vec2(id, 8.8))) + id * 40.0);
    if (up) {
        float j = max(0.0, sin(uTime * (6.0 + 2.5 * cjHash(vec2(id, 9.9))) + id * 31.0));
        bob += uJump * j * j;
    }
    float hh = h - bob;
    vec2 tuv = vec2(tcol / SEATS, (strip * STRIP_M + STRIP_M - hh) / (STRIPS * STRIP_M));
    // Gradients of the unbroken coordinates keep the mip level steady across the jumps.
    vec2 cuv = vec2(cs / SEATS, -h / (STRIPS * STRIP_M));
    vec2 gx = dFdx(cuv);
    vec2 gy = dFdy(cuv);
    float w = fwidth(cs) + 0.001;
    float wh = fwidth(h) + 0.001;
    // Above the tallest person a column can hold there is only air, and in a gangway nothing.
    float top = up ? CARD_M : uMosaic < 0.5 ? GRF_TOP
        : mod(floor(tcol), TALL_EVERY) < 0.5 ? CALM_TOP : SEATED_TOP;
    if (hh > top || open < 0.001) {
        discard;
    }
    vec4 fig;
    if (up) {
        fig = textureGrad(uFigB, tuv, gx, gy);
    } else {
        fig = textureGrad(uFigA, tuv, gx, gy);
    }
    float present = step(cjHash(vec2(id, 1.3)), uDensity) * step(0.0, hh) * step(hh, CARD_M);
    float a = fig.a * present;
    // Up close the edges are cut sharp, far away the coverage of the mip levels is kept.
    float texels = max(length(gx * uTexSize), length(gy * uTexSize));
    float sharp = clamp((a - 0.5) / max(fwidth(a), 0.001) + 0.5, 0.0, 1.0);
    a = uHard > 0.5 ? step(0.5, a) : mix(sharp, a, clamp(texels - 0.6, 0.0, 1.0));
    // An empty seat shows its plastic back.
    float sx = abs(fx - 0.5);
    float back = (1.0 - smoothstep(0.4 - w, 0.4 + w, sx)) * smoothstep(0.42 - wh, 0.42 + wh, h)
        * (1.0 - smoothstep(0.84 - wh, 0.84 + wh, h)) * uSeats;
    float alpha = max(a, back) * open;
    // Most of a card is air, leaving there skips the mask, the colors and the light of the rows
    // seen through it.
    if (alpha < 0.004) {
        discard;
    }
    vec3 people = fig.rgb;
    vec3 cloth = vec3(0.0);
    if (a > 0.004) {
        vec2 mk;
        if (up) {
            mk = textureGrad(uMaskB, tuv, gx, gy).rg;
        } else {
            mk = textureGrad(uMaskA, tuv, gx, gy).rg;
        }
        // Fans of the left team fill the left of the ground, mixing near the halfway line.
        float side = step(0.0, vCrowdX + (cjHash(vec2(id, 2.2)) - 0.5) * 2.0 * uMix);
        int n = int(mix(uCounts.x, uCounts.y, side) + 0.5);
        float dom = mix(uDominant.x, uDominant.y, side);
        int si = cjHash(vec2(id, 3.3)) < dom ? 0 : min(n - 1, int(cjHash(vec2(id, 5.5)) * float(n)));
        int ai = n > 1 ? (si + 1 + min(n - 2, int(cjHash(vec2(id, 6.6)) * float(n - 1)))) % n : si;
        vec3 shirt = side > 0.5 ? uRight[si] : uLeft[si];
        vec3 acc = side > 0.5 ? uRight[ai] : uLeft[ai];
        if (n < 2) {
            acc = dot(shirt, vec3(0.2126, 0.7152, 0.0722)) > 0.2 ? vec3(0.015) : vec3(0.8);
        }
        if (cjHash(vec2(id, 7.7)) < mix(uNeutral.x, uNeutral.y, side)) {
            int k = int(cjHash(vec2(id, 9.1)) * 5.0);
            acc = shirt;
            shirt = k == 0 ? vec3(0.82) : k == 1 ? vec3(0.3) : k == 2 ? vec3(0.018) : k == 3 ? vec3(0.03, 0.05, 0.12) : vec3(0.1, 0.18, 0.34);
        }
        float shade = fig.g / uShirt;
        cloth = (shirt * mk.r + acc * mk.g) * shade;
        people = fig.rgb * max(0.0, 1.0 - mk.r - mk.g) + cloth;
    }
    // Figures of light, dark bodies whose team colors glow, and up close outlines that shine.
    if (uNeon > 0.5) {
        float lum = dot(fig.rgb, vec3(0.2126, 0.7152, 0.0722));
        float edge = clamp(fwidth(fig.a) * 2.0, 0.0, 1.0) * (1.0 - clamp(texels - 0.6, 0.0, 1.0));
        people = vec3(0.012, 0.008, 0.03);
        cjEmit = (cloth * 1.7 + uNeonTint * (lum * 0.03 + edge * 0.5)) * a;
    }
    vec3 seat = uSeatColor * (0.55 + 0.45 * clamp((h - 0.42) / 0.42, 0.0, 1.0));
    diffuseColor.rgb *= mix(seat, people, a);
    diffuseColor.a = alpha;
}
`;

const STEP_DECL = `
uniform vec3 uConcrete;
uniform vec3 uWall;
uniform vec3 uFront;
uniform vec3 uNosing;
uniform float uAisleHalf;
uniform float uStepGlow;
`;

// Kinds of surface by the whole part of v, tread, riser, the wall under the ribbon, the walkway,
// the wall along the pitch and the back of the stand.
const STEP_BODY = `
{
    float aisleD = abs(fract(vAisle.x) - 0.5) * vAisle.y;
    float inA = 1.0 - smoothstep(uAisleHalf - 0.04, uAisleHalf + 0.04, aisleD);
    float v = vCrowd.y;
    float kind = floor(v);
    float t = v - kind;
    float fw = fwidth(v) + 0.001;
    vec3 c;
    if (kind < 0.5) {
        vec3 under = uConcrete * mix(0.7, 0.3, smoothstep(0.2, 0.95, t));
        float mid = smoothstep(0.5 - fw, 0.5 + fw, t);
        vec3 stair = uConcrete * mix(1.2, 0.95, mid);
        float nose = max(1.0 - smoothstep(0.07 - fw, 0.07 + fw, t), mid * (1.0 - smoothstep(0.57 - fw, 0.57 + fw, t)));
        c = mix(under, mix(stair, uNosing, nose), inA);
        cjEmit = uNosing * uStepGlow * (1.0 - smoothstep(0.07 - fw, 0.07 + fw, t));
    } else if (kind < 1.5) {
        vec3 riser = mix(uConcrete * 0.55, uConcrete * 0.9, smoothstep(0.82 - fw, 0.82 + fw, t));
        c = mix(riser, uConcrete * 0.85, inA);
    } else if (kind < 2.5) {
        float hm = t * vAisle.z;
        float door = (1.0 - smoothstep(1.05, 1.15, aisleD)) * (1.0 - smoothstep(2.05, 2.15, hm));
        c = mix(uWall, vec3(0.01), door);
    } else if (kind < 3.5) {
        c = uConcrete * 1.1;
    } else if (kind < 4.5) {
        c = mix(uFront, vec3(0.75), smoothstep(0.9 - fw, 0.9 + fw, t));
        cjEmit = uNosing * uStepGlow * smoothstep(0.9 - fw, 0.9 + fw, t);
    } else {
        c = uConcrete * (0.8 + 0.12 * step(0.5, fract(t * 7.0)));
    }
    diffuseColor.rgb *= c;
}
`;

const linear = (hex) => new THREE.Color(hex);

function fanColors(list) {
    return list.slice(0, MAX_COLORS).map((c) => (Array.isArray(c)
        ? new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace)
        : new THREE.Color(c)));
}

function validFans(list) {
    return Array.isArray(list) && list.length > 0
        && list.every((c) => Array.isArray(c) && c.length >= 3 && c.every((v) => Number.isFinite(v)));
}

// The last fans a stadium was given, so a stadium built again when the art arrives keeps them.
let lastFans = [null, null];

// The cuju logo in white for the dark boards, null until it loads. The logo is dark gray on
// transparent, so each channel is turned around and the alpha kept.
let boardLogo = null;
const boardLogoReady = new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < data.data.length; i += 4) {
            data.data[i] = 255 - data.data[i];
            data.data[i + 1] = 255 - data.data[i + 1];
            data.data[i + 2] = 255 - data.data[i + 2];
        }
        ctx.putImageData(data, 0, 0);
        boardLogo = canvas;
        resolve(canvas);
    };
    img.onerror = () => resolve(null);
    img.src = 'assets/cuju-logo.png';
});

// LED boards, the cuju logo and the team names on colored panels, with the asset boards between
// them. The logo panel shows the word until the logo has loaded.
function makeBoardTexture(names, colors, images, sponsors) {
    const h = 64;
    const cuju = (fill) => (boardLogo ? { logo: boardLogo, fill } : { word: 'cuju', fill });
    const panels = sponsors.length ? [
        cuju('#202124'),
        ...sponsors.map((image) => ({ image })),
    ] : [
        cuju('#0e1a2b'),
        { word: names[0], fill: colors[0] },
        images[0] ? { image: images[0] } : { word: 'Google Research Football', fill: '#14301c' },
        { word: names[1], fill: colors[1] },
    ];
    if (images[1]) {
        panels.push({ image: images[1] });
    }
    const widths = panels.map((p) => {
        if (p.image) {
            return Math.round(p.image.width * (h / p.image.height));
        }
        return p.logo ? 256 : 512;
    });
    const canvas = document.createElement('canvas');
    canvas.width = widths.reduce((a, b) => a + b, 0);
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    let x = 0;
    panels.forEach((p, k) => {
        const w = widths[k];
        if (p.image) {
            ctx.drawImage(p.image, x, 0, w, h);
        } else if (p.logo) {
            ctx.fillStyle = p.fill;
            ctx.fillRect(x, 0, w, h);
            const side = h - 12;
            const width = Math.round(side * (p.logo.width / p.logo.height));
            ctx.drawImage(p.logo, x + (w - width) / 2, 6, width, side);
        } else {
            ctx.fillStyle = p.fill;
            ctx.fillRect(x, 0, w, h);
            ctx.fillStyle = '#ffffff';
            ctx.font = '600 34px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(p.word || '', x + w / 2, h / 2 + 2, w - 24);
        }
        x += w;
    });
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    // The boards are 0.9 meters high, so the strip keeps its aspect at this length.
    texture.userData.meters = (canvas.width / h) * 0.9;
    return texture;
}

// A soft round glow, bright in the middle and fading out, for the glare of the lamps.
function makeHaloTexture() {
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.1, 'rgba(255,255,255,0.75)');
    g.addColorStop(0.3, 'rgba(255,255,255,0.25)');
    g.addColorStop(0.6, 'rgba(255,255,255,0.06)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
}

// The lamp face of a floodlight head, rows of round lamps on a dark bank.
function makeLampTexture() {
    const cols = 6;
    const rows = 3;
    const cell = 64;
    const canvas = document.createElement('canvas');
    canvas.width = cols * cell;
    canvas.height = rows * cell;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#15181c';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const x = (c + 0.5) * cell;
            const y = (r + 0.5) * cell;
            const glow = ctx.createRadialGradient(x, y, 2, x, y, cell * 0.42);
            glow.addColorStop(0, '#ffffff');
            glow.addColorStop(0.55, '#e8eef8');
            glow.addColorStop(1, '#5a6270');
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(x, y, cell * 0.42, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
}

// Four towers at the corners, each a mast with a bank of lamps and a spot light aimed across the
// pitch. They light up at dusk and at night, and the quality decides how many cast shadows.
class Floodlights {
    constructor(parent) {
        this.group = new THREE.Group();
        parent.add(this.group);
        this.level = 0;
        this.quality = 'medium';
        const metal = new THREE.MeshStandardMaterial({ color: 0x3b4047, roughness: 0.55, metalness: 0.6 });
        const lamps = makeLampTexture();
        this.lampMaterial = new THREE.MeshStandardMaterial({
            map: lamps, emissive: 0xfff8ee, emissiveMap: lamps, emissiveIntensity: 0, roughness: 0.3,
        });
        this.haloMaterial = new THREE.SpriteMaterial({
            map: makeHaloTexture(), color: 0xf4f7ff, transparent: true, opacity: 0, depthWrite: false,
            blending: THREE.AdditiveBlending, fog: false,
        });
        const mastGeo = new THREE.CylinderGeometry(0.45, 1.2, MAST_HEIGHT, 12);
        const frameGeo = new THREE.BoxGeometry(10, 5.2, 0.8);
        const panelGeo = new THREE.PlaneGeometry(9.4, 4.6);
        const deckGeo = new THREE.BoxGeometry(10.5, 0.25, 2.2);
        this.lights = [];
        // Opposite corners come first, so two shadows cross from both sides.
        for (const [sx, sz] of [[1, 1], [-1, -1], [-1, 1], [1, -1]]) {
            const x = sx * (INNER_HALF_X - CORNER + MAST_OUT * Math.SQRT1_2);
            const z = sz * (INNER_HALF_Z - CORNER + MAST_OUT * Math.SQRT1_2);
            const mast = new THREE.Mesh(mastGeo, metal);
            mast.position.set(x, MAST_HEIGHT / 2, z);
            const aim = new THREE.Vector3(-sx * 5, 0, -sz * 3.5);
            const head = new THREE.Group();
            head.position.set(x, MAST_HEIGHT + 2.6, z);
            head.lookAt(aim);
            const frame = new THREE.Mesh(frameGeo, metal);
            frame.position.z = -0.45;
            const panel = new THREE.Mesh(panelGeo, this.lampMaterial);
            panel.position.z = 0.02;
            const deck = new THREE.Mesh(deckGeo, metal);
            deck.position.set(0, -2.9, 0.4);
            // A halo in front of the lamps, the glare of the lights at every quality, with or
            // without the bloom.
            const halo = new THREE.Sprite(this.haloMaterial);
            halo.scale.set(44, 40, 1);
            halo.position.z = 1.5;
            head.add(frame, panel, deck, halo);
            head.updateMatrixWorld();
            const light = new THREE.SpotLight(0xf2f5ff, 0, 0, FLOOD_ANGLE, FLOOD_PENUMBRA, 2);
            light.position.copy(head.localToWorld(new THREE.Vector3(0, 0, 1)));
            light.target.position.copy(aim);
            light.visible = false;
            const shadow = light.shadow;
            shadow.camera.near = 40;
            shadow.camera.far = 240;
            // The pitch fills about 85 percent of the cone, the shadow map need not cover the rest.
            shadow.focus = 0.85;
            shadow.bias = -0.0006;
            shadow.normalBias = 0.05;
            // Four lights share the work, so each shadow alone only darkens the grass a little.
            shadow.intensity = 0.8;
            this.lights.push(light);
            this.group.add(mast, head, light, light.target);
        }
    }

    // Shows or hides the masts and their lamps, the light itself stays.
    setMasts(on) {
        for (const child of this.group.children) {
            if (child.isMesh || child.isGroup) {
                child.visible = on;
            }
        }
    }

    // How far the lights are up, from 0 for off to 1 for full.
    setLevel(level) {
        this.level = THREE.MathUtils.clamp(level, 0, 1);
        this.apply();
    }

    setQuality(quality) {
        this.quality = quality;
        this.apply();
    }

    apply() {
        // Faint light casts no shadow, and the count changes only at whole steps, since each
        // change rebuilds the shaders.
        const level = this.level < 0.02 ? 0 : this.level;
        const shadows = level >= 0.2 ? FLOOD_SHADOWS[level >= 0.75 ? 'night' : 'dusk'][this.quality] || 0 : 0;
        const size = FLOOD_SHADOW_MAP[this.quality] || 1024;
        this.lights.forEach((light, i) => {
            light.visible = level > 0;
            light.intensity = FLOOD_CANDELA * level;
            if (light.shadow.mapSize.x !== size) {
                light.shadow.mapSize.set(size, size);
                if (light.shadow.map) {
                    light.shadow.map.dispose();
                    light.shadow.map = null;
                }
            }
            light.shadow.radius = this.quality === 'high' ? 3 : 2;
            light.castShadow = i < shadows;
            light.shadow.autoUpdate = false;
            light.shadow.needsUpdate = true;
        });
        // Bright enough that the bloom of the highest quality picks the lamps out.
        this.lampMaterial.emissiveIntensity = level > 0 ? 4 + 10 * level : 0;
        this.haloMaterial.opacity = 0.8 * level * level;
    }

    // Each shadow map is redrawn every other frame, half of them on each, which halves the cost
    // of four shadows while players move only a few centimeters in between.
    update() {
        this.turn = 1 - (this.turn || 0);
        this.lights.forEach((light, i) => {
            if (light.castShadow && i % 2 === this.turn) {
                light.shadow.needsUpdate = true;
            }
        });
    }
}

export class Stadium {
    constructor(scene, crowdImage = null, adboards = []) {
        this.group = new THREE.Group();
        scene.add(this.group);
        this.scene = scene;
        this.adboards = adboards;
        this.crowdImage = crowdImage;
        this.quality = 'medium';
        this.mode = 'teams';
        this.field = null;
        this.fans = lastFans.slice();
        this.atlasKey = '';
        this.textures = null;
        this.uniforms = {
            uFigA: { value: null },
            uMaskA: { value: null },
            uFigB: { value: null },
            uMaskB: { value: null },
            uTexSize: { value: new THREE.Vector2(1, 1) },
            uTime: { value: 0 },
            uBob: { value: 0 },
            uJump: { value: 0 },
            uCheer: { value: 0 },
            uDensity: { value: 0.9 },
            uMosaic: { value: 1 },
            uSeats: { value: 1 },
            uLeft: { value: Array.from({ length: MAX_COLORS }, () => new THREE.Color()) },
            uRight: { value: Array.from({ length: MAX_COLORS }, () => new THREE.Color()) },
            uCounts: { value: new THREE.Vector2(1, 1) },
            uDominant: { value: new THREE.Vector2(0, 0) },
            uNeutral: { value: new THREE.Vector2(0, 0) },
            uMix: { value: 8 },
            uSeatColor: { value: linear('#26344a') },
            uLift: { value: 0 },
            uSky: { value: new THREE.Color(0xcfe3ff) },
            uGround: { value: new THREE.Color(0x3a5a2a) },
            uSun: { value: new THREE.Color(0xfff1dc).multiplyScalar(3) },
            uSunDir: { value: new THREE.Vector3(-60, 110, -70).normalize() },
            uRoofY: { value: (ROOF_BACK + ROOF_FRONT) / 2 },
            uRoofFront: { value: ROOF_REACH },
            uFlood: { value: 0 },
            uAisleHalf: { value: AISLE_HALF },
            uHard: { value: 0 },
            uShirt: { value: new THREE.Color().setRGB(ATLAS.shirtGrey / 255, 0, 0, THREE.SRGBColorSpace).r },
            uNeon: { value: 0 },
            uNeonTint: { value: linear('#2be8ff') },
            uStepGlow: { value: 0 },
            uConcrete: { value: linear('#8c8f94') },
            uWall: { value: linear('#3b424c') },
            uFront: { value: linear('#1f2a24') },
            uNosing: { value: linear('#d9b21e') },
        };
        const stands = buildStands();
        this.stepsMesh = new THREE.Mesh(stands.steps, standMaterial({ side: THREE.DoubleSide }, this.uniforms, STEP_DECL, STEP_BODY));
        this.cardMaterial = standMaterial({ side: THREE.DoubleSide, alphaToCoverage: true }, this.uniforms, CARD_DECL, CARD_BODY);
        this.crowdMesh = new THREE.Mesh(stands.cards, this.cardMaterial);
        // The people go first, so the steps behind them fail the depth test before shading.
        this.crowdMesh.renderOrder = -2;
        this.stepsMesh.renderOrder = -1;
        this.group.add(this.stepsMesh, this.crowdMesh);
        const roof = buildRoof();
        const metal = new THREE.MeshStandardMaterial({
            color: 0x7f8790, map: makeRibTexture(), roughness: 0.55, metalness: 0.45, side: THREE.DoubleSide,
        });
        // The underside faces the grass, a little light of its own keeps it from taking the
        // green of the ground.
        const under = new THREE.MeshStandardMaterial({
            color: 0xc9cdd2, emissive: 0x16181b, roughness: 0.8, metalness: 0.1, side: THREE.DoubleSide,
        });
        const dark = new THREE.MeshStandardMaterial({ color: 0x2a2f36, roughness: 0.5, metalness: 0.5, side: THREE.DoubleSide });
        const steel = new THREE.MeshStandardMaterial({ color: 0x8a9099, roughness: 0.5, metalness: 0.5, side: THREE.DoubleSide });
        const roofTop = new THREE.Mesh(roof.top, metal);
        roofTop.castShadow = true;
        const roofUnder = new THREE.Mesh(roof.under, under);
        roofUnder.receiveShadow = true;
        const fascia = new THREE.Mesh(roof.fascia, dark);
        fascia.castShadow = true;
        this.rafters = new THREE.Mesh(roof.rafters, steel);
        this.strip = new THREE.Mesh(roof.strip, new THREE.MeshStandardMaterial({
            color: 0x111111, emissive: 0xfff4dd, emissiveIntensity: 2.2, side: THREE.DoubleSide,
        }));
        this.group.add(roofTop, roofUnder, fascia, this.rafters, this.strip);
        this.floodlights = new Floodlights(this.group);
        this.boardMaterial = new THREE.MeshStandardMaterial({
            emissive: 0xffffff, emissiveIntensity: 0.55, roughness: 0.5, color: 0x222222, side: THREE.DoubleSide,
        });
        this.addBoards();
        const ribbon = new THREE.Mesh(buildRibbon(this.boardLength), this.boardMaterial);
        this.group.add(ribbon);
        // The bowl, everything an arena without stands hides.
        this.bowl = [this.stepsMesh, this.crowdMesh, roofTop, roofUnder, fascia, this.strip, ribbon];
        this.arena = 'stadium';
        // The stands of the other arenas, built when an arena first asks for them.
        this.arenaStands = new Map();
        this.arenaCardMaterials = [];
        this.setTeams(['Left', 'Right'], ['#b3262e', '#2a56b8']);
        this.applyCrowd();
    }

    addBoards() {
        const height = 0.9;
        const bx = FULL_HALF_W + 1.5;
        const bz = FULL_HALF_H + 1.5;
        this.boardLength = bx * 2 - 4;
        const make = (length, x, z, rotY) => {
            const geometry = new THREE.PlaneGeometry(length, height);
            // The end boards are shorter, their texture coordinates shrink to match so the strip
            // keeps its aspect on every side.
            const uv = geometry.attributes.uv;
            for (let i = 0; i < uv.count; i++) {
                uv.setX(i, uv.getX(i) * (length / this.boardLength));
            }
            const mesh = new THREE.Mesh(geometry, this.boardMaterial);
            mesh.position.set(x, height / 2, z);
            mesh.rotation.y = rotY;
            mesh.castShadow = true;
            this.group.add(mesh);
            const backing = new THREE.Mesh(
                new THREE.BoxGeometry(length, height, 0.15),
                new THREE.MeshStandardMaterial({ color: 0x15181d, roughness: 0.8 }),
            );
            backing.position.set(x, height / 2, z);
            backing.rotation.y = rotY;
            backing.translateZ(-0.09);
            this.group.add(backing);
        };
        make(bx * 2 - 4, 0, -bz, 0);
        make(bx * 2 - 4, 0, bz, Math.PI);
        make(bz * 2 - 4, -bx, 0, Math.PI / 2);
        make(bz * 2 - 4, bx, 0, -Math.PI / 2);
    }

    setTeams(names, colors, sponsors = []) {
        if (this.boardMaterial.map) {
            this.boardMaterial.map.dispose();
        }
        // Boards drawn before the logo loaded are drawn again once it has, unless newer teams
        // replaced them in between.
        const args = [names, colors, sponsors];
        this.lastTeams = args;
        if (!boardLogo) {
            boardLogoReady.then((logo) => {
                if (logo && !this.disposed && this.lastTeams === args) {
                    this.setTeams(names, colors, sponsors);
                }
            });
        }
        const tex = makeBoardTexture(names, colors, this.adboards, sponsors);
        // Repeat the strip at its natural size along a long side, the end boards and the ribbon
        // follow it.
        tex.repeat.set(Math.max(1, Math.round(this.boardLength / tex.userData.meters)), 1);
        this.boardMaterial.map = tex;
        this.boardMaterial.emissiveMap = tex;
        this.boardMaterial.needsUpdate = true;
    }

    // Frees every geometry, material and texture of the stands, for a stadium being replaced.
    dispose() {
        this.disposed = true;
        this.group.removeFromParent();
        const textures = new Set();
        this.group.traverse((obj) => {
            if (obj.geometry) {
                obj.geometry.dispose();
            }
            if (obj.material) {
                for (const key of ['map', 'emissiveMap']) {
                    if (obj.material[key]) {
                        textures.add(obj.material[key]);
                    }
                }
                obj.material.dispose();
            }
            if (obj.isLight && obj.shadow && obj.shadow.map) {
                obj.shadow.map.dispose();
            }
        });
        textures.forEach((texture) => texture.dispose());
        this.freeAtlas();
    }

    freeAtlas() {
        if (this.textures) {
            new Set(['calmColor', 'calmMask', 'cheerColor', 'cheerMask'].map((k) => this.textures[k]))
                .forEach((t) => t.dispose());
            this.textures = null;
        }
    }

    // The fans of each team in one to four colors as RGB from 0 to 255, or null when unknown.
    setFans(left, right) {
        this.fans = [validFans(left) ? left : null, validFans(right) ? right : null];
        lastFans = this.fans.slice();
        this.applyCrowd();
    }

    // Crowd from the field settings, the fans of the teams, painted people in chosen colors or
    // GRF's strip.
    setCrowd(field) {
        this.field = field;
        this.applyCrowd();
    }

    applyCrowd() {
        const field = this.field || {};
        let mode = field.crowd || 'teams';
        if (mode === 'grf' && !this.crowdImage) {
            mode = 'teams';
        }
        this.mode = mode;
        const scale = this.quality === 'low' ? 0.5 : 1;
        const key = `${mode === 'grf' ? 'grf' : 'painted'}:${scale}`;
        if (key !== this.atlasKey) {
            this.freeAtlas();
            const atlas = mode === 'grf' ? grfCrowd(this.crowdImage, scale) : paintCrowd(scale);
            this.textures = atlasTextures(atlas, this.quality === 'low' ? 2 : 4);
            this.atlasKey = key;
            const u = this.uniforms;
            u.uFigA.value = this.textures.calmColor;
            u.uMaskA.value = this.textures.calmMask;
            u.uFigB.value = this.textures.cheerColor;
            u.uMaskB.value = this.textures.cheerMask;
            u.uTexSize.value.copy(this.textures.size);
        }
        const u = this.uniforms;
        const painted = mode !== 'grf';
        u.uMosaic.value = painted ? 1 : 0;
        u.uSeats.value = painted ? 1 : 0;
        u.uDensity.value = painted ? (Number.isFinite(field.crowdDensity) ? field.crowdDensity : 0.9) : 1;
        let sides;
        if (mode === 'custom' && Array.isArray(field.crowdColors) && field.crowdColors.length) {
            const list = fanColors(field.crowdColors);
            sides = [{ list, dominant: 0, neutral: 0 }, { list, dominant: 0, neutral: 0 }];
        } else {
            // Shirts mostly in the first color of the team, the others as accents, and a few
            // neutral coats among them.
            sides = this.fans.map((fans) => (fans
                ? { list: fanColors(fans), dominant: 0.62, neutral: 0.1 }
                : { list: fanColors(MIXED), dominant: 0, neutral: 0 }));
        }
        [u.uLeft, u.uRight].forEach((uniform, k) => {
            const list = sides[k].list;
            uniform.value.forEach((c, i) => c.copy(list[Math.min(i, list.length - 1)]));
        });
        u.uCounts.value.set(sides[0].list.length, sides[1].list.length);
        u.uDominant.value.set(sides[0].dominant, sides[1].dominant);
        u.uNeutral.value.set(sides[0].neutral, sides[1].neutral);
    }

    // The lights under the roof glow brighter as the day goes, and the towers come up. Takes the
    // lighting from sky.js lightingAt, or an hour, or one of the old names day, dusk and night.
    setTime(time) {
        const L = time && typeof time === 'object' ? time : lightingAt(hourOf(time));
        const flood = L.flood;
        this.strip.material.emissiveIntensity = 2.2 + 1.8 * flood;
        this.boardMaterial.emissiveIntensity = 0.55 + 0.55 * flood;
        // Light from the concourses and the roof, and by day light bounced off the pitch, keeps
        // the stands in the shade of the roof readable.
        this.uniforms.uLift.value = L.lift;
        this.uniforms.uFlood.value = flood * STAND_FLOOD;
        this.followLights();
        this.floodlights.setLevel(flood);
    }

    // Arenas other than the stadium keep the boards and the floodlight towers and take the bowl
    // away, with its crowd, and show stands of their own. An arena may hide the masts too and keep
    // their light, see arenas.js.
    setArena(name, masts = true) {
        this.arena = name || 'stadium';
        const bowl = this.arena === 'stadium';
        for (const part of this.bowl) {
            part.visible = bowl;
        }
        if (ARENA_STANDS[this.arena] && !this.arenaStands.has(this.arena)) {
            this.arenaStands.set(this.arena, this.buildArenaStands(ARENA_STANDS[this.arena]));
        }
        for (const [n, group] of this.arenaStands) {
            group.visible = n === this.arena;
        }
        this.rafters.visible = bowl && this.quality !== 'low';
        this.floodlights.setMasts(bowl || masts);
    }

    // A small stand on each touch line, its people sharing the atlas, the fans and the cheer of the
    // bowl, with the seats, the density and the colors of its arena.
    buildArenaStands(spec) {
        const group = new THREE.Group();
        const geo = buildArenaStands(spec);
        const own = {
            uSeats: { value: spec.seats === undefined ? 1 : spec.seats },
            uSeatColor: { value: linear(spec.seat || '#26344a') },
            uNeon: { value: spec.neon ? 1 : 0 },
            uConcrete: { value: linear(spec.concrete || '#8c8f94') },
            uWall: { value: linear(spec.wall || '#3b424c') },
            uFront: { value: linear(spec.facing || '#1f2a24') },
            uNosing: { value: linear(spec.nosing || '#d9b21e') },
            uStepGlow: { value: spec.glow || 0 },
        };
        if (spec.density) {
            own.uDensity = { value: spec.density };
        }
        const uniforms = { ...this.uniforms, ...own };
        const material = standMaterial({ side: THREE.DoubleSide, alphaToCoverage: this.quality !== 'low' }, uniforms,
            CARD_DECL, CARD_BODY);
        this.arenaCardMaterials.push(material);
        const people = new THREE.Mesh(geo.cards, material);
        people.renderOrder = -2;
        group.add(people);
        if (geo.steps) {
            const steps = new THREE.Mesh(geo.steps, standMaterial({ side: THREE.DoubleSide }, uniforms, STEP_DECL, STEP_BODY));
            steps.renderOrder = -1;
            group.add(steps);
        }
        if (spec.glass) {
            // A pane of glass before each gallery, with a light along its top.
            const height = spec.front + 3.4;
            const pane = new THREE.MeshBasicMaterial({
                color: 0x9fdcff, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide,
            });
            const rail = new THREE.MeshBasicMaterial({ color: 0x39d0ff });
            for (const side of [-1, 1]) {
                const glass = new THREE.Mesh(new THREE.PlaneGeometry(spec.half * 2, height), pane);
                glass.position.set(0, height / 2, side * (spec.at - 0.3));
                glass.renderOrder = 1;
                const edge = new THREE.Mesh(new THREE.BoxGeometry(spec.half * 2, 0.08, 0.08), rail);
                edge.position.set(0, height, side * (spec.at - 0.3));
                group.add(glass, edge);
            }
        }
        this.group.add(group);
        return group;
    }

    setQuality(quality) {
        this.quality = quality;
        this.rafters.visible = quality !== 'low' && this.arena === 'stadium';
        // The low quality cuts the people out sharply.
        const low = quality === 'low';
        this.uniforms.uHard.value = low ? 1 : 0;
        for (const material of [this.cardMaterial, ...this.arenaCardMaterials]) {
            if (material.alphaToCoverage === low) {
                material.alphaToCoverage = !low;
                material.needsUpdate = true;
            }
        }
        this.floodlights.setQuality(quality);
        this.applyCrowd();
    }

    // The crowd sways a little, more when it is excited, and stands and jumps after a goal while
    // cheer is high.
    update(time, excitement = 0, cheer = 0) {
        const u = this.uniforms;
        const lively = this.quality !== 'low' && this.mode !== 'grf';
        const c = Number.isFinite(cheer) ? THREE.MathUtils.clamp(cheer, 0, 1) : 0;
        u.uTime.value = time % 3600;
        u.uBob.value = lively ? 0.008 + 0.025 * (excitement || 0) : 0;
        u.uJump.value = lively ? 0.16 * c : 0;
        u.uCheer.value = this.mode === 'grf' ? 0 : c;
        this.followLights();
        this.floodlights.update();
    }

    // The sky and the sun of the scene, as the view sets them for the time of day.
    followLights() {
        const u = this.uniforms;
        if (!this.sun || !this.sun.parent) {
            this.sun = this.scene.children.find((o) => o.isDirectionalLight) || null;
        }
        if (!this.hemi || !this.hemi.parent) {
            this.hemi = this.scene.children.find((o) => o.isHemisphereLight) || null;
        }
        if (this.hemi) {
            u.uSky.value.copy(this.hemi.color).multiplyScalar(this.hemi.intensity);
            u.uGround.value.copy(this.hemi.groundColor).multiplyScalar(this.hemi.intensity);
        }
        if (this.sun) {
            u.uSun.value.copy(this.sun.color).multiplyScalar(this.sun.visible ? this.sun.intensity : 0);
            u.uSunDir.value.subVectors(this.sun.position, this.sun.target.position).normalize();
        }
    }
}
