// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The pitch, its markings and the goals, sized as in GRF's gamedefines.hpp and its line overlay.

import * as THREE from 'three';

// Inside the lines, as pitchHalfW and pitchHalfH.
export const HALF_W = 55;
export const HALF_H = 36;
// Grass including the rim, as pitchFullHalfW and pitchFullHalfH.
export const FULL_HALF_W = 60;
export const FULL_HALF_H = 40;
export const LINE_HALF_W = 0.06;
export const GOAL_HALF_WIDTH = 3.7;
export const GOAL_HEIGHT = 2.5;
export const GOAL_DEPTH = 2.55;
export const POST_RADIUS = 0.07;
export const BALL_RADIUS = 0.11;
// Markings measured from GRF's overlay.bmp, the referee code uses the same box sizes.
const PENALTY_DEPTH = 16.5;
const PENALTY_HALF_WIDTH = 20.15;
const GOAL_AREA_DEPTH = 5.5;
const GOAL_AREA_HALF_WIDTH = 9.16;
const PENALTY_SPOT = 11;
const CIRCLE_RADIUS = 9.15;
const CORNER_RADIUS = 1;

function drawMarkings(ctx, scale, color, width = LINE_HALF_W * 2) {
    // Canvas x runs along engine x, canvas y along minus engine y, so the texture lies flat
    // with three's x and z axes once the plane is rotated onto the ground.
    const px = (x) => (x + FULL_HALF_W) * scale;
    const py = (y) => (FULL_HALF_H - y) * scale;
    ctx.globalAlpha = 0.92;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width * scale;
    ctx.lineCap = 'butt';
    const line = (x0, y0, x1, y1) => {
        ctx.beginPath();
        ctx.moveTo(px(x0), py(y0));
        ctx.lineTo(px(x1), py(y1));
        ctx.stroke();
    };
    const arc = (x, y, r, a0, a1) => {
        // Engine angles turn counterclockwise, the flipped canvas y turns them clockwise.
        ctx.beginPath();
        ctx.arc(px(x), py(y), r * scale, -a1, -a0);
        ctx.stroke();
    };
    const spot = (x, y, r) => {
        ctx.beginPath();
        ctx.arc(px(x), py(y), r * scale, 0, Math.PI * 2);
        ctx.fill();
    };
    ctx.strokeRect(px(-HALF_W), py(HALF_H), HALF_W * 2 * scale, HALF_H * 2 * scale);
    line(0, -HALF_H, 0, HALF_H);
    arc(0, 0, CIRCLE_RADIUS, 0, Math.PI * 2);
    spot(0, 0, 0.15);
    for (const side of [-1, 1]) {
        const gx = side * HALF_W;
        const px16 = side * (HALF_W - PENALTY_DEPTH);
        const px5 = side * (HALF_W - GOAL_AREA_DEPTH);
        line(gx, -PENALTY_HALF_WIDTH, px16, -PENALTY_HALF_WIDTH);
        line(px16, -PENALTY_HALF_WIDTH, px16, PENALTY_HALF_WIDTH);
        line(px16, PENALTY_HALF_WIDTH, gx, PENALTY_HALF_WIDTH);
        line(gx, -GOAL_AREA_HALF_WIDTH, px5, -GOAL_AREA_HALF_WIDTH);
        line(px5, -GOAL_AREA_HALF_WIDTH, px5, GOAL_AREA_HALF_WIDTH);
        line(px5, GOAL_AREA_HALF_WIDTH, gx, GOAL_AREA_HALF_WIDTH);
        const sx = side * (HALF_W - PENALTY_SPOT);
        spot(sx, 0, 0.12);
        // The arc outside the box, where its distance from the spot passes the box line.
        const half = Math.acos((PENALTY_DEPTH - PENALTY_SPOT) / CIRCLE_RADIUS);
        if (side < 0) {
            arc(sx, 0, CIRCLE_RADIUS, -half, half);
        } else {
            arc(sx, 0, CIRCLE_RADIUS, Math.PI - half, Math.PI + half);
        }
        for (const cy of [-1, 1]) {
            const start = side < 0 ? (cy < 0 ? 0 : -Math.PI / 2) : (cy < 0 ? Math.PI / 2 : Math.PI);
            arc(gx, cy * HALF_H, CORNER_RADIUS, start, start + Math.PI / 2);
        }
    }
}

function hash(x, y) {
    let h = (x * 374761393 + y * 668265263) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const SNOW = new THREE.Color(0xd8dee6);
const LIGHT = 'rgba(255, 255, 230, 0.07)';
const DARK = 'rgba(0, 20, 0, 0.08)';

// Mowing patterns, centered on the pitch so they look the same from both ends.
function drawMowing(ctx, scale, w, h, mowing, width) {
    const px = (x) => (x + FULL_HALF_W) * scale;
    const py = (y) => (FULL_HALF_H - y) * scale;
    if (mowing === 'stripes' || mowing === 'checkers') {
        // Band edges fall on the goal lines and every band width between them.
        const n = Math.ceil((FULL_HALF_W * 2) / width) + 2;
        for (let k = -2; k < n; k++) {
            const x0 = -HALF_W + k * width;
            if (mowing === 'stripes') {
                ctx.fillStyle = (k & 1) ? LIGHT : DARK;
                ctx.fillRect(px(x0), 0, width * scale, h);
                continue;
            }
            const m = Math.ceil((FULL_HALF_H * 2) / width) + 2;
            for (let j = -m; j < m; j++) {
                const y0 = j * width - width / 2;
                ctx.fillStyle = ((k + j) & 1) ? LIGHT : DARK;
                ctx.fillRect(px(x0), py(y0 + width), width * scale, width * scale);
            }
        }
        if (mowing === 'stripes') {
            // Weak cross bands, as GRF's normal map has.
            for (let k = -4; k <= 4; k += 2) {
                const y0 = -5.5 + k * 11;
                ctx.fillStyle = 'rgba(255, 255, 230, 0.025)';
                ctx.fillRect(0, py(y0 + 11), w, 11 * scale);
            }
        }
    } else if (mowing === 'circles') {
        // Each ring is painted once, as an annulus, since the colors are translucent and
        // stacked discs would build up towards the centre.
        const rings = Math.ceil(Math.hypot(FULL_HALF_W, FULL_HALF_H) / width) + 1;
        for (let k = 1; k <= rings; k++) {
            ctx.fillStyle = (k & 1) ? LIGHT : DARK;
            ctx.beginPath();
            ctx.arc(px(0), py(0), k * width * scale, 0, Math.PI * 2);
            ctx.arc(px(0), py(0), (k - 1) * width * scale, 0, Math.PI * 2, true);
            ctx.fill();
        }
    }
}

// Grass color and mowing, then fine noise, then the markings on top.
export function makePitchTexture(renderer, quality, field = {}) {
    const scale = quality === 'high' ? 32 : quality === 'medium' ? 20 : 12;
    const w = Math.round(FULL_HALF_W * 2 * scale);
    const h = Math.round(FULL_HALF_H * 2 * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = field.grass || '#3f7f2c';
    ctx.fillRect(0, 0, w, h);
    drawMowing(ctx, scale, w, h, field.mowing || 'stripes', field.stripeWidth || 5.5);
    // Large soft patches break up the uniform color, like wear and uneven growth.
    for (let k = 0; k < 140; k++) {
        const x = hash(k, 1) * w;
        const y = hash(k, 2) * h;
        const r = (2 + hash(k, 3) * 8) * scale;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        const dark = hash(k, 4) < 0.5;
        g.addColorStop(0, dark ? 'rgba(30, 40, 10, 0.06)' : 'rgba(180, 200, 120, 0.05)');
        g.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
    // Wear in the goalmouths and the centre circle.
    for (const [cx, rr] of [[-HALF_W + 3, 5], [HALF_W - 3, 5], [0, 4]]) {
        const x = (cx + FULL_HALF_W) * scale;
        const y = FULL_HALF_H * scale;
        const g = ctx.createRadialGradient(x, y, 0, x, y, rr * scale);
        g.addColorStop(0, 'rgba(120, 110, 60, 0.18)');
        g.addColorStop(1, 'rgba(120, 110, 60, 0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - rr * scale, y - rr * scale, rr * 2 * scale, rr * 2 * scale);
    }
    const image = ctx.getImageData(0, 0, w, h);
    const d = image.data;
    for (let i = 0; i < d.length; i += 4) {
        const n = (Math.random() - 0.5) * 14;
        d[i] += n;
        d[i + 1] += n * 1.2;
        d[i + 2] += n * 0.6;
    }
    ctx.putImageData(image, 0, 0);
    drawMarkings(ctx, scale, field.lines || '#ffffff');
    ctx.globalAlpha = 1;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    return texture;
}

// A tiled normal map of grass blades, it carries the fine detail the color map cannot hold.
function makeGrassNormal() {
    const size = 256;
    const heights = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            heights[y * size + x] = Math.random();
        }
    }
    // Smear along one axis, blades lean in the mowing direction.
    const smooth = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let s = 0;
            for (let k = -2; k <= 2; k++) {
                s += heights[((y + k + size) % size) * size + x];
            }
            smooth[y * size + x] = s / 5;
        }
    }
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const l = smooth[y * size + ((x - 1 + size) % size)];
            const r = smooth[y * size + ((x + 1) % size)];
            const u = smooth[((y - 1 + size) % size) * size + x];
            const dn = smooth[((y + 1) % size) * size + x];
            const nx = (l - r) * 1.5;
            const ny = (u - dn) * 1.5;
            const nz = 1;
            const len = Math.hypot(nx, ny, nz);
            const i = (y * size + x) * 4;
            data[i] = (nx / len * 0.5 + 0.5) * 255;
            data[i + 1] = (ny / len * 0.5 + 0.5) * 255;
            data[i + 2] = (nz / len * 0.5 + 0.5) * 255;
            data[i + 3] = 255;
        }
    }
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(FULL_HALF_W * 2 / 1.5, FULL_HALF_H * 2 / 1.5);
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    texture.needsUpdate = true;
    return texture;
}

// One cell of the net, strands along two edges, with its own mip levels. Each smaller level is
// the average of the one above, lifted a little, so the net keeps its coverage when it is far
// away and reads as a light haze rather than vanishing. The material turns the alpha into
// coverage of the multisampled pixels, so no edge of a strand is cut off hard.
const NET_CELL = 64;
const NET_STRAND = 5;

function makeNetTexture() {
    const levels = [];
    let size = NET_CELL;
    let alpha = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const on = x < NET_STRAND || y < NET_STRAND;
            // A faint tint between the strands, the light caught by the net as a whole.
            alpha[y * size + x] = on ? 1 : 0.03;
        }
    }
    let level = 0;
    for (;;) {
        // Far levels hold at least this much, about a fifth.
        const lift = Math.min(1, level / 5);
        const data = new Uint8Array(size * size * 4);
        for (let i = 0; i < size * size; i++) {
            const v = Math.min(1, alpha[i] * (1 + 0.6 * lift) + 0.04 * lift);
            data[i * 4] = 255;
            data[i * 4 + 1] = 255;
            data[i * 4 + 2] = 255;
            data[i * 4 + 3] = Math.round(v * 255);
        }
        levels.push({ data, width: size, height: size });
        if (size === 1) {
            break;
        }
        const half = size / 2;
        const next = new Float32Array(half * half);
        for (let y = 0; y < half; y++) {
            for (let x = 0; x < half; x++) {
                const i = y * 2 * size + x * 2;
                next[y * half + x] = (alpha[i] + alpha[i + 1] + alpha[i + size] + alpha[i + size + 1]) / 4;
            }
        }
        alpha = next;
        size = half;
        level++;
    }
    const texture = new THREE.DataTexture(levels[0].data, NET_CELL, NET_CELL, THREE.RGBAFormat);
    texture.mipmaps = levels;
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 4;
    texture.needsUpdate = true;
    return texture;
}

// The net bulges where the ball hits it and springs back over about a second, as GRF's net
// does. Each goal keeps its latest hit, a point, the time and a direction, and the vertex shader
// pushes the net round the point with a damped swing. The edges tied to the goal frame at the
// mouth and on the ground stay put.
const NET_DECL = `
uniform vec4 uNetHit[2];
uniform vec3 uNetDir[2];
uniform float uNetAmp[2];
uniform float uNetTime;
`;
const NET_VERTEX = `
vec3 cjNetW = (modelMatrix * vec4(transformed, 1.0)).xyz;
vec3 cjNetPush = vec3(0.0);
float cjNetTie = smoothstep(0.0, 0.6, abs(cjNetW.x) - ${HALF_W.toFixed(2)}) * smoothstep(0.0, 0.35, cjNetW.y);
for (int i = 0; i < 2; i++) {
    float t = uNetTime - uNetHit[i].w;
    if (t >= 0.0 && t < 2.0) {
        vec3 d = cjNetW - uNetHit[i].xyz;
        float swing = exp(-3.6 * t) * sin(11.3 * t);
        cjNetPush += uNetDir[i] * (uNetAmp[i] * swing * exp(-dot(d, d) / 1.6));
    }
}
transformed += inverse(mat3(modelMatrix)) * (cjNetPush * cjNetTie);
`;

// The net material of either goal, the shipped one or GRF's, with the strong texture and the
// bulge.
function dressNet(material, texture, uniforms) {
    material.map = texture;
    material.alphaMap = null;
    material.transparent = false;
    material.alphaTest = 0;
    material.alphaToCoverage = true;
    material.side = THREE.DoubleSide;
    material.roughness = 0.85;
    material.metalness = 0;
    material.emissive = material.emissive || new THREE.Color();
    material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', `#include <common>\n${NET_DECL}`)
            .replace('#include <begin_vertex>', `#include <begin_vertex>\n${NET_VERTEX}`);
    };
    material.needsUpdate = true;
}

// A four cornered panel of net, split into cells of about 30 centimeters so it can bulge, with
// texture coordinates that put one strand cell every 12 centimeters.
function netPanel(corners, uvs, metersU, metersV) {
    const nu = Math.max(1, Math.ceil(metersU / 0.3));
    const nv = Math.max(1, Math.ceil(metersV / 0.3));
    const pos = [];
    const uv = [];
    const index = [];
    const lerp2 = (a, b, c, d, u, v) => a * (1 - u) * (1 - v) + b * u * (1 - v) + c * u * v + d * (1 - u) * v;
    for (let j = 0; j <= nv; j++) {
        for (let i = 0; i <= nu; i++) {
            const u = i / nu;
            const v = j / nv;
            for (const k of ['x', 'y', 'z']) {
                pos.push(lerp2(corners[0][k], corners[1][k], corners[2][k], corners[3][k], u, v));
            }
            uv.push(lerp2(uvs[0][0], uvs[1][0], uvs[2][0], uvs[3][0], u, v), lerp2(uvs[0][1], uvs[1][1], uvs[2][1], uvs[3][1], u, v));
        }
    }
    for (let j = 0; j < nv; j++) {
        for (let i = 0; i < nu; i++) {
            const a = j * (nu + 1) + i;
            index.push(a, a + 1, a + nu + 2, a, a + nu + 2, a + nu + 1);
        }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.computeVertexNormals();
    return geo;
}

function makeGoal(side, netMat) {
    const group = new THREE.Group();
    const white = new THREE.MeshStandardMaterial({ color: 0xf4f4f4, roughness: 0.35, metalness: 0.1 });
    const postGeo = new THREE.CylinderGeometry(POST_RADIUS, POST_RADIUS, GOAL_HEIGHT + POST_RADIUS, 20);
    // In three, engine y maps to minus z.
    for (const y of [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH]) {
        const post = new THREE.Mesh(postGeo, white);
        post.position.set(side * HALF_W, (GOAL_HEIGHT + POST_RADIUS) / 2, -y);
        post.castShadow = true;
        group.add(post);
    }
    const bar = new THREE.Mesh(
        new THREE.CylinderGeometry(POST_RADIUS, POST_RADIUS, GOAL_HALF_WIDTH * 2 + POST_RADIUS * 2, 20),
        white,
    );
    bar.rotation.x = Math.PI / 2;
    bar.position.set(side * HALF_W, GOAL_HEIGHT, 0);
    bar.castShadow = true;
    group.add(bar);
    // Thin back frame, the stanchions that hold the net.
    const frameMat = new THREE.MeshStandardMaterial({ color: 0xcfd3d6, roughness: 0.5, metalness: 0.3 });
    const back = side * (HALF_W + GOAL_DEPTH);
    const rodGeo = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
    const rod = (a, b) => {
        const m = new THREE.Mesh(rodGeo, frameMat);
        const d = new THREE.Vector3().subVectors(b, a);
        m.scale.y = d.length();
        m.position.copy(a).addScaledVector(d, 0.5);
        m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
        m.castShadow = true;
        group.add(m);
    };
    for (const z of [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH]) {
        rod(new THREE.Vector3(side * HALF_W, GOAL_HEIGHT, z), new THREE.Vector3(back, GOAL_HEIGHT * 0.85, z));
        rod(new THREE.Vector3(back, 0, z), new THREE.Vector3(back, GOAL_HEIGHT * 0.85, z));
        rod(new THREE.Vector3(side * HALF_W, 0.02, z), new THREE.Vector3(back, 0.02, z));
    }
    rod(new THREE.Vector3(back, GOAL_HEIGHT * 0.85, -GOAL_HALF_WIDTH), new THREE.Vector3(back, GOAL_HEIGHT * 0.85, GOAL_HALF_WIDTH));
    const cell = 0.12;
    const quad = (corners, uLen, vLen) => {
        const geo = netPanel(corners, [[0, 0], [uLen / cell, 0], [uLen / cell, vLen / cell], [0, vLen / cell]], uLen, vLen);
        const mesh = new THREE.Mesh(geo, netMat);
        mesh.castShadow = true;
        group.add(mesh);
    };
    const v = (x, y, z) => new THREE.Vector3(x, y, z);
    const gx = side * HALF_W;
    const hb = GOAL_HEIGHT * 0.85;
    const W = GOAL_HALF_WIDTH;
    quad([v(back, 0, -W), v(back, 0, W), v(back, hb, W), v(back, hb, -W)], W * 2, hb);
    quad([v(gx, GOAL_HEIGHT, -W), v(gx, GOAL_HEIGHT, W), v(back, hb, W), v(back, hb, -W)], W * 2, GOAL_DEPTH);
    for (const z of [-W, W]) {
        const pts = [v(gx, 0, z), v(back, 0, z), v(back, hb, z), v(gx, GOAL_HEIGHT, z)];
        const geo = netPanel(pts, [[0, 0], [GOAL_DEPTH / cell, 0], [GOAL_DEPTH / cell, hb / cell], [0, GOAL_HEIGHT / cell]],
            GOAL_DEPTH, GOAL_HEIGHT);
        const mesh = new THREE.Mesh(geo, netMat);
        mesh.castShadow = true;
        group.add(mesh);
    }
    return group;
}

// A band around every line, which snow leaves clear so the markings stay visible, as when the
// ground staff sweep them.
function makeLineMask() {
    const scale = 6;
    const canvas = document.createElement('canvas');
    canvas.width = FULL_HALF_W * 2 * scale;
    canvas.height = FULL_HALF_H * 2 * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.filter = 'blur(1px)';
    drawMarkings(ctx, scale, '#ffffff', 0.6);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.NoColorSpace;
    return texture;
}

// Rain darkens the grass and pools in shallow puddles that catch the light, snow settles in
// patches that grow together, both from the same slow noise in meters across the pitch.
const SURFACE_DECL = `
uniform float uWet;
uniform float uSnow;
uniform vec3 uSnowColor;
uniform sampler2D uLineMask;
float cjHash2(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x);
}
float cjNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(cjHash2(i), cjHash2(i + vec2(1.0, 0.0)), f.x), mix(cjHash2(i + vec2(0.0, 1.0)), cjHash2(i + 1.0), f.x), f.y);
}
`;

const SURFACE_MAP = `
vec2 cjM = vMapUv * vec2(${(FULL_HALF_W * 2).toFixed(1)}, ${(FULL_HALF_H * 2).toFixed(1)});
float cjN = cjNoise(cjM * 0.12) * 0.6 + cjNoise(cjM * 0.45) * 0.3 + cjNoise(cjM * 2.1) * 0.1;
float cjPuddle = smoothstep(0.7, 0.86, cjNoise(cjM * 0.7) * 0.6 + cjN * 0.4) * smoothstep(0.3, 0.9, uWet) * 0.8;
diffuseColor.rgb *= 1.0 - 0.3 * uWet - 0.05 * cjPuddle;
float cjSnow = clamp(uSnow * 1.6 - (1.0 - cjN) * 0.6, 0.0, 1.0) * (1.0 - texture2D(uLineMask, vMapUv).r);
diffuseColor.rgb = mix(diffuseColor.rgb, uSnowColor, cjSnow);
`;

const SURFACE_ROUGH = `
roughnessFactor = mix(roughnessFactor, 0.55, uWet * 0.8);
roughnessFactor = mix(roughnessFactor, 0.3, cjPuddle);
roughnessFactor = mix(roughnessFactor, 0.8, cjSnow);
`;

export class Pitch {
    constructor(scene, renderer, quality) {
        this.renderer = renderer;
        this.group = new THREE.Group();
        scene.add(this.group);
        this.material = new THREE.MeshStandardMaterial({
            roughness: 0.92,
            metalness: 0,
            normalMap: makeGrassNormal(),
            normalScale: new THREE.Vector2(0.35, 0.35),
        });
        this.surface = {
            uWet: { value: 0 },
            uSnow: { value: 0 },
            uSnowColor: { value: new THREE.Color(0xe9eef4) },
            uLineMask: { value: makeLineMask() },
        };
        this.material.onBeforeCompile = (shader) => {
            Object.assign(shader.uniforms, this.surface);
            shader.fragmentShader = shader.fragmentShader
                .replace('#include <common>', `#include <common>
${SURFACE_DECL}`)
                .replace('#include <map_fragment>', `#include <map_fragment>
${SURFACE_MAP}`)
                .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
${SURFACE_ROUGH}`);
        };
        const grass = new THREE.Mesh(new THREE.PlaneGeometry(FULL_HALF_W * 2, FULL_HALF_H * 2), this.material);
        grass.rotation.x = -Math.PI / 2;
        grass.receiveShadow = true;
        this.group.add(grass);
        // Surround between the grass and the stands, a darker strip of artificial turf.
        this.surroundBase = new THREE.Color(0x24442a);
        this.surroundRough = 1;
        this.surroundMaterial = new THREE.MeshStandardMaterial({ color: 0x24442a, roughness: 1 });
        const surround = new THREE.Mesh(
            new THREE.PlaneGeometry(FULL_HALF_W * 2 + 24, FULL_HALF_H * 2 + 24),
            this.surroundMaterial,
        );
        surround.rotation.x = -Math.PI / 2;
        surround.position.y = -0.02;
        surround.receiveShadow = true;
        this.group.add(surround);
        // The net texture and the bulge belong to the pitch, both goal models share them.
        this.netTexture = makeNetTexture();
        this.netUniforms = {
            uNetHit: { value: [new THREE.Vector4(0, 0, 0, -10), new THREE.Vector4(0, 0, 0, -10)] },
            uNetDir: { value: [new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 0, 0)] },
            uNetAmp: { value: [0, 0] },
            uNetTime: { value: 0 },
        };
        this.netEpoch = performance.now();
        const netMaterial = new THREE.MeshStandardMaterial({ name: 'net', color: 0xffffff });
        dressNet(netMaterial, this.netTexture, this.netUniforms);
        this.goals = new THREE.Group();
        this.goals.add(makeGoal(-1, netMaterial), makeGoal(1, netMaterial));
        // Everything the procedural goals are made of belongs to them.
        this.goalsOwned = [];
        this.goals.traverse((obj) => {
            if (obj.isMesh) {
                this.goalsOwned.push(obj.geometry, obj.material);
            }
        });
        this.field = {};
        this.group.add(this.goals);
        this.classicMap = null;
        this.classic = false;
        this.setQuality(quality);
    }

    // GRF's goals, a model in the engine frame with both goals placed, turned into three's frame.
    // The clone shares geometries and textures with the loaded model, which other views use too,
    // so only its materials are copied, which lets the net color change without touching them.
    useGoals(scene) {
        const model = scene.clone();
        model.rotation.x = -Math.PI / 2;
        const owned = [];
        const copies = new Map();
        const copy = (mat) => {
            if (!copies.has(mat)) {
                const c = mat.clone();
                copies.set(mat, c);
                owned.push(c);
            }
            return copies.get(mat);
        };
        model.traverse((obj) => {
            if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
                obj.material = Array.isArray(obj.material) ? obj.material.map(copy) : copy(obj.material);
            }
        });
        // GRF's net is a cross of one texel in 32, which a far camera averages away.
        for (const m of copies.values()) {
            if (/net/i.test(m.name || '')) {
                dressNet(m, this.netTexture, this.netUniforms);
            }
        }
        this.group.remove(this.goals);
        for (const d of this.goalsOwned) {
            d.dispose();
        }
        this.goalsOwned = owned;
        this.goals = model;
        this.group.add(model);
        this.applyNet();
    }

    // The ball hits a net. Side is 0 for the left goal, at negative x, and 1 for the right, point
    // is where in three's frame, strength from 0 to 1. Age, in seconds, starts the swing that long
    // ago, for a hit replayed while scrubbing. The goal line is at x equal to plus or minus HALF_W.
    netImpact(side, point, strength = 1, age = 0) {
        const k = side ? 1 : 0;
        const sx = side ? 1 : -1;
        const u = this.netUniforms;
        const p = point && point.isVector3 ? point : new THREE.Vector3(point.x, point.y, point.z);
        u.uNetHit.value[k].set(p.x, p.y, p.z, this.netClock() - Math.max(0, age));
        // Outward from the middle of the goal mouth, so the back net goes back, a side net out
        // to the side and the roof up.
        const dir = u.uNetDir.value[k].set(p.x - sx * HALF_W, p.y - GOAL_HEIGHT * 0.4, p.z);
        dir.x = sx * Math.max(Math.abs(dir.x), 0.6);
        dir.normalize();
        u.uNetAmp.value[k] = 0.75 * THREE.MathUtils.clamp(Number(strength) || 0, 0, 1);
    }

    netClock() {
        return (performance.now() - this.netEpoch) / 1000;
    }

    // Every frame, the time of the net's swing.
    update() {
        this.netUniforms.uNetTime.value = this.netClock();
    }

    // How wet the ground is and how much snow lies on it, each from 0 to 1.
    setSurface(wet, snow) {
        const u = this.surface;
        if (Math.abs(u.uWet.value - wet) < 0.002 && Math.abs(u.uSnow.value - snow) < 0.002 && !this.groundChanged) {
            return;
        }
        u.uWet.value = wet;
        u.uSnow.value = snow;
        // Snow smooths the blades out of sight.
        const n = 0.35 * (1 - 0.8 * Math.min(1, snow * 1.5));
        this.material.normalScale.set(n, n);
        const m = this.surroundMaterial;
        m.color.copy(this.surroundBase).multiplyScalar(1 - 0.35 * wet).lerp(SNOW, Math.min(1, snow * 1.3));
        m.roughness = this.surroundRough * (1 - 0.5 * wet);
        this.groundChanged = false;
    }

    // The strip of ground between the grass and the stands, which arenas recolor.
    setGround(color = 0x24442a, roughness = 1, metalness = 0) {
        this.surroundBase.set(color);
        this.surroundRough = roughness;
        this.surroundMaterial.metalness = metalness;
        this.groundChanged = true;
        this.setSurface(this.surface.uWet.value, this.surface.uSnow.value);
    }

    // Grass, mowing and line colors, and the net color, from the field settings.
    setField(field) {
        const old = this.field;
        this.field = field;
        if (old.grass !== field.grass || old.mowing !== field.mowing || old.stripeWidth !== field.stripeWidth || old.lines !== field.lines) {
            this.quality = null;
            this.setQuality(this.lastQuality);
        }
        this.applyNet();
    }

    applyNet() {
        const color = new THREE.Color(this.field.net || '#ffffff');
        this.goals.traverse((obj) => {
            if (!obj.isMesh) {
                return;
            }
            for (const m of [].concat(obj.material)) {
                if (m && /net/i.test(m.name || '')) {
                    m.color.copy(color);
                    // A little light of its own, so the net reads against the crowd.
                    m.emissive.copy(color).multiplyScalar(0.06);
                }
            }
        });
    }

    // GRF's own pitch bake, or the sharper procedural pitch drawn to the same measurements.
    setClassic(on, map) {
        if (map) {
            this.classicMap = map;
        }
        this.classic = on && !!this.classicMap;
        this.material.map = this.classic ? this.classicMap : this.proceduralMap;
        this.material.needsUpdate = true;
    }

    setQuality(quality) {
        if (this.quality === quality) {
            return;
        }
        this.quality = quality;
        this.lastQuality = quality;
        if (this.proceduralMap) {
            this.proceduralMap.dispose();
        }
        this.proceduralMap = makePitchTexture(this.renderer, quality, this.field || {});
        this.material.map = this.classic ? this.classicMap : this.proceduralMap;
        this.material.needsUpdate = true;
    }
}
