// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Arenas, the surroundings of the pitch. The match always plays on GRF's pitch with its lines and
// goals, only what lies around it changes. The classic stadium is the bowl of stadium.js, the
// others take the bowl away and build their own world, all of it procedural and cheap, a few
// instanced meshes and small shaders each.

import * as THREE from 'three';
import { FULL_HALF_H, FULL_HALF_W } from './pitch.js';

export const ARENAS = ['stadium', 'beach', 'rooftop', 'space', 'neon'];
export const ARENA_LABELS = {
    stadium: 'Classic stadium',
    beach: 'Beach',
    rooftop: 'City rooftop',
    space: 'Space station',
    neon: 'Neon arcade',
};
export const ARENA_NOTES = {
    stadium: 'Two tiers of fans under a roof, floodlights on four towers.',
    beach: 'Sand, palms and the sea behind the far touch line, fans under umbrellas.',
    rooftop: 'A pitch on top of a tower, small stands of fans and the city lights all around.',
    space: 'A deck in orbit with galleries of fans, the Earth turning below.',
    neon: 'A glowing grid to the horizon, a crowd of light, neon pylons and a striped sun, best at night.',
};
// The sound bed of each arena, for app.sound.setAmbience.
export const ARENA_AMBIENCE = { beach: 'waves', rooftop: 'city', space: 'hum', neon: 'hum' };

const SHORE_Z = -150;
const SPACE_SUN = new THREE.Vector3(-0.55, 0.5, -0.67).normalize();

function rand(seed) {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function canvasTexture(w, h, draw, srgb = true) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    draw(canvas.getContext('2d'), w, h);
    const texture = new THREE.CanvasTexture(canvas);
    if (srgb) {
        texture.colorSpace = THREE.SRGBColorSpace;
    }
    return texture;
}

// Fine grain over a base color, tiled, for sand, concrete and the like.
function grainTexture(base, spread, seed) {
    const r = rand(seed);
    const texture = canvasTexture(256, 256, (ctx, w, h) => {
        ctx.fillStyle = base;
        ctx.fillRect(0, 0, w, h);
        const img = ctx.getImageData(0, 0, w, h);
        for (let i = 0; i < img.data.length; i += 4) {
            const n = (r() - 0.5) * spread;
            img.data[i] += n;
            img.data[i + 1] += n;
            img.data[i + 2] += n * 0.9;
        }
        ctx.putImageData(img, 0, 0);
    });
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    return texture;
}

// Shared code of the shader materials here, value noise and the scene's fog by hand.
const NOISE = `
float aHash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float aNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(aHash(i), aHash(i + vec3(1, 0, 0)), f.x), mix(aHash(i + vec3(0, 1, 0)), aHash(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(aHash(i + vec3(0, 0, 1)), aHash(i + vec3(1, 0, 1)), f.x), mix(aHash(i + vec3(0, 1, 1)), aHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float aFbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * aNoise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return s;
}
uniform vec3 uFogColor;
uniform float uFogDensity;
vec3 aFog(vec3 c, float dist) {
    float f = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);
    return mix(c, uFogColor, clamp(f, 0.0, 1.0));
}
`;

const FINISH = `
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
`;

function fogUniforms() {
    return { uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.001 } };
}

// The sea, waves from a sum of sines, the sky in its reflection, the sun's glitter and foam
// along the shore.
const SEA_VERTEX = `
varying vec3 vWorld;
void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
}`;
const SEA_FRAGMENT = `
${NOISE}
uniform float uTime;
uniform vec3 uSky;
uniform vec3 uDeep;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uShore;
varying vec3 vWorld;
void main() {
    vec2 p = vWorld.xz;
    float t = uTime;
    vec2 g = vec2(0.0);
    g += vec2(0.20, 0.90) * cos(dot(p, vec2(0.020, 0.090)) + t * 0.9) * 0.10;
    g += vec2(-0.6, 0.80) * cos(dot(p, vec2(-0.07, 0.10)) + t * 1.3) * 0.06;
    g += vec2(0.80, 0.50) * cos(dot(p, vec2(0.21, 0.13)) + t * 2.1) * 0.03;
    g += vec2(aNoise(vec3(p * 0.35, t * 0.4)) - 0.5, aNoise(vec3(p.yx * 0.35, t * 0.4 + 7.0)) - 0.5) * 0.12;
    vec3 n = normalize(vec3(-g.x, 1.0, -g.y));
    vec3 toCam = cameraPosition - vWorld;
    float dist = length(toCam);
    vec3 v = toCam / dist;
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    vec3 col = mix(uDeep, uSky, fres);
    float spec = pow(max(dot(reflect(-v, n), uSunDir), 0.0), 300.0);
    col += uSunColor * spec * 6.0;
    float d = vWorld.z - uShore;
    float foam = smoothstep(-14.0, -1.0, d) * smoothstep(0.35, 0.7, aNoise(vec3(p * 0.4, t * 0.8)) + 0.4 * sin(d * 0.8 + t * 1.6));
    col = mix(col, uSky * 1.1 + 0.05, foam * 0.75);
    gl_FragColor = vec4(aFog(col, dist), 1.0);
    ${FINISH}
}`;

// Towers of the city, each an instance of a box, with windows that light up as the day goes.
const CITY_VERTEX = `
varying vec3 vWorld;
varying vec3 vNormalW;
varying vec2 vSeed;
void main() {
    vec4 w = modelMatrix * instanceMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    vNormalW = normalize(mat3(modelMatrix * instanceMatrix) * normal);
    vSeed = instanceMatrix[3].xz;
    gl_Position = projectionMatrix * viewMatrix * w;
}`;
const CITY_FRAGMENT = `
${NOISE}
uniform vec3 uHemi;
uniform vec3 uSunColor;
uniform vec3 uSunDir;
uniform float uNight;
uniform float uTime;
varying vec3 vWorld;
varying vec3 vNormalW;
varying vec2 vSeed;
void main() {
    vec3 n = normalize(vNormalW);
    float tone = 0.55 + 0.45 * aHash(vec3(vSeed, 3.0));
    vec3 facade = vec3(0.075, 0.08, 0.09) * tone;
    vec3 col = facade * (uHemi * (0.55 + 0.45 * n.y) + uSunColor * max(dot(n, uSunDir), 0.0));
    if (abs(n.y) < 0.5) {
        float along = abs(n.x) > 0.5 ? vWorld.z : vWorld.x;
        vec2 cell = vec2(along / 3.4, vWorld.y / 3.8);
        vec2 f = fract(cell);
        float pane = step(0.18, f.x) * step(f.x, 0.82) * step(0.22, f.y) * step(f.y, 0.78);
        float h = aHash(vec3(floor(cell), vSeed.x * 0.37 + vSeed.y));
        float lit = step(0.62, h) * uNight;
        vec3 warm = mix(vec3(1.0, 0.72, 0.38), vec3(0.75, 0.85, 1.0), step(0.85, h));
        col = mix(col, vec3(0.02, 0.025, 0.035) + uHemi * 0.12, pane * (1.0 - lit));
        col += pane * lit * warm * (1.2 + 1.4 * h);
    }
    float dist = length(cameraPosition - vWorld);
    gl_FragColor = vec4(aFog(col, dist), 1.0);
    ${FINISH}
}`;

// The Earth below the station. Its surface is noise, continents, ice and clouds, painted once
// into a map on the graphics card, since the noise costs several milliseconds a frame drawn live.
// The map holds the color and, in its alpha, where cities light the night side.
const EARTH_BAKE = `
${NOISE}
varying vec2 vUv;
void main() {
    float lon = (vUv.x - 0.5) * 6.2831853;
    float lat = (vUv.y - 0.5) * 3.1415927;
    vec3 o = vec3(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
    float h = aFbm(o * 2.6 + 3.1);
    float land = smoothstep(0.5, 0.53, h);
    float ice = smoothstep(0.82, 0.9, abs(o.y) + 0.05 * aNoise(o * 9.0));
    vec3 ocean = mix(vec3(0.01, 0.05, 0.16), vec3(0.02, 0.12, 0.28), smoothstep(0.35, 0.5, h));
    vec3 ground = mix(vec3(0.10, 0.20, 0.06), vec3(0.42, 0.34, 0.20), smoothstep(0.58, 0.72, h));
    vec3 col = mix(ocean, ground, land);
    col = mix(col, vec3(0.9), ice);
    float cloud = smoothstep(0.5, 0.72, aFbm(o * 5.0));
    col = mix(col, vec3(0.95), cloud * 0.85);
    float cities = land * (1.0 - ice) * (1.0 - cloud) * step(0.7, aHash(floor(o * 700.0))) * smoothstep(0.45, 0.7, aNoise(o * 14.0));
    gl_FragColor = vec4(sqrt(col), cities);
}`;
const BAKE_VERTEX = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
}`;
const EARTH_VERTEX = `
varying vec3 vObj;
varying vec3 vNormalW;
varying vec3 vWorld;
void main() {
    vObj = normal;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
}`;
const EARTH_FRAGMENT = `
uniform vec3 uSunDir;
uniform sampler2D uMap;
varying vec3 vObj;
varying vec3 vNormalW;
varying vec3 vWorld;
void main() {
    vec3 o = normalize(vObj);
    vec2 uv = vec2(atan(o.z, o.x) / 6.2831853 + 0.5, asin(clamp(o.y, -1.0, 1.0)) / 3.1415927 + 0.5);
    vec4 m = texture2D(uMap, uv);
    vec3 col = m.rgb * m.rgb;
    vec3 n = normalize(vNormalW);
    float ndl = dot(n, uSunDir);
    float day = smoothstep(-0.08, 0.25, ndl);
    vec3 lit = col * (0.015 + 1.7 * max(ndl, 0.0));
    lit += vec3(1.0, 0.75, 0.4) * m.a * (1.0 - day) * 0.7;
    vec3 v = normalize(cameraPosition - vWorld);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    lit += vec3(0.25, 0.5, 1.0) * rim * (0.15 + 1.3 * smoothstep(-0.3, 0.4, ndl));
    gl_FragColor = vec4(lit, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
}`;

// The floor of the arcade, a grid of glowing lines that pulse outwards from the pitch, fading
// into the haze at the horizon.
const GRID_VERTEX = `
varying vec3 vWorld;
void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
}`;
const GRID_FRAGMENT = `
${NOISE}
uniform float uTime;
uniform vec3 uLine;
uniform vec3 uGlow;
varying vec3 vWorld;
void main() {
    vec2 p = vWorld.xz / 6.0;
    vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p);
    float r = length(vWorld.xz);
    // Far lines thin out, so the horizon reads as a glow rather than a moire.
    float line = (1.0 - min(min(g.x, g.y), 1.0)) * (1.0 - 0.8 * smoothstep(140.0, 650.0, r));
    float wave = 0.55 + 0.45 * sin(r * 0.045 - uTime * 1.6);
    vec3 col = vec3(0.012, 0.006, 0.03) + uLine * line * (0.9 + 1.6 * wave);
    col += uGlow * 0.05 * smoothstep(900.0, 120.0, r);
    float dist = length(cameraPosition - vWorld);
    gl_FragColor = vec4(aFog(col, dist), 1.0);
    ${FINISH}
}`;
// The sun of the arcade, a disc with bands cut out of its lower half, hung at the horizon.
const SUN_VERTEX = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const SUN_FRAGMENT = `
uniform float uTime;
varying vec2 vUv;
void main() {
    vec2 c = vUv * 2.0 - 1.0;
    if (length(c) > 1.0) {
        discard;
    }
    float y = vUv.y;
    float band = step(0.5, fract(y * 14.0 - uTime * 0.15));
    if (y < 0.5 && y > 0.08 && band < 0.5) {
        discard;
    }
    vec3 col = mix(vec3(1.0, 0.18, 0.55), vec3(1.0, 0.85, 0.25), smoothstep(0.1, 0.95, y));
    gl_FragColor = vec4(col * 1.6, 1.0);
    ${FINISH}
}`;

// A palm, a curved trunk and a crown of drooping fronds.
function palmGeometries() {
    const trunk = new THREE.CylinderGeometry(0.16, 0.3, 9, 7, 6);
    trunk.translate(0, 4.5, 0);
    const p = trunk.attributes.position;
    for (let i = 0; i < p.count; i++) {
        const y = p.getY(i);
        p.setX(i, p.getX(i) + 0.022 * y * y);
    }
    trunk.computeVertexNormals();
    const frond = new THREE.PlaneGeometry(1.5, 5, 1, 8);
    frond.translate(0, 2.5, 0);
    const f = frond.attributes.position;
    for (let i = 0; i < f.count; i++) {
        const y = f.getY(i);
        // Out along the leaf it bows down.
        f.setZ(i, -0.32 * y * y);
    }
    frond.rotateX(-Math.PI / 2 + 0.85);
    frond.computeVertexNormals();
    return { trunk, frond };
}

// A frond, leaflets filled in along a midrib, wide enough to stay visible from afar once the
// mipmaps average it.
function frondTexture() {
    return canvasTexture(64, 256, (ctx, w, h) => {
        ctx.clearRect(0, 0, w, h);
        for (let y = 4; y < h - 4; y += 9) {
            const reach = (w / 2 - 1) * Math.sin(Math.PI * Math.min(1, (h - y) / h * 1.1));
            ctx.fillStyle = (y / 9) % 2 < 1 ? '#4f8a2c' : '#3f7524';
            for (const s of [-1, 1]) {
                ctx.beginPath();
                ctx.moveTo(w / 2, y + 4);
                ctx.lineTo(w / 2 + s * reach, y - 6);
                ctx.lineTo(w / 2 + s * reach * 0.8, y + 2);
                ctx.lineTo(w / 2, y + 10);
                ctx.fill();
            }
        }
        ctx.fillStyle = '#5a7a30';
        ctx.fillRect(w / 2 - 2, 0, 4, h);
    });
}

function stripeTexture(a, b) {
    return canvasTexture(128, 16, (ctx, w, h) => {
        for (let i = 0; i < 8; i++) {
            ctx.fillStyle = i % 2 ? a : b;
            ctx.fillRect(i * w / 8, 0, w / 8, h);
        }
    });
}

// Places outside the pitch and its boards, clear of the main camera's line of sight from the
// near side.
function aroundPitch(r, count, minX, minZ, maxR, nearOk = false) {
    const out = [];
    let guard = 0;
    while (out.length < count && guard++ < count * 50) {
        const x = (r() * 2 - 1) * maxR;
        const z = (r() * 2 - 1) * maxR * 0.75;
        if (Math.abs(x) < minX && Math.abs(z) < minZ) {
            continue;
        }
        if (!nearOk && z > 0 && Math.abs(x) < minX + 10) {
            continue;
        }
        out.push([x, z]);
    }
    return out;
}

export class Arena {
    constructor(scene) {
        this.scene = scene;
        this.group = new THREE.Group();
        this.group.name = 'arena';
        scene.add(this.group);
        this.name = 'stadium';
        this.owned = [];
        this.shaders = [];
    }

    // Builds the named arena and tells the stadium and the pitch what to show.
    set(name, app) {
        const next = ARENAS.includes(name) ? name : 'stadium';
        if (next !== this.name) {
            this.clear();
            this.name = next;
            if (next === 'beach') {
                this.buildBeach();
            } else if (next === 'rooftop') {
                this.buildRooftop();
            } else if (next === 'space') {
                this.buildSpace(app.view.renderer);
            } else if (next === 'neon') {
                this.buildNeon();
            }
        }
        this.dress(app);
    }

    // The stadium's bowl and masts, and the ground around the grass, for the current arena. The
    // players wear glass helmets in space, see players.js.
    dress(app) {
        this.group.userData.helmets = this.name === 'space';
        this.group.userData.sun = SPACE_SUN;
        if (app.stadium && app.stadium.setArena) {
            app.stadium.setArena(this.name, this.name !== 'space');
        }
        if (app.pitch && app.pitch.setGround) {
            const ground = {
                stadium: [0x24442a, 1, 0], beach: [0xd9c79c, 1, 0], rooftop: [0x55585c, 0.9, 0], space: [0x2b3138, 0.45, 0.6],
                neon: [0x120a1e, 0.5, 0.3],
            }[this.name];
            app.pitch.setGround(...ground);
        }
    }

    clear() {
        for (const obj of this.owned) {
            obj.dispose();
        }
        this.owned = [];
        this.shaders = [];
        this.group.clear();
        this.earth = null;
        this.sea = null;
    }

    own(...things) {
        this.owned.push(...things);
        return things[0];
    }

    shader(uniforms, vertexShader, fragmentShader, options = {}) {
        const material = this.own(new THREE.ShaderMaterial({
            uniforms: { ...fogUniforms(), uTime: { value: 0 }, ...uniforms }, vertexShader, fragmentShader, ...options,
        }));
        this.shaders.push(material);
        return material;
    }

    mesh(geometry, material, props = {}) {
        this.own(geometry);
        const m = new THREE.Mesh(geometry, material);
        Object.assign(m, props);
        this.group.add(m);
        return m;
    }

    // Sand all round, the sea beyond the far touch line, palms and umbrellas.
    buildBeach() {
        const sandMap = this.own(grainTexture('#d6c39a', 26, 5));
        sandMap.repeat.set(160, 160);
        const sand = this.own(new THREE.MeshStandardMaterial({ map: sandMap, roughness: 1 }));
        const ground = new THREE.PlaneGeometry(1600, 700 + Math.abs(SHORE_Z));
        ground.rotateX(-Math.PI / 2);
        ground.translate(0, -0.06, (700 + SHORE_Z) / 2);
        this.mesh(ground, sand, { receiveShadow: true });
        // A slope down into the water, so the shore line is not a hard edge.
        const slope = new THREE.PlaneGeometry(1600, 30, 1, 1);
        slope.rotateX(-Math.PI / 2 - 0.05);
        slope.translate(0, -0.8, SHORE_Z - 15);
        this.mesh(slope, sand);
        const seaGeo = new THREE.PlaneGeometry(4000, 2400);
        seaGeo.rotateX(-Math.PI / 2);
        seaGeo.translate(0, -0.5, SHORE_Z - 1200 + 8);
        this.sea = this.mesh(seaGeo, this.shader({
            uSky: { value: new THREE.Color() }, uDeep: { value: new THREE.Color(0x05303a) },
            uSunDir: { value: new THREE.Vector3() }, uSunColor: { value: new THREE.Color() }, uShore: { value: SHORE_Z - 2 },
        }, SEA_VERTEX, SEA_FRAGMENT));
        const r = rand(17);
        const { trunk, frond } = palmGeometries();
        this.own(trunk, frond);
        const spots = aroundPitch(r, 34, FULL_HALF_W + 14, FULL_HALF_H + 12, 120).filter(([, z]) => z > SHORE_Z + 20);
        const bark = this.own(new THREE.MeshStandardMaterial({ map: this.own(grainTexture('#7a6248', 60, 9)), roughness: 0.9 }));
        const leaves = this.own(new THREE.MeshStandardMaterial({
            map: this.own(frondTexture()), alphaTest: 0.3, side: THREE.DoubleSide, roughness: 0.8,
        }));
        const trunks = new THREE.InstancedMesh(trunk, bark, spots.length);
        const fronds = new THREE.InstancedMesh(frond, leaves, spots.length * 11);
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const top = new THREE.Vector3();
        spots.forEach(([x, z], i) => {
            const s = 0.8 + 0.5 * r();
            const yaw = r() * Math.PI * 2;
            q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
            m.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(s, s, s));
            trunks.setMatrixAt(i, m);
            top.set(0.022 * 81 * s, 9 * s, 0).applyQuaternion(q).add(new THREE.Vector3(x, 0, z));
            for (let k = 0; k < 11; k++) {
                const fq = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw + k * Math.PI * 2 / 11 + r() * 0.3, 0));
                m.compose(top, fq, new THREE.Vector3(s, s, s * (0.8 + 0.4 * r())));
                fronds.setMatrixAt(i * 11 + k, m);
            }
        });
        trunks.castShadow = true;
        fronds.castShadow = true;
        this.group.add(this.own(trunks), this.own(fronds));
        // Umbrellas over the crowds along the touch lines, and scattered behind them and the ends.
        const cone = this.own(new THREE.ConeGeometry(1.5, 0.6, 10, 1, true));
        cone.translate(0, 2.4, 0);
        const pole = this.own(new THREE.CylinderGeometry(0.03, 0.03, 2.4, 5));
        pole.translate(0, 1.2, 0);
        const colors = [['#e8463a', '#f4efe2'], ['#2a6fd6', '#f4efe2'], ['#f2b33a', '#2a6fd6']];
        const spotsU = aroundPitch(r, 27, FULL_HALF_W + 8, FULL_HALF_H + 7, 95, true)
            .filter(([x, z]) => z > SHORE_Z + 12 && !(Math.abs(x) < 52 && Math.abs(z) < 53));
        for (const side of [-1, 1]) {
            for (let x = -42; x <= 42; x += 12) {
                spotsU.push([x + (side > 0 ? 0 : 6) + (r() - 0.5) * 2, side * (46.4 + r() * 0.8)]);
            }
        }
        colors.forEach(([a, b], c) => {
            const mat = this.own(new THREE.MeshStandardMaterial({ map: this.own(stripeTexture(a, b)), side: THREE.DoubleSide, roughness: 0.7 }));
            const list = spotsU.filter((_, i) => i % 3 === c);
            const tops = new THREE.InstancedMesh(cone, mat, list.length);
            const poles = new THREE.InstancedMesh(pole, bark, list.length);
            list.forEach(([x, z], i) => {
                m.makeRotationY(r() * 6.28).setPosition(x, 0, z);
                tops.setMatrixAt(i, m);
                poles.setMatrixAt(i, m);
            });
            tops.castShadow = true;
            this.group.add(this.own(tops), this.own(poles));
        });
    }

    // A roof deck with a parapet, a few machines, and the city around and below it.
    buildRooftop() {
        const concreteMap = this.own(grainTexture('#7d8084', 30, 21));
        concreteMap.repeat.set(60, 45);
        const concrete = this.own(new THREE.MeshStandardMaterial({ map: concreteMap, roughness: 0.92 }));
        const deck = new THREE.BoxGeometry(220, 4, 170);
        deck.translate(0, -2.06, 0);
        this.mesh(deck, concrete, { receiveShadow: true });
        const wall = this.own(new THREE.MeshStandardMaterial({ color: 0x5f6368, roughness: 0.85 }));
        for (const [w, d, x, z] of [[220, 0.5, 0, -85], [220, 0.5, 0, 85], [0.5, 170, -110, 0], [0.5, 170, 110, 0]]) {
            const g = new THREE.BoxGeometry(w, 1.2, d);
            g.translate(x, 0.6, z);
            this.mesh(g, wall, { castShadow: true, receiveShadow: true });
        }
        // The tower the deck stands on, down to the streets.
        const tower = new THREE.BoxGeometry(222, 420, 172);
        tower.translate(0, -214, 0);
        const r = rand(29);
        const city = this.shader({
            uHemi: { value: new THREE.Color() }, uSunColor: { value: new THREE.Color() }, uSunDir: { value: new THREE.Vector3() },
            uNight: { value: 0 },
        }, CITY_VERTEX, CITY_FRAGMENT);
        this.city = city;
        const towerMesh = new THREE.InstancedMesh(this.own(tower), city, 1);
        towerMesh.setMatrixAt(0, new THREE.Matrix4());
        this.group.add(this.own(towerMesh));
        // Machines on the roof, out of the way of play.
        const box = this.own(new THREE.BoxGeometry(1, 1, 1));
        const metal = this.own(new THREE.MeshStandardMaterial({ color: 0x8a8f96, roughness: 0.5, metalness: 0.5 }));
        const units = new THREE.InstancedMesh(box, metal, 10);
        const m = new THREE.Matrix4();
        for (let i = 0; i < 10; i++) {
            const sx = i < 5 ? -1 : 1;
            m.compose(new THREE.Vector3(sx * (96 + r() * 8), 1.2, -70 + (i % 5) * 35),
                new THREE.Quaternion(), new THREE.Vector3(4 + r() * 4, 2.4 + r() * 2, 3 + r() * 3));
            units.setMatrixAt(i, m);
        }
        units.castShadow = true;
        this.group.add(this.own(units));
        // The skyline, towers in rings that grow taller with distance, so the near ones sit
        // below the deck and the far ones rise over the horizon.
        const count = 420;
        const towers = new THREE.InstancedMesh(box, city, count);
        const q = new THREE.Quaternion();
        for (let i = 0; i < count; i++) {
            const a = r() * Math.PI * 2;
            const d = 190 + Math.pow(r(), 0.7) * 1100;
            const top = -120 + Math.min(1, d / 900) * 260 * (0.4 + r());
            const w = 18 + r() * 40;
            const base = -430;
            q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.round(r() * 3) * Math.PI / 2 + (r() < 0.2 ? r() : 0));
            m.compose(new THREE.Vector3(Math.cos(a) * d, (top + base) / 2, Math.sin(a) * d * 0.85), q,
                new THREE.Vector3(w, top - base, w * (0.6 + 0.8 * r())));
            towers.setMatrixAt(i, m);
        }
        this.group.add(this.own(towers));
        const streets = new THREE.PlaneGeometry(4000, 4000);
        streets.rotateX(-Math.PI / 2);
        streets.translate(0, -428, 0);
        this.mesh(streets, this.own(new THREE.MeshBasicMaterial({ color: 0x0c0d10 })));
    }

    // A grid floor to the horizon, pylons of light round the pitch, wire mountains and a sun.
    buildNeon() {
        const grid = new THREE.PlaneGeometry(3000, 3000);
        grid.rotateX(-Math.PI / 2);
        grid.translate(0, -0.07, 0);
        this.mesh(grid, this.shader({
            uLine: { value: new THREE.Color(0xff2bd6) }, uGlow: { value: new THREE.Color(0x6a2cff) },
        }, GRID_VERTEX, GRID_FRAGMENT));
        const r = rand(53);
        // Pylons in two colors, standing back from the boards, taller towards the corners.
        const box = this.own(new THREE.BoxGeometry(1, 1, 1));
        const pink = this.own(new THREE.MeshStandardMaterial({ color: 0x14061a, emissive: 0xff2bd6, emissiveIntensity: 2.4 }));
        const cyan = this.own(new THREE.MeshStandardMaterial({ color: 0x041418, emissive: 0x2be8ff, emissiveIntensity: 2.4 }));
        // The pylons keep clear of the stands along the touch lines.
        const spots = aroundPitch(r, 44, FULL_HALF_W + 10, FULL_HALF_H + 9, 70, true)
            .filter(([x, z]) => !(Math.abs(x) < 46 && Math.abs(z) < 54));
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        [pink, cyan].forEach((mat, c) => {
            const list = spots.filter((_, i) => i % 2 === c);
            const pylons = new THREE.InstancedMesh(box, mat, list.length);
            list.forEach(([x, z], i) => {
                const corner = Math.min(1, Math.abs(x) / 90) * Math.min(1, Math.abs(z) / 70);
                const h = 4 + 10 * corner + 5 * r();
                m.compose(new THREE.Vector3(x, h / 2, z), q, new THREE.Vector3(0.8, h, 0.8));
                pylons.setMatrixAt(i, m);
            });
            this.group.add(this.own(pylons));
        });
        // A band of light along the foot of the boards.
        const bands = [
            [2 * FULL_HALF_W + 6, 0.3, 0, -FULL_HALF_H - 3, cyan], [2 * FULL_HALF_W + 6, 0.3, 0, FULL_HALF_H + 3, cyan],
            [0.3, 2 * FULL_HALF_H + 6, -FULL_HALF_W - 3, 0, pink], [0.3, 2 * FULL_HALF_H + 6, FULL_HALF_W + 3, 0, pink],
        ];
        for (const [w, d, x, z, mat] of bands) {
            const g = new THREE.BoxGeometry(w, 0.12, d);
            g.translate(x, 0.06, z);
            this.mesh(g, mat);
        }
        // Mountains of wire on the horizon behind the far side and the ends.
        const peaks = this.own(new THREE.ConeGeometry(1, 1, 4, 1, true));
        const wire = this.own(new THREE.MeshBasicMaterial({ color: 0x8a3cff, wireframe: true }));
        const hills = new THREE.InstancedMesh(peaks, wire, 46);
        for (let i = 0; i < 46; i++) {
            const a = Math.PI * (0.05 + 0.9 * (i / 45)) + (r() - 0.5) * 0.05;
            const d = 520 + r() * 260;
            const hgt = 50 + r() * 120;
            const wid = 60 + r() * 90;
            q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r() * Math.PI);
            m.compose(new THREE.Vector3(Math.cos(a) * d * 1.3, hgt / 2 - 2, -Math.sin(a) * d), q, new THREE.Vector3(wid, hgt, wid));
            hills.setMatrixAt(i, m);
        }
        this.group.add(this.own(hills));
        const sunGeo = new THREE.PlaneGeometry(380, 380);
        sunGeo.translate(0, 120, -1150);
        const sun = this.mesh(sunGeo, this.shader({}, SUN_VERTEX, SUN_FRAGMENT, { depthWrite: false }));
        sun.renderOrder = -8;
    }

    // A metal deck in orbit with a ring arching behind the far stand line and the Earth below.
    buildSpace(renderer) {
        const plates = this.own(canvasTexture(256, 256, (ctx, w, h) => {
            ctx.fillStyle = '#5a636d';
            ctx.fillRect(0, 0, w, h);
            const r = rand(41);
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) {
                    const v = 80 + Math.floor(r() * 26);
                    ctx.fillStyle = `rgb(${v},${v + 8},${v + 18})`;
                    ctx.fillRect(x * 64 + 2, y * 64 + 2, 60, 60);
                    ctx.fillStyle = 'rgba(20,24,30,0.7)';
                    for (const [cx, cy] of [[6, 6], [58, 6], [6, 58], [58, 58]]) {
                        ctx.fillRect(x * 64 + cx - 1, y * 64 + cy - 1, 3, 3);
                    }
                }
            }
        }));
        plates.wrapS = THREE.RepeatWrapping;
        plates.wrapT = THREE.RepeatWrapping;
        plates.repeat.set(55, 42);
        const metal = this.own(new THREE.MeshStandardMaterial({ map: plates, roughness: 0.5, metalness: 0.55 }));
        const deck = new THREE.BoxGeometry(230, 3, 180);
        deck.translate(0, -1.56, 0);
        this.mesh(deck, metal, { receiveShadow: true });
        // Glowing edges round the deck, and a light rail along each touch line.
        const glow = this.own(new THREE.MeshStandardMaterial({ color: 0x0a1418, emissive: 0x39d0ff, emissiveIntensity: 2.2 }));
        for (const [w, d, x, z] of [[230, 0.4, 0, -90], [230, 0.4, 0, 90], [0.4, 180, -115, 0], [0.4, 180, 115, 0]]) {
            const g = new THREE.BoxGeometry(w, 0.5, d);
            g.translate(x, 0.25, z);
            this.mesh(g, glow);
        }
        const steel = this.own(new THREE.MeshStandardMaterial({ color: 0xa6adb6, roughness: 0.35, metalness: 0.8 }));
        // The ring, an arch over the far side.
        const ring = new THREE.TorusGeometry(260, 7, 10, 96, Math.PI);
        ring.translate(0, -60, -300);
        this.mesh(ring, steel);
        const inner = new THREE.TorusGeometry(260, 1.2, 6, 96, Math.PI);
        inner.translate(0, -60, -288);
        this.mesh(inner, glow);
        // Solar wings out from the ends.
        const cells = this.own(canvasTexture(128, 128, (ctx, w, h) => {
            ctx.fillStyle = '#0b1d3c';
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = '#3b5f9a';
            ctx.lineWidth = 2;
            for (let i = 0; i <= 8; i++) {
                ctx.beginPath();
                ctx.moveTo(i * 16, 0);
                ctx.lineTo(i * 16, h);
                ctx.moveTo(0, i * 16);
                ctx.lineTo(w, i * 16);
                ctx.stroke();
            }
        }));
        cells.wrapS = THREE.RepeatWrapping;
        cells.wrapT = THREE.RepeatWrapping;
        cells.repeat.set(8, 3);
        const panel = this.own(new THREE.MeshStandardMaterial({ map: cells, roughness: 0.25, metalness: 0.7, side: THREE.DoubleSide }));
        for (const sx of [-1, 1]) {
            const beam = new THREE.BoxGeometry(200, 2, 2);
            beam.translate(sx * 215, -6, 0);
            this.mesh(beam, steel);
            for (const z of [-40, 40]) {
                const wing = new THREE.BoxGeometry(170, 0.3, 50);
                wing.rotateX(0.35 * sx);
                wing.translate(sx * 220, -6, z);
                this.mesh(wing, panel);
            }
        }
        // Lamps on the deck corners, the light of the pitch comes from them.
        const lampGeo = new THREE.BoxGeometry(8, 1.2, 1.2);
        for (const [x, z] of [[-105, -80], [105, -80], [-105, 80], [105, 80]]) {
            const g = lampGeo.clone();
            g.lookAt(new THREE.Vector3(-x, 0, -z));
            g.translate(x, 6, z);
            this.mesh(g, glow);
            const post = new THREE.CylinderGeometry(0.3, 0.4, 6, 6);
            post.translate(x, 3, z);
            this.mesh(post, steel);
        }
        this.own(lampGeo);
        const map = this.own(new THREE.WebGLRenderTarget(1024, 512, { depthBuffer: false }));
        const bake = new THREE.ShaderMaterial({ vertexShader: BAKE_VERTEX, fragmentShader: EARTH_BAKE, uniforms: fogUniforms() });
        const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bake);
        quad.frustumCulled = false;
        const before = renderer.getRenderTarget();
        renderer.setRenderTarget(map);
        renderer.render(quad, new THREE.Camera());
        renderer.setRenderTarget(before);
        quad.geometry.dispose();
        bake.dispose();
        const earth = new THREE.SphereGeometry(760, 96, 48);
        this.earth = this.mesh(earth, this.shader({ uSunDir: { value: SPACE_SUN.clone() }, uMap: { value: map.texture } },
            EARTH_VERTEX, EARTH_FRAGMENT));
        this.earth.frustumCulled = false;
        this.earth.renderOrder = -9;
        this.earth.matrixAutoUpdate = false;
        // The Earth hangs at the same place in the sky wherever the camera goes.
        this.earthOffset = new THREE.Vector3(0.12, -0.22, -1).normalize().multiplyScalar(1600);
        this.earth.onBeforeRender = (renderer, scene, camera) => {
            const e = this.earth;
            e.matrix.compose(e.position.copy(camera.position).add(this.earthOffset), e.quaternion, e.scale);
            e.matrixWorld.copy(e.matrix);
            e.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, e.matrixWorld);
            e.normalMatrix.getNormalMatrix(e.modelViewMatrix);
        };
    }

    // Changes the light of an arena. Space has its own sun and black sky whatever the hour, the
    // hour turns the Earth instead, and no weather reaches it.
    adjust(L) {
        if (this.name === 'neon') {
            // A violet haze over the grid and a little more glow, whatever the hour.
            if (L.fogColor) {
                L.fogColor = L.fogColor.clone().lerp(new THREE.Color(0x2a0f3d), 0.7);
            }
            L.bloom = Math.max(L.bloom || 0, 0.35);
            return L;
        }
        if (this.name !== 'space') {
            return L;
        }
        L.spaceHour = L.hour;
        L.lightDir = SPACE_SUN.clone();
        L.lightColor = new THREE.Color(0xfff6ea);
        L.lightIntensity = 3.2;
        L.castShadow = true;
        L.shadowIntensity = 1;
        L.moon = false;
        L.hemiSky = new THREE.Color(0x3a4a6a);
        L.hemiGround = new THREE.Color(0x1a2030);
        L.hemiIntensity = 0.45;
        L.exposure = 1.0;
        L.bloom = 0.3;
        L.fogColor = new THREE.Color(0x000000);
        L.fogDensity = 0.00015;
        L.flood = 0.55;
        L.lift = 0.06;
        L.rain = 0;
        L.snow = 0;
        L.wind = 0;
        L.sky = {
            ...L.sky, sunDir: SPACE_SUN.clone(), cloud: 0, grey: 0, stars: 1, moon: 0, night: 0, fog: 0, space: 1,
            glow: new THREE.Color(0, 0, 0),
        };
        return L;
    }

    // Every frame, with the lighting in use.
    update(seconds, camera, L) {
        for (const m of this.shaders) {
            const u = m.uniforms;
            u.uTime.value = seconds % 10000;
            if (L) {
                u.uFogColor.value.copy(L.fogColor);
                u.uFogDensity.value = L.fogDensity;
            }
        }
        if (!L) {
            return;
        }
        if (this.sea) {
            const u = this.sea.material.uniforms;
            u.uSky.value.copy(L.fogColor).multiplyScalar(0.8);
            u.uDeep.value.setHex(0x0a4250).multiplyScalar(0.2 + 0.8 * L.day);
            u.uSunDir.value.copy(L.lightDir);
            u.uSunColor.value.copy(L.lightColor).multiplyScalar(L.lightIntensity / 3);
        }
        if (this.city) {
            const u = this.city.uniforms;
            // The towers stand far off, a thinner air keeps them from fading to white.
            u.uFogDensity.value = L.fogDensity * 0.55;
            u.uHemi.value.copy(L.hemiSky).multiplyScalar(L.hemiIntensity);
            u.uSunColor.value.copy(L.lightColor).multiplyScalar(L.lightIntensity * 0.4);
            u.uSunDir.value.copy(L.lightDir);
            u.uNight.value = THREE.MathUtils.clamp(1.15 - L.day * 1.1, 0, 1);
        }
        if (this.earth) {
            // The hour turns the Earth, a full turn a day.
            const spin = ((L.spaceHour || 0) / 24) * Math.PI * 2;
            this.earth.quaternion.setFromEuler(new THREE.Euler(0.4, spin, 0.2));
        }
    }

    dispose() {
        this.clear();
        this.group.removeFromParent();
    }
}

// A small picture of each arena for the Field page, painted rather than rendered.
export function arenaThumbnail(name, w = 160, h = 90) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const sky = (top, bottom) => {
        const g = ctx.createLinearGradient(0, 0, 0, h * 0.6);
        g.addColorStop(0, top);
        g.addColorStop(1, bottom);
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
    };
    const pitch = (y) => {
        ctx.fillStyle = '#3f7f2c';
        ctx.beginPath();
        ctx.moveTo(w * 0.12, h);
        ctx.lineTo(w * 0.3, y);
        ctx.lineTo(w * 0.7, y);
        ctx.lineTo(w * 0.88, h);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w * 0.5, y);
        ctx.lineTo(w * 0.5, h);
        ctx.stroke();
    };
    const r = rand(name.length * 7 + 3);
    if (name === 'beach') {
        sky('#4f93d8', '#cfe6f5');
        ctx.fillStyle = '#1e6f86';
        ctx.fillRect(0, h * 0.42, w, h * 0.12);
        ctx.fillStyle = '#e2d0a4';
        ctx.fillRect(0, h * 0.54, w, h);
        pitch(h * 0.6);
        for (const x of [0.08, 0.92]) {
            ctx.strokeStyle = '#6b5440';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(w * x, h * 0.95);
            ctx.quadraticCurveTo(w * x + 4, h * 0.6, w * x + 8, h * 0.3);
            ctx.stroke();
            ctx.strokeStyle = '#3f7a28';
            ctx.lineWidth = 3;
            for (let k = 0; k < 6; k++) {
                const a = k / 6 * Math.PI * 2;
                ctx.beginPath();
                ctx.moveTo(w * x + 8, h * 0.3);
                ctx.quadraticCurveTo(w * x + 8 + Math.cos(a) * 10, h * 0.3 - 6, w * x + 8 + Math.cos(a) * 16, h * 0.3 + 6 + Math.abs(Math.sin(a)) * 4);
                ctx.stroke();
            }
        }
    } else if (name === 'rooftop') {
        sky('#070b1c', '#2a2440');
        for (let i = 0; i < 26; i++) {
            const x = r() * w;
            const bw = 6 + r() * 12;
            const bh = h * (0.15 + r() * 0.35);
            ctx.fillStyle = '#10131a';
            ctx.fillRect(x, h * 0.62 - bh, bw, bh);
            ctx.fillStyle = '#ffcf7a';
            for (let k = 0; k < 8; k++) {
                if (r() < 0.5) {
                    ctx.fillRect(x + 1 + r() * (bw - 3), h * 0.62 - bh + r() * bh, 1.5, 1.5);
                }
            }
        }
        ctx.fillStyle = '#4a4d52';
        ctx.fillRect(0, h * 0.58, w, h);
        pitch(h * 0.64);
    } else if (name === 'neon') {
        sky('#12021f', '#4a1258');
        const sunG = ctx.createLinearGradient(0, h * 0.1, 0, h * 0.55);
        sunG.addColorStop(0, '#ffd640');
        sunG.addColorStop(1, '#ff2e8c');
        ctx.fillStyle = sunG;
        ctx.beginPath();
        ctx.arc(w * 0.5, h * 0.5, h * 0.3, Math.PI, 0);
        ctx.fill();
        ctx.fillStyle = '#12021f';
        ctx.fillRect(0, h * 0.5, w, h);
        ctx.strokeStyle = '#ff2bd6';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 12; i++) {
            ctx.beginPath();
            ctx.moveTo(w * 0.5, h * 0.5);
            ctx.lineTo(w * (i / 12) * 2 - w * 0.5, h);
            ctx.stroke();
        }
        for (let i = 1; i < 6; i++) {
            const y = h * 0.5 + (h * 0.5) * (i / 6) ** 1.8;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        pitch(h * 0.64);
        ctx.fillStyle = '#2be8ff';
        for (const x of [0.06, 0.18, 0.82, 0.94]) {
            ctx.fillRect(w * x - 1, h * 0.4, 2.5, h * 0.24);
        }
    } else if (name === 'space') {
        ctx.fillStyle = '#000005';
        ctx.fillRect(0, 0, w, h);
        for (let i = 0; i < 60; i++) {
            ctx.fillStyle = `rgba(255,255,255,${0.3 + r() * 0.7})`;
            ctx.fillRect(r() * w, r() * h * 0.6, 1, 1);
        }
        const g = ctx.createRadialGradient(w * 0.25, h * 0.95, h * 0.2, w * 0.25, h * 0.95, h * 0.75);
        g.addColorStop(0, '#2c6fb3');
        g.addColorStop(0.9, '#1d4f8a');
        g.addColorStop(1, 'rgba(120,180,255,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(w * 0.25, h * 0.95, h * 0.75, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#a6adb6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(w * 0.6, h * 0.75, h * 0.5, Math.PI, 0);
        ctx.stroke();
        ctx.fillStyle = '#4d5560';
        ctx.fillRect(0, h * 0.62, w, h);
        pitch(h * 0.66);
        ctx.fillStyle = '#39d0ff';
        ctx.fillRect(0, h * 0.62, w, 1.5);
    } else {
        sky('#3f7fca', '#c9dcec');
        ctx.fillStyle = '#2a2f36';
        ctx.fillRect(0, h * 0.28, w, h * 0.06);
        ctx.fillStyle = '#505865';
        ctx.fillRect(0, h * 0.34, w, h * 0.3);
        for (let i = 0; i < 260; i++) {
            ctx.fillStyle = ['#b3262e', '#2a56b8', '#d8d8d8', '#e0b33a'][i % 4];
            ctx.fillRect(r() * w, h * (0.36 + r() * 0.26), 1.5, 1.5);
        }
        ctx.fillStyle = '#3b4047';
        for (const x of [0.06, 0.94]) {
            ctx.fillRect(w * x - 1, h * 0.06, 2, h * 0.3);
            ctx.fillStyle = '#f4f6ff';
            ctx.fillRect(w * x - 4, h * 0.04, 8, 4);
            ctx.fillStyle = '#3b4047';
        }
        ctx.fillStyle = '#24442a';
        ctx.fillRect(0, h * 0.64, w, h);
        pitch(h * 0.64);
    }
    return canvas.toDataURL('image/png');
}
