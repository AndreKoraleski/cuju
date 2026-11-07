// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Rain, snow and splashes. Every particle is one instance of a small quad whose place comes from
// its seed and the time in the vertex shader, so the processor only sets a few uniforms a frame.
// The drops are anchored to the world and wrap around a box that follows the camera, so they
// never swim with it. Quality sets how many there are, the low quality draws none.

import * as THREE from 'three';
import { FULL_HALF_H, FULL_HALF_W } from './pitch.js';

const COUNTS = {
    low: { rain: 0, snow: 0, splash: 0 },
    medium: { rain: 6000, snow: 4000, splash: 900 },
    high: { rain: 16000, snow: 9000, splash: 2400 },
};
// The box of drops around the camera, its half size in meters.
const RAIN_HALF = new THREE.Vector3(36, 22, 36);
const SNOW_HALF = new THREE.Vector3(30, 18, 30);

const COMMON = `
uniform float uTime;
uniform vec3 uCenter;
uniform vec3 uHalf;
uniform vec3 uVel;
attribute vec4 seed;
varying vec2 vUv;
varying float vFade;
vec3 cjWrap(vec3 p) {
    vec3 rel = mod(p - uCenter + uHalf, uHalf * 2.0) - uHalf;
    vFade = 1.0 - smoothstep(0.75, 1.0, length(rel.xz) / uHalf.x);
    vFade *= 1.0 - smoothstep(0.8, 1.0, abs(rel.y) / uHalf.y);
    return uCenter + rel;
}
`;

const RAIN_VERTEX = `${COMMON}
uniform float uLen;
uniform float uWidth;
void main() {
    float speed = 0.85 + 0.3 * seed.w;
    vec3 head = cjWrap(seed.xyz * uHalf * 2.0 + uVel * speed * uTime);
    vec3 dir = normalize(uVel);
    vec3 p = head - dir * uLen * speed * position.y;
    vec3 toCam = cameraPosition - p;
    float dist = length(toCam);
    // Far drops widen to about a pixel and fade to match, so they shimmer rather than flicker.
    float width = max(uWidth, dist * 0.0022);
    vFade *= uWidth / width * 0.5 + 0.5;
    vFade *= step(0.0, p.y) * smoothstep(0.6, 3.0, dist);
    p += normalize(cross(dir, toCam)) * width * position.x;
    vUv = position.xy;
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}`;

const SNOW_VERTEX = `${COMMON}
uniform float uSize;
void main() {
    float t = uTime * (0.7 + 0.6 * seed.w);
    vec3 sway = vec3(sin(t * 1.3 + seed.x * 40.0), 0.0, cos(t * 1.1 + seed.z * 40.0)) * 0.5;
    vec3 head = cjWrap(seed.xyz * uHalf * 2.0 + uVel * t + sway);
    vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
    vec3 up = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
    float dist = length(cameraPosition - head);
    float size = max(uSize * (0.6 + 0.8 * seed.w), dist * 0.0016);
    vFade *= step(0.0, head.y) * smoothstep(0.4, 2.0, dist) * min(1.0, uSize * 1.5 / size + 0.25);
    vec3 p = head + (right * position.x + up * (position.y * 2.0 - 1.0)) * size;
    vUv = vec2(position.x, position.y * 2.0 - 1.0);
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}`;

// Rings spreading on the ground where drops land, each at a new place every cycle.
const SPLASH_VERTEX = `
uniform float uTime;
uniform vec2 uArea;
attribute vec4 seed;
varying vec2 vUv;
varying float vFade;
float cjHash(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x);
}
void main() {
    float period = 0.45 + 0.35 * seed.w;
    float c = uTime / period + seed.z * 7.0;
    float cycle = floor(c);
    float phase = fract(c);
    vec2 at = (vec2(cjHash(seed.xy * 91.0 + cycle), cjHash(seed.yx * 57.0 + cycle * 1.7)) * 2.0 - 1.0) * uArea;
    float r = 0.05 + 0.22 * phase;
    vec3 p = vec3(at.x + position.x * r, 0.03, at.y + position.y * r);
    vUv = position.xy;
    vFade = 1.0 - phase;
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}`;

const FRAGMENT_HEAD = `
uniform vec3 uColor;
uniform float uOpacity;
varying vec2 vUv;
varying float vFade;
`;
const FRAGMENT_TAIL = `
    if (a < 0.004) {
        discard;
    }
    gl_FragColor = vec4(uColor, a);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
}`;

const RAIN_FRAGMENT = `${FRAGMENT_HEAD}
void main() {
    float a = (1.0 - abs(vUv.x)) * (1.0 - 0.7 * vUv.y) * vFade * uOpacity;
${FRAGMENT_TAIL}`;

const SNOW_FRAGMENT = `${FRAGMENT_HEAD}
void main() {
    float a = smoothstep(1.0, 0.3, length(vUv)) * vFade * uOpacity;
${FRAGMENT_TAIL}`;

const SPLASH_FRAGMENT = `${FRAGMENT_HEAD}
void main() {
    float d = length(vUv);
    float a = smoothstep(0.65, 0.85, d) * smoothstep(1.0, 0.88, d) * vFade * uOpacity;
${FRAGMENT_TAIL}`;

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

function particles(count, quad, vertexShader, fragmentShader, uniforms, seedValue) {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(quad, 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const seeds = new Float32Array(count * 4);
    const r = rand(seedValue);
    for (let i = 0; i < seeds.length; i++) {
        seeds[i] = r();
    }
    geometry.setAttribute('seed', new THREE.InstancedBufferAttribute(seeds, 4));
    geometry.instanceCount = 0;
    const material = new THREE.ShaderMaterial({
        uniforms, vertexShader, fragmentShader, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.visible = false;
    mesh.renderOrder = 5;
    return mesh;
}

export class Weather {
    constructor(scene) {
        this.group = new THREE.Group();
        scene.add(this.group);
        this.quality = 'medium';
        this.state = { rain: 0, snow: 0, wind: 0 };
        this.rainU = {
            uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uHalf: { value: RAIN_HALF.clone() },
            uVel: { value: new THREE.Vector3(0, -11, 0) }, uLen: { value: 0.9 }, uWidth: { value: 0.012 },
            uColor: { value: new THREE.Color(0xc8d2dc) }, uOpacity: { value: 0.35 },
        };
        this.snowU = {
            uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uHalf: { value: SNOW_HALF.clone() },
            uVel: { value: new THREE.Vector3(0, -1.2, 0) }, uSize: { value: 0.035 },
            uColor: { value: new THREE.Color(0xffffff) }, uOpacity: { value: 0.85 },
        };
        this.splashU = {
            uTime: { value: 0 }, uArea: { value: new THREE.Vector2(FULL_HALF_W + 2, FULL_HALF_H + 2) },
            uColor: { value: new THREE.Color(0xdfe6ee) }, uOpacity: { value: 0.5 },
        };
        this.build();
    }

    build() {
        this.disposeMeshes();
        const c = COUNTS[this.quality] || COUNTS.medium;
        this.max = c;
        this.rain = particles(c.rain, [-1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 1, 0], RAIN_VERTEX, RAIN_FRAGMENT, this.rainU, 7);
        this.snow = particles(c.snow, [-1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 1, 0], SNOW_VERTEX, SNOW_FRAGMENT, this.snowU, 11);
        this.splash = particles(c.splash, [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], SPLASH_VERTEX, SPLASH_FRAGMENT, this.splashU, 13);
        this.group.add(this.rain, this.snow, this.splash);
        this.applyCounts();
    }

    setQuality(quality) {
        if (quality === this.quality && this.rain) {
            return;
        }
        this.quality = quality;
        this.build();
    }

    // Takes the lighting of the hour and weather, see lightingAt.
    set(L) {
        this.state = { rain: L.rain, snow: L.snow, wind: L.wind };
        const wind = L.wind;
        // Wind blows along the pitch from the right end, and a little towards the camera.
        const gust = new THREE.Vector3(-1, 0, 0.35).normalize();
        this.rainU.uVel.value.set(0, -11 - 3 * L.rain, 0).addScaledVector(gust, 9 * wind);
        this.snowU.uVel.value.set(0, -1.1, 0).addScaledVector(gust, 3.5 * wind);
        // Drops catch the light around them, the floodlights at night.
        const light = Math.min(1.2, 0.25 + 0.45 * L.hemiIntensity + 0.15 * L.lightIntensity + 0.8 * L.flood);
        this.rainU.uColor.value.setRGB(0.78, 0.82, 0.88).multiplyScalar(light);
        this.rainU.uOpacity.value = 0.3 + 0.25 * L.rain;
        this.snowU.uColor.value.setRGB(0.95, 0.96, 1.0).multiplyScalar(Math.min(1.1, 0.35 + 0.5 * L.hemiIntensity + 0.15 * L.lightIntensity + 0.7 * L.flood));
        this.splashU.uColor.value.copy(this.rainU.uColor.value);
        this.applyCounts();
    }

    applyCounts() {
        const c = this.max;
        const s = this.state;
        this.rain.geometry.instanceCount = Math.round(c.rain * Math.min(1, s.rain));
        this.snow.geometry.instanceCount = Math.round(c.snow * Math.min(1, s.snow));
        this.splash.geometry.instanceCount = Math.round(c.splash * Math.min(1, s.rain));
        this.rain.visible = this.rain.geometry.instanceCount > 0;
        this.snow.visible = this.snow.geometry.instanceCount > 0;
        this.splash.visible = this.splash.geometry.instanceCount > 0;
    }

    // The boxes sit a little ahead of the camera, where most of what it sees is.
    update(seconds, camera) {
        const t = seconds % 3600;
        this.rainU.uTime.value = t;
        this.snowU.uTime.value = t;
        this.splashU.uTime.value = t;
        if (!this.rain.visible && !this.snow.visible) {
            return;
        }
        const ahead = camera.getWorldDirection(this.ahead || (this.ahead = new THREE.Vector3()));
        ahead.y = 0;
        if (ahead.lengthSq() > 1e-6) {
            ahead.normalize();
        }
        this.rainU.uCenter.value.copy(camera.position).addScaledVector(ahead, RAIN_HALF.x * 0.55);
        this.rainU.uCenter.value.y = Math.max(RAIN_HALF.y - 2, camera.position.y - 4);
        this.snowU.uCenter.value.copy(camera.position).addScaledVector(ahead, SNOW_HALF.x * 0.55);
        this.snowU.uCenter.value.y = Math.max(SNOW_HALF.y - 2, camera.position.y - 4);
    }

    disposeMeshes() {
        for (const mesh of [this.rain, this.snow, this.splash]) {
            if (mesh) {
                mesh.removeFromParent();
                mesh.geometry.dispose();
                mesh.material.dispose();
            }
        }
    }

    dispose() {
        this.disposeMeshes();
        this.group.removeFromParent();
    }
}
