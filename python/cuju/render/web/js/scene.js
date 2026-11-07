// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Renderer, lights, the quality levels, and the environment that sets sky, light and weather
// for the hour of the day.

import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ARENA_AMBIENCE, Arena } from './arenas.js';
import { FULL_HALF_H, FULL_HALF_W } from './pitch.js';
import { SkyDome, hourOf, lightingAt } from './sky.js';
import { Weather } from './weather.js';

export const QUALITIES = {
    low: { shadow: 1024, maxRatio: 1, post: false, shadowRadius: 1 },
    medium: { shadow: 2048, maxRatio: 1.5, post: false, shadowRadius: 2.5 },
    high: { shadow: 4096, maxRatio: 2, post: true, shadowRadius: 3 },
};

// Integrated and software graphics, which share memory with the processor or have none.
const INTEGRATED = /intel|uhd|iris|radeon\(tm\) graphics|radeon graphics|vega \d+ graphics|apple|mali|adreno|powervr|videocore/i;
const SOFTWARE = /swiftshader|llvmpipe|software|microsoft basic render/i;
const DISCRETE = /nvidia|geforce|quadro|rtx|gtx|radeon rx|radeon pro|arc a\d|firepro/i;

// The name of the graphics processor, from a throwaway context.
function gpuName() {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) {
            return '';
        }
        const info = gl.getExtension('WEBGL_debug_renderer_info');
        const name = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || '');
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) {
            lose.loseContext();
        }
        return name;
    } catch {
        return '';
    }
}

// High needs a graphics card of its own, integrated graphics get Medium, phones and software
// rendering get Low.
export function defaultQuality() {
    const cores = navigator.hardwareConcurrency || 4;
    const memory = navigator.deviceMemory || 8;
    const touch = matchMedia('(pointer: coarse)').matches;
    const gpu = gpuName();
    if (touch || SOFTWARE.test(gpu) || memory <= 2) {
        return 'low';
    }
    if (DISCRETE.test(gpu) && !INTEGRATED.test(gpu.replace(/radeon rx|radeon pro/ig, ''))) {
        return cores >= 6 && memory >= 8 ? 'high' : 'medium';
    }
    if (INTEGRATED.test(gpu) || !gpu) {
        return cores <= 2 ? 'low' : 'medium';
    }
    return cores >= 8 && memory >= 8 ? 'high' : 'medium';
}

// The pitch and its surround, the region whose shadows matter, in three's frame.
const SHADOW_REGION = [];
for (const x of [-FULL_HALF_W - 4, FULL_HALF_W + 4]) {
    for (const z of [-FULL_HALF_H - 4, FULL_HALF_H + 4]) {
        for (const y of [0, 3]) {
            SHADOW_REGION.push(new THREE.Vector3(x, y, z));
        }
    }
}

export class View {
    constructor(container) {
        this.container = container;
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
        if (!this.renderer.capabilities.isWebGL2) {
            throw new Error('This viewer needs WebGL 2.');
        }
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowMap;
        container.appendChild(this.renderer.domElement);
        this.scene = new THREE.Scene();
        // The air is always there, thin on a clear day, so changing the weather never recompiles.
        this.scene.fog = new THREE.FogExp2(0xbccbd9, 0.0012);
        this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 2500);
        this.sun = new THREE.DirectionalLight(0xfff1dc, 3.0);
        this.sun.position.set(-60, 110, -70);
        this.sun.castShadow = true;
        const cam = this.sun.shadow.camera;
        cam.left = -78;
        cam.right = 78;
        cam.top = 58;
        cam.bottom = -58;
        cam.near = 10;
        cam.far = 600;
        this.sun.shadow.bias = -0.0004;
        this.sun.shadow.normalBias = 0.03;
        this.scene.add(this.sun, this.sun.target);
        this.hemi = new THREE.HemisphereLight(0xcfe3ff, 0x3a5a2a, 1.15);
        this.scene.add(this.hemi);
        this.quality = null;
        this.bloom = null;
        this.bloomStrength = 0.2;
        this.light = null;
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(container);
        this.resize();
    }

    // Sun or moon, sky light, exposure, bloom and fog from lightingAt.
    applyLight(L) {
        this.light = L;
        this.sun.color.copy(L.lightColor);
        this.sun.intensity = L.lightIntensity;
        this.sun.position.copy(L.lightDir).multiplyScalar(300);
        this.sun.target.position.set(0, 0, 0);
        this.sun.castShadow = L.castShadow;
        this.sun.shadow.intensity = THREE.MathUtils.clamp(L.shadowIntensity, 0, 1);
        this.fitShadow(L.lightDir);
        this.hemi.color.copy(L.hemiSky);
        this.hemi.groundColor.copy(L.hemiGround);
        this.hemi.intensity = L.hemiIntensity;
        this.renderer.toneMappingExposure = L.exposure;
        this.bloomStrength = L.bloom;
        if (this.bloom) {
            this.bloom.strength = this.bloomBoost(L.bloom);
        }
        this.scene.fog.color.copy(L.fogColor);
        this.scene.fog.density = L.fogDensity;
    }

    // The old names of the times of day still work.
    setTime(time) {
        this.applyLight(lightingAt(hourOf(time)));
    }

    // The sun's shadow covers the pitch and its surround and nothing else, so its texels stay
    // small at every height of the sun.
    fitShadow(dir) {
        const cam = this.sun.shadow.camera;
        this.sun.updateMatrixWorld();
        this.sun.target.updateMatrixWorld();
        const view = new THREE.Matrix4().lookAt(this.sun.position, this.sun.target.position, new THREE.Vector3(0, 1, 0));
        view.setPosition(this.sun.position);
        view.invert();
        const box = new THREE.Box3();
        const p = new THREE.Vector3();
        for (const corner of SHADOW_REGION) {
            box.expandByPoint(p.copy(corner).applyMatrix4(view));
        }
        cam.left = box.min.x - 1;
        cam.right = box.max.x + 1;
        cam.bottom = box.min.y - 1;
        cam.top = box.max.y + 1;
        cam.near = 10;
        cam.far = 300 + 200;
        cam.updateProjectionMatrix();
        this.sun.shadow.needsUpdate = true;
        return dir;
    }

    // Moves the canvas into another element, the Field page borrows the view for its preview.
    attach(container) {
        if (this.container === container) {
            return;
        }
        this.resizeObserver.unobserve(this.container);
        this.container = container;
        container.appendChild(this.renderer.domElement);
        this.resizeObserver.observe(container);
        this.resize();
    }

    // The bloom reads the finished picture, so it is a little stronger than one fed the light
    // before tone mapping.
    bloomBoost(strength) {
        return strength * 1.6;
    }

    setQuality(name) {
        const q = QUALITIES[name] || QUALITIES.medium;
        this.quality = name;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, q.maxRatio));
        this.sun.shadow.mapSize.set(q.shadow, q.shadow);
        this.sun.shadow.radius = q.shadowRadius;
        if (this.sun.shadow.map) {
            this.sun.shadow.map.dispose();
            this.sun.shadow.map = null;
        }
        this.disposePost();
        if (q.post) {
            // The scene draws straight to the canvas with its multisampling, and the bloom works
            // from a copy of the finished picture at half size. A multisampled half float target
            // for the whole scene cost about 7 ms a frame on integrated graphics.
            const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
            this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), this.bloomBoost(this.bloomStrength), 0.55, 0.96);
            this.quad = new FullScreenQuad(null);
        }
        this.resize();
    }

    disposePost() {
        if (this.bloom) {
            this.bloom.dispose();
            this.bloom = null;
        }
        if (this.quad) {
            this.quad.dispose();
            this.quad = null;
        }
        if (this.frame) {
            this.frame.dispose();
            this.frame = null;
        }
    }

    resize() {
        const w = Math.max(1, this.container.clientWidth);
        const h = Math.max(1, this.container.clientHeight);
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        if (this.bloom) {
            const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
            this.bloom.setSize(size.x, size.y);
            if (this.frame) {
                this.frame.dispose();
            }
            this.frame = new THREE.FramebufferTexture(size.x, size.y);
        }
    }

    get aspect() {
        return this.camera.aspect;
    }

    render() {
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);
        if (this.bloom && this.frame) {
            this.renderBloom();
        }
    }

    // UnrealBloomPass's own steps, fed from the canvas and added back onto it.
    renderBloom() {
        const r = this.renderer;
        const b = this.bloom;
        r.copyFramebufferToTexture(this.frame);
        const autoClear = r.autoClear;
        r.autoClear = false;
        const clear = r.getClearColor(new THREE.Color());
        const alpha = r.getClearAlpha();
        r.setClearColor(0x000000, 0);
        b.highPassUniforms.tDiffuse.value = this.frame;
        b.highPassUniforms.luminosityThreshold.value = b.threshold;
        this.quad.material = b.materialHighPassFilter;
        r.setRenderTarget(b.renderTargetBright);
        r.clear();
        this.quad.render(r);
        let input = b.renderTargetBright;
        for (let i = 0; i < b.nMips; i++) {
            const m = b.separableBlurMaterials[i];
            this.quad.material = m;
            m.uniforms.colorTexture.value = input.texture;
            m.uniforms.direction.value = UnrealBloomPass.BlurDirectionX;
            r.setRenderTarget(b.renderTargetsHorizontal[i]);
            r.clear();
            this.quad.render(r);
            m.uniforms.colorTexture.value = b.renderTargetsHorizontal[i].texture;
            m.uniforms.direction.value = UnrealBloomPass.BlurDirectionY;
            r.setRenderTarget(b.renderTargetsVertical[i]);
            r.clear();
            this.quad.render(r);
            input = b.renderTargetsVertical[i];
        }
        this.quad.material = b.compositeMaterial;
        b.compositeMaterial.uniforms.bloomStrength.value = b.strength;
        b.compositeMaterial.uniforms.bloomRadius.value = b.radius;
        b.compositeMaterial.uniforms.bloomTintColors.value = b.bloomTintColors;
        r.setRenderTarget(b.renderTargetsHorizontal[0]);
        r.clear();
        this.quad.render(r);
        this.quad.material = b.blendMaterial;
        b.copyUniforms.tDiffuse.value = b.renderTargetsHorizontal[0].texture;
        r.setRenderTarget(null);
        this.quad.render(r);
        r.setClearColor(clear, alpha);
        r.autoClear = autoClear;
    }
}

// Rain and snow wet and whiten the ground over time, at these rates per second.
const WET_RATE = 1 / 8;
const DRY_RATE = 1 / 20;
const SNOW_RATE = 1 / 45;
const MELT_RATE = 1 / 25;

// The hour and weather of the field settings, turned into the sky, the light, the stands'
// floodlights, the falling rain or snow and the state of the ground.
export class Environment {
    constructor(app) {
        this.app = app;
        this.view = app.view;
        this.sky = new SkyDome(app.view.scene);
        this.weather = new Weather(app.view.scene);
        this.weather.setQuality(app.settings ? app.settings.quality : 'medium');
        this.arena = new Arena(app.view.scene);
        this.wet = 0;
        this.snow = 0;
        this.seconds = 0;
        this.light = null;
        this.sound = {};
    }

    // Takes the field settings, or just the hour and weather while a slider moves.
    setField(field) {
        const f = field || {};
        const L = lightingAt(hourOf(f.time), f.weather || 'clear', Number.isFinite(f.weatherAmount) ? f.weatherAmount : 1, f.wind || 0);
        this.arena.set(f.arena, this.app);
        this.light = this.arena.adjust(L);
        this.apply();
    }

    // Applies the current light everywhere, also to a stadium that was just replaced.
    apply() {
        const L = this.light || (this.light = lightingAt(15));
        this.view.applyLight(L);
        this.sky.apply(L);
        this.weather.set(L);
        if (this.app.stadium) {
            this.app.stadium.setTime(L);
        }
        this.arena.dress(this.app);
        // The sound takes the arena and the daylight first, so the weather follows its ground.
        const sound = this.app.sound;
        if (sound && typeof sound.setSurroundings === 'function') {
            sound.setSurroundings(this.arena.name, L.day);
        }
        this.ambience('rain', L.rain > 0 ? Math.min(1, 0.25 + 0.75 * L.rain) : 0);
        this.ambience('wind', Math.min(1, L.wind * 0.8 + (L.weather === 'storm' ? 0.3 * L.amount : 0)));
        for (const [arena, name] of Object.entries(ARENA_AMBIENCE)) {
            this.ambience(name, arena === this.arena.name ? 0.5 : 0);
        }
    }

    ambience(name, level) {
        const sound = this.app.sound;
        const rounded = Math.round(level * 100) / 100;
        if (this.sound[name] === rounded || !sound || typeof sound.setAmbience !== 'function') {
            return;
        }
        this.sound[name] = rounded;
        sound.setAmbience(name, rounded);
    }

    setQuality(quality) {
        this.weather.setQuality(quality);
        if (this.light) {
            this.weather.set(this.light);
        }
    }

    // Every frame, the clouds drift, the drops fall and the ground wets or whitens.
    update(dt) {
        const L = this.light;
        if (!L) {
            return;
        }
        this.seconds += dt;
        this.sky.update(this.seconds);
        this.weather.update(this.seconds, this.view.camera);
        this.arena.update(this.seconds, this.view.camera, L);
        const wetTarget = Math.min(1, L.rain * 1.4);
        this.wet += THREE.MathUtils.clamp(wetTarget - this.wet, -DRY_RATE * dt, WET_RATE * dt);
        const snowTarget = L.snow;
        this.snow += THREE.MathUtils.clamp(snowTarget - this.snow, -MELT_RATE * dt, SNOW_RATE * dt);
        if (this.app.pitch && this.app.pitch.setSurface) {
            this.app.pitch.setSurface(this.wet, this.snow);
            this.app.pitch.update();
        }
    }
}
