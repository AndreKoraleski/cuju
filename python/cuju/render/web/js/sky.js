// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The sky and the light of every hour and weather. The sun follows a summer path, rising behind
// the right end, crossing behind the far stand and setting behind the left end, so its shadows
// fall towards the main camera in the afternoon as before. The dome is three's Preetham sky with
// soft clouds, dressed here with stars, a moon, an overcast grey, fog at the horizon and the glow
// the floodlights throw up into the air.

import * as THREE from 'three';
import { Sky as PreethamSky } from './vendor-sky.js';

// The old named times, and the hours they stand for.
export const PRESET_HOURS = { dawn: 6.5, day: 15, dusk: 19, night: 22 };

const SUNRISE = 6;
const SUNSET = 19.75;
const MAX_ELEVATION = THREE.MathUtils.degToRad(62);
// The moon hangs low over the far right corner, where the main camera and the Field preview
// can see it.
const MOON_DIR = new THREE.Vector3(0.62, 0.36, -0.7).normalize();

// The hour of a stored time, a number of hours or one of the old names.
export function hourOf(time) {
    if (typeof time === 'string' && time in PRESET_HOURS) {
        return PRESET_HOURS[time];
    }
    const h = Number(time);
    return Number.isFinite(h) ? THREE.MathUtils.clamp(h, 0, 24) : PRESET_HOURS.day;
}

export function formatHour(hour) {
    const minutes = Math.round(hourOf(hour) * 60) % (24 * 60);
    return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
}

// The direction towards the sun at an hour. Below the horizon the same arc runs on under the
// ground, deepest at midnight.
export function sunDirection(hour, out = new THREE.Vector3()) {
    const day = SUNSET - SUNRISE;
    const a = Math.PI * (hour - SUNRISE) / day;
    let elevation = MAX_ELEVATION * Math.sin(a);
    if (hour < SUNRISE || hour > SUNSET) {
        // Night runs through the remaining hours, so the arc is continuous at both ends.
        const night = 24 - day;
        const n = ((hour - SUNSET) % 24 + 24) % 24;
        elevation = -MAX_ELEVATION * 0.8 * Math.sin(Math.PI * n / night);
    }
    const t = THREE.MathUtils.clamp((hour - SUNRISE) / day, -0.3, 1.3);
    const azimuth = Math.PI * t;
    const c = Math.cos(elevation);
    return out.set(Math.cos(azimuth) * c, Math.sin(elevation), -Math.sin(azimuth) * c * 0.9 - 0.1 * c).normalize();
}

// Light keyed by the sine of the sun's elevation. Each row holds the sky and ground colors of
// the hemisphere light and its strength, the exposure, the bloom, the color of the air, and the
// sky colors the dome adds at night.
const KEYS = [
    { s: -0.4, hemiSky: 0x24304e, hemiGround: 0x0c120b, hemi: 0.22, exposure: 1.0, bloom: 0.32, air: 0x060a14 },
    { s: -0.1, hemiSky: 0x34426e, hemiGround: 0x12160f, hemi: 0.27, exposure: 1.02, bloom: 0.32, air: 0x141a2e },
    { s: 0.0, hemiSky: 0x8a7a98, hemiGround: 0x1e2218, hemi: 0.36, exposure: 1.06, bloom: 0.3, air: 0x5a4e5e },
    { s: 0.12, hemiSky: 0xe8a888, hemiGround: 0x2a3020, hemi: 0.52, exposure: 1.05, bloom: 0.28, air: 0xb08a78 },
    { s: 0.4, hemiSky: 0xcfe0f8, hemiGround: 0x38562a, hemi: 1.0, exposure: 1.0, bloom: 0.2, air: 0xa9bccf },
    { s: 1.0, hemiSky: 0xcfe3ff, hemiGround: 0x3a5a2a, hemi: 1.15, exposure: 1.0, bloom: 0.18, air: 0xbccbd9 },
];
const SUN_COLORS = [[-0.05, 0xff6a2a], [0.05, 0xff8a45], [0.18, 0xffa868], [0.45, 0xffe6c8], [1, 0xfff1dc]];

// Each weather at full strength. Cloud is the sky's cover, grey how much of its color goes,
// sun and shadow scale the sun and the darkness of its shadows, hemi the light of the sky,
// dark all light, fog the density of the air, flood how far the floodlights come on by day.
export const WEATHER = {
    clear: { cloud: 0.32, density: 0.4, grey: 0, sun: 1, shadow: 1, hemi: 1, dark: 1, fog: 0.0012, flood: 0, rain: 0, snow: 0, haze: 0 },
    overcast: { cloud: 0.97, density: 0.85, grey: 0.9, sun: 0.22, shadow: 0.3, hemi: 1.3, dark: 0.85, fog: 0.003, flood: 0.15, rain: 0, snow: 0, haze: 5 },
    rain: { cloud: 1, density: 0.95, grey: 1, sun: 0.14, shadow: 0.2, hemi: 1.2, dark: 0.68, fog: 0.0055, flood: 0.45, rain: 0.6, snow: 0, haze: 7 },
    storm: { cloud: 1, density: 1, grey: 1, sun: 0.06, shadow: 0, hemi: 1.05, dark: 0.5, fog: 0.0095, flood: 0.75, rain: 1, snow: 0, haze: 10 },
    snow: { cloud: 1, density: 0.9, grey: 1, sun: 0.18, shadow: 0.25, hemi: 1.35, dark: 0.85, fog: 0.0065, flood: 0.35, rain: 0, snow: 1, haze: 6 },
    fog: { cloud: 0.7, density: 0.7, grey: 0.85, sun: 0.3, shadow: 0.35, hemi: 1.2, dark: 0.85, fog: 0.016, flood: 0.4, rain: 0, snow: 0, haze: 8 },
};

const smooth = THREE.MathUtils.smoothstep;
const lerp = THREE.MathUtils.lerp;

function keyAt(s) {
    let i = 0;
    while (i < KEYS.length - 2 && s > KEYS[i + 1].s) {
        i++;
    }
    const a = KEYS[i];
    const b = KEYS[i + 1];
    return { a, b, t: THREE.MathUtils.clamp((s - a.s) / (b.s - a.s), 0, 1) };
}

function colorAt(list, s, out) {
    let i = 0;
    while (i < list.length - 2 && s > list[i + 1][0]) {
        i++;
    }
    const t = THREE.MathUtils.clamp((s - list[i][0]) / (list[i + 1][0] - list[i][0]), 0, 1);
    return out.setHex(list[i][1]).lerp(new THREE.Color(list[i + 1][1]), t);
}

const greyOf = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

// Everything the view, the stands, the sky and the weather need for one hour and weather.
// Amount scales the weather from clear to full, wind from calm to a gale.
export function lightingAt(hour, weather = 'clear', amount = 1, wind = 0) {
    const h = hourOf(hour);
    const kind = WEATHER[weather] ? weather : 'clear';
    const k = kind === 'clear' ? 0 : THREE.MathUtils.clamp(Number.isFinite(amount) ? amount : 1, 0, 1);
    const w = {};
    for (const key of Object.keys(WEATHER.clear)) {
        w[key] = lerp(WEATHER.clear[key], WEATHER[kind][key], k);
    }
    const sunDir = sunDirection(h);
    const s = sunDir.y;
    const { a, b, t } = keyAt(s);
    const day = smooth(s, -0.12, 0.2);
    const L = {
        hour: h, weather: kind, amount: k, wind: THREE.MathUtils.clamp(Number(wind) || 0, 0, 1), s, day, sunDir,
        rain: w.rain, snow: w.snow, cloud: w.cloud, grey: w.grey,
    };
    L.hemiSky = new THREE.Color(a.hemiSky).lerp(new THREE.Color(b.hemiSky), t);
    L.hemiGround = new THREE.Color(a.hemiGround).lerp(new THREE.Color(b.hemiGround), t);
    L.air = new THREE.Color(a.air).lerp(new THREE.Color(b.air), t);
    // Clouds take the color out of the sky light and the air.
    const g = w.grey;
    L.hemiSky.lerp(new THREE.Color().setScalar(greyOf(L.hemiSky) * 1.05), g);
    L.air.lerp(new THREE.Color().setScalar(greyOf(L.air)), g * 0.85);
    L.hemiIntensity = lerp(a.hemi, b.hemi, t) * lerp(1, w.hemi, day) * w.dark;
    L.exposure = lerp(a.exposure, b.exposure, t);
    L.bloom = lerp(a.bloom, b.bloom, t);
    // The sun by day, the moon by night, whichever is stronger lights the scene.
    const sunI = 3.0 * smooth(s, -0.01, 0.3) * w.sun * w.dark;
    const moonI = 0.12 * smooth(-s, 0.0, 0.12) * lerp(1, 0.3, w.cloud > 0.6 ? g : 0);
    L.moon = moonI > sunI;
    L.lightDir = L.moon ? MOON_DIR.clone() : sunDir.clone();
    L.lightColor = L.moon ? new THREE.Color(0x9fb4ff) : colorAt(SUN_COLORS, s, new THREE.Color());
    L.lightColor.lerp(new THREE.Color().setScalar(greyOf(L.lightColor)), g * 0.6);
    L.lightIntensity = Math.max(sunI, moonI);
    L.shadowIntensity = w.shadow * smooth(s, 0.03, 0.12);
    L.castShadow = !L.moon && L.shadowIntensity > 0.04;
    // Floodlights come up as the sun goes down, and in bad weather by day.
    L.flood = Math.min(1, Math.max(1 - smooth(s, 0.0, 0.33), w.flood));
    L.lift = lerp(0.035, 0.1, 1 - day);
    L.fogDensity = w.fog;
    L.fogColor = L.air.clone().multiplyScalar(lerp(1, 0.8, 1 - w.dark));
    // Under a cloud deck at night the air glows with the floodlights.
    L.fogColor.lerp(new THREE.Color(0x3a3a3c), L.flood * g * 0.5 * (1 - day));
    L.sky = {
        sunDir,
        turbidity: lerp(2, 6, smooth(0.35 - s, 0, 0.35)) + w.haze,
        rayleigh: lerp(1.0, 2.6, smooth(0.3 - s, 0, 0.3)),
        mie: lerp(0.004, 0.012, k),
        cloud: w.cloud,
        density: w.density,
        grey: Math.min(1, g * 1.35),
        tint: new THREE.Color(0xb4bcc8).multiplyScalar(w.dark),
        over: new THREE.Color(0xb4bcc8).multiplyScalar(w.dark * (0.05 + 0.93 * day)),
        bright: 0.33 * w.dark,
        night: 1 - day,
        stars: (1 - smooth(s, -0.18, -0.04)) * (1 - Math.min(1, g * 1.35)),
        moon: (1 - day) * (1 - Math.min(1, g * 1.35)),
        glow: new THREE.Color(0x2a2a30).multiplyScalar(L.flood * (1 - 0.6 * day) * (0.6 + g)),
        fog: smooth(w.fog, 0.0015, 0.012),
        air: L.fogColor,
        wind: L.wind,
    };
    return L;
}

const SKY_DECL = `
uniform float uBright;
uniform float uGrey;
uniform vec3 uTint;
uniform vec3 uOver;
uniform float uStars;
uniform vec3 uMoonDir;
uniform float uMoon;
uniform vec3 uGlow;
uniform vec3 uAir;
uniform float uFog;
uniform float uNight;
uniform float uSpace;
float cjHash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
void main() {`;

const SKY_FINISH = `
vec3 cjCol = texColor * uBright;
float cjUp = max(direction.y, 0.0);
float cjLum = dot(cjCol, vec3(0.2126, 0.7152, 0.0722));
// An overcast deck, flat grey and a little brighter overhead, with the clouds as its texture.
vec3 cjOver = uOver * mix(0.72, 1.0, cjUp) * (0.9 + 0.1 * cjCloud) + uTint * cjLum * 0.15;
cjCol = mix(cjCol, cjOver, uGrey);
// A deep blue night with a lighter band along the horizon.
cjCol += uNight * mix(vec3(0.012, 0.022, 0.05), vec3(0.002, 0.005, 0.016), pow(cjUp, 0.45));
// Stars, fixed to the sky, fewer and fainter low down and behind cloud.
vec3 cjSp = direction * 300.0;
vec3 cjId = floor(cjSp);
float cjH = cjHash3(cjId);
float cjStar = smoothstep(0.42, 0.05, length(fract(cjSp) - 0.5)) * smoothstep(0.994, 1.0, cjH);
cjStar *= (0.75 + 0.25 * sin(time * (2.0 + 5.0 * cjH) + cjH * 40.0));
cjCol += vec3(0.85, 0.9, 1.0) * cjStar * 1.4 * uStars * smoothstep(0.03, 0.25, direction.y) * (1.0 - cjCloud);
// The moon, a pale disc with darker seas and a halo.
float cjMd = dot(direction, uMoonDir);
float cjDisc = smoothstep(0.99975, 0.99979, cjMd);
vec3 cjMx = cross(uMoonDir, vec3(0.0, 1.0, 0.0));
vec2 cjMp = vec2(dot(direction, normalize(cjMx)), dot(direction, normalize(cross(cjMx, uMoonDir)))) * 250.0;
float cjSea = smoothstep(0.35, 0.65, cjHash3(floor(vec3(cjMp * 1.3, 1.0)))) * 0.25 + smoothstep(1.2, 0.0, length(cjMp - vec2(1.0, 0.8))) * 0.25;
cjCol += uMoon * (1.0 - 0.85 * cjCloud) * (cjDisc * vec3(1.5, 1.48, 1.4) * (1.0 - cjSea) + vec3(0.06, 0.08, 0.13) * pow(max(cjMd, 0.0), 900.0));
// Light from the floodlights hangs as a haze over the stands and under the clouds.
cjCol += uGlow * (exp(-cjUp * 9.0) + 0.6 * max(cjCloud, uGrey));
cjCol = mix(cjCol, uAir, uFog * exp(-cjUp * 5.0));
// Below the horizon the land fades into the air.
cjCol = mix(cjCol, uAir * 0.55, smoothstep(0.0, -0.08, direction.y));
// In orbit the sky is black, every star shows and the sun is a hard white disc.
float cjSun = smoothstep(0.99986, 0.9999, dot(direction, vSunDirection));
cjCol = mix(cjCol, vec3(0.0004, 0.0006, 0.0012) + vec3(0.85, 0.9, 1.0) * cjStar * 1.8 + vec3(40.0) * cjSun, uSpace);
gl_FragColor = vec4( cjCol, 1.0 );`;

function skyMaterial() {
    const base = PreethamSky.SkyShader;
    let fragment = base.fragmentShader;
    const patch = (from, to) => {
        if (!fragment.includes(from)) {
            throw new Error(`The sky shader changed, missing ${from}`);
        }
        fragment = fragment.replace(from, to);
    };
    patch('void main() {', SKY_DECL);
    patch('// Clouds', 'float cjCloud = 0.0;');
    patch('float alpha = ( 1.0 - exp( depth * cloudDensity * -12.0 ) ) * horizonFade;',
        'float alpha = ( 1.0 - exp( depth * cloudDensity * -12.0 ) ) * horizonFade;\ncjCloud = alpha;');
    patch('gl_FragColor = vec4( texColor, 1.0 );', SKY_FINISH);
    const uniforms = THREE.UniformsUtils.clone(base.uniforms);
    Object.assign(uniforms, {
        uBright: { value: 0.5 },
        uGrey: { value: 0 },
        uTint: { value: new THREE.Color(0xb4bcc8) },
        uOver: { value: new THREE.Color(0x808890) },
        uStars: { value: 0 },
        uMoonDir: { value: MOON_DIR.clone() },
        uMoon: { value: 0 },
        uGlow: { value: new THREE.Color(0, 0, 0) },
        uAir: { value: new THREE.Color(0xbccbd9) },
        uFog: { value: 0 },
        uNight: { value: 0 },
        uSpace: { value: 0 },
    });
    return new THREE.ShaderMaterial({
        name: 'cujuSky',
        uniforms,
        vertexShader: base.vertexShader,
        fragmentShader: fragment,
        side: THREE.BackSide,
        depthWrite: false,
    });
}

// The dome sits on the camera every frame, so it never slides or turns as the camera moves,
// and its vertex shader puts it on the far plane, behind everything.
export class SkyDome {
    constructor(scene) {
        this.material = skyMaterial();
        this.mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
        this.mesh.scale.setScalar(1000);
        this.mesh.frustumCulled = false;
        this.mesh.renderOrder = -10;
        this.mesh.matrixAutoUpdate = false;
        this.mesh.onBeforeRender = (renderer, s, camera) => {
            this.mesh.position.copy(camera.position);
            this.mesh.updateMatrix();
            this.mesh.matrixWorld.copy(this.mesh.matrix);
            this.mesh.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, this.mesh.matrixWorld);
        };
        scene.add(this.mesh);
        this.windSpeed = 0;
    }

    apply(L) {
        const u = this.material.uniforms;
        const k = L.sky;
        u.sunPosition.value.copy(k.sunDir);
        u.turbidity.value = k.turbidity;
        u.rayleigh.value = k.rayleigh;
        u.mieCoefficient.value = k.mie;
        u.mieDirectionalG.value = 0.8;
        u.cloudCoverage.value = k.cloud;
        u.cloudDensity.value = k.density;
        u.cloudScale.value = 0.00022;
        u.cloudElevation.value = 0.55;
        u.uBright.value = k.bright;
        u.uGrey.value = k.grey;
        u.uTint.value.copy(k.tint);
        u.uOver.value.copy(k.over);
        u.uStars.value = k.stars;
        u.uMoon.value = k.moon;
        u.uGlow.value.copy(k.glow);
        u.uAir.value.copy(k.air);
        u.uFog.value = k.fog;
        u.uNight.value = k.night;
        u.uSpace.value = k.space || 0;
        this.windSpeed = k.wind;
    }

    // Clouds drift slowly, faster in the wind. The time runs on wall seconds.
    update(seconds) {
        const u = this.material.uniforms;
        u.time.value = seconds % 100000;
        u.cloudSpeed.value = 0.000012 * (1 + 4 * this.windSpeed);
    }

    setVisible(on) {
        this.mesh.visible = on;
    }

    dispose() {
        this.mesh.removeFromParent();
        this.mesh.geometry.dispose();
        this.material.dispose();
    }
}
