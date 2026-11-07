// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Sound with the Web Audio API. Everything is synthesized unless assets/sounds lists files of its
// own, which then stand in for the synthesized sound of the same name, and teams may bring
// sounds of their own for their players' actions. Browsers only allow audio after a user
// gesture, so nothing starts until unlock runs from one.
//
// Each arena has a soundscape of its own and a crowd. The classic stadium has its big crowd, the
// beach the sea and a crowd in the open air, the rooftop the city below and a crowd whose voices
// come back off the towers, the space station the hum of its machines and a crowd in an enclosed
// gallery, and the neon arcade the buzz of its tubes and a crowd on the grid. The crowds of the
// other arenas play the stadium's recordings, smaller and colored by the place. Rain and wind
// are synthesized too and follow the weather. Switching arenas or weather crossfades over about
// a second.

// The optional files that replace synthesized sounds, by name without the extension. The list
// in assets/sounds/sounds.json names the files present, so absent ones are never requested, and
// each sound falls back to its synthesis when its file is absent.
//
// The crowd of every arena, crowd-loop, crowd-cheer, crowd-ooh and applause.
// Every arena, whistle-short and whistle-long.
// Weather, rain-loop and wind-loop, which no longer play in space.
// Arena beds played as loops over the arena, waves-loop on the beach, city-loop on the rooftop,
// hum-loop on the space station and the neon arcade. Each silences the synthesized bed it
// stands for, the sea, the city or the station hum.
// The beach, gulls for a gull's call and beach-wave-1 to beach-wave-4 for single breaking waves
// over the synthesized sea.
// The rooftop, city-horn for a far car horn and city-siren for a distant siren.
// The space station, space-beep for a console beep, space-radio for a radio chirp and space-hum, a loop in place of the
// station hum used when hum-loop is absent.
// The neon arcade, neon-sting for the goal sting.
export const SOUND_FILES = [
    'crowd-loop', 'crowd-cheer', 'crowd-ooh', 'applause', 'whistle-short', 'whistle-long',
    'rain-loop', 'wind-loop', 'waves-loop', 'city-loop', 'hum-loop',
    'gulls', 'city-horn', 'city-siren', 'space-beep', 'space-radio', 'space-hum',
    'neon-sting', 'beach-wave-1', 'beach-wave-2', 'beach-wave-3', 'beach-wave-4',
];
// The recorded waves, of which each breaking wave plays one at random when any is present.
const BEACH_WAVES = ['beach-wave-1', 'beach-wave-2', 'beach-wave-3', 'beach-wave-4'];
// The gain of each arena file at its level from loading, so it plays no louder than the sound
// it replaces. The shipped files were measured offline at the output of the page against their
// synthesized sounds, and each gain keeps the root mean square of a file and its busiest tenth
// of a second within a tenth above the synthesis and its peak at or below it, close to parity.
const ARENA_FILE_GAIN = {
    gulls: 0.0058, 'city-horn': 0.02, 'city-siren': 0.0105, 'space-beep': 0.014,
    'space-radio': 0.013, 'space-hum': 0.135, 'beach-wave-1': 0.1, 'beach-wave-2': 0.1,
    'beach-wave-3': 0.1, 'beach-wave-4': 0.1,
    'neon-sting': 0.2,
};

// The loudness of each synthesized sound that a file may replace, at a gain of one, as the root
// mean square over its busiest tenth of a second and its peak. A file is scaled once on loading
// to the lower of the two, so with the gains of the synthesized sound it never plays louder.
export const FILE_LEVELS = {
    // The stadium crowd, measured against the synthesized murmur at an intensity of 0.3 and 0.9.
    'crowd-loop': { rms: 0.24, peak: 0.825 },
    // The goal roar, held to the root mean square of the synthesized roar over its length, which
    // keeps its busiest tenth of a second and its peak well under the stadium goal.
    'crowd-cheer': { rms: 0.0305, peak: 0.094 },
    // The groan at a near miss, near the level of the crowd and well under the goal, since the
    // synthesized shouts it stands for lie far under the crowd.
    'crowd-ooh': { rms: 0.025, peak: 0.082 },
    // The applause of the full time show, near the level of the crowd and well under the goal,
    // since the synthesized claps it stands for lie far under the crowd.
    applause: { rms: 0.022, peak: 0.07 },
    // The whistles are held to the stadium goal, the loudest sound of the match.
    'whistle-short': { rms: 0.072, peak: 0.15 },
    'whistle-long': { rms: 0.072, peak: 0.15 },
    // Loops and arena sounds, chosen so that with their gains they play no louder than the
    // synthesized beds and reactions, which sit under the stadium crowd. Rain is held to the
    // quietest synthesized rain of the arenas, and the city to the city at night.
    'rain-loop': { rms: 0.0143, peak: 0.3 },
    'wind-loop': { rms: 0.0228, peak: 0.3 },
    'waves-loop': { rms: 0.035, peak: 0.4 },
    'city-loop': { rms: 0.028, peak: 0.32 },
    'hum-loop': { rms: 0.035, peak: 0.4 },
    'space-hum': { rms: 0.06, peak: 0.5 },
    gulls: { rms: 0.15, peak: 0.5 },
    'city-horn': { rms: 0.15, peak: 0.5 },
    'city-siren': { rms: 0.15, peak: 0.5 },
    'space-radio': { rms: 0.15, peak: 0.5 },
    'space-beep': { rms: 0.15, peak: 0.5 },
    'neon-sting': { rms: 0.15, peak: 0.5 },
    'beach-wave-1': { rms: 0.15, peak: 0.5 },
    'beach-wave-2': { rms: 0.15, peak: 0.5 },
    'beach-wave-3': { rms: 0.15, peak: 0.5 },
    'beach-wave-4': { rms: 0.15, peak: 0.5 },
};
// Files without a measured synthesized sound are brought to a quiet peak.
const FILE_PEAK = 0.5;

// The arenas the sound knows.
export const SOUND_ARENAS = ['stadium', 'beach', 'rooftop', 'space', 'neon'];
// How loud the fans' chant is in each arena against the stadium's.
const FAN_CHANT = { stadium: 1, beach: 0.45, rooftop: 0.45, space: 0.4, neon: 0.45 };
// The crowd of each arena other than the stadium, from the stadium's recordings. level is the
// murmur and react the cheers, groans and applause, each against the stadium's, low and high
// the band the crowd is heard through, dry and wet how much of it reaches the listener directly
// and through the echo, and slap the delay in seconds of a short echo off nearby walls. The
// beach crowd is in the open air, the rooftop crowd echoes off the towers around, the space
// crowd sits in an enclosed gallery and the neon crowd stands on an open grid.
const ARENA_CROWD = {
    beach: { level: 0.9, react: 0.8, low: 7000, high: 90, dry: 0.8, wet: 0.1, slap: 0 },
    rooftop: { level: 0.65, react: 0.75, low: 5000, high: 140, dry: 0.65, wet: 0.3, slap: 0.23 },
    space: { level: 0.85, react: 0.75, low: 3000, high: 190, dry: 0.5, wet: 0.7, slap: 0.06 },
    neon: { level: 0.85, react: 0.75, low: 5500, high: 110, dry: 0.7, wet: 0.35, slap: 0 },
};
// How much of the echo comes back in each arena, the stands ring, the beach is open air.
const ARENA_ECHO = { stadium: 0.5, beach: 0.12, rooftop: 0.26, space: 0.32, neon: 0.4 };
// The overall level of each arena's soundscape, measured against the stadium crowd so none of
// them plays louder.
const ARENA_TRIM = { stadium: 1, beach: 0.82, rooftop: 1, space: 1, neon: 0.85 };
// The overall level of the synthesized weather at full strength.
const WEATHER_GAIN = { rain: 1, wind: 1 };
// How rain sounds on the ground of each arena, the hiss, the patter of drops and the drumming
// of heavier drops on a roof.
const RAIN_SURFACE = {
    stadium: [0.55, 0.7, 1], beach: [1, 0.4, 0.1], rooftop: [0.75, 1, 0.3], space: [0, 0, 0], neon: [0.8, 0.8, 0.3],
};
// The time constant of a crossfade, so a switch settles in about a second.
const FADE = 0.3;

function midi(n) {
    return 440 * Math.pow(2, (n - 69) / 12);
}

// Scales a decoded file in place to a level, by its busiest tenth of a second and its peak.
export function levelBuffer(buffer, level) {
    const window = Math.max(1, Math.floor(buffer.sampleRate / 10));
    let peak = 0;
    let loudest = 0;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const d = buffer.getChannelData(ch);
        let sum = 0;
        for (let i = 0; i < d.length; i++) {
            const v = d[i];
            peak = Math.max(peak, Math.abs(v));
            sum += v * v;
            if (i >= window) {
                sum -= d[i - window] * d[i - window];
            }
            if (i >= window - 1) {
                loudest = Math.max(loudest, Math.sqrt(Math.max(0, sum) / window));
            }
        }
        if (d.length < window) {
            loudest = Math.max(loudest, Math.sqrt(sum / window));
        }
    }
    if (peak < 1e-5) {
        return buffer;
    }
    const byPeak = (level ? level.peak : FILE_PEAK) / peak;
    const byRms = level && loudest > 1e-6 ? level.rms / loudest : Infinity;
    const scale = Math.min(byPeak, byRms);
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const d = buffer.getChannelData(ch);
        for (let i = 0; i < d.length; i++) {
            d[i] *= scale;
        }
    }
    return buffer;
}

// A team's own sounds are cut to this length and their peaks brought to one level, so an
// upload can neither run long nor play louder than the rest of the page.
const TEAM_SOUND_SECONDS = 4;
const TEAM_SOUND_PEAK = 0.7;
const TEAM_SLOT_GAIN = { goal: 0.55, celebration: 0.5, chant: 0.3, shot: 0.4, pass: 0.3, tackle: 0.4, save: 0.45, kick: 0.35 };
// The shortest time between two plays of one team's slot, so a run of passes does not stutter.
const TEAM_SLOT_GAP = { kick: 0.25, pass: 0.5, shot: 0.8, tackle: 0.8, save: 1, goal: 3, celebration: 4, chant: 20 };

function noiseBuffer(ctx, seconds, kind) {
    const length = Math.floor(ctx.sampleRate * seconds);
    const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
        const data = buffer.getChannelData(ch);
        let b0 = 0;
        let b1 = 0;
        let b2 = 0;
        for (let i = 0; i < length; i++) {
            const white = Math.random() * 2 - 1;
            if (kind === 'pink') {
                // Paul Kellet's economical pink filter.
                b0 = 0.99765 * b0 + white * 0.099046;
                b1 = 0.963 * b1 + white * 0.2965164;
                b2 = 0.57 * b2 + white * 1.0526913;
                data[i] = (b0 + b1 + b2 + white * 0.1848) * 0.2;
            } else if (kind === 'brown') {
                // Integrated noise falls off with frequency, a distant rumble rather than a hiss.
                b0 = (b0 + 0.02 * white) / 1.02;
                data[i] = b0 * 3.5;
            } else {
                data[i] = white * 0.5;
            }
        }
        loopFade(data, Math.floor(ctx.sampleRate * 0.25));
    }
    return buffer;
}

// Drops of rain, rendered once. Many small ticks and a few round plips at random moments, most
// of them faint, so a loop of it sounds like rain on a surface rather than steady noise.
function dropsBuffer(ctx, seconds = 3, perSecond = 320) {
    const rate = ctx.sampleRate;
    const length = Math.floor(rate * seconds);
    const buffer = ctx.createBuffer(2, length, rate);
    for (let ch = 0; ch < 2; ch++) {
        const data = buffer.getChannelData(ch);
        const count = Math.floor(seconds * perSecond);
        for (let n = 0; n < count; n++) {
            const at = Math.floor(Math.random() * length);
            const amp = 0.1 + 0.9 * Math.random() ** 3;
            if (Math.random() < 0.75) {
                const decay = rate * (0.0008 + Math.random() * 0.0025);
                const span = Math.floor(decay * 5);
                for (let i = 0; i < span; i++) {
                    data[(at + i) % length] += amp * (Math.random() * 2 - 1) * Math.exp(-i / decay);
                }
            } else {
                const f = 1800 + Math.random() * 3200;
                const decay = rate * (0.002 + Math.random() * 0.004);
                const span = Math.floor(decay * 5);
                for (let i = 0; i < span; i++) {
                    data[(at + i) % length] += amp * 0.7 * Math.sin((2 * Math.PI * f * i) / rate) * Math.exp(-i / decay);
                }
            }
        }
    }
    let peak = 0;
    for (let ch = 0; ch < 2; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            peak = Math.max(peak, Math.abs(data[i]));
        }
    }
    for (let ch = 0; ch < 2; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            data[i] *= peak > 0 ? 0.8 / peak : 1;
        }
    }
    return buffer;
}

// The two ends of a loop fade into each other, so it has no click as it repeats.
function loopFade(data, fade) {
    const length = data.length;
    for (let i = 0; i < fade; i++) {
        const w = i / fade;
        data[i] = data[i] * w + data[length - fade + i] * (1 - w);
    }
}

// The reverberation of a large open stand, decaying noise that darkens as it fades, the way air
// and seats swallow the high end of an echo first.
function stadiumImpulse(ctx, seconds = 2.4) {
    const length = Math.floor(ctx.sampleRate * seconds);
    const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
        const data = buffer.getChannelData(ch);
        let low = 0;
        for (let i = 0; i < length; i++) {
            const f = i / length;
            const white = Math.random() * 2 - 1;
            const k = 0.5 - 0.45 * f;
            low += (white - low) * k;
            // A short gap before the first reflections, the stands lie some way off.
            const onset = Math.min(1, i / (ctx.sampleRate * 0.03));
            data[i] = low * onset * Math.pow(1 - f, 2.6) * 0.6;
        }
    }
    return buffer;
}

// Many people talking at once, rendered once offline. Each voice is noise shaped by a band in
// the range of speech and cut into syllables, phrases and pauses, so the bed sounds like a crowd
// murmuring rather than steady noise.
async function renderBabble(sampleRate, seconds = 9, voices = 34) {
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Offline) {
        return null;
    }
    const ctx = new Offline(2, Math.floor(sampleRate * seconds), sampleRate);
    const pink = noiseBuffer(ctx, 3, 'pink');
    const bus = ctx.createGain();
    const lowpass = ctx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 1700;
    lowpass.Q.value = 0.4;
    bus.connect(lowpass).connect(ctx.destination);
    for (let v = 0; v < voices; v++) {
        const src = ctx.createBufferSource();
        src.buffer = pink;
        src.loop = true;
        src.playbackRate.value = 0.8 + Math.random() * 0.5;
        const band = ctx.createBiquadFilter();
        band.type = 'bandpass';
        band.frequency.value = 280 + Math.random() * 820;
        band.Q.value = 1.6 + Math.random() * 2.4;
        const amp = ctx.createGain();
        amp.gain.value = 0;
        const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
        if (pan) {
            pan.pan.value = Math.random() * 1.8 - 0.9;
        }
        src.connect(band).connect(amp);
        (pan ? amp.connect(pan) : amp).connect(bus);
        let t = Math.random() * 1.5;
        while (t < seconds) {
            // A phrase of syllables, then a pause.
            const phraseEnd = t + 0.8 + Math.random() * 3;
            const level = 0.3 + Math.random() * 0.7;
            while (t < phraseEnd && t < seconds) {
                const syllable = 0.07 + Math.random() * 0.2;
                amp.gain.setTargetAtTime(level * (0.5 + Math.random() * 0.5), t, 0.02);
                amp.gain.setTargetAtTime(0.05 * level, t + syllable, 0.03);
                t += syllable + 0.03 + Math.random() * 0.12;
            }
            amp.gain.setTargetAtTime(0, t, 0.08);
            t += 0.4 + Math.random() * 2.5;
        }
        src.start(0, Math.random() * 2.5);
    }
    const buffer = await ctx.startRendering();
    let peak = 0;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < data.length; i++) {
            peak = Math.max(peak, Math.abs(data[i]));
        }
    }
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const data = buffer.getChannelData(ch);
        const scale = peak > 0 ? 0.9 / peak : 1;
        for (let i = 0; i < data.length; i++) {
            data[i] *= scale;
        }
        loopFade(data, Math.floor(sampleRate * 0.6));
    }
    return buffer;
}

// A team's sound, cut to its longest length with a short fade and scaled to one peak.
function tameBuffer(ctx, decoded) {
    const length = Math.min(decoded.length, Math.floor(decoded.sampleRate * TEAM_SOUND_SECONDS));
    const out = ctx.createBuffer(decoded.numberOfChannels, length, decoded.sampleRate);
    let peak = 0;
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        const src = decoded.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            peak = Math.max(peak, Math.abs(src[i]));
        }
    }
    const scale = peak > 1e-4 ? TEAM_SOUND_PEAK / peak : 0;
    const fade = Math.min(length, Math.floor(decoded.sampleRate * 0.2));
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        const src = decoded.getChannelData(ch);
        const dst = out.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            const tail = i > length - fade ? (length - i) / fade : 1;
            dst[i] = src[i] * scale * tail;
        }
    }
    return out;
}

function makeGain(ctx, value) {
    const g = ctx.createGain();
    g.gain.value = value;
    return g;
}

function makeFilter(ctx, type, frequency, q = 0.7) {
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = frequency;
    f.Q.value = q;
    return f;
}

// A stereo pan, or a plain pass through where the browser has none.
function makePan(ctx, value) {
    if (ctx.createStereoPanner) {
        const p = ctx.createStereoPanner();
        p.pan.value = value;
        return p;
    }
    return makeGain(ctx, 1);
}

// A noise buffer played from a random point, looping, until a given time.
function noiseShot(ctx, buffer, t, end, rate = 1) {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    src.playbackRate.value = rate;
    src.start(t, Math.random() * Math.max(0.01, buffer.duration - 0.01));
    src.stop(end);
    return src;
}

// A swell of a gain up to a peak and back to silence, smooth at both ends.
function swellGain(param, t, peak, attack, hold, release) {
    const top = Math.max(peak, 0.00011);
    param.setValueAtTime(0.0001, t);
    param.exponentialRampToValueAtTime(top, t + attack);
    param.setValueAtTime(top, t + attack + hold);
    param.exponentialRampToValueAtTime(0.0001, t + attack + hold + release);
}

// One soundscape, an arena's or a weather's. Everything in it reaches the listener through two
// fades, a direct one and one into the echo, which rise and fall together when it comes and
// goes, and its timers stop once it is done.
class Scape {
    constructor(sound, name, always = false) {
        const ctx = sound.ctx;
        this.sound = sound;
        this.ctx = ctx;
        this.name = name;
        this.always = always;
        this.fade = makeGain(ctx, 0);
        this.fade.connect(sound.master);
        this.wetFade = makeGain(ctx, 0);
        this.wetFade.connect(sound.reverb);
        this.sources = [];
        this.timers = new Set();
        this.live = true;
        this.target = 0;
        this.x = sound.intensity;
    }

    // Whether its sounds may start now, it must be the arena shown on a page that is playing.
    get current() {
        return this.live && this.sound.audible && (this.always || this.sound.arena === this.name);
    }

    setLevel(level) {
        if (!this.live || Math.abs(level - this.target) < 1e-4) {
            return;
        }
        this.target = level;
        const t = this.ctx.currentTime;
        this.fade.gain.setTargetAtTime(level, t, FADE);
        this.wetFade.gain.setTargetAtTime(level, t, FADE);
    }

    // Sends a node into the scape, directly and into the echo.
    send(node, dry = 1, wet = 0.3) {
        if (dry > 0) {
            node.connect(makeGain(this.ctx, dry)).connect(this.fade);
        }
        if (wet > 0) {
            node.connect(makeGain(this.ctx, wet)).connect(this.wetFade);
        }
        return node;
    }

    // A looping buffer from a random point, kept until the scape ends.
    loop(buffer, rate = 1) {
        const src = this.ctx.createBufferSource();
        src.buffer = buffer;
        src.loop = true;
        src.playbackRate.value = rate;
        src.start(0, Math.random() * Math.max(0.01, buffer.duration - 0.01));
        return this.keep(src);
    }

    keep(node) {
        this.sources.push(node);
        return node;
    }

    later(fn, ms) {
        const id = setTimeout(() => {
            this.timers.delete(id);
            if (this.live) {
                fn();
            }
        }, ms);
        this.timers.add(id);
    }

    // Runs a sound now and then while the scape is current, delay gives the next wait in
    // milliseconds.
    every(fn, delay, first = null) {
        const run = () => {
            if (this.current) {
                fn();
            }
            this.later(run, delay());
        };
        this.later(run, first === null ? delay() : first);
    }

    dispose() {
        this.live = false;
        for (const id of this.timers) {
            clearTimeout(id);
        }
        this.timers.clear();
        for (const src of this.sources) {
            try {
                src.stop();
            } catch {
                // A source already stopped has nothing to stop.
            }
        }
        this.sources = [];
        this.fade.disconnect();
        this.wetFade.disconnect();
    }
}

export class Sound {
    constructor(base = 'assets/sounds/') {
        this.base = base;
        this.ctx = null;
        this.volume = 0.5;
        this.muted = true;
        this.intensity = 0.3;
        this.ready = false;
        // Whether the match is playing on a visible page. A paused or hidden match fades out and
        // the context suspends, so nothing keeps sounding.
        this.active = true;
        this.files = new Map();
        this.teams = [null, null];
        this.teamBuffers = [new Map(), new Map()];
        this.lastPlayed = [new Map(), new Map()];
        this.ambience = new Map();
        this.ambienceWanted = new Map();
        // The arena shown and how far into the night it is, from 0 by day to 1 at night.
        this.arena = 'stadium';
        this.night = 0;
        this.scapes = new Map();
        this.weather = { rain: null, wind: null };
        this.lastOoh = -Infinity;
    }

    setActive(on) {
        on = !!on;
        if (on === this.active) {
            return;
        }
        this.active = on;
        if (!this.ctx) {
            return;
        }
        clearTimeout(this.suspendTimer);
        if (on) {
            this.ctx.resume().catch(() => {});
            this.applyGain();
        } else {
            this.master.gain.setTargetAtTime(0, this.ctx.currentTime, 0.06);
            this.suspendTimer = setTimeout(() => {
                if (!this.active && this.ctx.state === 'running') {
                    this.ctx.suspend().catch(() => {});
                }
            }, 400);
        }
    }

    // Whether a sound may start now, the page must be unmuted, visible and playing.
    get audible() {
        return this.ready && !this.muted && this.active;
    }

    // Creates the audio graph, must run inside a user gesture handler. A context may be handed
    // in instead, an offline one renders the sound for a test without playing it.
    unlock(context = null) {
        if (this.ctx) {
            if (this.ctx.state === 'suspended' && this.active) {
                this.ctx.resume();
            }
            return;
        }
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!context && !AudioCtx) {
            return;
        }
        const ctx = context || new AudioCtx();
        this.ctx = ctx;
        this.master = ctx.createGain();
        this.master.gain.value = this.muted || !this.active ? 0 : this.volume;
        const comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -14;
        comp.ratio.value = 3;
        this.master.connect(comp).connect(ctx.destination);
        // The crowd and its echo share a bus, the echo comes back quieter than the direct sound.
        this.reverb = ctx.createConvolver();
        this.reverb.buffer = stadiumImpulse(ctx);
        this.reverbReturn = ctx.createGain();
        this.reverbReturn.gain.value = ARENA_ECHO[this.arena];
        this.reverb.connect(this.reverbReturn).connect(this.master);
        this.pink = noiseBuffer(ctx, 4, 'pink');
        this.brown = noiseBuffer(ctx, 6, 'brown');
        this.white = noiseBuffer(ctx, 2, 'white');
        this.scapes.set('stadium', new Scape(this, 'stadium'));
        this.buildCrowd();
        this.ready = true;
        this.filesReady = this.loadFiles();
        this.loadTeamSounds();
        this.babbleReady = renderBabble(ctx.sampleRate).then((babble) => {
            if (babble && !this.files.has('crowd-loop')) {
                this.useCrowdBuffer(babble, false);
            }
        }).catch((err) => console.warn('crowd murmur not rendered', err));
        this.scheduleCrowd();
        this.applyArena();
        this.updateWeather();
    }

    // Sends a node to the master directly and to the stadium echo.
    route(node, dry = 1, wet = 0.4) {
        const d = this.ctx.createGain();
        d.gain.value = dry;
        node.connect(d).connect(this.master);
        if (wet > 0) {
            const w = this.ctx.createGain();
            w.gain.value = wet;
            node.connect(w).connect(this.reverb);
        }
    }

    async loadFiles() {
        let names = [];
        try {
            const res = await fetch(`${this.base}sounds.json`, { cache: 'no-cache' });
            if (res.ok) {
                const list = await res.json();
                names = Array.isArray(list.files) ? list.files.filter((f) => typeof f === 'string') : [];
            }
        } catch {
            // Without a list, every sound stays synthesized.
        }
        await Promise.all(names.map(async (file) => {
            const name = file.replace(/\.[a-z0-9]+$/i, '');
            if (!SOUND_FILES.includes(name) || /[/\\]/.test(file)) {
                return;
            }
            try {
                const res = await fetch(this.base + file);
                if (!res.ok) {
                    return;
                }
                const decoded = await this.ctx.decodeAudioData(await res.arrayBuffer());
                this.files.set(name, levelBuffer(decoded, FILE_LEVELS[name] || null));
            } catch (err) {
                console.warn('sound file not used', file, err);
            }
        }));
        if (this.files.has('crowd-loop')) {
            this.useCrowdBuffer(this.files.get('crowd-loop'), true);
        }
        for (const [name, level] of this.ambienceWanted) {
            this.setAmbience(name, level);
        }
        this.updateBeds();
        this.updateWeather();
    }

    // A distant crowd. Two copies of a murmur run apart in time, one darker for a calm crowd and
    // one brighter that swells with excitement, over a low rumble. Shouts and claps rise from it
    // now and then, and all of it reaches the listener through the echo of the stands.
    buildCrowd() {
        const ctx = this.ctx;
        this.crowdBus = ctx.createGain();
        this.crowdBus.gain.value = 1;
        this.scapes.get('stadium').send(this.crowdBus, 0.8, 0.5);
        const layer = (buffer, low, high, gain, rate = 1) => {
            const src = ctx.createBufferSource();
            src.buffer = buffer;
            src.loop = true;
            src.playbackRate.value = rate;
            const highpass = ctx.createBiquadFilter();
            highpass.type = 'highpass';
            highpass.frequency.value = high;
            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = low;
            filter.Q.value = 0.5;
            const amp = ctx.createGain();
            amp.gain.value = gain;
            const swell = ctx.createGain();
            swell.gain.value = 1;
            src.connect(highpass).connect(filter).connect(amp).connect(swell).connect(this.crowdBus);
            src.start(0, Math.random() * Math.max(0.1, buffer.duration - 0.1));
            return { src, filter, amp, swell, highpass };
        };
        this.rumble = layer(this.brown, 380, 60, 0.025);
        this.murmur = layer(this.brown, 520, 70, 0.04);
        this.roar = layer(this.pink, 700, 160, 0);
        this.setIntensity(this.intensity);
    }

    // Puts a murmur, rendered or from a file, in place of the noise layers.
    useCrowdBuffer(buffer, fromFile) {
        const ctx = this.ctx;
        const swap = (layer, rate, offset) => {
            const src = ctx.createBufferSource();
            src.buffer = buffer;
            src.loop = true;
            src.playbackRate.value = rate;
            src.connect(layer.highpass);
            const t = ctx.currentTime;
            src.start(t, offset % buffer.duration);
            try {
                layer.src.stop(t + 0.05);
            } catch {
                // A layer never started has nothing to stop.
            }
            layer.src = src;
        };
        swap(this.murmur, 1, Math.random() * buffer.duration);
        swap(this.roar, fromFile ? 1 : 1.12, Math.random() * buffer.duration);
        this.murmur.filter.frequency.value = fromFile ? 6000 : 1500;
        this.murmur.highpass.frequency.value = fromFile ? 30 : 140;
        this.roar.highpass.frequency.value = fromFile ? 120 : 220;
        this.crowdReal = true;
        this.setIntensity(this.intensity);
    }

    // A murmur of a few dozen people, rendered once for the crowds of the other arenas while their
    // recordings load or when they are absent.
    smallBabble() {
        if (!this.smallBabbleReady) {
            this.smallBabbleReady = renderBabble(this.ctx.sampleRate, 8, 9).catch((err) => {
                console.warn('small crowd not rendered', err);
                return null;
            });
        }
        return this.smallBabbleReady;
    }

    // Slow random swells of the murmur and the shouts and claps that rise from the crowd, each
    // timer reschedules itself and plays nothing while the page is silent or the stadium is
    // elsewhere.
    scheduleCrowd() {
        const here = () => this.audible && this.arena === 'stadium';
        const swell = () => {
            if (here()) {
                const t = this.ctx.currentTime;
                this.murmur.swell.gain.setTargetAtTime(0.7 + Math.random() * 0.6, t, 1.6);
                this.roar.swell.gain.setTargetAtTime(0.6 + Math.random() * 0.8, t, 1.2);
            }
            this.swellTimer = setTimeout(swell, 1500 + Math.random() * 3000);
        };
        const shout = () => {
            const x = this.intensity;
            if (here() && !this.files.has('crowd-loop')) {
                const count = Math.random() < 0.25 + x * 0.4 ? 2 + Math.floor(Math.random() * 3) : 1;
                for (let i = 0; i < count; i++) {
                    this.shout(this.ctx.currentTime + Math.random() * 0.35, 0.5 + x * 0.6);
                }
            }
            this.shoutTimer = setTimeout(shout, (2000 + Math.random() * 6000) / (0.6 + x));
        };
        const claps = () => {
            const x = this.intensity;
            if (here() && !this.files.has('crowd-loop') && Math.random() < 0.35 + x * 0.5) {
                this.claps(4 + Math.floor(Math.random() * (10 + x * 25)), 0.4 + x * 0.6);
            }
            this.clapTimer = setTimeout(claps, 3500 + Math.random() * 7000);
        };
        // Fans chant in every arena, softer in the smaller ones.
        const chant = () => {
            if (this.audible && FAN_CHANT[this.arena]) {
                const side = Math.random() < 0.5 ? 0 : 1;
                this.teamSound(side, 'chant', null, true) || this.teamSound(1 - side, 'chant', null, true);
            }
            this.chantTimer = setTimeout(chant, 45000 + Math.random() * 75000);
        };
        swell();
        this.shoutTimer = setTimeout(shout, 1500);
        this.clapTimer = setTimeout(claps, 3000);
        this.chantTimer = setTimeout(chant, 30000 + Math.random() * 30000);
    }

    // Where the stadium crowd's voices go, mostly into the echo of the stands.
    stadiumPlace(dry, wet) {
        return { scape: this.scapes.get('stadium'), dry, wet };
    }

    // One distant voice calling out, a buzzing tone through two vowel resonances, mostly heard
    // through the echo of the stands, or in the scape and with the length a place gives. A
    // rising voice is a cheer.
    shout(t, level = 1, place = null) {
        const p = place || this.stadiumPlace(0.25, 0.9);
        const ctx = this.ctx;
        const vowels = [[750, 1200], [500, 900], [450, 1900], [650, 1100], [350, 2200]];
        const [f1, f2] = vowels[Math.floor(Math.random() * vowels.length)];
        const f0 = 140 + Math.random() * 180;
        const length = p.length ? p.length * (0.7 + Math.random() * 0.6) : 0.25 + Math.random() * 0.55;
        const osc = ctx.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(f0, t);
        const end = p.rise ? f0 * (1.1 + Math.random() * 0.3) : f0 * (0.85 + Math.random() * 0.35);
        osc.frequency.linearRampToValueAtTime(end, t + length);
        const out = ctx.createGain();
        out.gain.setValueAtTime(0.0001, t);
        out.gain.exponentialRampToValueAtTime(0.006 * level, t + 0.05);
        out.gain.exponentialRampToValueAtTime(0.0001, t + length);
        const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
        for (const [f, q, g] of [[f1, 6, 1], [f2, 8, 0.6]]) {
            const band = ctx.createBiquadFilter();
            band.type = 'bandpass';
            band.frequency.value = f;
            band.Q.value = q;
            const bg = ctx.createGain();
            bg.gain.value = g;
            osc.connect(band).connect(bg).connect(out);
        }
        const dark = ctx.createBiquadFilter();
        dark.type = 'lowpass';
        dark.frequency.value = 1800;
        out.connect(dark);
        if (pan) {
            pan.pan.value = Math.random() * 1.6 - 0.8;
            dark.connect(pan);
            p.scape.send(pan, p.dry, p.wet);
        } else {
            p.scape.send(dark, p.dry, p.wet);
        }
        osc.start(t);
        osc.stop(t + length + 0.05);
    }

    // A ripple of hand claps, short bands of noise spread over a second or so.
    claps(count, level = 1, place = null) {
        const p = place || this.stadiumPlace(0.3, 0.9);
        const ctx = this.ctx;
        const t0 = ctx.currentTime + 0.02;
        const span = 0.4 + count * 0.05;
        const bus = ctx.createGain();
        bus.gain.value = 0.012 * level;
        const band = ctx.createBiquadFilter();
        band.type = 'bandpass';
        band.frequency.value = p.band || 1300 + Math.random() * 700;
        band.Q.value = 0.9;
        bus.connect(band);
        p.scape.send(band, p.dry, p.wet);
        for (let i = 0; i < count; i++) {
            const t = t0 + Math.random() * span;
            const src = ctx.createBufferSource();
            src.buffer = this.white;
            const g = ctx.createGain();
            g.gain.setValueAtTime(0.0001, t);
            g.gain.exponentialRampToValueAtTime(0.5 + Math.random() * 0.5, t + 0.002);
            g.gain.exponentialRampToValueAtTime(0.0001, t + 0.03);
            src.connect(g).connect(bus);
            src.start(t, Math.random() * 1.5);
            src.stop(t + 0.04);
        }
    }

    setVolume(v) {
        this.volume = v;
        this.applyGain();
    }

    setMuted(m) {
        this.muted = m;
        this.applyGain();
    }

    applyGain() {
        if (this.master) {
            this.master.gain.setTargetAtTime(this.muted || !this.active ? 0 : this.volume, this.ctx.currentTime, 0.05);
        }
    }

    // Moves the listener with the camera, so a sound placed on the pitch comes from its place.
    setListener(camera) {
        if (!this.ready) {
            return;
        }
        const l = this.ctx.listener;
        const p = camera.position;
        const f = camera.getWorldDirection(this._dir || (this._dir = camera.position.clone()));
        const t = this.ctx.currentTime;
        if (l.positionX) {
            l.positionX.setTargetAtTime(p.x, t, 0.05);
            l.positionY.setTargetAtTime(p.y, t, 0.05);
            l.positionZ.setTargetAtTime(p.z, t, 0.05);
            l.forwardX.setTargetAtTime(f.x, t, 0.05);
            l.forwardY.setTargetAtTime(f.y, t, 0.05);
            l.forwardZ.setTargetAtTime(f.z, t, 0.05);
            l.upX.value = 0;
            l.upY.value = 1;
            l.upZ.value = 0;
        } else {
            l.setPosition(p.x, p.y, p.z);
            l.setOrientation(f.x, f.y, f.z, 0, 1, 0);
        }
    }

    // Excitement from 0 to 1, eased in the audio thread. The levels stay soft, the crowd is a
    // bed under the match and never louder than the effects. Each arena answers in its own way.
    setIntensity(x) {
        this.intensity = x;
        if (!this.ready || !this.murmur) {
            return;
        }
        const t = this.ctx.currentTime;
        const real = this.crowdReal;
        this.rumble.amp.gain.setTargetAtTime(real ? 0.012 : 0.022 + x * 0.01, t, 0.8);
        this.murmur.amp.gain.setTargetAtTime((real ? 0.05 : 0.045) * (1 - x * 0.35) + 0.01, t, 0.8);
        this.roar.amp.gain.setTargetAtTime(x * x * (real ? 0.07 : 0.06), t, 0.6);
        this.roar.filter.frequency.setTargetAtTime((real ? 2200 : 900) + x * 900, t, 0.8);
        for (const s of this.scapes.values()) {
            if (s.intensity && Math.abs(x - s.x) > 0.01) {
                s.x = x;
                s.intensity(x);
            }
        }
    }

    // The arena shown and the daylight there, from 0 at night to 1 by day. The soundscape of
    // the arena fades in while the last one fades out.
    setSurroundings(arena, day = 1) {
        this.arena = SOUND_ARENAS.includes(arena) ? arena : 'stadium';
        this.night = Math.max(0, Math.min(1, 1 - (Number.isFinite(day) ? day : 1)));
        if (this.ctx) {
            this.applyArena();
            this.updateWeather();
        }
    }

    applyArena() {
        const name = this.arena;
        if (!this.scapes.has(name)) {
            this.scapes.set(name, this.buildScape(name));
        }
        for (const [n, s] of this.scapes) {
            const on = n === name;
            s.setLevel(on ? ARENA_TRIM[n] : 0);
            if (s.setNight) {
                s.setNight(this.night);
            }
            if (on) {
                clearTimeout(s.retireTimer);
            } else if (n !== 'stadium') {
                this.retire(s, () => this.arena === n, () => this.scapes.delete(n));
            }
        }
        this.reverbReturn.gain.setTargetAtTime(ARENA_ECHO[name], this.ctx.currentTime, FADE);
        this.updateBeds();
    }

    // Ends a scape once it is no longer wanted and has faded to silence.
    retire(s, wanted, done) {
        clearTimeout(s.retireTimer);
        const check = () => {
            if (wanted() || !s.live) {
                return;
            }
            if (s.fade.gain.value > 0.002) {
                s.retireTimer = setTimeout(check, 1000);
                return;
            }
            s.dispose();
            done();
        };
        s.retireTimer = setTimeout(check, 3000);
    }

    // A file for an arena's steady bed, when present, plays in place of the synthesized one.
    updateBeds() {
        if (!this.ctx) {
            return;
        }
        const t = this.ctx.currentTime;
        for (const s of this.scapes.values()) {
            if (s.crowdFile) {
                s.crowdFile();
            }
            let replaced = s.bedFile ? this.files.has(`${s.bedFile}-loop`) : false;
            // Loops of an arena's own, each in place of one synthesized part, unless a file
            // of the bed already plays over it.
            for (const fb of s.fileBeds || []) {
                const buffer = replaced ? null : this.files.get(fb.name);
                if (buffer && !fb.src) {
                    fb.src = s.loop(buffer, 1);
                    fb.src.connect(makeGain(this.ctx, ARENA_FILE_GAIN[fb.name] || 0.05)).connect(fb.out);
                }
                if (fb.mute === s.bed) {
                    replaced = replaced || !!buffer;
                } else {
                    fb.mute.gain.setTargetAtTime(buffer ? 0 : 1, t, FADE);
                }
            }
            if (s.bed) {
                s.bed.gain.setTargetAtTime(replaced ? 0 : 1, t, FADE);
            }
        }
    }

    // Plays an arena's file once into its scape, and returns whether there was one.
    arenaFile(s, name, dry, wet, { rate = 1, pan = 0, gain = 1 } = {}) {
        const buffer = this.files.get(name);
        if (!buffer) {
            return false;
        }
        const src = this.ctx.createBufferSource();
        src.buffer = buffer;
        src.playbackRate.value = rate;
        const p = makePan(this.ctx, pan);
        src.connect(makeGain(this.ctx, (ARENA_FILE_GAIN[name] || 0.03) * gain)).connect(p);
        s.send(p, dry, wet);
        src.start();
        return true;
    }

    buildScape(name) {
        const s = new Scape(this, name);
        if (name === 'beach') {
            this.buildBeach(s);
        } else if (name === 'rooftop') {
            this.buildRooftop(s);
        } else if (name === 'space') {
            this.buildSpace(s);
        } else if (name === 'neon') {
            this.buildNeon(s);
        }
        if (s.intensity) {
            s.intensity(this.intensity);
        }
        return s;
    }

    // The crowd of an arena other than the stadium, the stadium's recordings heard through the
    // band, the echo and the walls of the place, smaller and softer than the stadium crowd. The
    // murmur swells with the play and a second, brighter copy roars as play nears a goal. Until
    // the files load, or without them, a murmur of a few dozen people rendered once stands in,
    // with a shout or a few claps now and then.
    arenaCrowd(s, name) {
        const ctx = this.ctx;
        const c = ARENA_CROWD[name];
        s.crowd = makeGain(ctx, 0);
        s.cheer = makeGain(ctx, 1);
        const input = makeFilter(ctx, 'highpass', c.high, 0.5);
        const band = makeFilter(ctx, 'lowpass', c.low, 0.5);
        input.connect(band);
        const out = makeGain(ctx, 1);
        band.connect(out);
        if (c.slap > 0) {
            // A short echo off the walls around, darker with each return.
            const delay = ctx.createDelay(1);
            delay.delayTime.value = c.slap;
            const back = makeFilter(ctx, 'lowpass', 2400, 0.5);
            const feedback = makeGain(ctx, 0.28);
            band.connect(delay).connect(back).connect(feedback).connect(delay);
            back.connect(makeGain(ctx, 0.45)).connect(out);
        }
        s.send(out, c.dry, c.wet);
        // The crowd's own sound, before the place, which an offline render can measure alone.
        s.crowdOut = out;
        s.crowd.connect(s.cheer).connect(input);
        const babble = makeGain(ctx, 0.06 * c.level);
        babble.connect(s.crowd);
        this.smallBabble().then((buffer) => {
            if (buffer && s.live && !s.murmur) {
                s.loop(buffer, 1).connect(babble);
            }
        });
        // The recordings take over from the rendered murmur once they load.
        s.crowdFile = () => {
            const buffer = this.files.get('crowd-loop');
            if (!buffer || s.murmur) {
                return;
            }
            babble.gain.setTargetAtTime(0, ctx.currentTime, FADE);
            s.murmur = makeGain(ctx, 0.055 * c.level);
            s.loop(buffer, 1).connect(s.murmur).connect(s.crowd);
            s.roar = makeGain(ctx, 0);
            s.loop(buffer, 1).connect(makeFilter(ctx, 'highpass', 300, 0.5)).connect(s.roar).connect(s.crowd);
            s.crowdIntensity(this.intensity);
        };
        s.crowdIntensity = (x) => {
            const t = ctx.currentTime;
            s.crowd.gain.setTargetAtTime(0.75 + 0.5 * x, t, 0.8);
            if (s.roar) {
                s.roar.gain.setTargetAtTime(x * x * 0.06 * c.level, t, 0.6);
            }
        };
        const place = { scape: s, dry: c.dry * 0.8, wet: c.wet * 1.6 + 0.1 };
        const synthesized = () => s.current && !this.files.has('crowd-loop');
        s.every(() => {
            if (synthesized()) {
                const x = this.intensity;
                this.shout(ctx.currentTime + Math.random() * 0.3, (0.35 + x * 0.5) * 0.8, place);
            }
        }, () => (4000 + Math.random() * 10000) / (0.6 + this.intensity), 2500);
        s.every(() => {
            const x = this.intensity;
            if (synthesized() && Math.random() < 0.2 + x * 0.4) {
                this.claps(3 + Math.floor(Math.random() * (4 + x * 8)), 0.35 + x * 0.4, place);
            }
        }, () => 6000 + Math.random() * 9000, 5000);
        // A reaction from the stadium's recordings, at the stadium's gain times the arena's.
        const react = (file, gain) => {
            const buffer = this.files.get(file);
            if (!buffer) {
                return false;
            }
            const src = ctx.createBufferSource();
            src.buffer = buffer;
            src.connect(makeGain(ctx, gain * c.react)).connect(input);
            src.start();
            return true;
        };
        s.goal = (own) => {
            const t = ctx.currentTime;
            const k = own ? 0.7 : 1;
            s.cheer.gain.setTargetAtTime(1.6, t + 0.1, 0.4);
            s.cheer.gain.setTargetAtTime(1, t + 3, 1.5);
            if (react('crowd-cheer', own ? 0.35 : 0.55)) {
                return;
            }
            const count = 6 + Math.floor(Math.random() * 5);
            for (let i = 0; i < count; i++) {
                this.shout(t + 0.1 + Math.random() * 2.2, (1.1 + Math.random() * 0.6) * 0.8 * k, { ...place, length: 0.75, rise: true });
            }
            s.later(() => this.audible && this.claps(16 + Math.floor(Math.random() * 8), 0.9 * k, place), 700);
            s.later(() => this.audible && this.claps(10, 0.7 * k, place), 2200);
        };
        s.applause = () => {
            if (react('applause', 0.45)) {
                return;
            }
            this.claps(28, 1, place);
            s.later(() => this.audible && this.claps(20, 0.8, place), 900);
        };
        s.ooh = () => {
            if (react('crowd-ooh', 0.45)) {
                return;
            }
            const t = ctx.currentTime;
            for (let i = 0; i < 3; i++) {
                this.shout(t + Math.random() * 0.25, 0.9, place);
            }
        };
    }

    // The beach. The wash of the sea, waves that break and run back up the sand at irregular
    // times, a light breeze, gulls now and then by day, and a crowd on the sand.
    buildBeach(s) {
        const ctx = this.ctx;
        s.bed = makeGain(ctx, 1);
        s.bedFile = 'waves';
        s.send(s.bed, 1, 0.08);
        s.loop(this.brown, 0.85).connect(makeFilter(ctx, 'lowpass', 360, 0.5)).connect(makeGain(ctx, 0.035)).connect(s.bed);
        s.loop(this.pink, 0.9).connect(makeFilter(ctx, 'highpass', 1200, 0.5)).connect(makeFilter(ctx, 'lowpass', 3400, 0.5))
            .connect(makeGain(ctx, 0.003)).connect(s.bed);
        s.every(() => this.wave(s), () => 4500 + Math.random() * 7500, 600);
        // The sea breeze, a soft band of noise that rises and falls.
        const breeze = makeGain(ctx, 0.008);
        const breezeBand = makeFilter(ctx, 'bandpass', 520, 0.5);
        s.loop(this.pink, 0.7).connect(breezeBand).connect(breeze);
        s.send(breeze, 1, 0);
        s.every(() => {
            const t = ctx.currentTime;
            breeze.gain.setTargetAtTime(0.004 + Math.random() * 0.01, t, 1.2);
            breezeBand.frequency.setTargetAtTime(380 + Math.random() * 360, t, 1.5);
        }, () => 2000 + Math.random() * 4000, 0);
        s.every(() => this.gull(s), () => (14000 + Math.random() * 30000) * (1 + this.night * 4), 6000);
        this.arenaCrowd(s, 'beach');
        s.intensity = (x) => s.crowdIntensity(x);
    }

    // One wave, a low roar that brightens as it curls and breaks, then the fizz of foam running
    // back down the sand.
    wave(s) {
        const recorded = BEACH_WAVES.filter((n) => this.files.has(n));
        if (recorded.length) {
            const name = recorded[Math.floor(Math.random() * recorded.length)];
            if (this.arenaFile(s, name, 0.8, 0.3, { rate: 0.92 + Math.random() * 0.16, pan: Math.random() * 1.2 - 0.6 })) {
                return;
            }
        }
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05;
        const size = 0.45 + Math.random() * 0.55;
        const rise = 1.3 + Math.random() * 1.4;
        const crash = t + rise;
        const tail = 2.6 + Math.random() * 2.8;
        const end = crash + tail * 2.2;
        const pan = makePan(ctx, Math.random() * 1.2 - 0.6);
        pan.connect(s.bed);
        const body = noiseShot(ctx, this.brown, t, end, 0.9 + Math.random() * 0.2);
        const low = makeFilter(ctx, 'lowpass', 220, 0.6);
        low.frequency.setValueAtTime(220, t);
        low.frequency.exponentialRampToValueAtTime(480 + 520 * size, crash);
        low.frequency.setTargetAtTime(230, crash, tail * 0.45);
        const bodyGain = ctx.createGain();
        bodyGain.gain.setValueAtTime(0.0001, t);
        bodyGain.gain.exponentialRampToValueAtTime(0.07 * size, crash);
        bodyGain.gain.setTargetAtTime(0.0001, crash + 0.15, tail * 0.4);
        body.connect(low).connect(bodyGain).connect(pan);
        const foam = noiseShot(ctx, this.pink, crash - 0.2, end, 1);
        const band = makeFilter(ctx, 'bandpass', 2400, 0.6);
        band.frequency.setValueAtTime(2800, crash);
        band.frequency.setTargetAtTime(1800, crash, tail * 0.5);
        const foamGain = ctx.createGain();
        foamGain.gain.setValueAtTime(0.0001, crash - 0.2);
        foamGain.gain.exponentialRampToValueAtTime(0.01 * size, crash + 0.35);
        foamGain.gain.setTargetAtTime(0.0001, crash + 0.6, tail * 0.45);
        foam.connect(band).connect(makeFilter(ctx, 'lowpass', 4500, 0.5)).connect(foamGain).connect(pan);
    }

    // A gull far off, a short series of falling calls with a little rasp.
    gull(s) {
        if (this.arenaFile(s, 'gulls', 0.7, 0.3, { rate: 0.9 + Math.random() * 0.2, pan: Math.random() * 1.6 - 0.8 })) {
            return;
        }
        const ctx = this.ctx;
        const calls = 1 + Math.floor(Math.random() * 4);
        const base = 0.85 + Math.random() * 0.3;
        const level = 0.4 + Math.random() * 0.6;
        const pan = makePan(ctx, Math.random() * 1.6 - 0.8);
        const band = makeFilter(ctx, 'bandpass', 1700 * base, 1.1);
        band.connect(pan);
        s.send(pan, 0.7, 0.3);
        let t = ctx.currentTime + 0.05;
        for (let i = 0; i < calls; i++) {
            const length = i === 0 ? 0.32 + Math.random() * 0.12 : 0.16 + Math.random() * 0.1;
            const osc = ctx.createOscillator();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(1350 * base, t);
            osc.frequency.linearRampToValueAtTime(2050 * base, t + 0.045);
            osc.frequency.exponentialRampToValueAtTime(1150 * base, t + length);
            const vib = ctx.createOscillator();
            vib.frequency.value = 26 + Math.random() * 10;
            vib.connect(makeGain(ctx, 35)).connect(osc.frequency);
            const g = ctx.createGain();
            g.gain.setValueAtTime(0.0001, t);
            g.gain.exponentialRampToValueAtTime(0.008 * level, t + 0.03);
            g.gain.exponentialRampToValueAtTime(0.0001, t + length);
            osc.connect(g).connect(band);
            osc.start(t);
            osc.stop(t + length + 0.05);
            vib.start(t);
            vib.stop(t + length + 0.05);
            t += length + 0.08 + Math.random() * 0.1;
        }
    }

    // The city rooftop. Traffic far below and the wash of the city, a car passing, the odd horn
    // or siren a long way off, wind at height around the railings, and a crowd in a small stand
    // whose voices come back off the towers around. The city
    // quietens at night.
    buildRooftop(s) {
        const ctx = this.ctx;
        s.bed = makeGain(ctx, 1);
        s.bedFile = 'city';
        s.send(s.bed, 1, 0.12);
        const rumble = makeGain(ctx, 0.035);
        s.loop(this.brown, 0.7).connect(makeFilter(ctx, 'lowpass', 170, 0.5)).connect(rumble).connect(s.bed);
        const wash = makeGain(ctx, 0.006);
        s.loop(this.pink, 0.55).connect(makeFilter(ctx, 'bandpass', 520, 0.5)).connect(makeFilter(ctx, 'lowpass', 1200, 0.5))
            .connect(wash).connect(s.bed);
        s.setNight = (n) => {
            const t = ctx.currentTime;
            rumble.gain.setTargetAtTime(0.035 * (1 - 0.45 * n), t, FADE);
            wash.gain.setTargetAtTime(0.006 * (1 - 0.5 * n), t, FADE);
        };
        s.every(() => this.carPass(s), () => (5000 + Math.random() * 9000) * (1 + this.night * 0.8), 2500);
        s.every(() => this.horn(s), () => (16000 + Math.random() * 30000) * (1 + this.night * 1.5), 9000);
        s.every(() => this.siren(s), () => 60000 + Math.random() * 100000, 35000 + Math.random() * 30000);
        // Wind at height, a steady band of air with gusts, and a faint tone where it meets the
        // railings.
        const wind = makeGain(ctx, 0.006);
        const windBand = makeFilter(ctx, 'bandpass', 340, 0.8);
        s.loop(this.pink, 0.75).connect(windBand).connect(wind);
        s.send(wind, 1, 0);
        const rail = makeGain(ctx, 0);
        s.loop(this.pink, 1.1).connect(makeFilter(ctx, 'bandpass', 1250, 10)).connect(rail);
        s.send(rail, 1, 0.1);
        s.every(() => {
            const t = ctx.currentTime;
            const gust = Math.random();
            wind.gain.setTargetAtTime(0.004 + gust * 0.008, t, 1);
            windBand.frequency.setTargetAtTime(280 + gust * 220, t, 1.2);
            rail.gain.setTargetAtTime(gust > 0.7 ? 0.01 * (gust - 0.7) : 0, t, 0.9);
        }, () => 1800 + Math.random() * 3500, 0);
        this.arenaCrowd(s, 'rooftop');
        const goal = s.goal;
        s.goal = (own) => {
            goal(own);
            // The city honks along now and then.
            if (Math.random() < 0.5) {
                s.later(() => this.audible && this.horn(s, 2), 1600);
            }
        };
        s.intensity = (x) => s.crowdIntensity(x);
    }

    // A car passing in the street far below, a swell of tyre noise that sweeps across.
    carPass(s) {
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05;
        const length = 2.5 + Math.random() * 2;
        const level = 0.006 * (0.5 + Math.random() * 0.5) * (1 - 0.4 * this.night);
        const src = noiseShot(ctx, this.pink, t, t + length + 0.1, 0.8);
        const band = makeFilter(ctx, 'bandpass', 380, 0.8);
        band.frequency.setValueAtTime(380, t);
        band.frequency.linearRampToValueAtTime(820, t + length * 0.5);
        band.frequency.linearRampToValueAtTime(420, t + length);
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(level, t + length * 0.5);
        g.gain.exponentialRampToValueAtTime(0.0001, t + length);
        const from = Math.random() < 0.5 ? -0.7 : 0.7;
        const pan = makePan(ctx, from);
        if (pan.pan) {
            pan.pan.setValueAtTime(from, t);
            pan.pan.linearRampToValueAtTime(-from, t + length);
        }
        src.connect(band).connect(g).connect(pan).connect(s.bed);
    }

    // A car horn a few streets away, one or two toots, dulled by distance and echoing off the
    // towers.
    horn(s, toots = Math.random() < 0.4 ? 2 : 1) {
        if (this.arenaFile(s, 'city-horn', 0.5, 0.9, { rate: 0.92 + Math.random() * 0.16, pan: Math.random() * 1.6 - 0.8 })) {
            return;
        }
        const ctx = this.ctx;
        const f = 380 + Math.random() * 120;
        const pan = makePan(ctx, Math.random() * 1.6 - 0.8);
        const low = makeFilter(ctx, 'lowpass', 900, 0.6);
        low.connect(pan);
        s.send(pan, 0.5, 0.9);
        let t = ctx.currentTime + 0.05;
        for (let i = 0; i < toots; i++) {
            const length = 0.18 + Math.random() * 0.27;
            const g = ctx.createGain();
            g.gain.setValueAtTime(0.0001, t);
            g.gain.exponentialRampToValueAtTime(0.003, t + 0.02);
            g.gain.setValueAtTime(0.003, t + length);
            g.gain.exponentialRampToValueAtTime(0.0001, t + length + 0.06);
            g.connect(low);
            for (const m of [1, 1.26]) {
                const o = ctx.createOscillator();
                o.type = 'square';
                o.frequency.value = f * m;
                o.connect(g);
                o.start(t);
                o.stop(t + length + 0.1);
            }
            t += length + 0.12;
        }
    }

    // A siren far across the city, a slow wail that drifts past.
    siren(s) {
        if (this.arenaFile(s, 'city-siren', 0.3, 1, { rate: 0.95 + Math.random() * 0.1, pan: Math.random() * 1.6 - 0.8 })) {
            return;
        }
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05;
        const length = 6 + Math.random() * 3;
        const osc = ctx.createOscillator();
        osc.frequency.value = 900;
        const lfo = ctx.createOscillator();
        lfo.frequency.value = 0.3 + Math.random() * 0.1;
        lfo.connect(makeGain(ctx, 280)).connect(osc.frequency);
        const g = ctx.createGain();
        swellGain(g.gain, t, 0.002, 2, length - 4.5, 2.5);
        const pan = makePan(ctx, 0);
        if (pan.pan) {
            const from = Math.random() * 1.4 - 0.7;
            pan.pan.setValueAtTime(from, t);
            pan.pan.linearRampToValueAtTime(-from * 0.6, t + length);
        }
        osc.connect(makeFilter(ctx, 'lowpass', 1400, 0.5)).connect(g).connect(pan);
        s.send(pan, 0.4, 0.8);
        osc.start(t);
        osc.stop(t + length + 0.1);
        lfo.start(t);
        lfo.stop(t + length + 0.1);
    }

    // The space station. No air outside, so only the station itself, a deep slow hum, soft
    // machinery and airflow, gentle beeps and radio chirps, and a crowd in the enclosed galleries.
    // A goal is answered by a chime and the crowd's cheer.
    buildSpace(s) {
        const ctx = this.ctx;
        s.bed = makeGain(ctx, 1);
        s.bedFile = 'hum';
        s.send(s.bed, 1, 0.2);
        const humFile = makeGain(ctx, 1);
        s.send(humFile, 1, 0.2);
        s.fileBeds = [{ name: 'space-hum', mute: s.bed, out: humFile }];
        const hum = makeGain(ctx, 0.014);
        const humLow = makeFilter(ctx, 'lowpass', 380, 0.5);
        humLow.connect(hum).connect(s.bed);
        for (const [f, level] of [[55, 0.5], [110.4, 0.3], [165.2, 0.1], [220.7, 0.04]]) {
            const o = s.keep(ctx.createOscillator());
            o.frequency.value = f;
            o.connect(makeGain(ctx, level)).connect(humLow);
            o.start();
        }
        // A slow breath in the hum.
        const lfo = s.keep(ctx.createOscillator());
        lfo.frequency.value = 0.07;
        lfo.connect(makeGain(ctx, 0.004)).connect(hum.gain);
        lfo.start();
        s.loop(this.brown, 0.5).connect(makeFilter(ctx, 'lowpass', 150, 0.5)).connect(makeGain(ctx, 0.012)).connect(s.bed);
        const air = makeGain(ctx, 0.015);
        const airLow = makeFilter(ctx, 'lowpass', 1500, 0.5);
        s.loop(this.pink, 1).connect(makeFilter(ctx, 'highpass', 300, 0.5)).connect(airLow).connect(air).connect(s.bed);
        s.every(() => this.pump(s), () => 6000 + Math.random() * 6000, 3000);
        s.every(() => this.beep(s), () => 7000 + Math.random() * 13000, 4000);
        s.every(() => this.radio(s), () => 16000 + Math.random() * 24000, 11000);
        // A soft proximity ping that quickens as play nears a goal.
        s.every(() => {
            if (this.intensity > 0.55) {
                this.ping(s, (this.intensity - 0.55) / 0.45);
            }
        }, () => 700 + (1 - this.intensity) * 1600, 1000);
        this.arenaCrowd(s, 'space');
        s.intensity = (x) => {
            const t = ctx.currentTime;
            airLow.frequency.setTargetAtTime(1300 + x * 700, t, 0.8);
            air.gain.setTargetAtTime(0.012 + x * 0.006, t, 0.8);
            s.crowdIntensity(x);
        };
        const cheer = s.goal;
        s.goal = (own) => {
            this.chime(s, [84, 88, 91, 96], 0.14, 0.02);
            cheer(own);
        };
        s.fanfare = () => {
            this.chime(s, [79, 84, 88, 91, 88, 91, 96], 0.2, 0.02);
            cheer(false);
            s.later(() => this.audible && s.applause(), 900);
        };
    }

    // The machinery cycling, a soft whoosh of air through a duct and now and then a servo.
    pump(s) {
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05;
        const length = 2.2 + Math.random() * 1.2;
        const src = noiseShot(ctx, this.pink, t, t + length + 0.1, 0.7);
        const band = makeFilter(ctx, 'bandpass', 220, 1.2);
        band.frequency.setValueAtTime(220, t);
        band.frequency.linearRampToValueAtTime(480, t + length * 0.4);
        band.frequency.linearRampToValueAtTime(260, t + length);
        const g = ctx.createGain();
        swellGain(g.gain, t, 0.006, length * 0.4, 0, length * 0.6);
        src.connect(band).connect(g).connect(s.bed);
        if (Math.random() < 0.4) {
            const o = ctx.createOscillator();
            o.type = 'triangle';
            const at = t + 0.3 + Math.random();
            o.frequency.setValueAtTime(180, at);
            o.frequency.linearRampToValueAtTime(260, at + 0.8);
            const og = ctx.createGain();
            swellGain(og.gain, at, 0.003, 0.15, 0.5, 0.25);
            o.connect(makeFilter(ctx, 'lowpass', 800, 0.5)).connect(og).connect(s.bed);
            o.start(at);
            o.stop(at + 1);
        }
    }

    // A tone with a soft start and a ringing fade, the voice of the station's consoles.
    tone(s, f, t, length, level, dry = 0.7, wet = 0.4, type = 'sine') {
        const ctx = this.ctx;
        const o = ctx.createOscillator();
        o.type = type;
        o.frequency.value = f;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(Math.max(level, 0.00011), t + 0.008);
        g.gain.exponentialRampToValueAtTime(0.0001, t + length);
        o.connect(g);
        s.send(g, dry, wet);
        o.start(t);
        o.stop(t + length + 0.02);
    }

    beep(s) {
        if (this.arenaFile(s, 'space-beep', 0.7, 0.4, { pan: Math.random() * 1.2 - 0.6 })) {
            return;
        }
        const t = this.ctx.currentTime + 0.05;
        const tunes = [[91], [88, 93], [93, 88], [91, 91], [86, 91, 95]];
        const tune = tunes[Math.floor(Math.random() * tunes.length)];
        tune.forEach((n, i) => this.tone(s, midi(n), t + i * 0.13, 0.12, 0.006));
    }

    // A radio chirp, a breath of static and a quick rising tone, like a call from another deck.
    radio(s) {
        if (this.arenaFile(s, 'space-radio', 0.6, 0.5, { pan: Math.random() * 1.2 - 0.6 })) {
            return;
        }
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05;
        const src = noiseShot(ctx, this.white, t, t + 0.2);
        const g = ctx.createGain();
        swellGain(g.gain, t, 0.004, 0.02, 0.08, 0.06);
        src.connect(makeFilter(ctx, 'bandpass', 1800, 0.8)).connect(g);
        s.send(g, 0.7, 0.2);
        const count = Math.random() < 0.5 ? 1 : 2;
        for (let i = 0; i < count; i++) {
            const at = t + 0.18 + i * 0.16;
            const o = ctx.createOscillator();
            o.frequency.setValueAtTime(1400, at);
            o.frequency.exponentialRampToValueAtTime(2500, at + 0.06);
            o.frequency.exponentialRampToValueAtTime(1800, at + 0.12);
            const og = ctx.createGain();
            swellGain(og.gain, at, 0.004, 0.01, 0.08, 0.04);
            o.connect(og);
            s.send(og, 0.7, 0.3);
            o.start(at);
            o.stop(at + 0.15);
        }
    }

    ping(s, k) {
        this.tone(s, midi(86), this.ctx.currentTime + 0.03, 0.35, 0.002 + 0.004 * k, 0.6, 0.6);
    }

    // Soft bells, each note a few sine partials that ring and fade.
    chime(s, notes, gap, level) {
        const t0 = this.ctx.currentTime + 0.05;
        notes.forEach((n, i) => {
            const t = t0 + i * gap;
            const last = i === notes.length - 1;
            const ring = last ? 2.2 : 1.1;
            this.tone(s, midi(n), t, ring, level, 0.7, 0.5);
            this.tone(s, midi(n) * 2.0, t, ring * 0.6, level * 0.3, 0.7, 0.5);
            this.tone(s, midi(n) * 3.0, t, ring * 0.35, level * 0.08, 0.7, 0.5);
        });
    }

    // The neon arcade. The buzz of the tubes, a low electric hum with a fizz that flickers, now
    // and then a far arcade blip, a crowd on the grid, and arcade stings on goals.
    buildNeon(s) {
        const ctx = this.ctx;
        s.bed = makeGain(ctx, 1);
        s.bedFile = 'hum';
        s.send(s.bed, 1, 0.3);
        const hum = makeGain(ctx, 0.0035);
        const humLow = makeFilter(ctx, 'lowpass', 700, 0.7);
        humLow.connect(hum).connect(s.bed);
        for (const [f, level, type] of [[60, 1, 'sine'], [120, 0.5, 'sawtooth'], [180, 0.15, 'sawtooth']]) {
            const o = s.keep(ctx.createOscillator());
            o.type = type;
            o.frequency.value = f;
            o.connect(makeGain(ctx, level)).connect(humLow);
            o.start();
        }
        const fizz = makeGain(ctx, 0.001);
        s.loop(this.white, 1).connect(makeFilter(ctx, 'bandpass', 5200, 0.8)).connect(fizz).connect(s.bed);
        s.every(() => {
            const t = ctx.currentTime;
            fizz.gain.setTargetAtTime(0.0005 + Math.random() * 0.0015, t, 0.15);
        }, () => 300 + Math.random() * 1500, 0);
        // The stings go through an echo on the dotted eighth, darkening as it repeats.
        const fx = makeGain(ctx, 1);
        s.send(fx, 1, 0.35);
        const delay = ctx.createDelay(1);
        delay.delayTime.value = 0.47;
        const feedback = makeGain(ctx, 0.3);
        const delayLow = makeFilter(ctx, 'lowpass', 2400, 0.5);
        delay.connect(delayLow).connect(feedback).connect(delay);
        delayLow.connect(makeGain(ctx, 0.5)).connect(fx);
        s.fx = fx;
        s.echo = delay;
        // A far arcade cabinet blips now and then.
        s.every(() => {
            const t = ctx.currentTime + 0.05;
            const base = 72 + Math.floor(Math.random() * 12);
            const count = 2 + Math.floor(Math.random() * 3);
            for (let i = 0; i < count; i++) {
                this.tone(s, midi(base + (Math.random() < 0.5 ? 0 : 7) + i * 2), t + i * 0.09, 0.07, 0.0025, 0.4, 0.5, 'square');
            }
        }, () => 9000 + Math.random() * 16000, 5000);
        this.arenaCrowd(s, 'neon');
        s.intensity = (x) => s.crowdIntensity(x);
        const cheer = s.goal;
        s.goal = (own) => {
            if (!this.arenaFile(s, 'neon-sting', 1, 0.35, { gain: own ? 0.7 : 1 })) {
                this.sting(s, [72, 76, 79, 84, 88, 91, 96], 0.06, own ? 0.01 : 0.014);
                this.padSwell(s, [72, 76, 79, 84], 0.5, 2.6, own ? 0.012 : 0.016);
            }
            cheer(own);
        };
        s.fanfare = () => {
            this.sting(s, [67, 72, 76, 79], 0.14, 0.012, 0);
            this.sting(s, [76, 79, 84], 0.14, 0.012, 0.7);
            this.padSwell(s, [60, 67, 72, 76], 1.2, 3.2, 0.016);
            cheer(false);
            s.later(() => this.audible && s.applause(), 900);
        };
    }

    // A quick arcade run of notes, square and a softer octave above, through the echo.
    sting(s, notes, gap, level, at = 0) {
        const ctx = this.ctx;
        const t0 = ctx.currentTime + 0.05 + at;
        const low = makeFilter(ctx, 'lowpass', 3200, 0.6);
        low.connect(s.fx);
        low.connect(makeGain(ctx, 0.4)).connect(s.echo);
        notes.forEach((n, i) => {
            const t = t0 + i * gap;
            const last = i === notes.length - 1;
            const length = last ? 0.5 : gap * 1.6;
            for (const [type, mul, k] of [['square', 1, 1], ['triangle', 2, 0.4]]) {
                const o = ctx.createOscillator();
                o.type = type;
                o.frequency.value = midi(n) * mul;
                const g = ctx.createGain();
                g.gain.setValueAtTime(0.0001, t);
                g.gain.exponentialRampToValueAtTime(level * k, t + 0.005);
                g.gain.setValueAtTime(level * k, t + length * 0.5);
                g.gain.exponentialRampToValueAtTime(0.0001, t + length);
                o.connect(g).connect(low);
                o.start(t);
                o.stop(t + length + 0.02);
            }
        });
    }

    // A bright chord that swells and settles, under a sting.
    padSwell(s, notes, at, length, level) {
        const ctx = this.ctx;
        const t = ctx.currentTime + 0.05 + at;
        const low = makeFilter(ctx, 'lowpass', 800, 1);
        low.frequency.setValueAtTime(800, t);
        low.frequency.linearRampToValueAtTime(2800, t + length * 0.3);
        low.frequency.linearRampToValueAtTime(1000, t + length);
        const g = ctx.createGain();
        swellGain(g.gain, t, level, length * 0.25, length * 0.2, length * 0.55);
        low.connect(g).connect(s.fx);
        for (const n of notes) {
            for (const detune of [-8, 8]) {
                const o = ctx.createOscillator();
                o.type = 'sawtooth';
                o.frequency.value = midi(n);
                o.detune.value = detune;
                o.connect(low);
                o.start(t);
                o.stop(t + length + 0.05);
            }
        }
    }

    // Rain and wind follow the weather, from files when present and synthesized otherwise. The
    // station in orbit has no weather.
    updateWeather() {
        if (!this.ctx) {
            return;
        }
        for (const kind of ['rain', 'wind']) {
            const wanted = this.arena === 'space' ? 0 : Math.max(0, Math.min(1, this.ambienceWanted.get(kind) || 0));
            const file = this.files.has(`${kind}-loop`);
            this.fileLoop(kind, file ? wanted : 0);
            const level = file ? 0 : wanted;
            let w = this.weather[kind];
            if (level > 0 && !w) {
                w = this.weather[kind] = kind === 'rain' ? this.buildRain() : this.buildWind();
            }
            if (!w) {
                continue;
            }
            w.setLevel(level * WEATHER_GAIN[kind]);
            if (w.setStrength) {
                w.setStrength(level);
            }
            if (w.setSurface) {
                w.setSurface(this.arena);
            }
            if (level > 0) {
                clearTimeout(w.retireTimer);
            } else {
                this.retire(w, () => w.target > 0, () => {
                    if (this.weather[kind] === w) {
                        this.weather[kind] = null;
                    }
                });
            }
        }
    }

    // Rain, a soft hiss, the patter of drops and on the stadium roof the drumming of heavier
    // ones, balanced for the ground of each arena.
    buildRain() {
        const ctx = this.ctx;
        const w = new Scape(this, 'rain', true);
        if (!this.drops) {
            this.drops = dropsBuffer(ctx);
        }
        const hiss = makeGain(ctx, 0);
        w.loop(this.pink, 1).connect(makeFilter(ctx, 'highpass', 450, 0.5)).connect(makeFilter(ctx, 'lowpass', 3600, 0.5)).connect(hiss);
        w.send(hiss, 1, 0.1);
        const patter = makeGain(ctx, 0);
        w.loop(this.drops, 1).connect(makeFilter(ctx, 'lowpass', 6000, 0.5)).connect(patter);
        w.send(patter, 1, 0.12);
        const drum = makeGain(ctx, 0);
        w.loop(this.drops, 0.45).connect(makeFilter(ctx, 'lowpass', 1300, 0.6)).connect(drum);
        w.send(drum, 1, 0.3);
        w.setSurface = (arena) => {
            if (arena === w.arena) {
                return;
            }
            w.arena = arena;
            const [h, p, d] = RAIN_SURFACE[arena] || RAIN_SURFACE.stadium;
            const t = ctx.currentTime;
            hiss.gain.setTargetAtTime(0.032 * h, t, FADE);
            patter.gain.setTargetAtTime(0.025 * p, t, FADE);
            drum.gain.setTargetAtTime(0.05 * d, t, FADE);
        };
        return w;
    }

    // Wind, two bands of air that gust and shift apart, and a low rumble when it blows hard.
    buildWind() {
        const ctx = this.ctx;
        const w = new Scape(this, 'wind', true);
        const layers = [];
        const base = makeGain(ctx, 0.04);
        base.connect(w.fade);
        for (const [f, q, side] of [[320, 0.9, -0.5], [620, 1.3, 0.5]]) {
            const band = makeFilter(ctx, 'bandpass', f, q);
            const g = makeGain(ctx, 0.5);
            w.loop(this.pink, 0.8 + Math.random() * 0.3).connect(band).connect(g).connect(makePan(ctx, side)).connect(base);
            layers.push({ band, g, f });
        }
        const rumble = makeGain(ctx, 0);
        w.loop(this.brown, 0.6).connect(makeFilter(ctx, 'lowpass', 140, 0.5)).connect(rumble).connect(w.fade);
        w.strength = 0;
        w.setStrength = (k) => {
            w.strength = k;
            rumble.gain.setTargetAtTime(0.02 * k * k, ctx.currentTime, FADE);
        };
        w.every(() => {
            const t = ctx.currentTime;
            const k = w.strength;
            for (const { band, g, f } of layers) {
                g.gain.setTargetAtTime(0.35 + Math.random() * (0.4 + 0.5 * k), t, 0.6 + Math.random() * 1.2);
                band.frequency.setTargetAtTime(f * (0.7 + Math.random() * 0.7) * (1 + 0.3 * k), t, 1);
            }
        }, () => 1200 + Math.random() * 2800, 0);
        return w;
    }

    // An ambience file played in a loop, rain, wind, waves, city or hum, faded to a level.
    fileLoop(name, level) {
        const buffer = this.files.get(`${name}-loop`);
        let entry = this.ambience.get(name);
        if (!buffer || (!entry && level <= 0)) {
            return;
        }
        if (!entry) {
            const src = this.ctx.createBufferSource();
            src.buffer = buffer;
            src.loop = true;
            const g = this.ctx.createGain();
            g.gain.value = 0;
            src.connect(g);
            this.route(g, 1, 0.1);
            src.start(0, Math.random() * buffer.duration);
            entry = { src, gain: g };
            this.ambience.set(name, entry);
        }
        entry.gain.gain.setTargetAtTime(Math.max(0, Math.min(1, level)) * 0.35, this.ctx.currentTime, FADE);
    }

    // A panner at a point of three's frame, sounds there fall off with distance from the camera.
    panner(position, ref = 12) {
        const panner = this.ctx.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = ref;
        panner.rolloffFactor = 0.8;
        if (panner.positionX) {
            panner.positionX.value = position.x;
            panner.positionY.value = position.y;
            panner.positionZ.value = position.z;
        } else {
            panner.setPosition(position.x, position.y, position.z);
        }
        return panner;
    }

    // Plays a decoded buffer once through an optional panner, at a gain and rate.
    playBuffer(buffer, { gain = 1, rate = 1, position = null, wet = 0.15, when = 0 } = {}) {
        const ctx = this.ctx;
        const src = ctx.createBufferSource();
        src.buffer = buffer;
        src.playbackRate.value = rate;
        const g = ctx.createGain();
        g.gain.value = gain;
        src.connect(g);
        if (position) {
            g.connect(this.panner(position)).connect(this.master);
            if (wet > 0) {
                const w = ctx.createGain();
                w.gain.value = wet;
                g.connect(w).connect(this.reverb);
            }
        } else {
            this.route(g, 1, wet);
        }
        src.start(ctx.currentTime + when);
        return src;
    }

    // A pea whistle, two close tones trilled by the pea at about 30 Hz, or the whistle files.
    whistle(blasts = [0.35]) {
        if (!this.audible) {
            return;
        }
        const ctx = this.ctx;
        let t = ctx.currentTime + 0.02;
        for (const length of blasts) {
            const file = this.files.get(length >= 0.6 ? 'whistle-long' : 'whistle-short') || null;
            if (file) {
                this.playBuffer(file, { gain: 0.5, when: t - ctx.currentTime });
                t += Math.max(length, file.duration) + 0.12;
                continue;
            }
            const out = ctx.createGain();
            out.gain.setValueAtTime(0, t);
            out.gain.linearRampToValueAtTime(0.22, t + 0.02);
            out.gain.setValueAtTime(0.22, t + length - 0.05);
            out.gain.linearRampToValueAtTime(0, t + length);
            const filter = ctx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.value = 2900;
            filter.Q.value = 3;
            out.connect(filter);
            this.route(filter, 1, 0.2);
            const trill = ctx.createOscillator();
            trill.frequency.value = 28 + Math.random() * 6;
            const depth = ctx.createGain();
            depth.gain.value = 140;
            trill.connect(depth);
            for (const f of [2850, 3010]) {
                const o = ctx.createOscillator();
                o.type = 'triangle';
                o.frequency.value = f;
                depth.connect(o.frequency);
                o.connect(out);
                o.start(t);
                o.stop(t + length + 0.02);
            }
            trill.start(t);
            trill.stop(t + length + 0.02);
            t += length + 0.18;
        }
    }

    // The arena's answer to a goal, with the scoring team's own goal sound when it has one. In
    // the stadium a swelling roar over the bed, a few seconds long.
    goal(team = -1) {
        if (!this.audible) {
            return;
        }
        const own = team >= 0 && this.teamSound(team, 'goal');
        if (this.arena !== 'stadium') {
            const s = this.scapes.get(this.arena);
            if (s && s.goal) {
                s.goal(own);
            }
            return;
        }
        const stadium = this.scapes.get('stadium');
        if (this.files.has('crowd-cheer')) {
            const g = this.ctx.createGain();
            stadium.send(g, 1, 0.3);
            const src = this.ctx.createBufferSource();
            src.buffer = this.files.get('crowd-cheer');
            src.connect(makeGain(this.ctx, own ? 0.35 : 0.55)).connect(g);
            src.start();
            return;
        }
        const ctx = this.ctx;
        const t = ctx.currentTime;
        const src = ctx.createBufferSource();
        src.buffer = this.roar.src.buffer || this.pink;
        src.loop = true;
        src.playbackRate.value = 1.18;
        // A dark swell, low passed and well under the effects, so the roar never turns to hiss.
        const filter = ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(500, t);
        filter.frequency.linearRampToValueAtTime(1300, t + 0.8);
        filter.frequency.linearRampToValueAtTime(700, t + 6);
        filter.Q.value = 0.5;
        const highpass = ctx.createBiquadFilter();
        highpass.type = 'highpass';
        highpass.frequency.value = 140;
        const g = ctx.createGain();
        const peak = own ? 0.1 : 0.16;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(peak, t + 0.6);
        g.gain.setValueAtTime(peak, t + 2.5);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 7);
        src.connect(highpass).connect(filter).connect(g);
        stadium.send(g, 0.8, 0.6);
        src.start(t, Math.random() * 2);
        src.stop(t + 7.2);
        // Shouts and applause break out of the roar.
        for (let i = 0; i < 6; i++) {
            this.shout(t + 0.2 + Math.random() * 2.5, 1.6);
        }
        setTimeout(() => this.audible && this.arena === 'stadium' && this.claps(40, 1.2), 1500);
    }

    // A groan at a near miss, the stadium crowd's from a file when one is present, else a
    // falling murmur, or the arena's own.
    ooh() {
        if (!this.audible) {
            return;
        }
        // Never twice within a few seconds, a save and the ball running wide are one moment.
        const now = this.ctx.currentTime;
        if (now - this.lastOoh < 4) {
            return;
        }
        this.lastOoh = now;
        if (this.arena !== 'stadium') {
            const s = this.scapes.get(this.arena);
            if (s && s.ooh) {
                s.ooh();
            }
            return;
        }
        if (this.files.has('crowd-ooh')) {
            this.playBuffer(this.files.get('crowd-ooh'), { gain: 0.45, wet: 0.3 });
            return;
        }
        const t = this.ctx.currentTime;
        for (let i = 0; i < 5; i++) {
            this.shout(t + Math.random() * 0.2, 1.2);
        }
    }

    // Applause, the file when present, else a long ripple of claps, or the arena's own.
    applause() {
        if (!this.audible) {
            return;
        }
        if (this.arena !== 'stadium') {
            const s = this.scapes.get(this.arena);
            if (s && s.applause) {
                s.applause();
            }
            return;
        }
        if (this.files.has('applause')) {
            this.playBuffer(this.files.get('applause'), { gain: 0.45, wet: 0.3 });
            return;
        }
        this.claps(60, 1.2);
        setTimeout(() => this.audible && this.arena === 'stadium' && this.claps(50, 1), 900);
    }

    // A short fanfare like brass for the winners, detuned saws through a swelling low pass, and
    // the crowd cheering along. The space station and the neon arcade play their own.
    fanfare() {
        if (!this.audible) {
            return;
        }
        const own = this.scapes.get(this.arena);
        if (own && own.fanfare) {
            own.fanfare();
            return;
        }
        const ctx = this.ctx;
        const t0 = ctx.currentTime + 0.05;
        const notes = [
            [392, 0, 0.16], [392, 0.18, 0.16], [392, 0.36, 0.16], [523.3, 0.54, 0.7],
            [466.2, 1.3, 0.45], [523.3, 1.8, 0.22], [587.3, 2.05, 0.22], [659.3, 2.3, 1.6],
        ];
        const chord = [[261.6, 2.3, 1.6], [392, 2.3, 1.6], [523.3, 2.3, 1.6]];
        for (const [f, at, len] of [...notes, ...chord]) {
            const out = ctx.createGain();
            const start = t0 + at;
            out.gain.setValueAtTime(0.0001, start);
            out.gain.exponentialRampToValueAtTime(0.12, start + 0.04);
            out.gain.setValueAtTime(0.12, start + len * 0.7);
            out.gain.exponentialRampToValueAtTime(0.0001, start + len);
            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.Q.value = 1.2;
            filter.frequency.setValueAtTime(f * 1.5, start);
            filter.frequency.linearRampToValueAtTime(f * 5, start + 0.08);
            filter.frequency.linearRampToValueAtTime(f * 3, start + len);
            this.route(out, 1, 0.25);
            filter.connect(out);
            for (const detune of [-7, 0, 7]) {
                const o = ctx.createOscillator();
                o.type = 'sawtooth';
                o.frequency.value = f;
                o.detune.value = detune;
                o.connect(filter);
                o.start(start);
                o.stop(start + len + 0.05);
            }
        }
        // The crowd cheers along.
        this.goal();
        this.applause();
    }

    // The two teams of the match, registry entries with an id and their list of sounds, or
    // null for a team with none. Their sounds load once the audio is unlocked.
    setTeams(teams) {
        const key = (t) => (t && t.id ? `${t.id}:${(t.sounds || []).join(',')}:${t.soundsVersion || ''}` : '');
        const same = this.teams.every((t, i) => key(t) === key(teams[i]));
        this.teams = [teams[0] || null, teams[1] || null];
        if (!same) {
            this.teamBuffers = [new Map(), new Map()];
            this.loadTeamSounds();
        }
    }

    async loadTeamSounds() {
        if (!this.ctx) {
            return;
        }
        const teams = this.teams;
        await Promise.all(teams.map(async (team, side) => {
            if (!team || !team.id || !Array.isArray(team.sounds)) {
                return;
            }
            const buffers = this.teamBuffers[side];
            await Promise.all(team.sounds.map(async (slot) => {
                try {
                    const res = await fetch(`/api/teams/${encodeURIComponent(team.id)}/sounds/${encodeURIComponent(slot)}`, { cache: 'no-store' });
                    if (!res.ok) {
                        return;
                    }
                    const decoded = await this.ctx.decodeAudioData(await res.arrayBuffer());
                    if (this.teams[side] === team) {
                        buffers.set(slot, tameBuffer(this.ctx, decoded));
                    }
                } catch (err) {
                    console.warn('team sound not used', team.id, slot, err);
                }
            }));
        }));
    }

    // Plays a team's own sound for one of its slots, from a point of three's frame when given,
    // and returns whether the team had one, so the caller can skip its built-in sound. Chants come
    // from the team's end of the ground, where its fans stand, and only where there are fans.
    teamSound(team, slot, position = null, fans = false) {
        if (!this.audible || (team !== 0 && team !== 1)) {
            return false;
        }
        const buffer = this.teamBuffers[team].get(slot);
        if (!buffer) {
            return false;
        }
        const crowd = fans ? FAN_CHANT[this.arena] || 0 : 1;
        if (crowd <= 0) {
            return false;
        }
        const now = this.ctx.currentTime;
        const last = this.lastPlayed[team].get(slot) || -Infinity;
        if (now - last < (TEAM_SLOT_GAP[slot] || 0.5)) {
            return true;
        }
        this.lastPlayed[team].set(slot, now);
        const where = position || (fans ? { x: team === 0 ? -62 : 62, y: 12, z: 0 } : null);
        const wet = fans ? 0.6 * (this.arena === 'stadium' ? 1 : 0.5) : 0.2;
        this.playBuffer(buffer, { gain: (TEAM_SLOT_GAIN[slot] || 0.4) * crowd, position: where, wet });
        return true;
    }
}
