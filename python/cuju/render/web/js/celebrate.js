// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Goal celebrations on the pitch, and the dances of the show after the match. The scorer runs
// off and performs his celebration, teammates run to him and huddle or join in, the other team
// walks back. Poses come from the animation library of animate.js when it is there, else from
// simple procedural poses, and are written into the pose buffer in the engine frame.

import * as THREE from 'three';
import { NODE_COUNT, PLAYER_FLOATS, NODES_FLOAT } from './format.js';
import { HALF_H, HALF_W } from './pitch.js';
import { restPoseNodes, walkPoseNodes } from './players.js';
import { KAYKIT_CELEBRATIONS, KAYKIT_ROLES } from './kaykit.js';

// The library loads on its own, a page without it still celebrates with the simple poses.
let lib = null;
import('./animate.js').then((m) => {
    lib = m;
}).catch(() => {});

export function animationLibrary() {
    return lib;
}

// Names of the library's animations, from a list or a map of them.
function libraryNames() {
    const all = lib && lib.ANIMATIONS;
    if (!all) {
        return [];
    }
    return Array.isArray(all) ? all.map((a) => (typeof a === 'string' ? a : a.name)).filter(Boolean) : Object.keys(all);
}

function libraryEntry(name) {
    const all = lib && lib.ANIMATIONS;
    if (!all) {
        return null;
    }
    if (Array.isArray(all)) {
        return all.find((a) => a === name || (a && a.name === name)) || null;
    }
    return all[name] || null;
}

// Utility animations by role, the first name the library has wins.
const ROLES = {
    run: ['run', 'jog', 'sprint'],
    walk: ['walk', 'stroll', 'walk_back'],
    idle: ['idle', 'stand', 'breathe'],
    clap: ['clap', 'applause', 'applaud'],
    dejected: ['dejected', 'hands_on_head', 'sad', 'disappointed'],
    huddle: ['team_huddle', 'huddle', 'hug', 'embrace'],
};
const UTILITY = new Set(Object.values(ROLES).flat());

export function roleName(role) {
    const names = libraryNames();
    return (ROLES[role] || []).find((n) => names.includes(n)) || null;
}

// The library's celebrations, the entries marked as dances or celebrations, else every entry
// that is not a utility animation.
export function danceNames() {
    const names = libraryNames();
    const marked = names.filter((n) => {
        const e = libraryEntry(n);
        if (!e || typeof e !== 'object') {
            return false;
        }
        const kind = e.kind || e.type || e.category || e.group;
        return kind === 'celebration' || kind === 'dance' || kind === 'celebrations' || e.celebration === true || e.dance === true;
    });
    // Clapping and the huddle belong to the teammates, not to a scorer's solo.
    return (marked.length ? marked : names).filter((n) => !UTILITY.has(n));
}

// The library's view of an animation, its kind, length and whether it loops.
function libraryInfo(name) {
    const e = libraryEntry(name);
    if (!e || typeof e !== 'object') {
        return { kind: '', duration: 0, loop: true, speed: 0 };
    }
    return { kind: e.kind || '', duration: e.duration || 0, loop: e.loop !== false, speed: e.speed || 0 };
}

// The ground speed a gait of the library was made for, so a faster mover steps faster.
export function gaitSpeed(name) {
    return libraryInfo(name).speed;
}

// A small deterministic random source, so a goal celebrates the same way every time it plays.
export function seeded(seed) {
    let a = (seed * 2654435761) >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// The celebration a player performs, from his roster entry. Team picks from the team's list,
// an empty list or entry picks from every dance. A clip entry names an animation of the model.
export function celebrationFor(style, index, random) {
    const roster = style && style.players ? style.players[index] : null;
    let choice = roster && typeof roster.celebration === 'string' ? roster.celebration : 'team';
    const dances = danceNames();
    if (choice === 'team') {
        const list = style && Array.isArray(style.celebrations) ? style.celebrations : [];
        choice = list.length ? list[Math.floor(random() * list.length)] : '';
    }
    if (choice && (choice.startsWith('clip:') || dances.includes(choice))) {
        return choice;
    }
    return dances.length ? dances[Math.floor(random() * dances.length)] : 'dance';
}

// Whether a team or its player chose the celebrations, rather than leaving them to chance.
export function choseCelebration(style, index) {
    const roster = style && style.players ? style.players[index] : null;
    if (roster && typeof roster.celebration === 'string' && roster.celebration && roster.celebration !== 'team') {
        return true;
    }
    return !!(style && Array.isArray(style.celebrations) && style.celebrations.length);
}

// A KayKit celebration that player k's model plays, picked by random, or null.
export function kaykitCelebration(hasClip, k, random) {
    if (!hasClip) {
        return null;
    }
    const list = KAYKIT_CELEBRATIONS.filter((n) => hasClip(k, n));
    return list.length ? list[Math.floor(random() * list.length)] : null;
}

// The KayKit role that stands in for each name of the library on a model that plays clips.
const CLIP_ROLES = {
    idle: 'idle', idle_confident: 'calm', idle_bounce: 'calm', team_huddle: 'huddle', clap: 'wave',
    arms_up: 'cheer', jump_fist: 'jump', dance: 'celebrate', hands_on_head: 'crouch', point: 'handshake',
};
// Roles whose clips vary from player to player, so a crowd does not move as one.
const VARIED = new Set(['idle', 'celebrate', 'jump', 'cheer']);

// The clip player k of a choreography plays for a still pose, a clip name that his model has,
// the KayKit clip of the library name's role, or null for the library's own pose.
export function stillClip(group, k, still) {
    if (!still || !group.hasClip) {
        return null;
    }
    const cache = group.clipCache || (group.clipCache = new Map());
    const key = `${k}:${still}`;
    if (cache.has(key)) {
        return cache.get(key);
    }
    let out = null;
    try {
        if (still.startsWith('clip:')) {
            out = group.hasClip(k, still) ? still : null;
        } else {
            const role = CLIP_ROLES[still];
            const list = role ? (KAYKIT_ROLES[role] || []).filter((n) => group.hasClip(k, n)) : [];
            if (list.length) {
                out = VARIED.has(role) ? list[k % list.length] : list[0];
            }
        }
    } catch {
        out = null;
    }
    cache.set(key, out);
    return out;
}

// The flavor of the show after the match, the player's own celebration, else his team's first,
// else any dance.
export function showDance(style, index, random = Math.random) {
    const roster = style && style.players ? style.players[index] : null;
    const own = roster && roster.celebration;
    if (own && own !== 'team') {
        return celebrationFor(style, index, random);
    }
    const list = style && Array.isArray(style.celebrations) ? style.celebrations : [];
    if (list.length) {
        return list[0];
    }
    // Else a dance that keeps going, for a show that lasts.
    const loops = danceNames().filter((n) => libraryInfo(n).loop && libraryInfo(n).duration);
    if (loops.length) {
        return loops[Math.floor(random() * loops.length)];
    }
    return celebrationFor(style, index, random);
}

// The heading that faces engine direction dx, dy, as restPoseNodes turns a player.
export function headingTo(dx, dy) {
    return Math.atan2(dx, -dy);
}

const scratch = new Float32Array(PLAYER_FLOATS);
const scratchNext = new Float32Array(PLAYER_FLOATS);

function shift(out, o, x, y, z) {
    for (let n = 0; n < NODE_COUNT; n++) {
        out[o + n * 7] += x;
        out[o + n * 7 + 1] += y;
        out[o + n * 7 + 2] += z;
    }
}

// Writes one player's pose for animation name at time t into out at offset o, standing at x, y
// and turned to heading. Names of the library go through animate.pose, a clip of the model
// stands idle here while the model plays it, and the rest fall back to simple poses.
export function posePlayer(name, t, out, o, x, y, heading, mode = '') {
    if (lib && typeof lib.pose === 'function' && name && libraryEntry(name)) {
        try {
            const info = libraryInfo(name);
            const opts = { x, y, heading };
            const repeat = mode === 'show';
            if (info.loop || !info.duration) {
                lib.pose(name, t, scratch, opts);
            } else {
                // A celebration that plays once goes on into a loop of arms up where it ended,
                // or starts over after a pause for the show, other animations hold their last
                // pose.
                const end = info.duration;
                let local = t;
                let next = null;
                if (repeat) {
                    local = t % (end + 1.4);
                    next = roleName('idle');
                } else if (info.kind === 'celebration') {
                    next = ['arms_up', 'jump_fist'].find((n) => libraryEntry(n)) || null;
                }
                lib.pose(name, Math.min(local, end), scratch, opts);
                if (local > end + 0.5 && next) {
                    const u = local - end - 0.5;
                    lib.pose(next, u, scratchNext, { x: scratch[0], y: scratch[1], heading });
                    blendInto(scratch, 0, scratchNext, 0, Math.min(1, u / 0.45));
                }
            }
            // On a podium, a stage or beside the scorer, a dance that travels stays on its spot.
            if (mode === 'show' || mode === 'pin') {
                shift(scratch, 0, x - scratch[0], y - scratch[1], 0);
            }
            out.set(scratch, o);
            return;
        } catch {
            // A failing animation falls back to the simple poses below.
        }
    }
    const kind = (name && Object.keys(ROLES).find((r) => ROLES[r].includes(name))) || name;
    if (kind === 'run') {
        walkPoseNodes(t * Math.PI * 2 * 1.5, 6, heading, scratch);
        out.set(scratch, o);
        shift(out, o, x, y, 0);
    } else if (kind === 'walk') {
        walkPoseNodes(t * Math.PI * 2 * 0.9, 1.6, heading, scratch);
        out.set(scratch, o);
        shift(out, o, x, y, 0);
    } else if (kind === 'idle' || kind === 'clap' || kind === 'dejected' || kind === 'huddle' || (name && name.startsWith('clip:'))) {
        restPoseNodes(heading, scratch);
        out.set(scratch, o);
        shift(out, o, x, y, Math.sin(t * 2.2) * 0.008);
    } else {
        // A bouncing step on the spot, turning from side to side.
        walkPoseNodes(t * Math.PI * 2 * 1.8, 2.5, heading + Math.sin(t * 2.4) * 0.7, scratch);
        out.set(scratch, o);
        shift(out, o, x, y, Math.abs(Math.sin(t * Math.PI * 1.8)) * 0.12);
    }
}

// Blends one player's pose b into a by weight w, positions linearly and rotations along the
// shorter arc, normalized.
export function blendInto(a, ao, b, bo, w) {
    if (w <= 0) {
        return;
    }
    if (w >= 1) {
        for (let i = 0; i < PLAYER_FLOATS; i++) {
            a[ao + i] = b[bo + i];
        }
        return;
    }
    for (let n = 0; n < NODE_COUNT; n++) {
        const p = ao + n * 7;
        const q = bo + n * 7;
        a[p] += (b[q] - a[p]) * w;
        a[p + 1] += (b[q + 1] - a[p + 1]) * w;
        a[p + 2] += (b[q + 2] - a[p + 2]) * w;
        const ax = a[p + 3];
        const ay = a[p + 4];
        const az = a[p + 5];
        const aw = a[p + 6];
        let bx = b[q + 3];
        let by = b[q + 4];
        let bz = b[q + 5];
        let bw = b[q + 6];
        if (ax === 0 && ay === 0 && az === 0 && aw === 0) {
            a[p + 3] = bx;
            a[p + 4] = by;
            a[p + 5] = bz;
            a[p + 6] = bw;
            continue;
        }
        if (ax * bx + ay * by + az * bz + aw * bw < 0) {
            bx = -bx;
            by = -by;
            bz = -bz;
            bw = -bw;
        }
        const x = ax + (bx - ax) * w;
        const y = ay + (by - ay) * w;
        const z = az + (bz - az) * w;
        const s = aw + (bw - aw) * w;
        const len = Math.hypot(x, y, z, s) || 1;
        a[p + 3] = x / len;
        a[p + 4] = y / len;
        a[p + 5] = z / len;
        a[p + 6] = s / len;
    }
}

const smooth = (u) => {
    const s = Math.min(1, Math.max(0, u));
    return s * s * (3 - 2 * s);
};

// The ball after the last frame before a goal, flying on from its last velocity into the net,
// slowing in the netting and dropping to the grass. t is seconds after frame b.
export function ballIntoNet(a, b, team, sps, t, out) {
    const goalX = team === 0 ? HALF_W : -HALF_W;
    const dir = Math.sign(goalX);
    let vx = (b[0] - a[0]) * sps;
    let vy = (b[1] - a[1]) * sps;
    let vz = (b[2] - a[2]) * sps;
    // A ball that the frames show slow or going the other way still finishes in the net.
    if (vx * dir < 4) {
        vx = 8 * dir;
    }
    const stopX = goalX + dir * 1.6;
    const tStop = Math.max(0.05, (stopX - b[0]) / vx);
    const s = Math.min(t, tStop);
    // Linear speed until the net, a quick stop over 0.4 seconds after.
    let x = b[0] + vx * s;
    let y = b[1] + vy * s;
    let z = b[2] + vz * s - 4.9 * s * s;
    if (t > tStop) {
        // The netting gives a little and holds the ball.
        const u = Math.min(1, (t - tStop) / 0.4);
        x += dir * 0.35 * (1 - (1 - u) * (1 - u));
        y += Math.sign(vy) * Math.min(0.3, Math.abs(vy) * 0.03) * u;
        vz = 0;
    }
    // Inside the goal, never through its back or roof.
    const back = Math.abs(goalX) + 2.3;
    if (Math.abs(x) > back && Math.abs(b[0]) <= back) {
        x = dir * back;
    }
    const zStop = Math.max(0.11, b[2] + vz * tStop - 4.9 * tStop * tStop);
    if (t > tStop) {
        const fall = Math.min(1, (t - tStop) / 0.45);
        z = zStop + (0.11 - zStop) * fall * fall;
    }
    out[0] = x;
    out[1] = Math.max(-3.4, Math.min(3.4, y));
    out[2] = Math.max(0.11, Math.abs(x) > Math.abs(goalX) ? Math.min(2.3, z) : z);
}

// One goal's celebration, a pure function of the time since the frame before the goal. base is
// the frame's data, its players' nodes start at NODES_FLOAT.
export class Celebration {
    constructor({ frame, before, header, styles, team, scorer, sps, seed, hasClip = () => false, radius = null }) {
        this.hasClip = hasClip;
        this.radii = header.players.map((p, k) => (radius ? radius(k) : DEFAULT_RADIUS));
        this.clipped = new Set();
        this.header = header;
        this.team = team;
        this.scorer = scorer;
        this.sps = sps;
        this.frame = frame;
        this.ballA = before.data.slice(8, 11);
        this.ballB = frame.data.slice(8, 11);
        this.base = frame.data.subarray(NODES_FLOAT);
        this.count = header.players.length;
        this.work = new Float32Array(PLAYER_FLOATS);
        const random = seeded(seed);
        const at = (k) => ({ x: this.base[k * PLAYER_FLOATS], y: this.base[k * PLAYER_FLOATS + 1] });
        const goalX = team === 0 ? HALF_W : -HALF_W;
        const dir = Math.sign(goalX);
        this.plans = [];
        // The scorer runs towards the corner of the end he attacked, ten meters or so, and
        // turns to the pitch to celebrate.
        let spot = { x: goalX - dir * 8, y: 0 };
        let dance = 'dance';
        if (scorer >= 0) {
            const p = at(scorer);
            const side = Math.sign(p.y) || (random() < 0.5 ? -1 : 1);
            const corner = { x: goalX - dir * 3, y: side * (HALF_H - 4) };
            let dx = corner.x - p.x;
            let dy = corner.y - p.y;
            const len = Math.hypot(dx, dy) || 1;
            const run = Math.min(len, 9 + random() * 4);
            dx /= len;
            dy /= len;
            spot = {
                x: Math.max(-HALF_W + 1, Math.min(HALF_W - 1, p.x + dx * run)),
                y: Math.max(-HALF_H + 1, Math.min(HALF_H - 1, p.y + dy * run)),
            };
            const pl = header.players[scorer];
            const style = styles ? styles[pl.team] : null;
            dance = celebrationFor(style, pl.index, random);
            // A KayKit scorer left to chance plays one of KayKit's own celebrations.
            if (!choseCelebration(style, pl.index)) {
                dance = kaykitCelebration(hasClip, scorer, random) || dance;
            }
            // A clip the scorer's model does not carry gives way to a dance of the library.
            if (dance.startsWith('clip:') && !hasClip(scorer, dance)) {
                const dances = danceNames();
                dance = dances.length ? dances[Math.floor(random() * dances.length)] : 'dance';
            }
            const runT = Math.max(1.2, run / 5.2);
            this.plans[scorer] = {
                kind: 'scorer', from: p, to: spot, start: 0.15, runT, dance,
                heading: headingTo(-spot.x * 0.4, -spot.y),
            };
        }
        this.spot = spot;
        this.dance = dance;
        const plain = danceNames();
        this.plainDance = dance.startsWith('clip:') ? (plain[Math.floor(random() * plain.length)] || 'dance') : dance;
        // Teammates gather in a ring around the spot, all huddling, all joining the dance, or
        // a mix, chosen per goal.
        const join = random();
        const teamRadius = Math.max(DEFAULT_RADIUS, ...header.players.map((p, k) => (p.team === team ? this.radii[k] : 0)));
        const mates = header.players.map((p, k) => ({ p, k }))
            .filter((m) => m.p.team === team && m.k !== scorer && !m.p.goalkeeper)
            .sort((a, b) => Math.hypot(at(a.k).x - spot.x, at(a.k).y - spot.y) - Math.hypot(at(b.k).x - spot.x, at(b.k).y - spot.y));
        mates.forEach((m, i) => {
            const from = at(m.k);
            // Two rows behind the scorer and to his sides, so the cameras in front see him.
            const face = this.plans[scorer] ? this.plans[scorer].heading : 0;
            const behind = Math.atan2(Math.cos(face), -Math.sin(face));
            const back = i >= 5;
            const offsets = back ? [0.3, -0.3, 0.85, -0.85, 1.35, -1.35] : [0, 0.6, -0.6, 1.15, -1.15];
            const j = back ? i - 5 : i;
            // Rings wide enough for the team's biggest model, neighbours a body apart.
            const reach = 2 * teamRadius + 0.15;
            const front = Math.max(1.3, reach / (2 * Math.sin(0.3)));
            const ring = (back ? front + reach : front) + Math.floor(j / offsets.length) * reach;
            const angle = behind + offsets[j % offsets.length] + (random() - 0.5) * 0.12;
            const to = { x: spot.x + Math.cos(angle) * ring, y: spot.y + Math.sin(angle) * ring };
            const speed = 6 + random() * 1.6;
            const dist = Math.hypot(to.x - from.x, to.y - from.y);
            // Nobody beats the scorer to his spot, the nearest arrive just after him.
            const scorerPlan = this.plans[scorer];
            const arrive = scorerPlan ? scorerPlan.start + scorerPlan.runT + 0.3 + i * 0.2 : 0;
            const delay = Math.max(0.2 + random() * 0.5, arrive - dist / speed);
            let mode = 'huddle';
            if (join < 0.35 || (join < 0.7 && i % 2 === 1)) {
                mode = 'dance';
            }
            this.plans[m.k] = {
                kind: 'mate', from, to, start: delay, runT: dist / speed, speed, mode,
                heading: mode === 'dance' ? null : headingTo(spot.x - to.x, spot.y - to.y),
                offset: random() * 0.4,
            };
        });
        header.players.forEach((p, k) => {
            if (this.plans[k]) {
                return;
            }
            const from = at(k);
            if (p.team === team) {
                // The scorer's keeper takes a few steps out and applauds.
                this.plans[k] = { kind: 'keeper', from, to: { x: from.x + dir * 4, y: from.y * 0.8 }, start: 0.6, speed: 1.6, mode: 'clap' };
            } else if (p.goalkeeper) {
                this.plans[k] = { kind: 'beaten', from, to: from, start: 0, speed: 0, mode: 'dejected' };
            } else {
                // The others walk slowly back towards the halfway line for the kick-off.
                const to = { x: from.x * 0.55, y: from.y * 0.7 };
                this.plans[k] = { kind: 'back', from, to, start: 0.5 + random() * 0.8, speed: 1.3 + random() * 0.5, mode: 'idle' };
            }
        });
    }

    // Hands every model that played a clip back to the nodes.
    release(stopClip) {
        for (const k of this.clipped) {
            stopClip(k);
        }
        this.clipped.clear();
    }

    // The ground position and heading of player k at time t, and the animation he plays.
    state(k, t) {
        const plan = this.plans[k];
        if (!plan) {
            return null;
        }
        const { from, to } = plan;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const dist = Math.hypot(dx, dy);
        const runHeading = dist > 0.01 ? headingTo(dx, dy) : 0;
        if (plan.kind === 'scorer') {
            const u = Math.min(1, Math.max(0, (t - plan.start) / plan.runT));
            // Fast off the mark, slowing into the spot.
            const e = 1 - (1 - u) * (1 - u);
            const pos = { x: from.x + dx * e, y: from.y + dy * e };
            const turn = smooth((t - plan.start - plan.runT + 0.2) / 0.5);
            return {
                ...pos,
                heading: lerpAngle(runHeading, plan.heading, turn),
                moving: u < 1 ? 'run' : null,
                travelled: dist * e,
                still: plan.dance,
                mix: smooth((t - plan.start - plan.runT + 0.25) / 0.35),
                still_t: t - plan.start - plan.runT,
                anchor: true,
            };
        }
        if (plan.kind === 'mate') {
            const travelled = Math.max(0, (t - plan.start) * plan.speed);
            const f = dist > 0 ? Math.min(1, travelled / dist) : 1;
            const pos = { x: from.x + dx * f, y: from.y + dy * f };
            const arrive = plan.start + plan.runT;
            const scorerHeading = this.plans[this.scorer] ? this.plans[this.scorer].heading : runHeading;
            const copy = this.dance.startsWith('clip:') && !this.hasClip(k, this.dance) ? this.plainDance : this.dance;
            const still = plan.mode === 'dance' ? copy : (roleName('huddle') || roleName('clap') || 'idle');
            const heading = plan.mode === 'dance' ? scorerHeading : plan.heading;
            return {
                ...pos,
                heading: lerpAngle(runHeading, heading, smooth((t - arrive + 0.2) / 0.5)),
                moving: f < 1 ? 'run' : null,
                travelled: Math.min(dist, travelled),
                still,
                mix: smooth((t - arrive + 0.2) / 0.35),
                still_t: t - arrive + plan.offset,
                pin: true,
            };
        }
        const travelled = Math.max(0, (t - plan.start) * plan.speed);
        const f = dist > 0.01 ? Math.min(1, travelled / dist) : 1;
        const pos = { x: from.x + dx * f, y: from.y + dy * f };
        const still = roleName(plan.mode) || roleName('idle') || 'idle';
        let heading = runHeading;
        if (plan.kind === 'beaten') {
            heading = headingTo(-from.x, -from.y * 0.2);
        }
        const walking = f < 1 && t > plan.start;
        const head = plan.kind === 'back' || plan.kind === 'beaten' ? 0.35 : 0;
        return { ...pos, heading, moving: walking ? 'walk' : null, travelled: Math.min(dist, travelled), still, mix: walking ? 0 : 1, still_t: t - plan.start, head };
    }

    // The scorer's place, for the cameras.
    focus(t) {
        const s = this.scorer >= 0 ? this.state(this.scorer, t) : null;
        return s || { x: this.spot.x, y: this.spot.y, heading: 0 };
    }

    // Writes every player and the ball at time t, blended in from the frame's own pose over the
    // first 0.3 seconds, and by weight overall, which a stop may ease out.
    apply(pose, t, weight = 1, playClip = null) {
        writeGroup(this, pose, t, weight, playClip);
        if (pose.ball) {
            const b = ballScratch;
            ballIntoNet(this.ballA, this.ballB, this.team, this.sps, t, b);
            const bw = Math.min(1, weight);
            pose.ball[0] += (b[0] - pose.ball[0]) * bw;
            pose.ball[1] += (b[1] - pose.ball[1]) * bw;
            pose.ball[2] += (b[2] - pose.ball[2]) * bw;
        }
    }
}

// The horizontal radius of a player when the figures cannot tell, GRF's player with his arms in.
export const DEFAULT_RADIUS = 0.35;

// Keeps the players apart by their radii, pushing each pair that overlaps along the line between
// them, an anchored player stays put. Three rounds settle a crowd well enough, and the result
// follows from the states alone, so it holds when the clock runs backwards too.
function separate(states, radii) {
    for (let round = 0; round < 3; round++) {
        for (let i = 0; i < states.length; i++) {
            const a = states[i];
            if (!a) {
                continue;
            }
            for (let j = i + 1; j < states.length; j++) {
                const b = states[j];
                if (!b) {
                    continue;
                }
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                const min = radii[i] + radii[j] + 0.05;
                const d2 = dx * dx + dy * dy;
                if (d2 >= min * min) {
                    continue;
                }
                let d = Math.sqrt(d2);
                if (d < 1e-4) {
                    // Two on one spot part sideways, the same way every time.
                    dx = Math.cos(i * 2.399);
                    dy = Math.sin(i * 2.399);
                    d = 0;
                } else {
                    dx /= d;
                    dy /= d;
                }
                const push = min - d;
                let wa = 0.5;
                let wb = 0.5;
                if (a.anchor) {
                    wa = 0;
                    wb = b.anchor ? 0 : 1;
                } else if (b.anchor) {
                    wa = 1;
                    wb = 0;
                }
                a.x -= dx * push * wa;
                a.y -= dy * push * wa;
                b.x += dx * push * wb;
                b.y += dy * push * wb;
            }
        }
    }
}

// Tips the head of the player at o forward by angle radians, a turn of the neck node about its
// own x axis, which leaves the shoulders where they are.
function bowHead(nodes, o, angle) {
    const n = o + 3 * 7;
    const s = Math.sin(angle / 2);
    const c = Math.cos(angle / 2);
    const x = nodes[n + 3];
    const y = nodes[n + 4];
    const z = nodes[n + 5];
    const w = nodes[n + 6];
    if (x === 0 && y === 0 && z === 0 && w === 0) {
        return;
    }
    nodes[n + 3] = w * s + x * c;
    nodes[n + 4] = y * c + z * s;
    nodes[n + 5] = z * c - y * s;
    nodes[n + 6] = w * c - x * s;
}

// Poses every player of a choreography at time t from its states, kept apart, blended in from
// the frame's own pose over the first 0.3 seconds.
function writeGroup(group, pose, t, weight, playClip) {
    const nodes = pose.nodes;
    const w = Math.min(weight, smooth(t / 0.3));
    const work = group.work;
    const states = [];
    for (let k = 0; k < group.count; k++) {
        states.push(group.state(k, t));
    }
    separate(states, group.radii);
    for (let k = 0; k < group.count; k++) {
        const s = states[k];
        if (!s) {
            continue;
        }
        const o = k * PLAYER_FLOATS;
        const still = s.still;
        const mode = s.pin ? 'pin' : '';
        // A model with clips plays the clip of the still pose over the nodes, as strongly as the
        // pose shows, every frame it shows.
        const clip = playClip ? stillClip(group, k, still) : null;
        const stillT = Math.max(0, s.still_t);
        if (s.moving && s.mix < 1) {
            const gait = roleName(s.moving) || s.moving;
            const base = gaitSpeed(gait);
            // The library's gaits step for their own speed, so the steps follow the ground
            // covered and the feet do not slide.
            posePlayer(gait, base > 0 ? s.travelled / base : t, work, 0, s.x, s.y, s.heading);
            // A player slowing down passes from one gait to the next, each stepping for the
            // ground it covers.
            if (s.moving2 && s.gaitMix > 0) {
                const gait2 = roleName(s.moving2) || s.moving2;
                const base2 = gaitSpeed(gait2);
                posePlayer(gait2, base2 > 0 ? s.travelled / base2 : t, scratchGait, 0, s.x, s.y, s.heading);
                blendInto(work, 0, scratchGait, 0, s.gaitMix);
            }
            if (s.mix > 0) {
                posePlayer(still, stillT, scratchStill, 0, s.x, s.y, s.heading, mode);
                blendInto(work, 0, scratchStill, 0, s.mix);
                if (clip) {
                    group.clipped.add(k);
                    playClip(k, clip, stillT, s.mix * w);
                }
            }
        } else {
            posePlayer(still, stillT, work, 0, s.x, s.y, s.heading, mode);
            if (clip) {
                group.clipped.add(k);
                playClip(k, clip, stillT, w);
            }
        }
        if (s.head) {
            bowHead(work, 0, s.head);
        }
        blendInto(nodes, o, work, 0, w);
    }
}

const scratchStill = new Float32Array(PLAYER_FLOATS);
const scratchGait = new Float32Array(PLAYER_FLOATS);
const ballScratch = new Float32Array(3);

function lerpAngle(a, b, w) {
    let d = b - a;
    while (d > Math.PI) {
        d -= Math.PI * 2;
    }
    while (d < -Math.PI) {
        d += Math.PI * 2;
    }
    return a + d * Math.min(1, Math.max(0, w));
}

// The size of player k of a set of figures, the horizontal radius and the height in meters, from
// the figures when they can tell, else measured once from the figure's own bounds.
const sizeCache = new WeakMap();

export function figureSize(players, k) {
    if (!players) {
        return { radius: DEFAULT_RADIUS, height: 1.85 };
    }
    let cache = sizeCache.get(players);
    if (!cache) {
        cache = new Map();
        sizeCache.set(players, cache);
    }
    if (cache.has(k)) {
        return cache.get(k);
    }
    let radius = null;
    let height = null;
    try {
        if (typeof players.radius === 'function') {
            radius = players.radius(k);
        }
        if (typeof players.height === 'function') {
            height = players.height(k);
        }
        if (!(radius > 0) || !(height > 0)) {
            const part = players.where ? players.where[k] : null;
            const entry = part && part.byK ? part.byK.get(k) : null;
            const root = entry ? (entry.instance ? entry.instance.root : entry.root) : null;
            if (root) {
                const box = new THREE.Box3().setFromObject(root);
                const size = box.getSize(new THREE.Vector3());
                if (Number.isFinite(size.x) && size.y > 0.3) {
                    radius = radius > 0 ? radius : Math.max(size.x, size.z) / 2;
                    height = height > 0 ? height : size.y;
                }
            }
        }
    } catch {
        // The bounds are a nicety, the defaults below still keep players apart.
    }
    const out = {
        radius: Math.max(0.25, Math.min(1.2, radius > 0 ? radius : DEFAULT_RADIUS)),
        height: Math.max(1, Math.min(3.5, height > 0 ? height : 1.85)),
    };
    cache.set(k, out);
    return out;
}

// Where a player of a frame faces, the yaw of his root node.
function yawOf(base, k) {
    const o = k * PLAYER_FLOATS;
    const z = base[o + 5];
    const w = base[o + 6];
    return z === 0 && w === 0 ? 0 : 2 * Math.atan2(z, w);
}

// Up to this far a player walks towards the fans in the time the shots leave him.
const STAND_WALK = 9;

// The players after the final whistle, a pure function of the time since the last frame. The
// winners celebrate, jumping, punching the air, gathering in a huddle, a few running to the fans.
// The losers slump with their hands on their heads and walk slowly with heads down. A draw mixes
// both and has handshakes. Later everyone walks towards the far stand and applauds the fans.
export class Reactions {
    constructor({ frame, header, winner, seed, radius = null, applause = 7.2, hasClip = null, hero = -1, prev = null, sps = 10, card = 0 }) {
        this.hasClip = hasClip;
        this.hero = hero;
        this.header = header;
        this.count = header.players.length;
        this.base = frame.data.subarray(NODES_FLOAT);
        this.flags = frame.playerFlags;
        this.work = new Float32Array(PLAYER_FLOATS);
        this.clipped = new Set();
        this.radii = header.players.map((p, k) => (radius ? radius(k) : DEFAULT_RADIUS));
        const random = seeded(seed);
        // The play winds down after the whistle, each player slowing from his last stride to a
        // stand over two to four seconds, and the ball rolls on until the grass stops it.
        this.windDown(frame, prev, sps, seeded(seed + 977));
        // A card of card seconds comes up once play has stopped, and the players react as it
        // goes, standing still behind it.
        this.titleAt = Math.max(1.5, Math.min(4.4, this.windEnd + 0.4));
        this.hold = card > 0 ? this.titleAt + card - 0.7 : this.windEnd;
        applause += this.hold;
        const at = (k) => ({ x: this.wind[k].sx, y: this.wind[k].sy });
        const pick = (names) => {
            const have = names.filter((n) => !lib || libraryEntry(n));
            return have.length ? have[Math.floor(random() * have.length)] : names[0];
        };
        const role = (names, fallback) => names.find((n) => libraryEntry(n)) || fallback;
        const dejected = () => role(['slump', 'crouch', 'kneel', 'hands_on_head'], 'idle');
        const handshake = role(['handshake', 'shake_hands', 'point'], 'idle');
        const clap = roleName('clap') || 'idle';
        this.plans = [];
        const on = (k) => !!(this.flags[k] & 1);
        // Everyone applauds the far stand in the end, from wherever the earlier part left him.
        const toStand = (from, extra = 0) => {
            const dy = HALF_H - 3 - (random() * 3 + extra) - from.y;
            const dx = -from.x * 0.25;
            const len = Math.hypot(dx, dy) || 1;
            const go = Math.min(len, STAND_WALK);
            return { x: from.x + (dx / len) * go, y: from.y + (dy / len) * go };
        };
        const standHeading = headingTo(0, 1);
        header.players.forEach((p, k) => {
            if (!on(k)) {
                return;
            }
            const from = at(k);
            const w = this.wind[k];
            const plan = { from, heading: w.T > 0 ? headingTo(w.vx, w.vy) : yawOf(this.base, k), react: 'idle', reactAt: 0.3 + random() * 0.6, legs: [], head: 0 };
            const won = winner >= 0 && p.team === winner;
            const lost = winner >= 0 && p.team !== winner;
            if (won && k === hero) {
                // The hero jumps and punches the air where he stands, facing the main stand, a
                // KayKit hero cheers with his arms high.
                plan.react = hasClip && hasClip(k, 'clip:Cheering') ? 'clip:Cheering' : role(['jump_fist', 'arms_up'], 'idle');
                plan.legs.push({ to: from, start: plan.reactAt + 0.2, speed: 1, gait: 'walk', still: plan.react, heading: headingTo(0, -1) });
            } else if (won) {
                plan.react = p.goalkeeper ? role(['arms_up', 'jump_fist'], 'idle') : pick(['jump_fist', 'arms_up', 'dance', 'jump_fist']);
            } else if (lost) {
                plan.react = p.goalkeeper || random() < 0.7 ? dejected() : 'idle';
                plan.head = 0.5;
                // A few slow steps towards the middle, heads down.
                const to = { x: from.x * 0.85, y: from.y * 0.85 };
                const leg = { to, start: 2.6 + random() * 1.4, speed: 0.8, gait: 'walk', still: 'idle', heading: null };
                // A KayKit loser sinks to the grass, the keeper flat on his back, and gets up
                // again before he walks.
                const floor = plan.react !== 'idle' ? this.floorClip(k, p.goalkeeper, random) : null;
                if (floor) {
                    plan.react = floor.down;
                    plan.getUp = { name: floor.up, at: Math.max(leg.start, plan.reactAt + floor.hold), length: floor.length };
                    leg.start = plan.getUp.at + floor.length;
                }
                plan.legs.push(leg);
            } else {
                plan.react = pick(['idle_confident', 'hands_on_head', 'clap', 'idle_confident']);
                plan.head = plan.react === 'hands_on_head' ? 0.3 : 0;
            }
            this.plans[k] = plan;
        });
        if (winner >= 0) {
            this.bundleWinners(winner, at, on, random);
        } else {
            this.shakeHands(at, random, handshake);
        }
        // Everything a player does after the whistle waits until he has come to a stand.
        this.plans.forEach((plan, k) => {
            const wait = plan ? Math.max(this.wind[k].T, this.hold) : 0;
            if (!wait) {
                return;
            }
            plan.reactAt += wait;
            for (const leg of plan.legs) {
                leg.start += wait;
            }
            if (plan.getUp) {
                plan.getUp.at += wait;
            }
        });
        // The applause, every leg timed after the one before.
        header.players.forEach((p, k) => {
            const plan = this.plans[k];
            if (!plan) {
                return;
            }
            let pos = plan.from;
            let free = 0;
            for (const leg of plan.legs) {
                leg.from = pos;
                leg.start = Math.max(leg.start, free);
                leg.dist = Math.hypot(leg.to.x - pos.x, leg.to.y - pos.y);
                leg.arrive = leg.start + leg.dist / leg.speed;
                pos = leg.to;
                free = leg.arrive + 0.4;
            }
            const lost = winner >= 0 && p.team !== winner;
            const to = toStand(pos, lost ? 1.5 : 0);
            const leg = {
                from: pos, to, start: Math.max(free, applause + (lost ? 0.8 : 0) + random() * 0.6),
                speed: lost ? 1.0 : 1.45, gait: 'walk', still: clap, heading: standHeading,
            };
            leg.dist = Math.hypot(to.x - pos.x, to.y - pos.y);
            leg.arrive = leg.start + leg.dist / leg.speed;
            plan.legs.push(leg);
        });
    }

    // How each player and the ball wind down from the last frame, from the velocities between it
    // and the frame before. A player stops at an even deceleration over his own time, the ball
    // flies, bounces and rolls, kept inside the net or the boards.
    windDown(frame, prev, sps, random) {
        const base = this.base;
        const before = prev && prev.data ? prev.data.subarray(NODES_FLOAT) : null;
        this.wind = [];
        this.windEnd = 0;
        for (let k = 0; k < this.count; k++) {
            const o = k * PLAYER_FLOATS;
            const x = base[o];
            const y = base[o + 1];
            let vx = before ? (x - before[o]) * sps : 0;
            let vy = before ? (y - before[o + 1]) * sps : 0;
            let v = Math.hypot(vx, vy);
            // A jump between the two frames is a cut, not a stride.
            if (!(v < 11)) {
                vx = 0;
                vy = 0;
                v = 0;
            }
            let T = v > 0.35 ? 2 + random() * 2 : 0;
            // A sprinter covers at most eight meters while he pulls up.
            if (v * T / 2 > 8) {
                T = 16 / v;
            }
            const run = v * T / 2;
            const sx = Math.max(-HALF_W - 1, Math.min(HALF_W + 1, x + (v > 0 ? (vx / v) * run : 0)));
            const sy = Math.max(-HALF_H - 1, Math.min(HALF_H + 1, y + (v > 0 ? (vy / v) * run : 0)));
            this.wind.push({ x, y, vx, vy, v, T, sx, sy, run: Math.hypot(sx - x, sy - y) });
            this.windEnd = Math.max(this.windEnd, T);
        }
        this.ballPath = rollBall(frame.data.subarray(8, 11), prev && prev.data ? prev.data.subarray(8, 11) : null, sps);
    }

    // The ball at time t after the last frame, into out.
    ball(t, out) {
        const path = this.ballPath;
        const f = Math.max(0, Math.min(path.length / 3 - 1, t / BALL_STEP));
        const i = Math.floor(f);
        const j = Math.min(path.length / 3 - 1, i + 1);
        const u = f - i;
        for (let c = 0; c < 3; c++) {
            out[c] = path[i * 3 + c] + (path[j * 3 + c] - path[i * 3 + c]) * u;
        }
        return out;
    }

    // The clips that take a beaten player of a KayKit model down to the grass and up again, the
    // time he stays down at least and the length of getting up, or null.
    floorClip(k, keeper, random) {
        const has = this.hasClip;
        if (!has) {
            return null;
        }
        const options = [
            { down: 'clip:Sit_Floor_Down>Sit_Floor_Idle', up: 'clip:Sit_Floor_StandUp', length: 1.15, hold: 2.4 },
            { down: 'clip:Lie_Down>Lie_Idle', up: 'clip:Lie_StandUp', length: 2.35, hold: 7.6 },
        ].filter((o) => has(k, o.down) && has(k, o.up));
        if (!options.length) {
            return null;
        }
        const lie = options.find((o) => o.down.startsWith('clip:Lie'));
        if (keeper && lie) {
            return lie;
        }
        const sit = options.find((o) => o.down.startsWith('clip:Sit'));
        // Most sit, a few stay on their feet with their hands on their heads.
        return sit && random() < 0.75 ? sit : null;
    }

    // A bundle of the winners nearest one another, around their middle, and one or two who run
    // off to the fans of the nearest stand with their arms up.
    bundleWinners(winner, at, on, random) {
        const header = this.header;
        const mates = header.players.map((p, k) => ({ p, k })).filter((m) => m.p.team === winner && !m.p.goalkeeper && on(m.k) && m.k !== this.hero);
        if (mates.length < 3) {
            return;
        }
        const cx = mates.reduce((a, m) => a + at(m.k).x, 0) / mates.length;
        const cy = mates.reduce((a, m) => a + at(m.k).y, 0) / mates.length;
        mates.sort((a, b) => Math.hypot(at(a.k).x - cx, at(a.k).y - cy) - Math.hypot(at(b.k).x - cx, at(b.k).y - cy));
        const bundle = mates.slice(0, Math.min(6, Math.max(3, Math.floor(mates.length / 2))));
        const bx = Math.max(-HALF_W + 3, Math.min(HALF_W - 3, bundle.reduce((a, m) => a + at(m.k).x, 0) / bundle.length));
        const by = Math.max(-HALF_H + 3, Math.min(HALF_H - 3, bundle.reduce((a, m) => a + at(m.k).y, 0) / bundle.length));
        const r = Math.max(...bundle.map((m) => this.radii[m.k]));
        const ring = Math.max(0.9, (2 * r + 0.1) / (2 * Math.sin(Math.PI / bundle.length)));
        const huddle = roleName('huddle') || 'arms_up';
        bundle.forEach((m, i) => {
            const a = (i / bundle.length) * Math.PI * 2;
            const to = { x: bx + Math.cos(a) * ring, y: by + Math.sin(a) * ring };
            const plan = this.plans[m.k];
            plan.react = libraryEntry('jump_fist') ? 'jump_fist' : plan.react;
            plan.legs.push({ to, start: 0.5 + random() * 0.5, speed: 5.2, gait: 'run', still: huddle, heading: headingTo(bx - to.x, by - to.y) });
        });
        this.bundle = { x: bx, y: by };
        mates.slice(bundle.length).slice(0, 2).forEach((m) => {
            const f = at(m.k);
            const side = f.y >= 0 ? 1 : -1;
            const to = { x: f.x, y: side * (HALF_H - 1.2) };
            if (Math.abs(to.y - f.y) < 22) {
                this.plans[m.k].legs.push({ to, start: 0.4, speed: 5, gait: 'run', still: 'arms_up', heading: headingTo(0, side) });
            }
        });
    }

    // Handshakes between near opponents on a draw, up to four pairs.
    shakeHands(at, random, handshake) {
        const header = this.header;
        const used = new Set();
        const pairs = [];
        for (let a = 0; a < this.count && pairs.length < 4; a++) {
            if (!this.plans[a] || used.has(a) || header.players[a].team !== 0 || header.players[a].goalkeeper) {
                continue;
            }
            let best = -1;
            let bestD = 12;
            for (let b = 0; b < this.count; b++) {
                if (!this.plans[b] || used.has(b) || header.players[b].team !== 1 || header.players[b].goalkeeper) {
                    continue;
                }
                const d = Math.hypot(at(a).x - at(b).x, at(a).y - at(b).y);
                if (d < bestD) {
                    bestD = d;
                    best = b;
                }
            }
            if (best >= 0) {
                used.add(a);
                used.add(best);
                pairs.push([a, best]);
            }
        }
        for (const [a, b] of pairs) {
            const pa = at(a);
            const pb = at(b);
            const mx = (pa.x + pb.x) / 2;
            const my = (pa.y + pb.y) / 2;
            const len = Math.hypot(pb.x - pa.x, pb.y - pa.y) || 1;
            const ux = (pb.x - pa.x) / len;
            const uy = (pb.y - pa.y) / len;
            const gap = (this.radii[a] + this.radii[b]) / 2 + 0.35;
            const start = 0.9 + random() * 0.6;
            this.plans[a].legs.push({ to: { x: mx - ux * gap, y: my - uy * gap }, start, speed: 1.5, gait: 'walk', still: handshake, heading: headingTo(ux, uy) });
            this.plans[b].legs.push({ to: { x: mx + ux * gap, y: my + uy * gap }, start, speed: 1.5, gait: 'walk', still: handshake, heading: headingTo(-ux, -uy) });
            this.plans[a].react = 'idle';
            this.plans[b].react = 'idle';
        }
        this.pairs = pairs;
    }

    release(stopClip) {
        for (const k of this.clipped) {
            stopClip(k);
        }
        this.clipped.clear();
    }

    state(k, t) {
        const plan = this.plans[k];
        if (!plan) {
            return null;
        }
        const w = this.wind[k];
        if (w && w.T > 0 && t < w.T) {
            // Slowing evenly, from a run through a jog to a walk and a stand.
            const covered = w.v * (t - (t * t) / (2 * w.T));
            const f = w.v * w.T > 0 ? Math.min(1, covered / (w.v * w.T / 2)) : 1;
            const speed = w.v * (1 - t / w.T);
            return {
                x: w.x + (w.sx - w.x) * f,
                y: w.y + (w.sy - w.y) * f,
                heading: plan.heading,
                moving: 'run',
                moving2: 'walk',
                gaitMix: 1 - smooth((speed - 1.5) / 1.6),
                travelled: w.run * f,
                still: 'idle',
                mix: smooth((0.7 - speed) / 0.7),
                still_t: 0,
                head: plan.head * smooth((t - w.T + 1) / 1),
            };
        }
        let x = plan.from.x;
        let y = plan.from.y;
        let still = t >= plan.reactAt ? plan.react : 'idle';
        let stillT = t - plan.reactAt;
        if (plan.getUp && t >= plan.getUp.at) {
            still = plan.getUp.name;
            stillT = t - plan.getUp.at;
        }
        let heading = plan.heading;
        let moving = null;
        let travelled = 0;
        let mix = 1;
        for (const leg of plan.legs) {
            if (t < leg.start) {
                break;
            }
            const d = Math.min(leg.dist, (t - leg.start) * leg.speed);
            const f = leg.dist > 0.01 ? d / leg.dist : 1;
            x = leg.from.x + (leg.to.x - leg.from.x) * f;
            y = leg.from.y + (leg.to.y - leg.from.y) * f;
            const walkHeading = leg.dist > 0.01 ? headingTo(leg.to.x - leg.from.x, leg.to.y - leg.from.y) : heading;
            if (f < 1) {
                moving = leg.gait;
                travelled = d;
                // Out of the pose before into the walk, and into the pose after near the end.
                const out = 1 - smooth((t - leg.start) / 0.35);
                const into = smooth((t - leg.arrive + 0.3) / 0.35);
                mix = Math.max(out, into);
                if (into > out) {
                    still = leg.still;
                    stillT = t - leg.arrive;
                }
                heading = lerpAngle(heading, walkHeading, smooth((t - leg.start) / 0.4));
            } else {
                moving = null;
                still = leg.still;
                stillT = t - leg.arrive;
                const settle = smooth((t - leg.arrive + 0.2) / 0.5);
                heading = lerpAngle(walkHeading, leg.heading == null ? walkHeading : leg.heading, settle);
                mix = 1;
            }
        }
        return { x, y, heading, moving, travelled, still, mix, still_t: stillT, head: plan.head, pin: true };
    }

    apply(pose, t, weight = 1, playClip = null) {
        writeGroup(this, pose, t, weight, playClip);
        if (pose.ball && this.ballPath) {
            const b = this.ball(t, ballScratch);
            const bw = Math.min(1, weight);
            pose.ball[0] += (b[0] - pose.ball[0]) * bw;
            pose.ball[1] += (b[1] - pose.ball[1]) * bw;
            pose.ball[2] += (b[2] - pose.ball[2]) * bw;
            // The ball spins as it rolls, from its own motion.
            pose.ballKnownRotation = false;
        }
    }
}

const BALL_STEP = 1 / 50;
const BALL_RADIUS_M = 0.11;

// The flight of the ball after the last frame, from its last velocity, as positions every
// BALL_STEP seconds for eight seconds. It falls, bounces lower each time, rolls with the drag of
// the grass, and stops in the netting of a goal or bounces back off the boards.
function rollBall(b, a, sps) {
    let x = b[0];
    let y = b[1];
    let z = Math.max(BALL_RADIUS_M, b[2]);
    let vx = a ? (b[0] - a[0]) * sps : 0;
    let vy = a ? (b[1] - a[1]) * sps : 0;
    let vz = a ? (b[2] - a[2]) * sps : 0;
    if (!(Math.hypot(vx, vy, vz) < 45)) {
        vx = 0;
        vy = 0;
        vz = 0;
    }
    const steps = Math.round(8 / BALL_STEP);
    const out = new Float32Array((steps + 1) * 3);
    const netX = HALF_W + 1.6;
    const boardX = HALF_W + 4;
    const boardY = HALF_H + 3.5;
    for (let i = 0; i <= steps; i++) {
        out[i * 3] = x;
        out[i * 3 + 1] = y;
        out[i * 3 + 2] = z;
        const dt = BALL_STEP;
        const ground = z <= BALL_RADIUS_M + 1e-3 && Math.abs(vz) < 0.5;
        if (ground) {
            z = BALL_RADIUS_M;
            vz = 0;
            // Rolling friction on grass, about a third of the speed lost each second.
            const drag = Math.exp(-0.45 * dt);
            vx = vx * drag - Math.sign(vx) * Math.min(Math.abs(vx), 0.35 * dt);
            vy = vy * drag - Math.sign(vy) * Math.min(Math.abs(vy), 0.35 * dt);
        } else {
            vz -= 9.81 * dt;
            vx *= Math.exp(-0.05 * dt);
            vy *= Math.exp(-0.05 * dt);
        }
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
        if (z < BALL_RADIUS_M) {
            z = BALL_RADIUS_M;
            vz = vz < -0.5 ? -vz * 0.5 : 0;
            vx *= 0.85;
            vy *= 0.85;
        }
        // Into the goal the netting holds the ball, wide of it the boards send it back softly.
        if (Math.abs(x) > HALF_W) {
            const mouth = Math.abs(y) < 3.66 && z < 2.44;
            if (mouth && Math.abs(x) > netX) {
                x = Math.sign(x) * netX;
                vx = 0;
                vy *= 0.3;
            } else if (!mouth && Math.abs(x) > boardX) {
                x = Math.sign(x) * boardX;
                vx = -vx * 0.25;
            }
        }
        if (Math.abs(y) > boardY) {
            y = Math.sign(y) * boardY;
            vy = -vy * 0.25;
        }
        if (z > 2.44 && Math.abs(x) > HALF_W && Math.abs(y) < 3.66 && Math.abs(x) < netX) {
            // Over the bar onto the roof of the net, it drops behind.
            vz = Math.min(vz, 0);
        }
    }
    return out;
}
