// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Procedural animations on GRF's skeleton of 14 nodes, celebrations, idles, entrances and the
// small gestures of restarts. Every animation writes one player's nodes in the engine frame, z up
// in meters with the player facing engine minus y at heading 0, the same buffer a recording
// holds, so every model follows it through its own retargeting.
//
// An animation is either keys or a function of time. A key holds joint angles in degrees, local
// to the parent node and applied x first, then y, then z, and a few other channels, the lift of
// the lowest point of the body above the ground, a move of the whole player forward and to its
// left, a turn, and targets for the hands and feet that two bone inverse kinematics reaches.
// Keys blend along cubic Hermite curves. Without planted feet the body is lowered or raised
// until its lowest point, a heel, toe, knee, hand, hip, chest or head, sits at the lift, so
// crouches and kneels stay on the ground by themselves.
//
// A turn of node body bends at the hips and leaves the legs upright, unless the animation sets
// trunk false, then the whole body turns about the pelvis, as in a flip.
//
// Angle signs in the engine frame. On x, a positive angle tips the body, the chest and the head
// forward, swings a thigh back, bends a knee, swings an arm back, straightens an elbow and points
// the toes down. On y, a positive angle tilts the trunk to the player's left and swings the right
// arm or leg out, a negative one the left arm or leg out. On z, a positive angle turns left.
// Targets are meters as left, forward and up, from the chest node for the hands and from the
// ground point for the feet.

import * as THREE from 'three';
import { NODE_COUNT, NODE_PARENTS, PLAYER_FLOATS } from './format.js';

export const REST_LOCAL = [
    [0, 0, 0], [0, 0, 0.96], [0, 0, 0.15], [0, -0.03, 0.5], [0.16, -0.01, 0.48], [-0.01, 0, -0.33],
    [-0.16, -0.01, 0.48], [0.01, 0, -0.33], [0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
    [-0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
];

// Node indices.
const BODY = 1;
const MIDDLE = 2;
const NECK = 3;
const L_SHOULDER = 4;
const L_ELBOW = 5;
const R_SHOULDER = 6;
const R_ELBOW = 7;
const L_THIGH = 8;
const L_KNEE = 9;
const L_ANKLE = 10;
const R_THIGH = 11;
const R_KNEE = 12;
const R_ANKLE = 13;

// Channels of an animation sample.
const LIFT = 42;
const FWD = 43;
const SIDE = 44;
const YAW = 45;
const HZ = 46;
const LH = 47;
const RH = 50;
const LF = 53;
const RF = 56;
const LP = 59;
const RP = 62;
const CHANNELS = 65;

const UPPER_ARM = 0.33;
const FOREARM = 0.30;
const THIGH = 0.42;
const SHIN = Math.hypot(0.04, 0.44);
const ANKLE_HEIGHT = 0.09;
const DEG = Math.PI / 180;

// The hands at rest, seen from the chest node, and the default poles of the elbows and knees.
const HAND_REST_L = [0.19, 0.0, -0.13];
const HAND_REST_R = [-0.19, 0.0, -0.13];
const POLE_L = [1, -0.6, -0.5];
const POLE_R = [-1, -0.6, -0.5];
const FOOT_L = [0.11, 0.0, 0];
const FOOT_R = [-0.11, 0.0, 0];

// Scratch objects, the library allocates nothing per call.
const euler = new THREE.Euler();
const qa = new THREE.Quaternion();
const qb = new THREE.Quaternion();
const qYaw = new THREE.Quaternion();
const va = new THREE.Vector3();
const vb = new THREE.Vector3();
const vc = new THREE.Vector3();
const vd = new THREE.Vector3();
const Z_AXIS = new THREE.Vector3(0, 0, 1);
const localQ = Array.from({ length: NODE_COUNT }, () => new THREE.Quaternion());
const localT = Array.from({ length: NODE_COUNT }, () => new THREE.Vector3());
const worldQ = Array.from({ length: NODE_COUNT }, () => new THREE.Quaternion());
const worldP = Array.from({ length: NODE_COUNT }, () => new THREE.Vector3());
const sample = new Float32Array(CHANNELS);
const scratchA = new Float32Array(CHANNELS);
const scratchB = new Float32Array(CHANNELS);
const scratchC = new Float32Array(CHANNELS);

function smoothstep(u) {
    return u * u * (3 - 2 * u);
}

// Key authoring. A key names joints by short names and lists their angles in degrees.
const JOINT = {
    body: BODY, middle: MIDDLE, neck: NECK, ls: L_SHOULDER, le: L_ELBOW, rs: R_SHOULDER, re: R_ELBOW,
    lt: L_THIGH, lk: L_KNEE, la: L_ANKLE, rt: R_THIGH, rk: R_KNEE, ra: R_ANKLE,
};
const VECTOR = { lh: LH, rh: RH, lf: LF, rf: RF, lp: LP, rp: RP };

function defaults(out) {
    out.fill(0);
    out.set(HAND_REST_L, LH);
    out.set(HAND_REST_R, RH);
    out.set(FOOT_L, LF);
    out.set(FOOT_R, RF);
    out.set(POLE_L, LP);
    out.set(POLE_R, RP);
    return out;
}

function writeKey(spec, out) {
    for (const [name, value] of Object.entries(spec)) {
        if (name in JOINT) {
            const j = JOINT[name] * 3;
            out[j] = value[0] || 0;
            out[j + 1] = value[1] || 0;
            out[j + 2] = value[2] || 0;
        } else if (name in VECTOR) {
            out.set(value, VECTOR[name]);
        } else if (name === 'lift') {
            out[LIFT] = value;
        } else if (name === 'f') {
            out[FWD] = value;
        } else if (name === 's') {
            out[SIDE] = value;
        } else if (name === 'yaw') {
            out[YAW] = value;
        } else if (name === 'hz') {
            out[HZ] = value;
        }
    }
    return out;
}

// Turns an animation's keys into flat channel vectors, each key over the animation's base.
function compile(anim) {
    if (anim.compiled || !anim.keys) {
        return;
    }
    const base = writeKey(anim.base || {}, defaults(new Float32Array(CHANNELS)));
    anim.compiled = anim.keys.map((k) => ({
        t: k.t,
        e: k.e || anim.ease || 'smooth',
        v: writeKey(k, Float32Array.from(base)),
    }));
}

// Samples keys at time t into out with Hermite curves through the keys. A looping animation
// wraps around its duration, any other holds its first and last keys.
function sampleKeys(anim, t, out) {
    compile(anim);
    const keys = anim.compiled;
    const n = keys.length;
    const D = anim.duration;
    if (n === 1) {
        out.set(keys[0].v);
        return out;
    }
    if (anim.loop) {
        t = ((t % D) + D) % D;
    } else if (t <= keys[0].t) {
        out.set(keys[0].v);
        return out;
    } else if (t >= keys[n - 1].t) {
        out.set(keys[n - 1].v);
        return out;
    }
    let i = n - 1;
    for (let k = 0; k < n; k++) {
        if (keys[k].t > t) {
            i = k - 1;
            break;
        }
    }
    const at = (k) => {
        if (anim.loop) {
            const wrap = Math.floor(k / n);
            const key = keys[((k % n) + n) % n];
            return { t: key.t + wrap * D, v: key.v, e: key.e };
        }
        const key = keys[Math.max(0, Math.min(n - 1, k))];
        return { t: key.t, v: key.v, e: key.e, edge: k < 0 || k > n - 1 };
    };
    if (i < 0) {
        // Before the first key of a loop, the segment from the last key of the previous turn.
        i = -1;
    }
    const k0 = at(i - 1);
    const k1 = at(i);
    const k2 = at(i + 1);
    const k3 = at(i + 2);
    let tt = t;
    if (tt < k1.t) {
        tt += D;
    }
    const span = k2.t - k1.t || 1;
    let u = Math.max(0, Math.min(1, (tt - k1.t) / span));
    const ease = k2.e;
    if (ease === 'hold') {
        out.set(k1.v);
        return out;
    }
    if (ease === 'linear') {
        for (let c = 0; c < CHANNELS; c++) {
            out[c] = k1.v[c] + (k2.v[c] - k1.v[c]) * u;
        }
        return out;
    }
    let zeroTangents = ease === 'inout' || ease === 'snap';
    if (ease === 'snap') {
        u = smoothstep(smoothstep(u));
    }
    const u2 = u * u;
    const u3 = u2 * u;
    const h00 = 2 * u3 - 3 * u2 + 1;
    const h10 = u3 - 2 * u2 + u;
    const h01 = -2 * u3 + 3 * u2;
    const h11 = u3 - u2;
    const d0 = k2.t - k0.t || 1;
    const d1 = k3.t - k1.t || 1;
    // The first and last keys of a single play start and end at rest.
    const m1s = zeroTangents || k0.edge || (!anim.loop && i === 0) ? 0 : span / d0;
    const m2s = zeroTangents || k3.edge || (!anim.loop && i + 1 === n - 1) ? 0 : span / d1;
    for (let c = 0; c < CHANNELS; c++) {
        const p1 = k1.v[c];
        const p2 = k2.v[c];
        const m1 = (p2 - k0.v[c]) * m1s;
        const m2 = (k3.v[c] - p1) * m2s;
        out[c] = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2;
    }
    return out;
}

// Two bone inverse kinematics. Root, target and pole in one frame, the result is the middle
// joint's position in out.
function reach(root, target, pole, a, b, out) {
    vd.subVectors(target, root);
    let d = vd.length();
    if (d < 1e-6) {
        vd.set(0, 0, -1);
        d = 1e-6;
    }
    vd.divideScalar(d);
    d = Math.max(Math.abs(a - b) + 1e-3, Math.min(a + b - 1e-4, d));
    const along = (a * a - b * b + d * d) / (2 * d);
    const h = Math.sqrt(Math.max(0, a * a - along * along));
    vc.copy(pole).addScaledVector(vd, -pole.dot(vd));
    if (vc.lengthSq() < 1e-10) {
        vc.set(0, 0, 1).addScaledVector(vd, -vd.z);
    }
    vc.normalize();
    return out.copy(root).addScaledVector(vd, along).addScaledVector(vc, h);
}

const REST_UPPER_L = new THREE.Vector3(...REST_LOCAL[L_ELBOW]).normalize();
const REST_UPPER_R = new THREE.Vector3(...REST_LOCAL[R_ELBOW]).normalize();
const REST_SHIN = new THREE.Vector3(...REST_LOCAL[L_ANKLE]).normalize();
const DOWN = new THREE.Vector3(0, 0, -1);

// Engine vector from left, forward and up.
function lfu(v, o, out) {
    return out.set(v[o], -v[o + 1], v[o + 2]);
}

// Sets the shoulder and elbow rotations of one arm so its hand reaches a target in the chest
// frame, or in the ground frame for an animation that plants its feet.
function armIK(v, shoulder, hand, pole, ground) {
    const elbow = shoulder + 1;
    const S = vb.set(...REST_LOCAL[shoulder]);
    const T = lfu(v, hand, new THREE.Vector3());
    if (ground) {
        // A target on the ground, brought into the chest frame of the trunk posed so far.
        T.applyQuaternion(qYaw).add(worldP[0]).sub(worldP[MIDDLE]);
        T.applyQuaternion(qa.copy(worldQ[MIDDLE]).invert());
    }
    const P = lfu(v, pole, new THREE.Vector3());
    const E = reach(S, T, P, UPPER_ARM, FOREARM, new THREE.Vector3());
    const upper = va.subVectors(E, S).normalize();
    localQ[shoulder].setFromUnitVectors(shoulder === L_SHOULDER ? REST_UPPER_L : REST_UPPER_R, upper);
    const fore = T.sub(E).normalize().applyQuaternion(qa.copy(localQ[shoulder]).invert());
    localQ[elbow].setFromUnitVectors(DOWN, fore);
}

// Sets the thigh and knee rotations of one leg so its ankle reaches a target on the ground.
function legIK(v, thigh, foot, side) {
    const knee = thigh + 1;
    const H = new THREE.Vector3(...REST_LOCAL[thigh]).applyQuaternion(worldQ[BODY]).add(worldP[BODY]);
    const T = lfu(v, foot, new THREE.Vector3());
    T.z += ANKLE_HEIGHT + v[LIFT];
    T.applyQuaternion(qYaw).add(worldP[0]);
    const P = new THREE.Vector3(side * 0.15, -1, 0.1).applyQuaternion(qYaw);
    const K = reach(H, T, P, THIGH, SHIN, new THREE.Vector3());
    const thighDir = va.subVectors(K, H).normalize();
    const qThigh = new THREE.Quaternion().setFromUnitVectors(DOWN, thighDir);
    // Keep the thigh's twist with the body, a turn of the hips turns the knees.
    const facing = new THREE.Vector3(0, -1, 0).applyQuaternion(worldQ[BODY]);
    const now = new THREE.Vector3(0, -1, 0).applyQuaternion(qThigh);
    const projNow = now.addScaledVector(thighDir, -now.dot(thighDir)).normalize();
    const projWant = facing.addScaledVector(thighDir, -facing.dot(thighDir)).normalize();
    if (projNow.lengthSq() > 0.5 && projWant.lengthSq() > 0.5) {
        qb.setFromUnitVectors(projNow, projWant);
        qThigh.premultiply(qb);
    }
    localQ[thigh].copy(worldQ[BODY]).invert().multiply(qThigh);
    const shin = vb.subVectors(T, K).normalize().applyQuaternion(qa.copy(qThigh).invert());
    localQ[knee].setFromUnitVectors(REST_SHIN, shin);
}

// Forward kinematics from node from on, ankles level with the ground unless the animation
// moves them with the legs.
function forward(v, from, footLocal) {
    for (let i = from; i < NODE_COUNT; i++) {
        const p = NODE_PARENTS[i];
        if (p < 0) {
            continue;
        }
        worldP[i].copy(localT[i]).applyQuaternion(worldQ[p]).add(worldP[p]);
        if ((i === L_ANKLE || i === R_ANKLE) && !footLocal) {
            euler.set(v[i * 3] * DEG, v[i * 3 + 1] * DEG, v[i * 3 + 2] * DEG, 'ZYX');
            worldQ[i].setFromEuler(euler).premultiply(qYaw);
        } else {
            worldQ[i].multiplyQuaternions(worldQ[p], localQ[i]);
        }
    }
}

// The lowest point of the body above the ground, heels, toes, knees, hands, hips, chest and head.
const CONTACTS = [
    [L_ANKLE, 0, 0.06, -0.085, 0], [L_ANKLE, 0, -0.17, -0.085, 0],
    [R_ANKLE, 0, 0.06, -0.085, 0], [R_ANKLE, 0, -0.17, -0.085, 0],
    [L_KNEE, 0, 0, 0, 0.06], [R_KNEE, 0, 0, 0, 0.06],
    [L_ELBOW, 0, 0, -FOREARM, 0.04], [R_ELBOW, 0, 0, -FOREARM, 0.04],
    [L_ELBOW, 0, 0, 0, 0.05], [R_ELBOW, 0, 0, 0, 0.05],
    [BODY, 0, 0, 0, 0.11], [MIDDLE, 0, 0, 0.2, 0.13], [NECK, 0, -0.02, 0.14, 0.12],
];

function lowest() {
    let low = Infinity;
    for (const [node, x, y, z, r] of CONTACTS) {
        va.set(x, y, z).applyQuaternion(worldQ[node]).add(worldP[node]);
        low = Math.min(low, va.z - r);
    }
    return low;
}

// Poses one player from a channel vector, writing its nodes into out from offset on.
function solve(v, anim, opts, out, offset) {
    const heading = (opts.heading || 0) + v[YAW] * DEG;
    qYaw.setFromAxisAngle(Z_AXIS, heading);
    const h0 = opts.heading || 0;
    // Root motion is along the heading the animation started with.
    const fx = Math.sin(h0);
    const fy = -Math.cos(h0);
    const px = (opts.x || 0) + v[FWD] * fx + v[SIDE] * Math.cos(h0);
    const py = (opts.y || 0) + v[FWD] * fy + v[SIDE] * Math.sin(h0);
    worldP[0].set(px, py, 0);
    worldQ[0].copy(qYaw);
    for (let i = 1; i < NODE_COUNT; i++) {
        euler.set(v[i * 3] * DEG, v[i * 3 + 1] * DEG, v[i * 3 + 2] * DEG, 'ZYX');
        localQ[i].setFromEuler(euler);
        localT[i].set(...REST_LOCAL[i]);
    }
    const plant = !!anim.plant;
    if (plant) {
        localT[BODY].z += v[HZ] + v[LIFT];
    } else if (anim.trunk !== false) {
        // A turn of the body bends at the hips, the thighs keep their angles from upright.
        qa.copy(localQ[BODY]).invert();
        localQ[L_THIGH].premultiply(qa);
        localQ[R_THIGH].premultiply(qa);
    }
    forward(v, 1, anim.footLocal);
    if (anim.hands) {
        if (anim.hands.includes('l')) {
            armIK(v, L_SHOULDER, LH, LP, anim.handFrame === 'ground');
        }
        if (anim.hands.includes('r')) {
            armIK(v, R_SHOULDER, RH, RP, anim.handFrame === 'ground');
        }
    }
    if (plant) {
        legIK(v, L_THIGH, LF, 1);
        legIK(v, R_THIGH, RF, -1);
    }
    forward(v, 1, anim.footLocal);
    if (!plant) {
        const dz = v[LIFT] - lowest();
        for (let i = 1; i < NODE_COUNT; i++) {
            worldP[i].z += dz;
        }
    }
    for (let i = 0; i < NODE_COUNT; i++) {
        const o = offset + i * 7;
        const p = worldP[i];
        const q = worldQ[i];
        out[o] = p.x;
        out[o + 1] = p.y;
        out[o + 2] = p.z;
        out[o + 3] = q.x;
        out[o + 4] = q.y;
        out[o + 5] = q.z;
        out[o + 6] = q.w;
    }
    return out;
}

// Channel helpers for animations written as functions.
function ang(v, node, x, y = 0, z = 0) {
    const j = node * 3;
    v[j] = x;
    v[j + 1] = y;
    v[j + 2] = z;
}

function addAng(v, node, x, y = 0, z = 0) {
    const j = node * 3;
    v[j] += x;
    v[j + 1] += y;
    v[j + 2] += z;
}

function setVec(v, o, a, b, c) {
    v[o] = a;
    v[o + 1] = b;
    v[o + 2] = c;
}

// A running gait into v, phase in turns, amount from 0 for a walk to 1 for a sprint.
function gait(v, phase, amount) {
    const p = phase * Math.PI * 2;
    const s = Math.sin(p);
    const c = Math.cos(p);
    const swingL = Math.max(0, c);
    const swingR = Math.max(0, -c);
    const thigh = 22 + 26 * amount;
    ang(v, L_THIGH, -thigh * s, -2);
    ang(v, R_THIGH, thigh * s, 2);
    ang(v, L_KNEE, 6 + (30 + 60 * amount) * swingL ** 1.5 + 8 * amount);
    ang(v, R_KNEE, 6 + (30 + 60 * amount) * swingR ** 1.5 + 8 * amount);
    ang(v, L_ANKLE, -8 * s);
    ang(v, R_ANKLE, 8 * s);
    const arm = 16 + 30 * amount;
    ang(v, L_SHOULDER, arm * s, -6);
    ang(v, R_SHOULDER, -arm * s, 6);
    ang(v, L_ELBOW, -25 - 55 * amount - 10 * Math.max(0, -s));
    ang(v, R_ELBOW, -25 - 55 * amount - 10 * Math.max(0, s));
    ang(v, BODY, 3 + 9 * amount, 0, 5 * s * (1 - amount * 0.5));
    ang(v, MIDDLE, 2, 0, -7 * s);
    ang(v, NECK, -4 - 6 * amount, 0, 2 * s);
    // A flight phase while running.
    v[LIFT] = amount > 0.3 ? 0.07 * amount * Math.max(0, Math.sin(p * 2)) : 0;
}

// A quiet stance with breathing, t in seconds, repeating every period seconds.
function stance(v, t, period = 6) {
    const f = (Math.PI * 2) / period;
    const b = Math.sin(t * f * Math.max(1, Math.round(3 / f)));
    const w = Math.sin(t * f);
    ang(v, BODY, 1, 1.6 * w, 1.5 * w);
    ang(v, MIDDLE, 1.2 * b, -1.2 * w);
    ang(v, NECK, -1 - b, 0, 3 * Math.sin(t * f + 1));
    ang(v, L_SHOULDER, 2 + 2 * b, -7);
    ang(v, R_SHOULDER, 2 + 2 * b, 7);
    ang(v, L_ELBOW, -14 - 2 * b);
    ang(v, R_ELBOW, -14 - 2 * b);
    ang(v, L_THIGH, -2, -4 + 1.5 * w);
    ang(v, R_THIGH, -2, 4 + 1.5 * w);
    ang(v, L_KNEE, 4 - 2 * w);
    ang(v, R_KNEE, 4 + 2 * w);
}

const HEART = 0.36;

// The library. Each entry has a label, a kind, a duration in seconds and whether it loops, and
// keys or a function fn of time and the channel vector.
const LIBRARY = {
    // Celebrations.
    arms_up: {
        label: 'Arms up', kind: 'celebration', duration: 1.6, loop: true, hands: 'lr',
        base: { lp: [1, -0.3, -0.4], rp: [-1, -0.3, -0.4], neck: [-18], middle: [-6] },
        keys: [
            { t: 0, lh: [0.42, 0.08, 1.0], rh: [-0.42, 0.08, 1.0], lift: 0 },
            { t: 0.4, lh: [0.5, 0.1, 0.92], rh: [-0.5, 0.1, 0.92], lift: 0.08, lk: [10], rk: [10] },
            { t: 0.8, lh: [0.4, 0.06, 1.02], rh: [-0.4, 0.06, 1.02], lift: 0 },
            { t: 1.2, lh: [0.5, 0.1, 0.92], rh: [-0.5, 0.1, 0.92], lift: 0.08, lk: [10], rk: [10] },
        ],
    },
    knee_slide: {
        label: 'Knee slide', kind: 'celebration', duration: 3.2, loop: false, footLocal: true, hands: 'lr',
        base: { lp: [1, -0.5, 0], rp: [-1, -0.5, 0] },
        keys: [
            {
                t: 0, f: 0, body: [10], lt: [-30], rt: [25], lk: [40], rk: [70], la: [10], ra: [30],
                lh: [0.2, -0.15, -0.05], rh: [-0.2, 0.3, 0.05],
            },
            {
                t: 0.35, f: 1.6, body: [-5], lt: [-5], rt: [-5], lk: [88], rk: [88], la: [60], ra: [60],
                lh: [0.45, 0.2, 0.35], rh: [-0.45, 0.2, 0.35], lift: 0.1,
            },
            {
                t: 0.7, f: 3.0, body: [-12], middle: [-6], neck: [-16], lt: [4], rt: [4], lk: [90], rk: [90],
                la: [70], ra: [70], lh: [0.6, 0.0, 0.6], rh: [-0.6, 0.0, 0.6], e: 'inout',
            },
            {
                t: 1.5, f: 4.2, body: [-16], middle: [-8], neck: [-20], lt: [4], rt: [4], lk: [88], rk: [88],
                la: [70], ra: [70], lh: [0.5, 0.05, 0.95], rh: [-0.5, 0.05, 0.95], e: 'inout',
            },
            {
                t: 2.1, f: 4.3, body: [-13], middle: [-6], neck: [-18], lt: [4], rt: [4], lk: [88], rk: [88],
                la: [70], ra: [70], lh: [0.55, 0.1, 0.8], rh: [-0.55, 0.1, 0.8],
            },
            {
                t: 2.6, f: 4.3, body: [-16], middle: [-8], neck: [-22], lt: [4], rt: [4], lk: [88], rk: [88],
                la: [70], ra: [70], lh: [0.5, 0.05, 0.98], rh: [-0.5, 0.05, 0.98],
            },
            {
                t: 3.2, f: 4.3, body: [-14], middle: [-7], neck: [-20], lt: [4], rt: [4], lk: [88], rk: [88],
                la: [70], ra: [70], lh: [0.55, 0.1, 0.85], rh: [-0.55, 0.1, 0.85],
            },
        ],
    },
    jump_fist: {
        label: 'Jump and punch', kind: 'celebration', duration: 1.5, loop: true, plant: true, hands: 'lr',
        base: { lp: [1, -0.6, -0.3], rp: [-1, -0.3, -0.2] },
        keys: [
            { t: 0, hz: -0.05, lh: [0.2, 0.15, -0.05], rh: [-0.25, 0.1, 0.15] },
            { t: 0.3, hz: -0.32, body: [24], neck: [-10], lh: [0.22, 0.05, -0.15], rh: [-0.22, -0.05, -0.12], e: 'inout' },
            { t: 0.55, hz: 0.02, lift: 0.4, body: [-6], neck: [-20], lf: [0.12, 0.05, 0.12], rf: [-0.12, -0.05, 0.05], lh: [0.18, 0.2, 0.05], rh: [-0.18, 0.12, 1.1] },
            { t: 0.75, hz: -0.1, lift: 0.5, body: [-4], neck: [-22], lf: [0.12, 0.1, 0.3], rf: [-0.12, 0.0, 0.25], lh: [0.18, 0.22, 0.05], rh: [-0.2, 0.15, 1.08] },
            { t: 1.0, hz: -0.3, lift: 0, body: [18], neck: [-10], lh: [0.22, 0.1, -0.1], rh: [-0.3, 0.25, 0.4], e: 'inout' },
            { t: 1.3, hz: -0.05, body: [2], lh: [0.2, 0.15, -0.05], rh: [-0.25, 0.1, 0.15] },
        ],
    },
    team_huddle: {
        label: 'Team huddle', kind: 'celebration', duration: 1.0, loop: true, plant: true, hands: 'lr',
        base: {
            body: [30], middle: [12], neck: [-25], lp: [1, -0.2, 0.6], rp: [-1, -0.2, 0.6],
            lf: [0.2, 0.05, 0], rf: [-0.2, 0.05, 0],
        },
        keys: [
            { t: 0, hz: -0.14, lh: [0.62, 0.15, 0.42], rh: [-0.62, 0.15, 0.42] },
            { t: 0.5, hz: -0.03, lift: 0.12, lh: [0.6, 0.18, 0.48], rh: [-0.6, 0.18, 0.48] },
        ],
    },
    dance: {
        label: 'Dance', kind: 'celebration', duration: 2.0, loop: true, hands: 'lr',
        fn(t, v) {
            const beat = (t / 0.5) * Math.PI;
            const s = Math.sin(beat);
            const c = Math.cos(beat);
            const bar = Math.sin(t * Math.PI);
            ang(v, BODY, 4, 8 * s, 14 * bar);
            ang(v, MIDDLE, 2, -10 * s, -16 * bar);
            ang(v, NECK, -6 + 6 * Math.abs(c), 6 * s, 8 * bar);
            ang(v, L_THIGH, -18 - 12 * Math.max(0, s), -8);
            ang(v, R_THIGH, -18 - 12 * Math.max(0, -s), 8);
            ang(v, L_KNEE, 28 + 30 * Math.max(0, s));
            ang(v, R_KNEE, 28 + 30 * Math.max(0, -s));
            v[SIDE] = 0.12 * bar;
            v[LIFT] = 0.03 * Math.abs(s);
            setVec(v, LH, 0.3 + 0.12 * bar, 0.28 + 0.1 * c, 0.25 + 0.25 * Math.max(0, s));
            setVec(v, RH, -0.3 + 0.12 * bar, 0.28 - 0.1 * c, 0.25 + 0.25 * Math.max(0, -s));
            setVec(v, LP, 1, -0.5, -0.6);
            setVec(v, RP, -1, -0.5, -0.6);
        },
    },
    airplane: {
        label: 'Airplane', kind: 'celebration', duration: 6.0, loop: true,
        fn(t, v) {
            const r = 3.2;
            const turn = (t / 6.0) * Math.PI * 2;
            gait(v, t / 0.6, 0.55);
            v[FWD] = r * Math.sin(turn);
            v[SIDE] = r * (1 - Math.cos(turn));
            v[YAW] = (turn / DEG);
            // Lean into the turn with the wings tipped.
            ang(v, BODY, 10, 14, 0);
            ang(v, MIDDLE, 4, 6, 0);
            ang(v, NECK, -10, -10);
            ang(v, L_SHOULDER, 10, -88);
            ang(v, R_SHOULDER, 10, 88);
            ang(v, L_ELBOW, -6);
            ang(v, R_ELBOW, -6);
        },
    },
    backflip: {
        label: 'Backflip', kind: 'celebration', duration: 2.2, loop: false, footLocal: true, trunk: false, hands: 'lr',
        base: { lp: [1, -0.4, -0.2], rp: [-1, -0.4, -0.2] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.35, body: [25], lt: [-50], rt: [-50], lk: [80], rk: [80], la: [-25], ra: [-25], lh: [0.2, -0.35, -0.1], rh: [-0.2, -0.35, -0.1], e: 'inout' },
            { t: 0.55, body: [-15], lift: 0.25, la: [40], ra: [40], lh: [0.25, 0.1, 1.0], rh: [-0.25, 0.1, 1.0] },
            { t: 0.75, body: [-130], lift: 0.75, lt: [-100], rt: [-100], lk: [125], rk: [125], lh: [0.2, 0.35, 0.05], rh: [-0.2, 0.35, 0.05], e: 'linear' },
            { t: 0.95, body: [-250], lift: 0.65, lt: [-105], rt: [-105], lk: [125], rk: [125], lh: [0.2, 0.35, 0.05], rh: [-0.2, 0.35, 0.05], e: 'linear' },
            { t: 1.15, body: [-330], lift: 0.12, lt: [-40], rt: [-40], lk: [50], rk: [50], la: [20], ra: [20], lh: [0.45, 0.2, 0.6], rh: [-0.45, 0.2, 0.6] },
            { t: 1.3, body: [-340], lift: 0, lt: [-55], rt: [-55], lk: [85], rk: [85], la: [-30], ra: [-30], lh: [0.45, 0.3, 0.45], rh: [-0.45, 0.3, 0.45] },
            { t: 1.8, body: [-360], neck: [-15], lt: [-3], rt: [-3], lk: [4], rk: [4], la: [0], ra: [0], lh: [0.42, 0.08, 1.0], rh: [-0.42, 0.08, 1.0], e: 'inout' },
            { t: 2.2, body: [-360], neck: [-18], lh: [0.45, 0.08, 0.98], rh: [-0.45, 0.08, 0.98] },
        ],
    },
    salute: {
        label: 'Salute', kind: 'celebration', duration: 2.4, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.6, -0.5], rp: [-1, 0.1, 0.35], middle: [-4], neck: [-4] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R, rp: POLE_R },
            { t: 0.5, lh: [0.15, 0.0, -0.13], rh: [-0.11, 0.1, 0.71], neck: [-6] },
            { t: 2.4, lh: [0.15, 0.0, -0.13], rh: [-0.11, 0.1, 0.71], neck: [-6] },
        ],
    },
    shush: {
        label: 'Shush', kind: 'celebration', duration: 2.4, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.6, -0.5], rp: [-1, -0.2, -0.8] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.45, rh: [-0.01, 0.16, 0.56], lh: [0.28, 0.12, -0.12], body: [4], neck: [6], middle: [3] },
            { t: 2.4, rh: [-0.01, 0.16, 0.56], lh: [0.28, 0.12, -0.12], body: [4], neck: [8], middle: [3] },
        ],
    },
    heart_hands: {
        label: 'Heart hands', kind: 'celebration', duration: 2.6, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.1, -0.6], rp: [-1, -0.1, -0.6] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.5, lh: [0.06, HEART, 0.3], rh: [-0.06, HEART, 0.3], neck: [6], middle: [-3] },
            { t: 1.5, lh: [0.06, HEART + 0.03, 0.34], rh: [-0.06, HEART + 0.03, 0.34], neck: [4], middle: [-4] },
            { t: 2.6, lh: [0.06, HEART, 0.3], rh: [-0.06, HEART, 0.3], neck: [6], middle: [-3] },
        ],
    },
    robot_dance: {
        label: 'Robot dance', kind: 'celebration', duration: 2.4, loop: true, hands: 'lr', ease: 'snap',
        base: { lp: [1, 0, -0.2], rp: [-1, 0, -0.2], lk: [12], rk: [12], lt: [-6], rt: [-6] },
        keys: [
            { t: 0, lh: [0.38, 0.22, 0.42], rh: [-0.3, 0.28, 0.0], middle: [0, 0, 20], neck: [0, 0, -20] },
            { t: 0.6, lh: [0.3, 0.28, 0.0], rh: [-0.38, 0.22, 0.42], middle: [0, 0, -20], neck: [0, 0, 20] },
            { t: 1.2, lh: [0.45, 0.0, 0.5], rh: [-0.45, 0.0, 0.5], middle: [0, 0, 0], neck: [0, 0, 0], lp: [1, 0, -1], rp: [-1, 0, -1], lift: 0.0 },
            { t: 1.8, lh: [0.3, 0.32, 0.25], rh: [-0.3, 0.32, 0.25], middle: [0, 0, 0], neck: [12, 0, 0], body: [0, 0, 30] },
        ],
    },
    bow: {
        label: 'Bow', kind: 'celebration', duration: 2.6, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, 0.3, -0.3], rp: [-1, -0.2, -0.3] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.6, rh: [-0.05, 0.2, 0.05], lh: [0.05, -0.2, 0.0], rt: [10, 0, 0], lt: [-10, 0, 0], rk: [12], la: [0] },
            { t: 1.3, body: [42], middle: [10], neck: [20], rh: [-0.05, 0.2, 0.05], lh: [0.05, -0.2, 0.0], rt: [10], lt: [-10], rk: [12] },
            { t: 1.9, body: [42], middle: [10], neck: [20], rh: [-0.05, 0.2, 0.05], lh: [0.05, -0.2, 0.0], rt: [10], lt: [-10], rk: [12] },
            { t: 2.6, body: [0], middle: [0], neck: [-6], rh: [-0.05, 0.2, 0.05], lh: [0.05, -0.2, 0.0], rt: [10], lt: [-10], rk: [12] },
        ],
    },
    worm: {
        label: 'The worm', kind: 'celebration', duration: 4.0, loop: false, footLocal: true, trunk: false,
        fn(t, v) {
            const down = smoothstep(Math.min(1, t / 0.7));
            const wave = Math.max(0, t - 0.7);
            const p = wave * Math.PI * 2 * 1.1;
            const amp = Math.min(1, wave * 2);
            stance(v, t, 6);
            for (let c = 0; c < 42; c++) {
                v[c] *= 1 - down;
            }
            ang(v, BODY, 90 * down + amp * 14 * Math.sin(p), 0, 0);
            ang(v, MIDDLE, -amp * 22 * Math.sin(p - 0.9), 0, 0);
            ang(v, NECK, -25 * down - amp * 15 * Math.sin(p - 1.8), 0, 0);
            ang(v, L_THIGH, amp * 20 * Math.sin(p + 1.0), -4);
            ang(v, R_THIGH, amp * 20 * Math.sin(p + 1.0), 4);
            ang(v, L_KNEE, 10 * down + amp * (22 + 22 * Math.sin(p + 2.0)));
            ang(v, R_KNEE, 10 * down + amp * (22 + 22 * Math.sin(p + 2.0)));
            ang(v, L_ANKLE, 60 * down);
            ang(v, R_ANKLE, 60 * down);
            // Hands flat under the shoulders, push ups make the wave.
            ang(v, L_SHOULDER, -70 * down + amp * 10 * Math.sin(p), -12 * down);
            ang(v, R_SHOULDER, -70 * down + amp * 10 * Math.sin(p), 12 * down);
            ang(v, L_ELBOW, -60 * down - amp * 25 * Math.max(0, Math.sin(p)));
            ang(v, R_ELBOW, -60 * down - amp * 25 * Math.max(0, Math.sin(p)));
            v[FWD] = 0.9 * down + 0.35 * wave;
            v[LIFT] = 0.05 * amp * Math.max(0, Math.sin(p - 0.4));
        },
    },
    clap: {
        label: 'Clap', kind: 'celebration', duration: 0.6, loop: true, hands: 'lr',
        base: { lp: [1, -0.3, -0.7], rp: [-1, -0.3, -0.7], middle: [-2], neck: [-5] },
        keys: [
            { t: 0, lh: [0.03, 0.32, 0.36], rh: [-0.03, 0.32, 0.36] },
            { t: 0.3, lh: [0.24, 0.26, 0.4], rh: [-0.24, 0.26, 0.4] },
        ],
    },

    // Utility.
    idle: {
        label: 'Idle', kind: 'utility', duration: 6.0, loop: true,
        fn(t, v) {
            stance(v, t, 6);
        },
    },
    walk: {
        label: 'Walk', kind: 'utility', duration: 1.1, loop: true, speed: 1.4,
        fn(t, v) {
            gait(v, t / 1.1, 0);
        },
    },
    jog: {
        label: 'Jog', kind: 'utility', duration: 0.7, loop: true, speed: 3.4,
        fn(t, v) {
            gait(v, t / 0.7, 0.6);
        },
    },
    card_raise: {
        label: 'Card raised', kind: 'utility', duration: 2.0, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.6, -0.5], rp: [-1, 0.2, -0.4] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.25, rh: [-0.24, 0.3, 0.3], neck: [-4] },
            { t: 0.55, rh: [-0.16, 0.12, 1.1], neck: [-10], middle: [-4] },
            { t: 2.0, rh: [-0.16, 0.12, 1.1], neck: [-10], middle: [-4] },
        ],
    },
    point: {
        label: 'Point', kind: 'utility', duration: 1.6, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.6, -0.5], rp: [-1, 0, -0.2] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.4, rh: [-0.42, 0.55, 0.52], neck: [0, 0, -14], middle: [0, 0, -8] },
            { t: 1.6, rh: [-0.42, 0.55, 0.52], neck: [0, 0, -14], middle: [0, 0, -8] },
        ],
    },
    whistle: {
        label: 'Whistle', kind: 'utility', duration: 1.6, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.2, -0.3], rp: [-1, -0.3, -0.7] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R },
            { t: 0.3, rh: [-0.02, 0.16, 0.55], lh: [0.35, 0.5, 0.6], neck: [-6] },
            { t: 1.6, rh: [-0.02, 0.16, 0.55], lh: [0.35, 0.5, 0.62], neck: [-6] },
        ],
    },
    throw_in_hold: {
        label: 'Throw-in hold', kind: 'utility', duration: 1.2, loop: true, hands: 'lr',
        base: { lp: [1, -0.4, 0.5], rp: [-1, -0.4, 0.5], middle: [-8], neck: [-6], lt: [-4], rt: [6], lk: [6] },
        keys: [
            { t: 0, lh: [0.11, -0.12, 0.84], rh: [-0.11, -0.12, 0.84] },
            { t: 0.6, lh: [0.11, -0.1, 0.86], rh: [-0.11, -0.1, 0.86], middle: [-10] },
        ],
    },
    place_ball: {
        label: 'Place the ball', kind: 'utility', duration: 2.2, loop: false, plant: true, hands: 'lr', handFrame: 'ground', ease: 'inout',
        base: { lp: [1, 0.2, 0.2], rp: [-1, 0.2, 0.2], lf: [0.15, -0.05, 0], rf: [-0.15, 0.1, 0] },
        keys: [
            { t: 0, hz: 0, lh: [0.12, 0.2, 1.0], rh: [-0.12, 0.2, 1.0] },
            { t: 0.8, hz: -0.42, body: [38], middle: [12], neck: [-10], lh: [0.12, 0.42, 0.13], rh: [-0.12, 0.42, 0.13] },
            { t: 1.3, hz: -0.42, body: [38], middle: [12], neck: [-10], lh: [0.14, 0.42, 0.14], rh: [-0.14, 0.42, 0.14] },
            { t: 2.2, hz: 0, body: [2], lh: [0.2, 0.05, 0.8], rh: [-0.2, 0.05, 0.8] },
        ],
    },
    hands_on_head: {
        label: 'Hands on head', kind: 'utility', duration: 2.0, loop: false, hands: 'lr', ease: 'inout',
        base: { lp: [1, -0.1, 0.4], rp: [-1, -0.1, 0.4] },
        keys: [
            { t: 0, lh: HAND_REST_L, rh: HAND_REST_R, lp: POLE_L, rp: POLE_R },
            { t: 0.45, lh: [0.08, 0.04, 0.78], rh: [-0.08, 0.04, 0.78], neck: [-12], body: [-3] },
            { t: 2.0, lh: [0.08, 0.04, 0.78], rh: [-0.08, 0.04, 0.78], neck: [-16], body: [-3] },
        ],
    },

    // Idles and entrances for the podium and the stage of the player of the match.
    idle_confident: {
        label: 'Hands on hips', kind: 'idle', duration: 4.0, loop: true, hands: 'lr',
        base: { lp: [1, -0.6, 0.1], rp: [-1, -0.6, 0.1] },
        fn(t, v) {
            stance(v, t, 4);
            ang(v, MIDDLE, -5 + Math.sin(t * Math.PI) * 1.2, 0, 0);
            ang(v, NECK, -6, 0, 6 * Math.sin(t * Math.PI / 2));
            ang(v, L_THIGH, -2, -8);
            ang(v, R_THIGH, -2, 8);
            setVec(v, LH, 0.22, 0.02, -0.04);
            setVec(v, RH, -0.22, 0.02, -0.04);
            setVec(v, LP, 1, -0.6, 0.1);
            setVec(v, RP, -1, -0.6, 0.1);
        },
    },
    idle_bounce: {
        label: 'Bouncing', kind: 'idle', duration: 4.0, loop: true,
        fn(t, v) {
            const p = (t / 0.8) * Math.PI * 2;
            stance(v, t, 4);
            v[LIFT] = 0.05 * Math.max(0, Math.sin(p));
            ang(v, L_KNEE, 14 - 10 * Math.max(0, Math.sin(p)));
            ang(v, R_KNEE, 14 - 10 * Math.max(0, Math.sin(p)));
            ang(v, L_THIGH, -6, -5);
            ang(v, R_THIGH, -6, 5);
            ang(v, L_ANKLE, 18 * Math.max(0, Math.sin(p)));
            ang(v, R_ANKLE, 18 * Math.max(0, Math.sin(p)));
            ang(v, L_SHOULDER, 4 * Math.sin(p * 2), -10);
            ang(v, R_SHOULDER, -4 * Math.sin(p * 2), 10);
            ang(v, L_ELBOW, -20 + 8 * Math.sin(p * 2));
            ang(v, R_ELBOW, -20 - 8 * Math.sin(p * 2));
        },
    },
    walkon_wave: {
        label: 'Walk on and wave', kind: 'walkon', duration: 2.2, loop: true, hands: 'r', speed: 1.4,
        fn(t, v) {
            gait(v, t / 1.1, 0);
            const w = Math.sin(t * Math.PI * 2 / 0.55);
            setVec(v, RH, -0.38 + 0.12 * w, 0.12, 0.9);
            setVec(v, RP, -1, 0.2, -0.2);
            addAng(v, NECK, 0, 0, -10);
        },
    },
};

// The animations by name, each with label, kind, duration in seconds, loop and speed in meters
// per second for the gaits.
export const ANIMATIONS = Object.fromEntries(Object.entries(LIBRARY).map(([name, a]) => [name, {
    label: a.label, kind: a.kind, duration: a.duration, loop: !!a.loop, speed: a.speed || 0,
}]));

export function names(kind = null) {
    return Object.keys(LIBRARY).filter((n) => !kind || LIBRARY[n].kind === kind);
}

export function has(name) {
    return Object.prototype.hasOwnProperty.call(LIBRARY, name);
}

export function duration(name) {
    return has(name) ? LIBRARY[name].duration : 0;
}

export function loops(name) {
    return has(name) && !!LIBRARY[name].loop;
}

// The ground speed that matches the gait, for callers that move the player.
export function speed(name) {
    return has(name) ? LIBRARY[name].speed || 0 : 0;
}

function channels(anim, t, out) {
    if (anim.keys) {
        sampleKeys(anim, t, out);
    } else {
        defaults(out);
        if (anim.base) {
            writeKey(anim.base, out);
        }
    }
    if (anim.fn) {
        anim.fn(t, out);
    }
    return out;
}

// Writes one player's nodes for animation name at time t seconds, standing at x, y of the
// engine frame and facing heading, radians about engine z with 0 facing engine minus y. Root
// motion of the animation moves along that heading. Returns out, or null for an unknown name.
export function pose(name, t, out = new Float32Array(PLAYER_FLOATS), opts = {}) {
    const anim = LIBRARY[name];
    if (!anim) {
        return null;
    }
    channels(anim, Math.max(0, t), sample);
    return solve(sample, anim, opts, out, opts.offset || 0);
}

// Blends two animations, w from 0 for the first alone to 1 for the second alone, a crossfade.
export function poseBlend(a, ta, b, tb, w, out = new Float32Array(PLAYER_FLOATS), opts = {}) {
    const A = LIBRARY[a];
    const B = LIBRARY[b];
    if (!A || !B) {
        return pose(A ? a : b, A ? ta : tb, out, opts);
    }
    channels(A, ta, scratchA);
    channels(B, tb, scratchB);
    for (let c = 0; c < CHANNELS; c++) {
        scratchC[c] = scratchA[c] + (scratchB[c] - scratchA[c]) * w;
    }
    // The flags of the animation that weighs more decide the solve.
    return solve(scratchC, w < 0.5 ? A : B, opts, out, opts.offset || 0);
}

// Blends two node buffers of players, positions linearly and rotations spherically, w of b.
export function blendNodes(a, b, w, out, offset = 0, count = 1) {
    const n = count * PLAYER_FLOATS;
    for (let o = offset; o < offset + n; o += 7) {
        out[o] = a[o] + (b[o] - a[o]) * w;
        out[o + 1] = a[o + 1] + (b[o + 1] - a[o + 1]) * w;
        out[o + 2] = a[o + 2] + (b[o + 2] - a[o + 2]) * w;
        qa.set(a[o + 3], a[o + 4], a[o + 5], a[o + 6]);
        qb.set(b[o + 3], b[o + 4], b[o + 5], b[o + 6]);
        if (qa.lengthSq() < 1e-8) {
            qa.copy(qb);
        }
        if (qb.lengthSq() < 1e-8) {
            qb.copy(qa);
        }
        qa.slerp(qb, w);
        out[o + 3] = qa.x;
        out[o + 4] = qa.y;
        out[o + 5] = qa.z;
        out[o + 6] = qa.w;
    }
    return out;
}

// The hands of a posed player in the engine frame, left then right, from the elbows of any
// pose buffer, a match frame included. base is the player's first float.
export function handPoints(nodes, base = 0, out = new Float32Array(6)) {
    for (let s = 0; s < 2; s++) {
        const o = base + (s ? R_ELBOW : L_ELBOW) * 7;
        qa.set(nodes[o + 3], nodes[o + 4], nodes[o + 5], nodes[o + 6]);
        va.set(0, 0, -FOREARM).applyQuaternion(qa);
        out[s * 3] = nodes[o] + va.x;
        out[s * 3 + 1] = nodes[o + 1] + va.y;
        out[s * 3 + 2] = nodes[o + 2] + va.z;
    }
    return out;
}

const hands = new Float32Array(6);

// Where a ball held in both hands sits, between the hands, for throw ins and placing the ball.
export function holdPoint(nodes, base = 0, out = new THREE.Vector3()) {
    handPoints(nodes, base, hands);
    return out.set((hands[0] + hands[3]) / 2, (hands[1] + hands[4]) / 2, (hands[2] + hands[5]) / 2);
}

// The heading that faces from a point towards another, both in the engine frame.
export function headingTo(dx, dy) {
    return Math.atan2(dx, -dy);
}

function hash(a, b) {
    let h = Math.imul((a | 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((b | 0) + 0x632be5ab, 0xc2b2ae35);
    h ^= h >>> 15;
    h = Math.imul(h, 0x2c1b3c6d);
    return (h ^ (h >>> 13)) >>> 0;
}

// The celebration a player performs after a goal, its own choice, else one of its team's, else
// one of every celebration the library holds, picked by seed so a replay shows the same one.
// Clip names come back as they are, the caller plays them where the model has that clip.
export function pickCelebration(style, index, seed = 0) {
    const roster = style && style.players ? style.players[index] : null;
    if (roster && roster.celebration && roster.celebration !== 'team') {
        return roster.celebration;
    }
    let list = style && Array.isArray(style.celebrations) ? style.celebrations : [];
    if (!list.length) {
        list = names('celebration');
    }
    return list[hash(seed, index) % list.length];
}

// The procedural stand in for a celebration name the library lacks, a clip on a model without
// it among them.
export function fallbackCelebration(name, seed = 0) {
    if (has(name)) {
        return name;
    }
    const list = names('celebration');
    return list[hash(seed, name ? name.length : 0) % list.length];
}

