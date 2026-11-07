// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Restarts as on television. GRF moves every player to the spots of a restart within one step,
// right after the ball leaves the pitch. The viewer holds play on the last frame before that step
// while the ball runs on off the pitch and the players slow down and walk towards their spots, a
// logo wipe hides the step, the camera cuts to a view of the restart, the taker sets the ball, and
// play goes on. Half time gets a title card, and fouls, cards and offsides get the referee's
// replay from decisions.js before the restart.

import * as THREE from 'three';
import { NODE_COUNT, NODE_FLOATS, NODE_PARENTS, NODES_FLOAT, PLAYER_FLOATS, episodeBoundary, isCut } from './format.js';
import { teamNames } from './hud.js';
import { BALL_RADIUS, FULL_HALF_H, FULL_HALF_W, GOAL_DEPTH, GOAL_HALF_WIDTH, GOAL_HEIGHT, HALF_H, HALF_W, POST_RADIUS } from './pitch.js';
import * as figures from './players.js';
import * as timeline from './timeline.js';

export const HALF_TIME_MS = 45 * 60 * 1000;
// Seconds of the logo wipe, of the walk before it, of the walk before a goal's wipe and of the
// taker setting the ball after it, all in match time, so the playback speed scales them.
export const WIPE = 0.72;
const STAGE = 1.35;
const GOAL_STAGE = 1.8;
// The pre-roll after a restart is revealed, between these many seconds, before the first step.
const PREROLL_MIN = 1.5;
const PREROLL_MAX = 3;
const TITLE = 3.6;
// The farthest a player walks towards a new spot before the wipe hides the rest.
const PULL = 2.6;
const TAKER_PULL = 4.5;
// A player or the ball that moves farther than this in one step was moved by the engine.
const JUMP = 3;
const BOARD_X = FULL_HALF_W + 1.5;
const BOARD_Y = FULL_HALF_H + 1.5;
const BOARD_H = 0.9;
const BALL_RATE = 120;
const BALL_SECONDS = 6;

export const RESTART_NAMES = {
    kickoff: 'Kick-off', goal_kick: 'Goal kick', free_kick: 'Free kick', corner: 'Corner',
    throw_in: 'Throw-in', penalty: 'Penalty', halftime: 'Half time',
};
const MODE_KINDS = [null, 'kickoff', 'goal_kick', 'free_kick', 'corner', 'throw_in', 'penalty'];

// The animation library of the figures may arrive later, the local poses stand in until then.
let animate = null;
import('./animate.js').then((m) => {
    animate = m;
}).catch(() => {});

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const smooth = (x) => {
    const t = clamp(x, 0, 1);
    return t * t * (3 - 2 * t);
};

// Poses.

// Local rest positions of GRF's player skeleton, as players.js keeps them.
const REST = [
    [0, 0, 0], [0, 0, 0.96], [0, 0, 0.15], [0, -0.03, 0.5], [0.16, -0.01, 0.48], [-0.01, 0, -0.33],
    [-0.16, -0.01, 0.48], [0.01, 0, -0.33], [0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
    [-0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
];
const ANKLE_Z = 0.09;
const FOREARM = 0.29;

// Local joint turns of the poses that stand in for the animation library, each joint as turns
// about its own x and y axes. A turn about x below zero swings a limb forward.
const POSES = {
    idle: { 4: [-0.05, -0.12], 6: [-0.05, 0.12], 5: [-0.3], 7: [-0.3], 9: [0.08], 12: [0.08] },
    throw_in_hold: { 2: [-0.16], 3: [0.12], 4: [-2.75, -0.2], 6: [-2.75, 0.2], 5: [1.05], 7: [1.05], 9: [0.1], 12: [0.1] },
    place_ball: {
        1: [0.3], 2: [0.7], 3: [-0.45], 4: [-1.0, -0.12], 6: [-1.0, 0.12], 5: [-0.25], 7: [-0.25],
        8: [-1.05], 9: [1.45], 10: [-0.35], 11: [-0.7], 12: [1.2], 13: [-0.45],
    },
    point: { 4: [-0.05, -0.12], 5: [-0.25], 6: [-1.5, 0.1], 7: [-0.05], 9: [0.05], 12: [0.05] },
    whistle: { 3: [0.1], 4: [-0.05, -0.12], 5: [-0.25], 6: [-1.05, 0.5], 7: [-2.25], 9: [0.05], 12: [0.05] },
    card_raise: { 2: [-0.05], 4: [-0.05, -0.12], 5: [-0.25], 6: [-2.95, 0.1], 7: [-0.08], 9: [0.05], 12: [0.05] },
    hands_on_head: { 2: [0.15], 3: [0.25], 4: [-2.45, -0.6], 6: [-2.45, 0.6], 5: [-2.2], 7: [-2.2], 9: [0.12], 12: [0.12] },
};

const wq = Array.from({ length: NODE_COUNT }, () => new THREE.Quaternion());
const wp = Array.from({ length: NODE_COUNT }, () => new THREE.Vector3());
const lq = new THREE.Quaternion();
const eul = new THREE.Euler();
const tv = new THREE.Vector3();
const ZAXIS = new THREE.Vector3(0, 0, 1);
const NO_TURN = [0, 0];

// Writes one player's nodes into out at offset o from local joint turns, standing at x, y and
// facing heading, where heading zero faces engine minus y. With ground set, the lower ankle rests
// on the grass, so bent legs lower the body. Lift raises or lowers the body as a walk bobs.
export function fkPose(turns, heading, x, y, out, o = 0, ground = true, lift = 0) {
    wq[0].setFromAxisAngle(ZAXIS, heading);
    wp[0].set(x, y, 0);
    for (let i = 1; i < NODE_COUNT; i++) {
        const a = turns[i] || NO_TURN;
        eul.set(a[0] || 0, a[1] || 0, a[2] || 0, 'XYZ');
        lq.setFromEuler(eul);
        const p = NODE_PARENTS[i];
        tv.set(REST[i][0], REST[i][1], REST[i][2] + (i === 1 ? lift : 0)).applyQuaternion(wq[p]);
        wp[i].copy(wp[p]).add(tv);
        wq[i].copy(wq[p]).multiply(lq);
    }
    const dz = ground ? ANKLE_Z - Math.min(wp[10].z, wp[13].z) : 0;
    for (let i = 0; i < NODE_COUNT; i++) {
        const k = o + i * NODE_FLOATS;
        out[k] = wp[i].x;
        out[k + 1] = wp[i].y;
        out[k + 2] = i === 0 ? 0 : wp[i].z + dz;
        out[k + 3] = wq[i].x;
        out[k + 4] = wq[i].y;
        out[k + 5] = wq[i].z;
        out[k + 6] = wq[i].w;
    }
    return out;
}

const walkTurns = Array.from({ length: NODE_COUNT }, () => [0, 0]);

// A walk or jog at speed meters per second, phase in radians, as players.js draws its walk.
export function walkPose(phase, speed, heading, x, y, out, o = 0) {
    const stride = Math.min(1, speed / 6);
    const s = Math.sin(phase);
    const c = Math.cos(phase);
    const bob = 0.035 * stride * Math.abs(c);
    const t = walkTurns;
    t[1][0] = 0.05 * stride;
    t[2][0] = 0.1 * stride;
    t[3][0] = -0.1 * stride;
    t[4][0] = 0.55 * stride * s;
    t[4][1] = -0.1;
    t[5][0] = -0.45 - 0.6 * stride;
    t[6][0] = -0.55 * stride * s;
    t[6][1] = 0.1;
    t[7][0] = -0.45 - 0.6 * stride;
    t[8][0] = -0.6 * stride * s;
    t[9][0] = 0.1 + 1.0 * stride * Math.max(0, -c);
    t[10][0] = -0.2 * stride * s;
    t[11][0] = 0.6 * stride * s;
    t[12][0] = 0.1 + 1.0 * stride * Math.max(0, c);
    t[13][0] = 0.2 * stride * s;
    return fkPose(t, heading, x, y, out, o, false, -bob - 0.02 * stride);
}

// A named pose, from the animation library when it has one, else from the stand ins above.
export function namedPose(name, time, heading, x, y, out, o = 0) {
    if (animate && typeof animate.pose === 'function' && (typeof animate.has !== 'function' || animate.has(name))) {
        try {
            const tmp = namedPose.tmp || (namedPose.tmp = new Float32Array(PLAYER_FLOATS));
            if (animate.pose(name, time, tmp, { heading, x, y })) {
                out.set(tmp, o);
                return out;
            }
        } catch {
            // A pose the library does not know falls back to the stand in.
        }
    }
    return fkPose(POSES[name] || POSES.idle, heading, x, y, out, o, true);
}

// Blends two poses of one player, positions linearly and rotations by normalized lerp.
export function blendPose(a, ao, b, bo, w, out, oo) {
    for (let i = 0; i < NODE_COUNT; i++) {
        const pa = ao + i * NODE_FLOATS;
        const pb = bo + i * NODE_FLOATS;
        const po = oo + i * NODE_FLOATS;
        out[po] = a[pa] + (b[pb] - a[pa]) * w;
        out[po + 1] = a[pa + 1] + (b[pb + 1] - a[pa + 1]) * w;
        out[po + 2] = a[pa + 2] + (b[pb + 2] - a[pa + 2]) * w;
        const dot = a[pa + 3] * b[pb + 3] + a[pa + 4] * b[pb + 4] + a[pa + 5] * b[pb + 5] + a[pa + 6] * b[pb + 6];
        const sb = dot < 0 ? -w : w;
        const x = a[pa + 3] * (1 - w) + b[pb + 3] * sb;
        const y = a[pa + 4] * (1 - w) + b[pb + 4] * sb;
        const z = a[pa + 5] * (1 - w) + b[pb + 5] * sb;
        const q = a[pa + 6] * (1 - w) + b[pb + 6] * sb;
        const n = Math.hypot(x, y, z, q) || 1;
        out[po + 3] = x / n;
        out[po + 4] = y / n;
        out[po + 5] = z / n;
        out[po + 6] = q / n;
    }
}

// The heading a player faces, read from the body node, since GRF keeps the root node turned the
// same way for a whole team.
export function headingOf(nodes, o) {
    const b = o + NODE_FLOATS;
    const x = nodes[b + 3];
    const y = nodes[b + 4];
    const z = nodes[b + 5];
    const w = nodes[b + 6];
    // The body's minus y axis, turned by its quaternion.
    const fx = -(2 * (x * y - w * z));
    const fy = -(1 - 2 * (x * x + z * z));
    if (Math.abs(fx) + Math.abs(fy) < 1e-6) {
        return 0;
    }
    return Math.atan2(fx, -fy);
}

export function headingTo(dx, dy) {
    return Math.atan2(dx, -dy);
}

export function lerpAngle(a, b, w) {
    let d = b - a;
    while (d > Math.PI) {
        d -= 2 * Math.PI;
    }
    while (d < -Math.PI) {
        d += 2 * Math.PI;
    }
    return a + d * w;
}

// A hand of a pose, the end of the forearm below the elbow node, side 0 left and 1 right.
const hq = new THREE.Quaternion();
export function handOf(nodes, o, side, out) {
    const e = o + (side ? 7 : 5) * NODE_FLOATS;
    hq.set(nodes[e + 3], nodes[e + 4], nodes[e + 5], nodes[e + 6]);
    out.set(0, 0, -FOREARM).applyQuaternion(hq);
    out.x += nodes[e];
    out.y += nodes[e + 1];
    out.z += nodes[e + 2];
    return out;
}

// Restarts in a store.

function rootJumps(a, b, players) {
    let count = 0;
    for (let k = 0; k < players; k++) {
        if (!(a.playerFlags[k] & 1) || !(b.playerFlags[k] & 1)) {
            continue;
        }
        const o = NODES_FLOAT + k * PLAYER_FLOATS;
        if (Math.hypot(b.data[o] - a.data[o], b.data[o + 1] - a.data[o + 1]) > JUMP) {
            count++;
        }
    }
    return count;
}

// The referee's call behind a restart at index i, a foul that started within the three seconds
// before it, or an offside, which is a free kick without a foul while an attacker stands offside.
function findDecision(store, header, i, teleport, mode) {
    for (let j = i + 1; j >= i - 30; j--) {
        const f = store.get(j);
        const g = store.get(j - 1);
        if (!f || !g || isCut(g, f)) {
            if (j <= i) {
                break;
            }
            continue;
        }
        if (f.foul.type > 0 && g.foul.type === 0) {
            const off = f.foul.player;
            const secondYellow = f.foul.type === 2 && off >= 0 && !!(g.playerFlags[off] & 2);
            const type = f.foul.type === 3 || secondYellow ? 'red' : (f.foul.type === 2 ? 'yellow' : 'foul');
            return {
                type,
                offender: off,
                victim: f.foul.victim,
                call: j,
                contact: j < i ? j : (teleport ? i - 1 : i),
                secondYellow,
            };
        }
    }
    if (mode !== 3) {
        return null;
    }
    let offender = -1;
    for (const j of [i, i - 1, i + 1]) {
        const f = store.get(j);
        if (f && f.offside >= 0 && f.foul.type === 0 && (!header.offside_called || (f.flags & 4))) {
            offender = f.offside;
            break;
        }
    }
    if (offender < 0 || offender >= header.players.length) {
        return null;
    }
    // The pass. The referee marks the offside position from the pass until the call, so the
    // pass is where that mark begins, else the last touch of a teammate before the call.
    const team = header.players[offender].team;
    let pass = -1;
    let passer = -1;
    for (let j = i - 1; j >= Math.max(store.first + 1, i - 80); j--) {
        const f = store.get(j);
        const g = store.get(j - 1);
        if (!f || f.offside !== offender) {
            break;
        }
        pass = j;
        if (!g || isCut(g, f)) {
            break;
        }
    }
    if (pass >= 0) {
        const f = store.get(pass);
        for (let k = 0; k < header.players.length; k++) {
            if (k !== offender && header.players[k].team === team && f.touches[k]) {
                passer = k;
            }
        }
    }
    for (let j = i; pass < 0 && j >= Math.max(store.first + 1, i - 80); j--) {
        const f = store.get(j);
        const g = store.get(j - 1);
        if (!f || !g || isCut(g, f)) {
            break;
        }
        for (let k = 0; k < header.players.length; k++) {
            if (k !== offender && header.players[k].team === team && f.touches[k]) {
                pass = j;
                passer = k;
                break;
            }
        }
    }
    // The frame the engine moves things on shows the restart, so the pass is shown before it.
    const contact = pass >= 0 && !(teleport && pass >= i) ? pass : i - 1;
    return { type: 'offside', offender, victim: -1, passer, call: i, contact };
}

// Appends the restarts between frames from and to, each where the engine moves players or the
// ball to a restart, or where the referee stops play. Plain changes of game mode without a
// move need no transition and are left out.
export function scanRestarts(store, header, from, to, list) {
    const players = header.players.length;
    for (let i = Math.max(from, store.first + 1); i <= to; i++) {
        const a = store.get(i - 1);
        const b = store.get(i);
        if (!a || !b || isCut(a, b)) {
            // A new episode opens with its kick-off, which gets a pre-roll of its own.
            if (a && b && episodeBoundary(a, b)) {
                list.push(startEvent(i, b));
            }
            continue;
        }
        const last = list[list.length - 1];
        if (last && i - last.index <= 2 && i > last.index) {
            continue;
        }
        const next = store.get(i + 1);
        const nextOk = next && !isCut(b, next);
        const prev = store.get(i - 2);
        const prevOk = prev && !isCut(prev, a);
        const jumps = rootJumps(a, b, players);
        const px = prevOk ? 2 * a.data[8] - prev.data[8] : a.data[8];
        const py = prevOk ? 2 * a.data[9] - prev.data[9] : a.data[9];
        const deviation = Math.hypot(b.data[8] - px, b.data[9] - py);
        const goal = b.scoreLeft !== a.scoreLeft || b.scoreRight !== a.scoreRight;
        let mode = b.gameMode !== a.gameMode && b.gameMode > 0 ? b.gameMode : 0;
        if (!mode && nextOk && a.gameMode === 0 && b.gameMode === 0 && next.gameMode > 0 && (jumps || deviation > JUMP)) {
            // Some sources write the new game mode one step after the move.
            mode = next.gameMode;
        }
        const teleport = jumps >= 3 || ((jumps >= 1 || deviation > JUMP) && (mode > 0 || goal));
        if (!teleport && !mode) {
            continue;
        }
        const decision = findDecision(store, header, i, teleport, mode);
        if (!teleport && !decision) {
            continue;
        }
        list.push({
            index: i,
            mode,
            kind: MODE_KINDS[mode] || 'kickoff',
            teleport,
            goal,
            goalTeam: goal ? (b.scoreLeft > a.scoreLeft ? 0 : 1) : -1,
            decision,
            half: false,
            taker: -1,
        });
    }
    return list;
}

// The kick-off that opens an episode. Nothing moves into it, so it has no wipe, only a pre-roll.
export function startEvent(i, f) {
    return {
        index: i, mode: f.gameMode, kind: 'kickoff', teleport: false, start: true, goal: false,
        goalTeam: -1, decision: null, half: false, taker: -1,
    };
}

// A number from 0 to 1 that stays the same for the same seeds, for looks that replay alike.
export function hash01(a, b = 0) {
    const v = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
    return v - Math.floor(v);
}

// The player who takes a restart, the first owner or toucher after it.
function takerOf(store, header, ev) {
    if (ev.taker >= 0) {
        return ev.taker;
    }
    for (let j = ev.index; j <= ev.index + 25; j++) {
        const f = store.get(j);
        if (!f) {
            break;
        }
        if (f.ownedPlayer >= 0) {
            ev.taker = f.ownedPlayer;
            return ev.taker;
        }
        for (let k = 0; k < header.players.length; k++) {
            if (f.touches[k]) {
                ev.taker = k;
                return ev.taker;
            }
        }
    }
    const f = store.get(ev.index);
    let best = -1;
    let bestD = Infinity;
    for (let k = 0; f && k < header.players.length; k++) {
        const o = NODES_FLOAT + k * PLAYER_FLOATS;
        const d = Math.hypot(f.data[o] - f.data[8], f.data[o + 1] - f.data[9]);
        if ((f.playerFlags[k] & 1) && d < bestD) {
            bestD = d;
            best = k;
        }
    }
    return best;
}

// The ball after it leaves play. A short simulation from its last position and velocity, with
// gravity, bounces, rolling on the grass, the advertising boards and the goal nets, sampled at
// BALL_RATE per second with the distance rolled for its spin.
export function simulateBall(px, py, pz, vx, vy, vz, goal) {
    const steps = BALL_SECONDS * BALL_RATE;
    const out = new Float32Array((steps + 1) * 4);
    const r = BALL_RADIUS;
    const h = 1 / BALL_RATE;
    let x = px;
    let y = py;
    let z = Math.max(r, pz);
    let dist = 0;
    let inNet = false;
    let stopped = false;
    // Touches of the net, each with its time, goal, point in three's frame and strength.
    const impacts = [];
    const touchNet = (s, side, px2, py2, pz2, speed) => {
        const last = impacts[impacts.length - 1];
        if (speed > 0.8 && (!last || s / BALL_RATE - last.time > 0.25)) {
            impacts.push({ time: s / BALL_RATE, side: side > 0 ? 1 : 0, point: [px2, pz2, -py2], strength: clamp(speed / 22, 0.1, 1) });
        }
    };
    const hit = POST_RADIUS + r;
    // Knocks on the posts and the bar, for their sound.
    const knocks = [];
    const knock = (s, px2, py2, pz2, speed) => {
        const last = knocks[knocks.length - 1];
        if (speed > 1 && (!last || s / BALL_RATE - last.time > 0.2)) {
            knocks.push({ time: s / BALL_RATE, point: [px2, pz2, -py2], strength: clamp(speed / 20, 0.15, 1) });
        }
    };
    for (let s = 0; s <= steps; s++) {
        out[s * 4] = x;
        out[s * 4 + 1] = y;
        out[s * 4 + 2] = z;
        out[s * 4 + 3] = dist;
        if (stopped) {
            continue;
        }
        vz -= 9.81 * h;
        const drag = 1 - 0.012 * h * Math.hypot(vx, vy, vz);
        vx *= drag;
        vy *= drag;
        let nx = x + vx * h;
        let ny = y + vy * h;
        let nz = z + vz * h;
        // The posts and the crossbar send the ball back, a goal from the frames always goes in.
        if (!goal && Math.abs(Math.abs(nx) - HALF_W) < hit + 0.05) {
            const gx = (Math.sign(nx) || 1) * HALF_W;
            for (const py0 of [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH]) {
                const dx = nx - gx;
                const dy = ny - py0;
                const d = Math.hypot(dx, dy);
                if (d < hit && nz < GOAL_HEIGHT + POST_RADIUS && d > 1e-6) {
                    const ux = dx / d;
                    const uy = dy / d;
                    const vn = vx * ux + vy * uy;
                    if (vn < 0) {
                        knock(s, gx, py0, nz, -vn);
                        vx -= 1.6 * vn * ux;
                        vy -= 1.6 * vn * uy;
                    }
                    nx = gx + ux * hit;
                    ny = py0 + uy * hit;
                }
            }
            if (Math.abs(ny) < GOAL_HALF_WIDTH) {
                const dx = nx - gx;
                const dz = nz - GOAL_HEIGHT;
                const d = Math.hypot(dx, dz);
                if (d < hit && d > 1e-6) {
                    const ux = dx / d;
                    const uz = dz / d;
                    const vn = vx * ux + vz * uz;
                    if (vn < 0) {
                        knock(s, gx, ny, GOAL_HEIGHT, -vn);
                        vx -= 1.6 * vn * ux;
                        vz -= 1.6 * vn * uz;
                    }
                    nx = gx + ux * hit;
                    nz = GOAL_HEIGHT + uz * hit;
                }
            }
        }
        if (!inNet && Math.abs(nx) > HALF_W && Math.abs(x) <= HALF_W) {
            if (goal || (Math.abs(ny) < GOAL_HALF_WIDTH - r && nz < GOAL_HEIGHT - r)) {
                inNet = true;
                ny = clamp(ny, -GOAL_HALF_WIDTH + r + 0.15, GOAL_HALF_WIDTH - r - 0.15);
                nz = Math.min(nz, GOAL_HEIGHT - r - 0.1);
            }
        }
        if (inNet) {
            const side = Math.sign(nx) || 1;
            const back = HALF_W + GOAL_DEPTH * 0.92 - r;
            if (Math.abs(nx) > back) {
                // The net takes the pace off the ball and bulges where it was hit.
                touchNet(s, side, side * (back + r), ny, nz, Math.hypot(vx, vy, vz));
                nx = side * back;
                vx = -vx * 0.1;
                vy *= 0.5;
                vz *= 0.4;
            }
            if (Math.abs(ny) > GOAL_HALF_WIDTH - r) {
                touchNet(s, side, nx, Math.sign(ny) * GOAL_HALF_WIDTH, nz, Math.abs(vy) * 1.5);
                ny = Math.sign(ny) * (GOAL_HALF_WIDTH - r);
                vy = -vy * 0.2;
                vx *= 0.7;
            }
            const roof = GOAL_HEIGHT * (1 - 0.15 * clamp((Math.abs(nx) - HALF_W) / GOAL_DEPTH, 0, 1)) - r;
            if (nz > roof) {
                touchNet(s, side, nx, ny, roof + r, Math.abs(vz) * 1.5);
                nz = roof;
                vz = Math.min(0, -vz * 0.2);
            }
        } else {
            if (Math.abs(nx) > BOARD_X - r - 0.1 && Math.abs(x) <= BOARD_X - r - 0.1 && nz < BOARD_H) {
                nx = Math.sign(nx) * (BOARD_X - r - 0.1);
                vx = -vx * 0.4;
                vy *= 0.8;
            }
            if (Math.abs(ny) > BOARD_Y - r - 0.1 && Math.abs(y) <= BOARD_Y - r - 0.1 && nz < BOARD_H) {
                ny = Math.sign(ny) * (BOARD_Y - r - 0.1);
                vy = -vy * 0.4;
                vx *= 0.8;
            }
            // Over the boards the ball drops among the photographers and stays.
            if (Math.abs(nx) > BOARD_X + 3 || Math.abs(ny) > BOARD_Y + 3) {
                nx = clamp(nx, -BOARD_X - 3, BOARD_X + 3);
                ny = clamp(ny, -BOARD_Y - 3, BOARD_Y + 3);
                vx = 0;
                vy = 0;
            }
        }
        if (nz < r) {
            nz = r;
            if (vz < -0.8) {
                vz = -vz * 0.5;
                vx *= 0.85;
                vy *= 0.85;
            } else {
                vz = 0;
                const sp = Math.hypot(vx, vy);
                const slow = Math.max(0, sp - (inNet ? 6 : 2.2) * h);
                const k = sp > 1e-6 ? slow / sp : 0;
                vx *= k;
                vy *= k;
                if (slow === 0) {
                    stopped = true;
                }
            }
        }
        dist += Math.hypot(nx - x, ny - y);
        x = nx;
        y = ny;
        z = nz;
    }
    out.impacts = impacts;
    out.knocks = knocks;
    return out;
}

function sampleBall(track, time, out) {
    const steps = track.length / 4 - 1;
    const f = clamp(time * BALL_RATE, 0, steps);
    const i = Math.min(steps - 1, Math.floor(f));
    const w = f - i;
    for (let c = 0; c < 4; c++) {
        out[c] = track[i * 4 + c] + (track[(i + 1) * 4 + c] - track[i * 4 + c]) * w;
    }
    return out;
}

// The overlays, built here so the page's markup stays with its owner.

function el(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) {
        e.className = cls;
    }
    if (text != null) {
        e.textContent = text;
    }
    if (parent) {
        parent.appendChild(e);
    }
    return e;
}

export function rgb(c, alpha = 1) {
    const [r, g, b] = Array.isArray(c) ? c : [128, 128, 128];
    return alpha === 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}


// Shows the pointer for a moment that asks for a click or Enter, out of any pointer lock.
export function showPointer(on) {
    if (on && document.pointerLockElement && document.exitPointerLock) {
        document.exitPointerLock();
    }
    const app = document.getElementById('app');
    if (app) {
        app.classList.toggle('show-pointer', !!on);
    }
}

class Wipe {
    constructor(watch, hud) {
        this.root = el('div', 'restart-wipe');
        this.root.setAttribute('aria-hidden', 'true');
        this.bands = [el('div', 'wipe-band lead', this.root), el('div', 'wipe-band mid', this.root), el('div', 'wipe-band trail', this.root)];
        this.plate = el('div', 'wipe-plate', this.bands[1]);
        const img = el('img', 'wipe-logo', this.plate);
        img.src = 'assets/cuju-logo.png';
        img.alt = '';
        this.label = el('div', 'wipe-label', this.plate);
        watch.insertBefore(this.root, hud);
        this.shown = false;
    }

    setTeams(styles) {
        const c = (s, fallback) => (s && s.kit ? rgb(s.kit.primary) : fallback);
        this.bands[0].style.background = c(styles && styles[0], '#b3262e');
        this.bands[2].style.background = c(styles && styles[1], '#2a56b8');
    }

    // Progress p from 0 to 1, the screen is fully covered from about 0.3 to 0.7.
    show(p, label) {
        if (p == null || p <= 0 || p >= 1) {
            if (this.shown) {
                this.root.classList.remove('show');
                this.shown = false;
            }
            return;
        }
        if (!this.shown) {
            this.root.classList.add('show');
            this.shown = true;
        }
        if (label != null && this.label.textContent !== label) {
            this.label.textContent = label;
        }
        const w = this.root.clientWidth || window.innerWidth;
        // The team bands lead and trail the middle band.
        const lags = [0.07, 0, -0.07];
        let mid = 0;
        this.bands.forEach((band, i) => {
            const q = clamp(p + lags[i], 0, 1);
            let c;
            // A band covers the screen while its centre is within a fifth of the width.
            if (q < 0.33) {
                c = 1.6 - 1.45 * smooth(q / 0.33);
            } else if (q < 0.67) {
                c = 0.15 - 0.3 * ((q - 0.33) / 0.34);
            } else {
                c = -0.15 - 1.45 * smooth((q - 0.67) / 0.33);
            }
            band.style.transform = `translateX(${(c * w).toFixed(1)}px) skewX(-14deg)`;
            if (i === 1) {
                mid = c;
            }
        });
        const vis = smooth((p - 0.24) / 0.1) * (1 - smooth((p - 0.66) / 0.1));
        this.plate.style.opacity = vis.toFixed(3);
        // The logo rides the band in and out but holds near the middle while it covers.
        const hold = -mid * w * 0.85;
        this.plate.style.transform = `translate(calc(-50% + ${hold.toFixed(1)}px), -50%) skewX(14deg) scale(${(0.86 + 0.2 * p).toFixed(3)})`;
    }
}

// The title card of half time and full time.
class TitleCard {
    constructor(watch) {
        this.root = el('div', 'title-card', watch);
        this.root.hidden = true;
        this.root.setAttribute('role', 'status');
        const inner = el('div', 'tc-inner', this.root);
        const logo = el('img', 'tc-logo', inner);
        logo.src = 'assets/cuju-logo.png';
        logo.alt = '';
        this.kicker = el('div', 'tc-kicker', inner);
        const row = el('div', 'tc-row', inner);
        this.sides = [0, 1].map((side) => {
            const team = el('div', `tc-team ${side ? 'right' : 'left'}`, row);
            const badge = el('b', 'tc-badge', team);
            const name = el('span', 'tc-name', team);
            return { team, badge, name };
        });
        const score = el('div', 'tc-score', row);
        this.scores = [el('span', '', score), el('span', '', score)];
        row.insertBefore(score, this.sides[1].team);
        this.foot = el('div', 'tc-foot', inner);
        this.actions = el('div', 'tc-actions', inner);
    }

    // Shows the card, with buttons under it when actions are given.
    show(kicker, header, styles, frame, foot, actions = []) {
        this.kicker.textContent = kicker;
        const names = header ? teamNames(header) : ['Left', 'Right'];
        for (const side of [0, 1]) {
            const s = this.sides[side];
            const team = (header && header.teams[side]) || {};
            const style = styles && styles[side];
            s.name.textContent = names[side];
            s.badge.textContent = (style && style.badge) || (team.name || '?').slice(0, 1).toUpperCase();
            const color = style ? rgb(style.kit.primary) : (side ? '#2a56b8' : '#b3262e');
            const trim = style ? rgb(style.kit.secondary) : '#ffffff';
            s.badge.style.background = color;
            s.badge.style.color = trim;
            s.team.style.setProperty('--team', color);
        }
        this.root.style.setProperty('--left', styles ? rgb(styles[0].kit.primary) : '#b3262e');
        this.root.style.setProperty('--right', styles ? rgb(styles[1].kit.primary) : '#2a56b8');
        this.scores[0].textContent = frame ? frame.scoreLeft : 0;
        this.scores[1].textContent = frame ? frame.scoreRight : 0;
        this.foot.textContent = foot || '';
        this.actions.replaceChildren(...actions);
        this.root.hidden = false;
        showPointer(true);
        // A frame later, so the entrance animates.
        requestAnimationFrame(() => this.root.classList.add('show'));
    }

    hide() {
        if (!this.root.hidden) {
            showPointer(false);
        }
        this.root.classList.remove('show');
        this.root.hidden = true;
    }
}

// The referee, the figure from players.js when it has one, else a plain figure of capsules.
class SimpleReferee {
    constructor(scene) {
        this.group = new THREE.Group();
        this.group.visible = false;
        scene.add(this.group);
        const shirt = new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.7 });
        const skin = new THREE.MeshStandardMaterial({ color: 0xc89f7e, roughness: 0.8 });
        const sock = new THREE.MeshStandardMaterial({ color: 0x0c0d10, roughness: 0.8 });
        const limb = new THREE.CylinderGeometry(1, 1, 1, 10);
        // Bones as pairs of nodes, with a radius and a material.
        this.bones = [[1, 3, 0.15, shirt], [4, 5, 0.055, shirt], [6, 7, 0.055, shirt], [5, -5, 0.045, skin],
            [7, -7, 0.045, skin], [8, 9, 0.075, shirt], [11, 12, 0.075, shirt], [9, 10, 0.055, sock], [12, 13, 0.055, sock]]
            .map(([a, b, r, mat]) => {
                const m = new THREE.Mesh(limb, mat);
                m.castShadow = true;
                m.userData = { a, b, r };
                this.group.add(m);
                return m;
            });
        this.head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 10), skin);
        this.head.castShadow = true;
        this.group.add(this.head);
        this.card = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.105, 0.005), new THREE.MeshBasicMaterial({ color: 0xffd400 }));
        this.card.visible = false;
        this.group.add(this.card);
        this.a = new THREE.Vector3();
        this.b = new THREE.Vector3();
        this.up = new THREE.Vector3(0, 1, 0);
        this.hand = new THREE.Vector3();
    }

    point(nodes, i, out) {
        if (i < 0) {
            handOf(nodes, 0, i === -7 ? 1 : 0, tv);
            return out.set(tv.x, tv.z, -tv.y);
        }
        const o = i * NODE_FLOATS;
        return out.set(nodes[o], nodes[o + 2], -nodes[o + 1]);
    }

    update(nodes, visible) {
        this.group.visible = !!visible;
        if (!visible) {
            return;
        }
        for (const m of this.bones) {
            const { a, b, r } = m.userData;
            this.point(nodes, a, this.a);
            this.point(nodes, b, this.b);
            const len = this.a.distanceTo(this.b);
            m.position.addVectors(this.a, this.b).multiplyScalar(0.5);
            m.quaternion.setFromUnitVectors(this.up, this.b.sub(this.a).normalize());
            m.scale.set(r, len, r);
        }
        this.point(nodes, 3, this.a);
        this.head.position.set(this.a.x, this.a.y + 0.14, this.a.z);
        if (this.card.visible) {
            this.point(nodes, -7, this.hand);
            this.card.position.set(this.hand.x, this.hand.y + 0.06, this.hand.z);
        }
    }

    setCard(kind) {
        this.card.visible = !!kind;
        if (kind) {
            this.card.material.color.set(kind === 'red' ? 0xe0242b : 0xffd400);
        }
    }

    dispose() {
        this.group.removeFromParent();
    }
}

// The referee's path through the match. Each frame's place follows from the frame before, a
// lagging average of the ball turned into a spot on the diagonal that referees run, kept clear of
// the ball and the players and reached through a spring of limited speed. Restarts put him at a
// spot near the ball, where he waits until play resumes. Every frame is computed once in order
// and kept, so any time on the timeline finds him where playing up to it leaves him.
const REF_TOP_SPEED = 7;
const REF_SPRING = 1.3;

export function refereeAnchor(ev, f) {
    const sx = f.data[8];
    const sy = f.data[9];
    if (ev.kind === 'kickoff') {
        return [sx - 3, sy - 11];
    }
    let cx = -sx;
    let cy = -sy;
    const cl = Math.hypot(cx, cy);
    if (cl < 1) {
        cx = 0;
        cy = -1;
    } else {
        cx /= cl;
        cy /= cl;
    }
    return [clamp(sx + cx * 9 - cy * 3, -HALF_W + 1, HALF_W - 1), clamp(sy + cy * 9 + cx * 3, -HALF_H + 1, HALF_H - 1)];
}

class RefereePath {
    constructor() {
        this.states = new WeakMap();
        this.next = -1;
        this.prev = null;
        this.wait = null;
    }

    // The spot a referee wants for a ball, on the diagonal from the left corner of the far
    // touch line to the right corner of the near one, leaning towards the ball and 9 m off it.
    target(bx, by, ball, f, players) {
        const ax = -32;
        const ay = 22;
        const dx = 64;
        const dy = -44;
        const t = clamp(((bx - ax) * dx + (by - ay) * dy) / (dx * dx + dy * dy), -0.1, 1.1);
        let tx = ax + dx * t;
        let ty = ay + dy * t;
        tx += (bx - tx) * 0.35;
        ty += (by - ty) * 0.35;
        let ox = tx - ball[0];
        let oy = ty - ball[1];
        const od = Math.hypot(ox, oy);
        if (od < 9) {
            if (od < 1e-3) {
                ox = 0;
                oy = -1;
            } else {
                ox /= od;
                oy /= od;
            }
            tx = ball[0] + ox * 9;
            ty = ball[1] + oy * 9;
        }
        // Clear of the players, loosely.
        for (let k = 0; k < players; k++) {
            if (!(f.playerFlags[k] & 1)) {
                continue;
            }
            const o = NODES_FLOAT + k * PLAYER_FLOATS;
            const px = tx - f.data[o];
            const py = ty - f.data[o + 1];
            const d = Math.hypot(px, py);
            if (d < 2.5 && d > 1e-3) {
                tx += (px / d) * (2.5 - d);
                ty += (py / d) * (2.5 - d);
            }
        }
        return [clamp(tx, -HALF_W + 1, HALF_W - 1), clamp(ty, -HALF_H + 1, HALF_H - 1)];
    }

    // Computes the frames from the last one done up to to.
    extend(store, header, to, eventAt, sps) {
        const players = header.players.length;
        const dt = 1 / sps;
        let i = Math.max(this.next, store.first);
        for (; i <= to; i++) {
            const f = store.get(i);
            if (!f) {
                continue;
            }
            const g = this.prev && this.prev.index === i - 1 ? store.get(i - 1) : null;
            const ball = [f.data[8], f.data[9]];
            const ev = eventAt(i);
            const s = new Float32Array(7);
            const p = this.prev ? this.prev.state : null;
            if (ev && (ev.teleport || ev.start)) {
                // A restart places him, and he waits there until the ball is played.
                const anchor = refereeAnchor(ev, f);
                s.set([anchor[0], anchor[1], 0, 0, ball[0], ball[1], p ? p[6] : 0]);
                let until = i + 8;
                for (let j = i + 1; j <= i + 40; j++) {
                    const h = store.get(j);
                    if (!h) {
                        break;
                    }
                    if (h.gameMode === 0) {
                        until = j;
                        break;
                    }
                }
                this.wait = { anchor, until };
            } else if (!p || !g || isCut(g, f)) {
                const t = this.target(ball[0], ball[1], ball, f, players);
                s.set([t[0], t[1], 0, 0, ball[0], ball[1], p ? p[6] : 0]);
                this.wait = null;
            } else {
                const k = 1 - Math.exp(-dt / 1.1);
                const bx = p[4] + (ball[0] - p[4]) * k;
                const by = p[5] + (ball[1] - p[5]) * k;
                let t;
                if (this.wait && i <= this.wait.until) {
                    t = this.wait.anchor;
                } else {
                    this.wait = null;
                    t = this.target(bx, by, ball, f, players);
                }
                let x = p[0];
                let y = p[1];
                let vx = p[2];
                let vy = p[3];
                const w = REF_SPRING;
                const h = dt / 4;
                for (let n = 0; n < 4; n++) {
                    vx += (w * w * (t[0] - x) - 2 * w * vx) * h;
                    vy += (w * w * (t[1] - y) - 2 * w * vy) * h;
                    const sp = Math.hypot(vx, vy);
                    if (sp > REF_TOP_SPEED) {
                        vx *= REF_TOP_SPEED / sp;
                        vy *= REF_TOP_SPEED / sp;
                    }
                    x += vx * h;
                    y += vy * h;
                }
                s.set([x, y, vx, vy, bx, by, p[6] + Math.hypot(x - p[0], y - p[1])]);
            }
            this.states.set(f, s);
            this.prev = { index: i, state: s };
        }
        this.next = Math.max(this.next, i);
    }

    get(store, i) {
        const f = store.get(i);
        return f ? this.states.get(f) || null : null;
    }
}

export class Transitions {
    constructor(app) {
        this.app = app;
        this.store = null;
        this.events = [];
        this.scanned = 0;
        this.seq = null;
        this.after = null;
        this.passed = -1;
        this.prevT = null;
        this.ownSeek = false;
        this.referee = null;
        this.refNodes = new Float32Array(PLAYER_FLOATS);
        this.tmpA = new Float32Array(PLAYER_FLOATS);
        this.tmpB = new Float32Array(PLAYER_FLOATS);
        this.ballTmp = new Float32Array(4);
        this.vec = new THREE.Vector3();
        this.camera = null;
        this.decisions = null;
        this.refPath = new RefereePath();
        this.refDrawn = false;
        const watch = document.getElementById('watch');
        const hud = document.getElementById('hud');
        this.wipe = new Wipe(watch, hud);
        this.title = new TitleCard(watch);
        app.poseHooks.push((pose, sample, dt) => this.dress(pose, sample, dt));
    }

    get active() {
        return !!this.seq;
    }

    get skippable() {
        return !!this.seq;
    }

    // Whether a sound or banner that main.js plays for frame j was already played here, the
    // restart whistle, the half time whistle, the goal and the kick of a teleported ball.
    suppress(j, kind) {
        const s = this.seq || this.after;
        if (kind === 'kick') {
            return this.events.some((e) => e.teleport && (j === e.index || j === e.index + 1));
        }
        if (!s) {
            return false;
        }
        if (kind === 'goal') {
            return !!s.goalAnnounced && s.T === j;
        }
        if (kind === 'half') {
            return !!s.halfWhistled && Math.abs(s.T - j) <= 25;
        }
        return s.T === j;
    }

    // Event lists.

    reset(store) {
        this.abort(false);
        this.store = store;
        this.events = [];
        this.scanned = store ? store.first + 1 : 0;
        this.passed = -1;
        this.after = null;
        this.halfIndex = -1;
        this.halfReal = false;
        this.halfPending = -1;
        this.halfTimes = new Set();
        this.refPath = new RefereePath();
        this.scan();
        // A match opened at its start plays the pre-roll of its first kick-off.
        const tl = this.app.timeline;
        this.prevT = tl && store && tl.t <= store.first + 1 ? store.first - 1e-3 : null;
    }

    scan() {
        const store = this.store;
        if (!store || store.size < 3) {
            return;
        }
        const header = this.app.source.header;
        if (this.events.length && this.events[0].index < store.first) {
            this.events = this.events.filter((e) => e.index >= store.first);
        }
        // The last frame waits for its successor, which may carry the new game mode.
        const to = store.last - 1;
        if (to < this.scanned) {
            return;
        }
        if (this.scanned === store.first + 1 && !this.events.length) {
            const f = store.get(store.first);
            if (f && f.step === 0) {
                this.events.push(startEvent(store.first, f));
            }
        }
        scanRestarts(store, header, this.scanned, to, this.events);
        this.findHalf(this.scanned, to);
        this.scanned = to + 1;
        // A recording's last frame gets his place too, a live one waits for the next frame,
        // which may show a restart there.
        const tl = this.app.timeline;
        this.refPath.extend(store, header, tl && tl.live ? to : to + 1, (i) => this.eventAt(i), tl ? tl.sps : 10);
    }

    // Half time, the first frame of the second half by the clock. Only a scenario with a second
    // half, such as 11_vs_11_kaggle, stops play there, and the engine moves the players to the
    // kick-off on the next step, while the game mode it reports stays at normal play. That move
    // shows the title card in place of the wipe. Elsewhere the clock runs on through 45 minutes
    // and the match has no half time.
    findHalf(from, to) {
        const store = this.store;
        const half = this.app.source.header.second_half;
        if (!(half > 0)) {
            return;
        }
        const halfTime = half * 1800;
        const start = this.halfPending >= 0 ? Math.min(from, this.halfPending) : from;
        this.halfPending = -1;
        for (let i = Math.max(start, store.first + 1); i <= to; i++) {
            const a = store.get(i - 1);
            const b = store.get(i);
            if (a && b && (b.step <= a.step || (b.flags & 2))) {
                this.halfIndex = -1;
                this.halfReal = false;
            }
            if (this.halfIndex < 0 && a && b && !isCut(a, b) && a.timeMs < halfTime && b.timeMs >= halfTime) {
                const near = this.events.find((e) => e.teleport && !e.goal && e.index >= i - 2 && e.index - i <= 20);
                if (!near && to - i < 20) {
                    // The kick-off may still be on its way, so the frame is looked at again on the
                    // next scan.
                    this.halfPending = i;
                    return;
                }
                this.halfIndex = i;
                this.halfReal = !!near;
                if (near) {
                    near.half = true;
                    this.halfTimes.add(i);
                }
            }
        }
    }

    // Whether the frame is the start of a real half time, for the double whistle.
    isHalfTime(j) {
        return this.halfTimes.has(j);
    }

    eventAt(index) {
        const list = this.events;
        let lo = 0;
        let hi = list.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (list[mid].index === index) {
                return list[mid];
            }
            if (list[mid].index < index) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return null;
    }

    // The frame loop, called after the timeline moves and before it is sampled.

    update(dt) {
        const app = this.app;
        const tl = app.timeline;
        if (!tl || !app.source) {
            return;
        }
        if (tl.store !== this.store) {
            this.reset(tl.store);
        }
        if (tl.live) {
            this.scan();
        }
        this.ownSeek = false;
        if (this.seq) {
            if (app.director.running || app.postMatch.running) {
                this.abort(true);
            } else {
                this.runSeq(dt);
            }
        } else {
            this.maybeStart();
        }
        this.updateAfter(dt);
        this.prevT = tl.t;
    }

    enabled() {
        const s = this.app.settings;
        return s.transitions !== false;
    }

    maybeStart() {
        const app = this.app;
        const tl = app.timeline;
        const t0 = this.prevT;
        const t1 = tl.t;
        if (t0 == null || !tl.playing || tl.direction <= 0 || t1 <= t0 || t1 - t0 > 8) {
            if (t0 != null && t1 < this.passed - 1) {
                this.passed = -1;
            }
            return;
        }
        if (app.director.running || app.postMatch.running) {
            return;
        }
        for (const ev of this.events) {
            if (ev.index - 1 > t1) {
                break;
            }
            if (ev.index <= this.passed) {
                continue;
            }
            // Play parked on a restart's first frame, as after a goal's replays, or the kick-off
            // that opens an episode, plays the pre-roll alone.
            const parked = (ev.teleport && Math.abs(t0 - ev.index) < 1e-6) || (ev.start && t0 <= ev.index && t1 > ev.index);
            const crossing = !ev.start && ev.index > t0 && t1 >= ev.index - 1;
            if (parked || crossing) {
                if (parked ? this.enabled() : this.wanted(ev)) {
                    this.start(ev, parked);
                }
                this.passed = Math.max(this.passed, ev.index);
                return;
            }
        }
    }

    // Decision replays and restart transitions each have a setting of their own.
    wanted(ev) {
        if (ev.decision && this.decisions && this.app.settings.decisionReplays !== false) {
            return true;
        }
        return this.enabled() && (ev.kind === 'halftime' || ev.teleport);
    }

    // Sequences. A sequence is a run of scenes, each holding or driving the timeline and
    // dressing the pose, with a wipe on the boundaries that ask for one.

    start(ev, prerollOnly = false) {
        const app = this.app;
        const tl = app.timeline;
        const store = tl.store;
        const header = app.source.header;
        const T = ev.index;
        takerOf(store, header, ev);
        const seq = {
            ev, T, clock: 0, idx: -1, scenes: [], starts: [], local: 0,
            following: tl.following, label: RESTART_NAMES[ev.kind] || '',
            preroll: PREROLL_MIN + (PREROLL_MAX - PREROLL_MIN) * hash01(T, 7),
        };
        this.wipe.setTeams(app.styles);
        this.prepareStage(seq);
        if (prerollOnly) {
            seq.prerollOnly = true;
            seq.scenes = [this.setupScene(seq, false)];
        } else if (ev.decision && this.decisions && app.settings.decisionReplays !== false) {
            seq.scenes = this.decisions.scenes(seq);
        } else {
            seq.scenes = this.restartScenes(seq);
        }
        let at = 0;
        for (const s of seq.scenes) {
            seq.starts.push(at);
            at += s.dur;
        }
        seq.total = at;
        this.seq = seq;
        this.after = null;
        this.enter(seq, 0);
        tl.t = seq.scenes[0].at(seq, 0);
    }

    // The scenes of a restart. The walk, the wipe with the cut, the taker setting the ball.
    restartScenes(seq, options = {}) {
        const ev = seq.ev;
        const T = seq.T;
        const scenes = [];
        if (ev.kind === 'halftime') {
            // The clock reaches half time in open play. The card covers a held frame, and play
            // runs into the frame again under the wipe that takes it away.
            const sps = this.app.timeline.sps;
            return [
                { name: 'hold', dur: WIPE / 2, at: () => T - 1, dress: (pose, l) => this.dressStage(pose, seq, l, 0) },
                this.titleScene(seq, 'Half time'),
                { name: 'resume', dur: WIPE / 2, wipe: true, at: (s, l) => T - 1 - (WIPE / 2 - l) * sps, dress: () => {} },
            ];
        }
        if (!options.skipStage) {
            const stage = ev.goal ? GOAL_STAGE : (ev.teleport ? STAGE : 0.4);
            scenes.push({
                name: 'stage', dur: stage + WIPE / 2,
                at: () => T - 1,
                dress: (pose, l) => this.dressStage(pose, seq, l, 1),
                enter: () => {
                    if (options.stopWhistle) {
                        this.whistle([0.3]);
                    }
                },
            });
        }
        if (ev.half) {
            scenes.push(this.titleScene(seq, 'Half time'));
        }
        scenes.push(this.setupScene(seq));
        return scenes;
    }

    // The restart after the cut, with the camera on it, then the pre-roll, where the players idle
    // and wander a little, the taker sets the ball, and every one of them is back on the pose of
    // the restart's first step when the referee whistles. Without a wipe the scene is the
    // pre-roll alone, as at the kick-off that opens a match.
    setupScene(seq, wipe = true) {
        const tl = this.app.timeline;
        const lead = wipe ? WIPE / 2 : 0;
        seq.setupLead = lead;
        return {
            name: 'setup', dur: lead + seq.preroll, wipe, wipeLabel: seq.label,
            at: () => (seq.liveCut ? seq.liveT : seq.T),
            dress: (pose, l) => {
                if (!seq.liveCut) {
                    this.dressSetup(pose, seq, l);
                }
            },
            enter: () => {
                // A live view that fell far behind during a replay goes back to live under the
                // wipe, as a broadcast returns to live play after a replay.
                const delay = typeof timeline.LIVE_DELAY_FRAMES === 'number' ? timeline.LIVE_DELAY_FRAMES : 4;
                if (seq.following && tl.live && seq.decision && tl.last - seq.T > delay + 40) {
                    seq.liveCut = true;
                    seq.liveT = Math.max(tl.first, tl.last - delay);
                    seq.wipeLabel = 'Live';
                    this.giveCamera(true);
                    return;
                }
                if (wipe && !seq.ev.start) {
                    this.takeCamera(this.restartShot(seq), true);
                }
            },
        };
    }

    titleScene(seq, kicker) {
        const T = seq.T;
        return {
            name: 'title', dur: WIPE / 2 + TITLE + WIPE / 2, wipe: true,
            at: () => (seq.ev.teleport ? T : T - 1),
            dress: () => {},
            enter: () => {
                const app = this.app;
                const frame = app.timeline.store.get(T);
                if (!seq.halfWhistled) {
                    seq.halfWhistled = true;
                    this.whistle([0.45, 0.9]);
                }
                this.title.show(kicker, app.source.header, app.styles, frame, kicker === 'Half time' ? '45′, the second half starts' : '');
            },
            exit: () => this.title.hide(),
        };
    }

    enter(seq, idx) {
        seq.idx = idx;
        const scene = seq.scenes[idx];
        if (scene.enter) {
            scene.enter(seq);
        }
    }

    runSeq(dt) {
        const app = this.app;
        const tl = app.timeline;
        const seq = this.seq;
        if (tl.playing && tl.direction < 0) {
            this.abort(true);
            return;
        }
        const d = tl.playing ? dt * Math.max(0.05, tl.speed) : 0;
        seq.clock += d;
        while (seq.idx < seq.scenes.length && seq.clock >= seq.starts[seq.idx] + seq.scenes[seq.idx].dur) {
            const done = seq.scenes[seq.idx];
            if (done.exit) {
                done.exit(seq);
            }
            if (seq.idx + 1 >= seq.scenes.length) {
                this.release(seq);
                return;
            }
            this.enter(seq, seq.idx + 1);
        }
        const scene = seq.scenes[seq.idx];
        seq.local = seq.clock - seq.starts[seq.idx];
        const t = scene.at(seq, seq.local);
        if (Math.abs(t - tl.t) > 5) {
            this.ownSeek = true;
        }
        tl.t = t;
        // The wipe, centred on each boundary that asks for one.
        let p = null;
        let label = seq.label;
        for (let i = 0; i < seq.scenes.length; i++) {
            if (seq.scenes[i].wipe && Math.abs(seq.clock - seq.starts[i]) < WIPE / 2) {
                p = (seq.clock - seq.starts[i] + WIPE / 2) / WIPE;
                if (seq.scenes[i].wipeLabel != null) {
                    label = seq.scenes[i].wipeLabel;
                }
                if (seq.liveCut && i === seq.idx) {
                    label = seq.wipeLabel;
                }
                break;
            }
        }
        this.wipe.show(p, label);
        if (scene.update) {
            scene.update(seq, seq.local, d);
        }
    }

    // The end of a sequence, play goes on from the restart.
    release(seq) {
        const app = this.app;
        const tl = app.timeline;
        const ev = seq.ev;
        this.seq = null;
        this.wipe.show(null);
        this.title.hide();
        this.clearScore(seq);
        if (seq.liveCut) {
            if (this.decisions) {
                this.decisions.end(seq);
            }
            tl.goLive();
            this.ownSeek = true;
            this.passed = Math.max(this.passed, Math.ceil(tl.t));
            this.after = null;
            return;
        }
        tl.t = seq.T;
        this.passed = Math.max(this.passed, seq.T);
        const store = tl.store;
        // Play resumes once the game mode returns to normal after the restart.
        let resume = seq.T + 6;
        for (let j = seq.T; j <= seq.T + 40; j++) {
            const f = store.get(j);
            if (!f) {
                break;
            }
            if (f.gameMode === 0 && j > seq.T) {
                resume = j;
                break;
            }
        }
        this.after = {
            ev, T: seq.T, clock: 0, resume: resume + 4, taker: seq.setupTaker || null,
            ball: seq.setupBall || null, goalAnnounced: seq.goalAnnounced, halfWhistled: seq.halfWhistled,
        };
        if (this.decisions) {
            this.decisions.end(seq);
        }
        // The referee's whistle ends the pre-roll and play starts.
        if (ev.kind === 'kickoff') {
            this.whistle([0.35]);
        } else if (ev.mode > 0) {
            this.whistle([0.3]);
        }
    }

    // Stops a sequence at once, after a seek, a reverse or a replay of the director. The view
    // stays where the timeline is.
    abort(restoreCamera) {
        const seq = this.seq;
        this.seq = null;
        this.after = null;
        this.wipe.show(null);
        this.title.hide();
        if (seq && this.decisions) {
            this.decisions.end(seq);
        }
        this.clearScore(seq);
        if (restoreCamera !== false) {
            this.giveCamera(true);
        } else {
            this.camera = null;
        }
    }

    clearScore(seq) {
        if (seq && seq.scoreShown && typeof this.app.hud.scoreOf === 'function') {
            this.app.hud.scoreOf(null);
        }
    }

    afterSeek() {
        if (this.ownSeek) {
            return;
        }
        if (this.seq || this.after) {
            this.abort(true);
        }
        this.passed = -1;
        this.prevT = this.app.timeline ? this.app.timeline.t : null;
        if (this.fullShown) {
            // A seek away from the end takes the full time card down and lets it show again.
            clearTimeout(this.fullTimer);
            this.title.hide();
            this.fullShown = null;
        }
    }

    skip() {
        const seq = this.seq;
        if (!seq) {
            return;
        }
        for (let i = seq.idx; i < seq.scenes.length; i++) {
            if (seq.scenes[i].exit) {
                seq.scenes[i].exit(seq);
            }
        }
        this.release(seq);
        // A skip from a replay lands on the restart with its own view.
        if (this.camera && !seq.prerollOnly) {
            this.takeCamera(this.restartShot(seq), true);
        }
        this.app.cameras.jump();
    }

    // After the release, the camera stays on the restart until play resumes, and the taker and
    // the ball blend back to the frames.
    updateAfter(dt) {
        const a = this.after;
        if (!a) {
            return;
        }
        const tl = this.app.timeline;
        a.clock += tl.playing ? dt * tl.speed : 0;
        if (this.camera && (tl.t >= a.resume || a.clock > 5)) {
            this.giveCamera(false);
        }
        if (a.clock > 2 && !this.camera) {
            this.after = null;
        }
    }

    // A sound of the ball, which only a sequence playing forward makes.
    sound(name, position, strength) {
        const sound = this.app.sound;
        const tl = this.app.timeline;
        try {
            if (sound && typeof sound[name] === 'function' && tl && tl.playing && tl.speed <= 2.5) {
                sound[name](position, strength);
            }
        } catch {
            // Sound is a nicety.
        }
    }

    whistle(blasts) {
        const sound = this.app.sound;
        try {
            if (sound && typeof sound.whistle === 'function') {
                sound.whistle(blasts);
            }
        } catch {
            // Sound is a nicety, a failure must not stop the show.
        }
    }

    // Cameras.

    takeCamera(shot, snap) {
        const cams = this.app.cameras;
        if (!this.camera) {
            if (cams.mode === 'scripted') {
                return;
            }
            this.camera = { mode: cams.mode };
        }
        cams.scripted = {
            pos: shot.pos, target: shot.target, fov: shot.fov || 40, rate: shot.rate || 3, snap: !!snap,
        };
        cams.setMode('scripted');
    }

    // Hands the camera back to the viewer's mode, which eases there from the restart view.
    giveCamera(snap) {
        const cams = this.app.cameras;
        if (this.camera) {
            if (cams.mode === 'scripted') {
                cams.setMode(this.camera.mode);
            }
            if (snap) {
                cams.jump();
            }
        }
        this.camera = null;
    }

    // A view of the restart, in three's frame, which is y up with engine y as minus z.
    restartShot(seq) {
        const ev = seq.ev;
        const app = this.app;
        const f = app.timeline.store.get(seq.T);
        const v = (x, y, z) => new THREE.Vector3(x, z, -y);
        const sx = f ? f.data[8] : 0;
        const sy = f ? f.data[9] : 0;
        const header = app.source.header;
        const taker = ev.taker >= 0 ? header.players[ev.taker] : null;
        const attack = taker ? (taker.team === 0 ? 1 : -1) : 1;
        switch (ev.kind) {
        case 'corner': {
            const ox = Math.sign(sx) || 1;
            const oy = Math.sign(sy) || 1;
            // From behind the flag, over the taker towards the near side of the box.
            return { pos: v(sx + ox * 6.5, sy + oy * 6.5, 4.2), target: v(ox * 47, oy * 17, 0.5), fov: 50 };
        }
        case 'throw_in': {
            const side = Math.sign(sy) || 1;
            return { pos: v(sx - attack * 15, side * (HALF_H + 4.5), 3.6), target: v(sx + attack * 7, side * (HALF_H - 9), 0.8), fov: 44 };
        }
        case 'goal_kick': {
            const side = Math.sign(sx) || 1;
            return { pos: v(side * (HALF_W + 9), sy * 0.3, 4.6), target: v(side * (HALF_W - 30), 0, 0.5), fov: 46 };
        }
        case 'penalty':
        case 'free_kick': {
            const gx = attack * HALF_W;
            const dx = gx - sx;
            const dy = -sy;
            const d = Math.hypot(dx, dy) || 1;
            const back = ev.kind === 'penalty' ? 6 : 10;
            const lift = ev.kind === 'penalty' ? 2.2 : 3.6;
            const far = Math.min(18, d * 0.75);
            return {
                pos: v(sx - (dx / d) * back, sy - (dy / d) * back, lift),
                target: v(sx + (dx / d) * far, sy + (dy / d) * far, ev.kind === 'penalty' ? 1.1 : 0.5),
                fov: ev.kind === 'penalty' ? 36 : 42,
            };
        }
        default:
            return { pos: new THREE.Vector3(0, 21, 60), target: new THREE.Vector3(0, 0, 0), fov: 30 };
        }
    }

    // Staging before the cut.

    // Reads what the walk needs from the frames around the step, positions, velocities, the
    // spots the players walk towards and the ball's flight on.
    prepareStage(seq) {
        const app = this.app;
        const store = app.timeline.store;
        const header = app.source.header;
        const T = seq.T;
        const a = store.get(T - 1);
        const prev = store.get(T - 2);
        const b = store.get(T);
        // The kick-off that opens an episode has nothing before it to walk from.
        if (!a || !b || isCut(a, b)) {
            seq.stage = null;
            return;
        }
        const n = header.players.length;
        const sps = app.timeline.sps;
        const prevOk = prev && !isCut(prev, a);
        const st = {
            n, frame: a, x: new Float32Array(n), y: new Float32Array(n), vx: new Float32Array(n),
            vy: new Float32Array(n), dx: new Float32Array(n), dy: new Float32Array(n),
            h0: new Float32Array(n), vis: new Uint8Array(n),
        };
        const taker = seq.ev.taker;
        for (let k = 0; k < n; k++) {
            const o = NODES_FLOAT + k * PLAYER_FLOATS;
            st.vis[k] = a.playerFlags[k] & 1;
            st.x[k] = a.data[o];
            st.y[k] = a.data[o + 1];
            if (prevOk) {
                let vx = (a.data[o] - prev.data[o]) * sps;
                let vy = (a.data[o + 1] - prev.data[o + 1]) * sps;
                const sp = Math.hypot(vx, vy);
                if (sp > 9) {
                    vx *= 9 / sp;
                    vy *= 9 / sp;
                }
                st.vx[k] = vx;
                st.vy[k] = vy;
            }
            st.h0[k] = headingOf(a.data, o);
            // Each player walks towards the spot the engine gives him, the taker towards the ball.
            let tx = b && seq.ev.teleport ? b.data[o] : a.data[o];
            let ty = b && seq.ev.teleport ? b.data[o + 1] : a.data[o + 1];
            if (k === taker && b) {
                tx = clamp(b.data[8], -HALF_W, HALF_W);
                ty = clamp(b.data[9], -HALF_H, HALF_H);
            }
            const ddx = tx - st.x[k];
            const ddy = ty - st.y[k];
            const dist = Math.hypot(ddx, ddy);
            const cap = k === taker ? TAKER_PULL : PULL;
            const m = dist > 1e-3 ? Math.min(dist, cap) / dist : 0;
            st.dx[k] = ddx * m;
            st.dy[k] = ddy * m;
        }
        let bvx = 0;
        let bvy = 0;
        let bvz = 0;
        if (prevOk) {
            bvx = (a.data[8] - prev.data[8]) * sps;
            bvy = (a.data[9] - prev.data[9]) * sps;
            bvz = (a.data[10] - prev.data[10]) * sps;
        }
        st.ball = simulateBall(a.data[8], a.data[9], a.data[10], bvx, bvy, bvz, seq.ev.goal);
        st.ballAxis = new THREE.Vector3(-bvy, bvx, 0);
        st.ballAxis.lengthSq() > 1e-6 ? st.ballAxis.normalize() : st.ballAxis.set(1, 0, 0);
        st.ballQ = new THREE.Quaternion(a.data[11], a.data[12], a.data[13], a.data[14]);
        st.ballKnown = st.ballQ.lengthSq() > 1e-6;
        // The referee runs from his place on the path towards play, stops short and signals.
        const fx = clamp(a.data[8], -HALF_W + 3, HALF_W - 3);
        const fy = clamp(a.data[9], -HALF_H + 3, HALF_H - 3);
        const ref = this.refPath.get(store, T - 1);
        const from = ref ? [ref[0], ref[1]] : [fx, fy - 10];
        const rx = fx - from[0];
        const ry = fy - from[1];
        const rd = Math.hypot(rx, ry) || 1;
        const run = clamp(rd - 7, 0, 7);
        st.refFrom = from;
        st.refTo = [from[0] + (rx / rd) * run, from[1] + (ry / rd) * run];
        st.refDur = seq.ev.decision ? 0.8 : 1.25;
        st.spot = b ? [b.data[8], b.data[9]] : [fx, fy];
        st.focus = [fx, fy];
        seq.stage = st;
    }

    // Root of player k at time l of the walk, its speed, and the distance walked.
    stagePos(st, k, l, pull, out) {
        const tc = 0.45;
        const dec = tc * (1 - Math.exp(-l / tc));
        const span = Math.max(0.3, st.span || 1.5);
        const w = smooth((l - 0.2) / span) * pull;
        out[0] = st.x[k] + st.vx[k] * dec + st.dx[k] * w;
        out[1] = st.y[k] + st.vy[k] * dec + st.dy[k] * w;
        out[2] = Math.hypot(st.vx[k], st.vy[k]) * dec + Math.hypot(st.dx[k], st.dy[k]) * w;
        return out;
    }

    // The walk before the cut. Players ease from their last frame into a jog or walk, slow down
    // and head towards their spots by a few meters, the ball runs on and the referee comes in.
    dressStage(pose, seq, l, pull) {
        const st = seq.stage;
        if (!st) {
            return;
        }
        st.span = seq.scenes[0] ? seq.scenes[0].dur - 0.2 : 1.5;
        const nodes = pose.nodes;
        const p = this.stageP || (this.stageP = new Float64Array(3));
        const q = this.stageQ || (this.stageQ = new Float64Array(3));
        const fa = st.frame.data;
        const blendIn = smooth(l / 0.35);
        const bpos = sampleBall(st.ball, l, this.ballTmp);
        for (let k = 0; k < st.n; k++) {
            if (!st.vis[k]) {
                continue;
            }
            this.stagePos(st, k, l, pull, p);
            this.stagePos(st, k, l + 0.05, pull, q);
            const vx = (q[0] - p[0]) / 0.05;
            const vy = (q[1] - p[1]) / 0.05;
            const speed = Math.hypot(vx, vy);
            const toBall = headingTo(bpos[0] - p[0], bpos[1] - p[1]);
            const moving = smooth((speed - 0.25) / 0.8);
            let heading = lerpAngle(toBall, headingTo(vx, vy), moving);
            heading = lerpAngle(st.h0[k], heading, smooth(l / 0.5));
            const o = k * PLAYER_FLOATS;
            walkPose(p[2] * (2 * Math.PI / 1.45) + k * 1.3, speed, heading, p[0], p[1], this.tmpA, 0);
            // The frame's own pose, carried along with the root.
            const src = NODES_FLOAT + k * PLAYER_FLOATS;
            const sx = p[0] - st.x[k];
            const sy = p[1] - st.y[k];
            for (let i = 0; i < PLAYER_FLOATS; i += NODE_FLOATS) {
                this.tmpB[i] = fa[src + i] + sx;
                this.tmpB[i + 1] = fa[src + i + 1] + sy;
                this.tmpB[i + 2] = fa[src + i + 2];
                this.tmpB[i + 3] = fa[src + i + 3];
                this.tmpB[i + 4] = fa[src + i + 4];
                this.tmpB[i + 5] = fa[src + i + 5];
                this.tmpB[i + 6] = fa[src + i + 6];
            }
            // GRF turns the root node by team, the walk keeps that turn and the root's height.
            for (let c = 2; c < 7; c++) {
                this.tmpA[c] = this.tmpB[c];
            }
            blendPose(this.tmpB, 0, this.tmpA, 0, blendIn, nodes, o);
        }
        this.dressBall(pose, st, l);
        this.announceGoal(seq, bpos);
        this.stageReferee(seq, l);
    }

    dressBall(pose, st, l) {
        const b = sampleBall(st.ball, l, this.ballTmp);
        // The net bulges where the ball hits it, once per touch.
        const impacts = st.ball.impacts || [];
        const pitch = this.app.pitch;
        while ((st.fired || 0) < impacts.length && impacts[st.fired || 0].time <= l) {
            const hit = impacts[st.fired || 0];
            st.fired = (st.fired || 0) + 1;
            const at = new THREE.Vector3(...hit.point);
            if (pitch && typeof pitch.netImpact === 'function') {
                try {
                    pitch.netImpact(hit.side, at, hit.strength);
                } catch {
                    // The net is a nicety.
                }
            }
        }
        pose.ball[0] = b[0];
        pose.ball[1] = b[1];
        pose.ball[2] = b[2];
        if (st.ballKnown) {
            const spin = this.spinQ || (this.spinQ = new THREE.Quaternion());
            spin.setFromAxisAngle(st.ballAxis, b[3] / BALL_RADIUS).multiply(st.ballQ);
            pose.ball[3] = spin.x;
            pose.ball[4] = spin.y;
            pose.ball[5] = spin.z;
            pose.ball[6] = spin.w;
        }
    }

    // A goal is announced when the ball crosses the line, not when the engine moves the
    // players back for the kick-off.
    announceGoal(seq, b) {
        const ev = seq.ev;
        if (!ev.goal || seq.goalAnnounced || Math.abs(b[0]) < HALF_W - 0.05) {
            return;
        }
        seq.goalAnnounced = true;
        const app = this.app;
        try {
            if (app.sound && typeof app.sound.goal === 'function') {
                app.sound.goal(ev.goalTeam);
            }
        } catch {
            // Sound is optional.
        }
        app.goalBoost = 1;
        app.hud.showGoal(ev.goalTeam, true);
        // The scoreboard counts the goal now, though the frames add it at the kick-off.
        if (typeof app.hud.scoreOf === 'function') {
            app.hud.scoreOf(app.timeline.store.get(seq.T));
            seq.scoreShown = true;
        }
    }

    // The referee. Created on first use, from players.js when it builds one.
    ensureReferee() {
        if (this.referee) {
            return this.referee;
        }
        const app = this.app;
        if (typeof figures.createReferee === 'function') {
            try {
                this.referee = figures.createReferee(app.view.scene, app.assets);
            } catch (err) {
                console.warn('referee figure failed, using a plain one', err);
            }
        }
        if (!this.referee) {
            this.referee = new SimpleReferee(app.view.scene);
        }
        return this.referee;
    }

    showReferee(visible, card = null) {
        this.refDrawn = true;
        const ref = this.ensureReferee();
        if (this.refCard !== card) {
            this.refCard = card;
            ref.setCard(card);
        }
        ref.update(this.refNodes, visible);
    }

    // Moves the referee along a path from a to b over dur seconds, then holds a gesture.
    refAlong(a, b, l, dur, faceX, faceY, gesture, gestureAt) {
        const w = smooth(l / dur);
        const x = a[0] + (b[0] - a[0]) * w;
        const y = a[1] + (b[1] - a[1]) * w;
        const w2 = smooth((l + 0.05) / dur);
        const speed = Math.hypot(b[0] - a[0], b[1] - a[1]) * Math.abs(w2 - w) / 0.05;
        const travel = Math.hypot(b[0] - a[0], b[1] - a[1]) * w;
        const moveHeading = headingTo(b[0] - a[0], b[1] - a[1]);
        const faceHeading = headingTo(faceX - x, faceY - y);
        const settle = smooth((l - dur + 0.25) / 0.35);
        const heading = lerpAngle(moveHeading, faceHeading, settle);
        walkPose(travel * (2 * Math.PI / 1.6), speed, heading, x, y, this.refNodes, 0);
        if (gesture && l > gestureAt) {
            namedPose(gesture, l - gestureAt, heading, x, y, this.tmpA, 0);
            blendPose(this.refNodes, 0, this.tmpA, 0, smooth((l - gestureAt) / 0.3), this.refNodes, 0);
        }
        return [x, y];
    }

    stageReferee(seq, l) {
        const st = seq.stage;
        const ev = seq.ev;
        if (ev.goal || ev.kind === 'halftime') {
            return;
        }
        const gesture = ev.decision ? 'whistle' : 'point';
        const dur = st.refDur || 1;
        this.refAlong(st.refFrom, st.refTo, l, dur, ev.decision ? st.focus[0] : st.spot[0], ev.decision ? st.focus[1] : st.spot[1], gesture, dur - 0.05);
        this.showReferee(true);
    }

    // After the cut, the pre-roll. Players idle, shift and glance about, wander up to about a
    // meter and come back, the taker sets the ball, and at the end every pose is the frame's own.
    dressSetup(pose, seq, l) {
        const app = this.app;
        const ev = seq.ev;
        const f = app.timeline.store.get(seq.T);
        if (!f) {
            return;
        }
        const end = seq.scenes[seq.idx] ? seq.scenes[seq.idx].dur : seq.preroll;
        const k = ev.taker;
        const dressed = k >= 0 && (f.playerFlags[k] & 1) && ev.kind !== 'kickoff';
        this.dressIdle(pose, seq, l, end, f, dressed ? k : -1);
        if (dressed) {
            const o = k * PLAYER_FLOATS;
            const x = pose.nodes[o];
            const y = pose.nodes[o + 1];
            const heading = headingOf(pose.nodes, o);
            const players = app.players;
            // A model with clips picks the ball up for a throw-in before holding it overhead.
            const lift = ev.kind === 'throw_in' ? smooth(l / 0.2) * (1 - smooth((l - 0.75) / 0.3)) : 0;
            const lifting = lift > 0 && players && typeof players.playClip === 'function'
                && players.playClip(k, 'clip:PickUp', Math.max(0, l - 0.05), lift);
            // The taker bounces the ball twice, then crouches to place it and stands behind it.
            const bounceEnd = 0.42 * end;
            const placeEnd = 0.78 * end;
            let name = null;
            let w = 0;
            if (ev.kind === 'throw_in') {
                name = 'throw_in_hold';
                w = lifting ? smooth((l - 0.75) / 0.3) : smooth(l / 0.3);
            } else {
                name = 'place_ball';
                w = smooth((l - bounceEnd + 0.15) / 0.3) * (1 - smooth((l - placeEnd) / 0.35));
            }
            if (w > 0) {
                namedPose(name, l, heading, x, y, this.tmpA, 0);
                for (let c = 2; c < 7; c++) {
                    this.tmpA[c] = pose.nodes[o + c];
                }
                blendPose(pose.nodes, o, this.tmpA, 0, w, pose.nodes, o);
            }
            seq.setupTaker = seq.setupTaker || new Float32Array(PLAYER_FLOATS);
            seq.setupTaker.set(pose.nodes.subarray(o, o + PLAYER_FLOATS));
            if (ev.kind === 'throw_in') {
                const hl = handOf(pose.nodes, o, 0, this.vec).clone();
                const hr = handOf(pose.nodes, o, 1, this.vec);
                // From the ground into the hands as the taker stands up with it.
                const hold = lifting ? w : 1;
                pose.ball[0] += ((hl.x + hr.x) / 2 - pose.ball[0]) * hold;
                pose.ball[1] += ((hl.y + hr.y) / 2 - pose.ball[1]) * hold;
                pose.ball[2] += ((hl.z + hr.z) / 2 - pose.ball[2]) * hold;
                seq.setupBall = seq.setupBall || new Float32Array(3);
                seq.setupBall.set([pose.ball[0], pose.ball[1], pose.ball[2]]);
            } else if (l < placeEnd) {
                this.bounceBall(pose, x, y, heading, l, bounceEnd, placeEnd);
            }
        }
        if (ev.kind === 'halftime') {
            return;
        }
        this.setupReferee(seq, l, f, end);
    }

    // The ball of a set piece before it is placed, two bounces beside the taker, then carried
    // low to the spot that the frame gives it.
    bounceBall(pose, x, y, heading, l, bounceEnd, placeEnd) {
        const fx = Math.sin(heading);
        const fy = -Math.cos(heading);
        const bx = x + fx * 0.38 - fy * 0.18;
        const by = y + fy * 0.38 + fx * 0.18;
        const sx = pose.ball[0];
        const sy = pose.ball[1];
        const sz = pose.ball[2];
        if (l < bounceEnd) {
            const half = bounceEnd / 2;
            const height = 0.75 * (1 - 0.35 * (l / bounceEnd));
            pose.ball[0] = bx;
            pose.ball[1] = by;
            pose.ball[2] = BALL_RADIUS + height * Math.abs(Math.sin(Math.PI * l / half));
            return;
        }
        const u = smooth((l - bounceEnd) / Math.max(0.1, placeEnd - bounceEnd));
        pose.ball[0] = bx + (sx - bx) * u;
        pose.ball[1] = by + (sy - by) * u;
        pose.ball[2] = BALL_RADIUS + (sz - BALL_RADIUS) * u + 0.22 * Math.sin(Math.PI * u);
    }

    // Idle motion of every player but the taker, zero at the start and the end of the pre-roll.
    dressIdle(pose, seq, l, end, f, skip) {
        const n = this.app.source.header.players.length;
        const env = smooth(l / 0.8) * (1 - smooth((l - end + 0.95) / 0.8));
        if (env <= 0) {
            return;
        }
        const T = seq.T;
        const tau = 2 * Math.PI;
        for (let k = 0; k < n; k++) {
            if (k === skip || !(f.playerFlags[k] & 1)) {
                continue;
            }
            const r1 = hash01(T, k * 4 + 1);
            const r2 = hash01(T, k * 4 + 2);
            const r3 = hash01(T, k * 4 + 3);
            const r4 = hash01(T, k * 4 + 4);
            const amp = k === seq.ev.taker ? 0.15 : 0.35 + 0.35 * r1;
            const f1 = 0.14 + 0.12 * r2;
            const f2 = 0.11 + 0.12 * r3;
            const wander = (t) => [amp * Math.sin(tau * f1 * t + tau * r3), amp * Math.sin(tau * f2 * t + tau * r4)];
            const [ox, oy] = wander(l);
            const [qx, qy] = wander(l + 0.05);
            const speed = Math.hypot(qx - ox, qy - oy) / 0.05 * env;
            const o = k * PLAYER_FLOATS;
            const x0 = pose.nodes[o];
            const y0 = pose.nodes[o + 1];
            const h0 = headingOf(pose.nodes, o);
            // Glances about, a turn of the body that comes back.
            const heading = h0 + 0.5 * Math.sin(tau * (0.09 + 0.08 * r4) * l + tau * r1);
            const x = x0 + ox * env;
            const y = y0 + oy * env;
            const idle = r1 < 0.18 ? 'idle_confident' : (r1 > 0.85 ? 'idle_bounce' : 'idle');
            namedPose(idle, l + 5 * r2, heading, x, y, this.tmpA, 0);
            const step = smooth((speed - 0.1) / 0.35) * 0.8;
            if (step > 0) {
                walkPose(l * 6.5 + k, speed * 2, heading, x, y, this.tmpB, 0);
                blendPose(this.tmpA, 0, this.tmpB, 0, step, this.tmpA, 0);
            }
            for (let c = 2; c < 7; c++) {
                this.tmpA[c] = pose.nodes[o + c];
            }
            blendPose(pose.nodes, o, this.tmpA, 0, env, pose.nodes, o);
            // A model with clips shifts its weight with its own idle, over the same envelope, so
            // at the whistle it stands on the frame's pose.
            const players = this.app.players;
            if (players && typeof players.playClip === 'function') {
                players.playClip(k, r2 < 0.5 ? 'clip:Idle_A' : 'clip:Idle_B', l + 3 * r3, env * 0.9);
            }
        }
    }

    // The referee during the pre-roll, at his place on the path, pointing to the spot, then
    // idling, then with the whistle up as it ends.
    setupReferee(seq, l, f, end) {
        const tl = this.app.timeline;
        const ref = this.refPath.get(tl.store, seq.T);
        if (!ref) {
            return;
        }
        const x = ref[0];
        const y = ref[1];
        const heading = headingTo(f.data[8] - x, f.data[9] - y);
        const idleT = seq.T / tl.sps;
        namedPose('idle', idleT, heading, x, y, this.refNodes, 0);
        const lead = seq.setupLead || 0;
        const point = smooth((l - lead - 0.1) / 0.3) * (1 - smooth((l - lead - 1.1) / 0.35));
        if (point > 0 && !seq.ev.start) {
            namedPose('point', l, heading, x, y, this.tmpB, 0);
            blendPose(this.refNodes, 0, this.tmpB, 0, point, this.refNodes, 0);
        }
        const whistleAt = Math.max(0.2, end - 0.6);
        // The whistle comes down again in the last moment so his pose meets the path's.
        const blow = smooth((l - whistleAt) / 0.25) * (1 - smooth((l - end + 0.12) / 0.1));
        if (blow > 0) {
            namedPose('whistle', l - whistleAt, heading, x, y, this.tmpB, 0);
            blendPose(this.refNodes, 0, this.tmpB, 0, blow, this.refNodes, 0);
        }
        this.showReferee(true);
    }

    // The referee in open play, at his place on the path, jogging or walking with the distance
    // covered, facing the ball when slow and his way when fast.
    drawPathRef(pose, t) {
        const app = this.app;
        const tl = app.timeline;
        const store = tl.store;
        if (app.postMatch.running) {
            // He blows the final whistle where play left him and lowers it as play winds down,
            // then leaves the picture under the full time card.
            const r = app.postMatch.react;
            const at = r ? this.refPath.get(store, r.index) : null;
            if (!at || r.clock >= r.titleAt) {
                this.ensureReferee().update(this.refNodes, false);
                return;
            }
            const heading = headingTo(pose.ball[0] - at[0], pose.ball[1] - at[1]);
            namedPose('idle', r.clock, heading, at[0], at[1], this.refNodes, 0);
            const down = Math.max(0.3, r.titleAt - 0.8);
            const blow = smooth(r.clock / 0.25) * (1 - smooth((r.clock - down) / 0.8));
            if (blow > 0) {
                namedPose('whistle', r.clock, heading, at[0], at[1], this.tmpB, 0);
                blendPose(this.refNodes, 0, this.tmpB, 0, blow, this.refNodes, 0);
            }
            this.showReferee(true);
            return;
        }
        const i = Math.floor(t);
        const s0 = this.refPath.get(store, i);
        if (!s0) {
            this.ensureReferee().update(this.refNodes, false);
            return;
        }
        let s1 = this.refPath.get(store, i + 1);
        const next = store.get(i + 1);
        const ev = this.eventAt(i + 1);
        if (!s1 || !next || isCut(store.get(i), next) || (ev && (ev.teleport || ev.start))) {
            s1 = s0;
        }
        const w = s1 === s0 ? 0 : t - i;
        const x = s0[0] + (s1[0] - s0[0]) * w;
        const y = s0[1] + (s1[1] - s0[1]) * w;
        const vx = s0[2] + (s1[2] - s0[2]) * w;
        const vy = s0[3] + (s1[3] - s0[3]) * w;
        const dist = s0[6] + (s1[6] - s0[6]) * w;
        const speed = Math.hypot(vx, vy);
        const toBall = headingTo(pose.ball[0] - x, pose.ball[1] - y);
        const heading = speed > 0.05 ? lerpAngle(toBall, headingTo(vx, vy), smooth((speed - 2.5) / 2.5)) : toBall;
        walkPose(dist * (2 * Math.PI / 1.6), speed, heading, x, y, this.refNodes, 0);
        const still = 1 - smooth((speed - 0.15) / 0.5);
        if (still > 0) {
            // Held frames keep breathing on the sequence's clock.
            const breath = t / tl.sps + (this.seq ? this.seq.clock : 0);
            namedPose('idle', breath, heading, x, y, this.tmpB, 0);
            blendPose(this.refNodes, 0, this.tmpB, 0, still, this.refNodes, 0);
        }
        this.showReferee(true);
    }

    // The pose hook.

    dress(pose, sample, dt) {
        void dt;
        if (!this.store || !sample || !sample.a) {
            return;
        }
        this.refDrawn = false;
        // Never slide players across the pitch between the two frames around a teleport.
        if (sample.w > 0 && this.events.length) {
            const ev = this.eventAt(sample.index + 1);
            if (ev && ev.teleport) {
                pose.compute(sample.a, null, 0);
            }
        }
        const seq = this.seq;
        if (seq) {
            const scene = seq.scenes[seq.idx];
            if (scene && scene.dress) {
                scene.dress(pose, seq.local, seq, sample);
            }
        } else {
            this.dressAfter(pose, sample);
        }
        // The referee is on the pitch all match, wherever a scene does not place him itself.
        if (!this.refDrawn) {
            this.drawPathRef(pose, this.app.timeline.t);
        }
    }

    dressAfter(pose, sample) {
        const a = this.after;
        if (!a || sample.index < a.T || sample.index > a.T + 30) {
            return;
        }
        const w = 1 - smooth(a.clock / 0.3);
        const k = a.ev.taker;
        if (w > 0 && a.taker && k >= 0) {
            blendPose(pose.nodes, k * PLAYER_FLOATS, a.taker, 0, w, pose.nodes, k * PLAYER_FLOATS);
        }
        if (w > 0 && a.ball) {
            for (let c = 0; c < 3; c++) {
                pose.ball[c] += (a.ball[c] - pose.ball[c]) * w;
            }
        }
    }

    // Full time, a title card when no show after the match follows.
    fullTime() {
        const app = this.app;
        if (!app.source || app.director.running || (app.settings.podium && app.source.kind !== 'live')) {
            return;
        }
        const tl = app.timeline;
        if (this.fullShown === tl) {
            return;
        }
        this.fullShown = tl;
        const frame = tl ? tl.store.get(tl.index) : null;
        this.title.show('Full time', app.source.header, app.styles, frame, '',
            app.source.kind === 'live' ? [] : [app.postMatch.statsButton()]);
        clearTimeout(this.fullTimer);
        this.fullTimer = setTimeout(() => this.title.hide(), 4500);
    }
}
