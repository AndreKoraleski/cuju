// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Camera modes. Each mode computes a wanted view and the rig eases towards it, with critically
// damped springs for the framed shots, so cuts between modes and jumps of the ball stay smooth.
// The cameras a user steers keep to a box inside the stadium, above the pitch and the lower tier,
// under the roof and out of the stands.

import * as THREE from 'three';
import { FULL_HALF_H, FULL_HALF_W, HALF_W } from './pitch.js';
import { NODE_FLOATS, PLAYER_FLOATS } from './format.js';
import * as stadium from './stadium.js';

export const CAMERA_MODES = [
    { id: 'broadcast', label: 'Broadcast', key: 'V' },
    { id: 'tactical', label: 'Tactical', key: 'V' },
    { id: 'goal', label: 'Behind the goal', key: 'V' },
    { id: 'player', label: 'Follow player', key: 'V' },
    { id: 'free', label: 'Free camera', key: 'V' },
    { id: 'ball', label: 'Follow the ball', key: 'B' },
    { id: 'owner', label: 'Follow the player on the ball', key: 'P' },
];

// Modes used by replays, first person and the podium, not offered in the camera list.
export const EXTRA_MODES = ['fpp', 'lowside', 'orbit', 'podium', 'scripted'];
// Views of one player, cycled by pressing that player's key again.
export const PLAYER_VIEWS = ['over the shoulder', 'first person', 'side', 'face'];

const FOLLOW = new Set(['ball', 'owner']);
// Modes whose shot lives in the shared springs, which carry on from one such mode to the next.
const FRAMED = new Set(['broadcast', 'tactical', 'goal', 'player', 'lowside', 'orbit', 'podium', 'scripted']);

// The shape of the bowl, as stadium.js builds it, out from the inner edge of the stands and up
// from the grass. stadium.js may export BOWL to keep the two in step.
const SHAPE = {
    halfX: FULL_HALF_W + 6,
    halfZ: FULL_HALF_H + 6,
    corner: 12,
    frontWall: 1.5,
    tread: 0.8,
    rise: 0.38,
    lowerRows: 18,
    roofReach: 10,
    roofLow: 29.1,
    ...(stadium.BOWL || {}),
};

// The box of the steerable cameras. They stay above the lower tier at most outMax beyond the
// inner edge, under the roof unless inside its front edge, clear of the seats and the front wall.
export const CAMERA_BOX = {
    minY: 0.4,
    maxY: 60,
    outMax: SHAPE.lowerRows * SHAPE.tread - 1.5,
    open: SHAPE.roofReach - 2,
    roofY: SHAPE.roofLow - 1.5,
    wallIn: -0.3,
    seatClear: 1.3,
    pitchLimit: THREE.MathUtils.degToRad(80),
    followMin: 4,
    followMax: 45,
    speedMin: 3,
    speedMax: 30,
};

// Advertising boards stand 1.5 meters outside the full pitch, 0.9 meters high.
const BOARD_X = FULL_HALF_W + 1.5;
const BOARD_Z = FULL_HALF_H + 1.5;
const BOARD_TOP = 1.3;
const BOARD_BAND = 0.45;

// Other players are capsules of this radius around their trunk, up to this height.
const CAPSULE_R = 0.35;
const CAPSULE_TOP = 2.1;
// In the player views, other players closer than this to the camera are hidden.
const NEAR_HIDE = 0.9;

const NEAR_PLAYER = 0.05;
const NEAR_DEFAULT = 0.1;
const UP = new THREE.Vector3(0, 1, 0);
const N = { x: 0, z: 0 };

function ease(rate, dt) {
    return 1 - Math.exp(-rate * dt);
}

function smoother(s) {
    s = Math.min(1, Math.max(0, s));
    return s * s * s * (s * (s * 6 - 15) + 10);
}

// Signed distance of a ground point from the inner edge of the bowl, a rounded rectangle, with
// the outward direction written to n.
function bowlDistance(x, z, n) {
    const qx = Math.abs(x) - (SHAPE.halfX - SHAPE.corner);
    const qz = Math.abs(z) - (SHAPE.halfZ - SHAPE.corner);
    let d;
    let nx;
    let nz;
    if (qx > 0 && qz > 0) {
        const l = Math.hypot(qx, qz);
        d = l - SHAPE.corner;
        nx = qx / l;
        nz = qz / l;
    } else if (qx > qz) {
        d = qx - SHAPE.corner;
        nx = 1;
        nz = 0;
    } else {
        d = qz - SHAPE.corner;
        nx = 0;
        nz = 1;
    }
    n.x = x < 0 ? -nx : nx;
    n.z = z < 0 ? -nz : nz;
    return d;
}

function seatFloor(d) {
    return SHAPE.frontWall + Math.max(0, d) / SHAPE.tread * SHAPE.rise + SHAPE.rise + CAMERA_BOX.seatClear;
}

// Moves p to the nearest place inside the box, sliding along whichever limit is closer, so a
// camera pushed into a wall glides along it rather than jumping. With prev, the place the camera
// comes from, a camera below the seats or above the roof line meets a wall instead, so a fast
// camera neither climbs the stands nor dives under the roof.
export function confine(p, prev = null) {
    const B = CAMERA_BOX;
    p.y = Math.min(B.maxY, Math.max(B.minY, p.y));
    let d = bowlDistance(p.x, p.z, N);
    const shift = (to) => {
        p.x += N.x * (to - d);
        p.z += N.z * (to - d);
        d = to;
    };
    if (d > B.outMax) {
        shift(B.outMax);
    }
    if (d > B.open && p.y > B.roofY) {
        if (prev ? prev.y <= B.roofY : p.y - B.roofY < d - B.open) {
            p.y = B.roofY;
        } else {
            shift(B.open);
        }
    }
    if (d > B.wallIn) {
        const floor = seatFloor(d);
        if (p.y < floor) {
            // The face of the terrace at this height, the front wall below the first row.
            const base = seatFloor(0);
            const wall = p.y < base ? B.wallIn : (p.y - base) / SHAPE.rise * SHAPE.tread;
            if (prev ? prev.y >= floor - 1e-4 : floor - p.y < d - wall) {
                p.y = floor;
            } else {
                shift(wall);
            }
        }
    }
    if (p.y < BOARD_TOP) {
        const ax = Math.abs(p.x);
        const az = Math.abs(p.z);
        const lift = BOARD_TOP - p.y;
        if (ax < BOARD_X - 1.5 && Math.abs(az - BOARD_Z) < BOARD_BAND) {
            const out = BOARD_BAND - Math.abs(az - BOARD_Z);
            if (lift < out) {
                p.y = BOARD_TOP;
            } else {
                p.z = Math.sign(p.z || 1) * (az < BOARD_Z ? BOARD_Z - BOARD_BAND : BOARD_Z + BOARD_BAND);
            }
        } else if (az < BOARD_Z - 1.5 && Math.abs(ax - BOARD_X) < BOARD_BAND) {
            const out = BOARD_BAND - Math.abs(ax - BOARD_X);
            if (lift < out) {
                p.y = BOARD_TOP;
            } else {
                p.x = Math.sign(p.x || 1) * (ax < BOARD_X ? BOARD_X - BOARD_BAND : BOARD_X + BOARD_BAND);
            }
        }
    }
    return p;
}

// Pushes p out of the trunk capsule of every visible player but skip.
function pushOutOfPlayers(p, pose, players, skip) {
    if (!pose || !pose.nodes || p.y > CAPSULE_TOP) {
        return p;
    }
    const n = pose.nodes;
    const visible = players && players.visible;
    for (let j = 0; j < pose.players; j++) {
        if (j === skip || (visible && !visible[j])) {
            continue;
        }
        // Node 2 is the middle of the trunk.
        const o = j * PLAYER_FLOATS + 2 * NODE_FLOATS;
        const x = n[o];
        const z = -n[o + 1];
        const dx = p.x - x;
        const dz = p.z - z;
        const d2 = dx * dx + dz * dz;
        if (d2 < CAPSULE_R * CAPSULE_R) {
            const d = Math.sqrt(d2);
            if (d < 1e-4) {
                p.x = x + CAPSULE_R;
            } else {
                p.x = x + (dx / d) * CAPSULE_R;
                p.z = z + (dz / d) * CAPSULE_R;
            }
        }
    }
    return p;
}

// A critically damped spring per axis, solved exactly so it is stable at any time step and
// starts and ends without a kick.
class Spring {
    constructor(size) {
        this.x = new Float64Array(size);
        this.v = new Float64Array(size);
    }

    reset(values) {
        for (let i = 0; i < this.x.length; i++) {
            this.x[i] = values[i];
            this.v[i] = 0;
        }
    }

    step(target, omega, dt) {
        const e = Math.exp(-omega * dt);
        for (let i = 0; i < this.x.length; i++) {
            const change = this.x[i] - target[i];
            const temp = (this.v[i] + omega * change) * dt;
            this.v[i] = (this.v[i] - omega * temp) * e;
            this.x[i] = target[i] + (change + temp) * e;
        }
    }
}

const arr3 = (v) => [v.x, v.y, v.z];

// Named views of the Field page. Each is an array of position, target and field of view that also
// carries them as the fields pos, target and fov, so flyTo takes a view spread or whole.
function view(pos, target, fov) {
    return Object.freeze(Object.assign([pos, target, fov], { pos, target, fov }));
}

export const FIELD_VIEWS = Object.freeze({
    overview: view(new THREE.Vector3(0, 34, 53), new THREE.Vector3(0, 0, -4), 58),
    grass: view(new THREE.Vector3(-30, 5, 26), new THREE.Vector3(-2, 0, 2), 60),
    lines: view(new THREE.Vector3(-57.5, 4.5, 37.5), new THREE.Vector3(10, 0, 35.2), 55),
    nets: view(new THREE.Vector3(60, 2.8, 5.5), new THREE.Vector3(55, 1.1, 0), 55),
    boards: view(new THREE.Vector3(-26, 1.6, 37.5), new THREE.Vector3(6, 0.6, 41.5), 45),
    crowd: view(new THREE.Vector3(0, 8, 12), new THREE.Vector3(0, 13, -70), 55),
    sky: view(new THREE.Vector3(-42, 18, 49), new THREE.Vector3(10, 16, -40), 65),
});

export class CameraRig {
    constructor(camera, dom) {
        this.camera = camera;
        this.mode = 'broadcast';
        this.target = new THREE.Vector3();
        this.wantPos = new THREE.Vector3();
        this.wantTarget = new THREE.Vector3();
        this.fov = 30;
        this.goalSide = 1;
        this.followIndex = 0;
        this.focusIndex = 0;
        this.playerView = 0;
        this.snap = true;
        this.scripted = null;
        // The shot of the framed modes, position, target and field of view.
        this.shotPos = new Spring(3);
        this.shotTarget = new Spring(3);
        this.shotFov = new Spring(1);
        // The ball as the broadcast camera sees it, on the ground and smoothed, and its lead.
        this.ballSpring = new Spring(3);
        this.leadSpring = new Spring(2);
        this.zoomSpring = new Spring(1);
        this.haveBall = false;
        // The raw view of the current mode before any blend between modes.
        this.rawPos = new THREE.Vector3();
        this.rawQuat = new THREE.Quaternion();
        this.rawFov = 30;
        this.blend = null;
        this.nextBlend = 0;
        // The velocity of the camera as last drawn, which a blend between modes carries on.
        this.camVel = new THREE.Vector3();
        this.lastCamPos = new THREE.Vector3();
        // Free camera, its place and velocity, the wanted and shown yaw and pitch, keys held,
        // speed in meters per second, its field of view and a glide set by flyTo.
        this.freePos = new THREE.Vector3(0, 30, 53);
        this.freeVel = new THREE.Vector3();
        this.yaw = 0;
        this.pitch = 0;
        this.viewYaw = 0;
        this.viewPitch = 0;
        this.freeFov = 50;
        this.flight = null;
        this.keys = new Set();
        this.speed = 14;
        this.fast = false;
        // Seconds without input in the free camera before onIdle runs, zero for never.
        this.idleSeconds = 0;
        this.idle = 0;
        this.onIdle = null;
        this.offset = new THREE.Vector3(0, 12, 22);
        this.followSpring = new Spring(3);
        this.fppPos = new THREE.Vector3();
        this.fppQuat = new THREE.Quaternion();
        this.hidden = { players: null, modes: new Map() };
        this.dragging = false;
        this.bindLook(dom);
        camera.position.set(0, 30, 70);
        camera.lookAt(0, 0, 0);
        this.rawPos.copy(camera.position);
        this.rawQuat.copy(camera.quaternion);
        this.rawFov = camera.fov;
        this.shotPos.reset(arr3(camera.position));
        this.shotTarget.reset([0, 0, 0]);
        this.shotFov.reset([camera.fov]);
    }

    setMode(mode) {
        if (!CAMERA_MODES.some((m) => m.id === mode) && !EXTRA_MODES.includes(mode)) {
            return;
        }
        const old = this.mode;
        const seconds = this.nextBlend || 0;
        this.nextBlend = 0;
        if (mode === old) {
            return;
        }
        const cam = this.camera;
        if (mode !== 'scripted') {
            // The view glides from where it is, carrying on with its own motion as it fades, so a
            // moving camera does not stop dead when the mode changes. A blend of no set length
            // takes longer for a longer way.
            this.blend = {
                pos: cam.position.clone(), vel: this.camVel.clone(), quat: cam.quaternion.clone(), fov: cam.fov,
                t: 0, dur: seconds || null, from: this.blend && this.blend.from === 'fpp' ? 'fpp' : old,
            };
            // The blend alone carries the view, so the new shot starts where it wants to be.
            this.shotCut = FRAMED.has(mode);
        }
        if (FRAMED.has(mode) && !FRAMED.has(old)) {
            // Start the shot from where the camera is, which a scripted shot eases on from.
            this.shotPos.reset(arr3(cam.position));
            this.shotTarget.reset(arr3(this.target));
            this.shotFov.reset([cam.fov]);
        }
        if (mode === 'fpp') {
            this.fppPos.copy(cam.position);
            this.fppQuat.copy(cam.quaternion);
            this.fov = cam.fov;
        }
        if (FOLLOW.has(mode) && !FOLLOW.has(old)) {
            // Keep the current angle and distance as the offset from the target, within limits.
            this.offset.subVectors(cam.position, this.target);
            const len = this.offset.length();
            if (len < CAMERA_BOX.followMin || this.offset.y < 1) {
                this.offset.set(0, 12, 22);
            }
            this.limitOffset();
            this.followSpring.reset(arr3(this.target));
        }
        if (mode === 'free') {
            // Start from the current view, so switching feels immediate.
            this.flight = null;
            this.freePos.copy(cam.position);
            confine(this.freePos);
            this.freeVel.set(0, 0, 0);
            const e = new THREE.Euler().setFromQuaternion(cam.quaternion, 'YXZ');
            this.yaw = e.y;
            this.pitch = THREE.MathUtils.clamp(e.x, -CAMERA_BOX.pitchLimit, CAMERA_BOX.pitchLimit);
            this.viewYaw = this.yaw;
            this.viewPitch = this.pitch;
            this.freeFov = THREE.MathUtils.clamp(cam.fov, 35, 60);
            this.idle = 0;
        }
        this.mode = mode;
        this.dom.classList.toggle('free-camera', mode === 'free' || FOLLOW.has(mode));
    }

    // Glides the free camera to a view and leaves free control there, the yaw and pitch turned
    // to match. Takes pos, target and fov, or one view of FIELD_VIEWS.
    flyTo(pos, target, fov = 50, seconds = 1.6) {
        if (pos && pos.pos) {
            seconds = target !== undefined && typeof target === 'number' ? target : seconds;
            ({ pos, target, fov } = pos);
        }
        if (this.mode !== 'free') {
            this.setMode('free');
        }
        const to = confine(pos.clone());
        const dir = new THREE.Vector3().subVectors(target, to);
        let yaw = Math.atan2(-dir.x, -dir.z);
        const pitch = THREE.MathUtils.clamp(Math.atan2(dir.y, Math.hypot(dir.x, dir.z)),
            -CAMERA_BOX.pitchLimit, CAMERA_BOX.pitchLimit);
        // Turn the short way round.
        yaw = this.viewYaw + Math.atan2(Math.sin(yaw - this.viewYaw), Math.cos(yaw - this.viewYaw));
        this.flight = {
            fromPos: this.freePos.clone(), fromYaw: this.viewYaw, fromPitch: this.viewPitch, fromFov: this.freeFov,
            toPos: to, toYaw: yaw, toPitch: pitch, toFov: fov || 50, t: 0, dur: Math.max(0.05, seconds),
        };
        this.idle = 0;
    }

    // Drag with the left button to look around, the wheel changes the flying speed or the
    // distance of a follow camera. A right click captures the pointer only when the settings
    // allow it, since a captured pointer stays confined to the window until released.
    bindLook(dom) {
        this.dom = dom;
        const steerable = () => this.mode === 'free' || FOLLOW.has(this.mode);
        dom.addEventListener('contextmenu', (e) => {
            if (!steerable()) {
                return;
            }
            e.preventDefault();
            if (document.pointerLockElement === dom) {
                document.exitPointerLock();
            } else if (this.allowLock && e.isTrusted && document.hasFocus() && dom.requestPointerLock) {
                dom.requestPointerLock();
            }
        });
        dom.addEventListener('pointerdown', (e) => {
            if (steerable() && e.button === 0) {
                this.dragging = true;
                dom.setPointerCapture(e.pointerId);
            }
        });
        const end = () => {
            this.dragging = false;
        };
        dom.addEventListener('pointerup', end);
        dom.addEventListener('pointercancel', end);
        dom.addEventListener('pointermove', (e) => {
            if (!steerable() || !(this.dragging || document.pointerLockElement === dom)) {
                return;
            }
            this.touch();
            if (FOLLOW.has(this.mode)) {
                // Orbit the offset around the target.
                const sph = new THREE.Spherical().setFromVector3(this.offset);
                sph.theta -= e.movementX * 0.005;
                sph.phi -= e.movementY * 0.005;
                this.offset.setFromSpherical(sph);
                this.limitOffset();
                return;
            }
            this.yaw -= e.movementX * 0.0035;
            this.pitch = THREE.MathUtils.clamp(this.pitch - e.movementY * 0.0035, -CAMERA_BOX.pitchLimit, CAMERA_BOX.pitchLimit);
        });
        dom.addEventListener('wheel', (e) => {
            if (!steerable()) {
                return;
            }
            e.preventDefault();
            this.touch();
            if (FOLLOW.has(this.mode)) {
                this.offset.multiplyScalar(e.deltaY > 0 ? 1.12 : 0.89);
                this.limitOffset();
                return;
            }
            this.speed = THREE.MathUtils.clamp(this.speed * (e.deltaY > 0 ? 0.85 : 1.18), CAMERA_BOX.speedMin, CAMERA_BOX.speedMax);
        }, { passive: false });
    }

    // Any input in a steered camera ends a glide and restarts the idle count.
    touch() {
        this.idle = 0;
        if (this.flight) {
            this.flight = null;
            this.yaw = this.viewYaw;
            this.pitch = this.viewPitch;
        }
    }

    // Ends a glide of flyTo where it is and hands the camera back to the user.
    stopFlight() {
        this.touch();
    }

    // The follow offset looks down between 2 and 80 degrees from a sensible distance.
    limitOffset() {
        const sph = new THREE.Spherical().setFromVector3(this.offset);
        sph.radius = THREE.MathUtils.clamp(sph.radius, CAMERA_BOX.followMin, CAMERA_BOX.followMax);
        sph.phi = THREE.MathUtils.clamp(sph.phi, Math.PI / 2 - CAMERA_BOX.pitchLimit, Math.PI / 2 - 0.035);
        this.offset.setFromSpherical(sph);
    }

    // Keys W, A, S, D move on the ground plane, E and Q go up and down, C also down.
    press(key, down, shift) {
        this.fast = shift;
        if (down) {
            this.keys.add(key);
            this.touch();
        } else {
            this.keys.delete(key);
        }
    }

    releaseAll() {
        this.keys.clear();
    }

    keyMove(forward) {
        const right = new THREE.Vector3(-forward.z, 0, forward.x);
        const move = new THREE.Vector3();
        if (this.keys.has('w')) move.add(forward);
        if (this.keys.has('s')) move.sub(forward);
        if (this.keys.has('d')) move.add(right);
        if (this.keys.has('a')) move.sub(right);
        if (this.keys.has('e')) move.y += 1;
        if (this.keys.has('q') || this.keys.has('c')) move.y -= 1;
        return move;
    }

    // Follow modes. The target glides to the wanted point, the offset stays as the user set it,
    // and W, A, S, D, E and Q nudge the offset within the box.
    followUpdate(dt, want) {
        const move = this.keyMove(new THREE.Vector3(-this.offset.x, 0, -this.offset.z).normalize());
        const moved = move.lengthSq() > 0;
        if (moved) {
            this.offset.addScaledVector(move.normalize(), this.speed * (this.fast ? 3 : 1) * dt);
            this.limitOffset();
        }
        if (this.snap) {
            this.followSpring.reset(arr3(want));
        } else {
            this.followSpring.step(arr3(want), 5, dt);
        }
        const f = this.followSpring.x;
        this.target.set(f[0], f[1], f[2]);
        this.rawPos.copy(this.target).add(this.offset);
        confine(this.rawPos);
        if (moved) {
            // The user cannot push the offset past the box, it keeps what the box allows.
            this.offset.subVectors(this.rawPos, this.target);
            this.limitOffset();
        }
        this.rawQuat.setFromRotationMatrix(new THREE.Matrix4().lookAt(this.rawPos, this.target, UP));
        this.rawFov += (40 - this.rawFov) * ease(3, dt);
    }

    flyUpdate(dt) {
        const f = this.flight;
        if (f) {
            f.t += dt;
            const s = smoother(f.t / f.dur);
            this.freePos.lerpVectors(f.fromPos, f.toPos, s);
            // A gentle arc over the pitch on longer glides.
            this.freePos.y += Math.sin(Math.PI * s) * Math.min(8, f.fromPos.distanceTo(f.toPos) * 0.12);
            confine(this.freePos);
            this.viewYaw = f.fromYaw + (f.toYaw - f.fromYaw) * s;
            this.viewPitch = f.fromPitch + (f.toPitch - f.fromPitch) * s;
            this.freeFov = f.fromFov + (f.toFov - f.fromFov) * s;
            this.yaw = this.viewYaw;
            this.pitch = this.viewPitch;
            if (f.t >= f.dur) {
                this.flight = null;
            }
        } else {
            const forward = new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
            const move = this.keyMove(forward);
            if (move.lengthSq() > 0) {
                move.normalize().multiplyScalar(this.speed * (this.fast ? 2.5 : 1));
            }
            // Speed builds and fades quickly, like a game camera.
            this.freeVel.lerp(move, ease(7, dt));
            const prev = this.freePos.clone();
            this.freePos.addScaledVector(this.freeVel, dt);
            confine(this.freePos, prev);
            this.viewYaw += (this.yaw - this.viewYaw) * ease(22, dt);
            this.viewPitch += (this.pitch - this.viewPitch) * ease(22, dt);
        }
        this.rawPos.copy(this.freePos);
        this.rawQuat.setFromEuler(new THREE.Euler(this.viewPitch, this.viewYaw, 0, 'YXZ'));
        this.rawFov += (this.freeFov - this.rawFov) * ease(5, dt);
        this.target.copy(this.rawPos).addScaledVector(new THREE.Vector3(0, 0, -1).applyQuaternion(this.rawQuat), 20);
        if (this.idleSeconds > 0 && !f && this.keys.size === 0 && !this.dragging) {
            this.idle += dt;
            if (this.idle >= this.idleSeconds) {
                this.idle = 0;
                this.nextBlend = 2.5;
                if (this.onIdle) {
                    this.onIdle();
                } else {
                    this.setMode('broadcast');
                }
                this.nextBlend = 0;
            }
        }
    }

    // Called after a seek or a cut, the next update jumps instead of easing.
    jump() {
        this.snap = true;
    }

    // The ball on the ground, followed by a critically damped spring so bounces and the corners
    // of the interpolation between frames never reach the broadcast camera, with a lead in the
    // direction of play and a zoom with its speed.
    trackBall(ball, dt) {
        const g = [ball.x, 0, ball.z];
        if (this.snap || !this.haveBall) {
            this.ballSpring.reset(g);
            this.leadSpring.reset([0, 0]);
            this.zoomSpring.reset([0]);
            this.haveBall = true;
            return;
        }
        this.ballSpring.step(g, 3, dt);
        const v = this.ballSpring.v;
        const speed = Math.hypot(v[0], v[2]);
        const lead = Math.min(1, 9 / Math.max(1e-6, speed * 0.6));
        this.leadSpring.step([v[0] * 0.6 * lead, v[2] * 0.6 * lead], 1.4, dt);
        this.zoomSpring.step([THREE.MathUtils.clamp((speed - 3) / 14, 0, 1)], 1.2, dt);
    }

    // follow is the ground point of the player to follow in owner and orbit modes, or null.
    update(dt, ball, players, pose, aspect, follow = null) {
        if (!(dt > 0)) {
            dt = 1 / 60;
        }
        this.trackBall(ball, dt);
        if (this.snap) {
            this.blend = null;
        }
        if (this.mode === 'free') {
            this.flyUpdate(dt);
        } else if (FOLLOW.has(this.mode)) {
            const b = this.ballSpring.x;
            const want = this.mode === 'owner' && follow
                ? new THREE.Vector3(follow.x, 1, follow.z)
                : new THREE.Vector3(b[0], 0.3, b[2]);
            this.followUpdate(dt, want);
        } else if (this.mode === 'fpp' && pose.nodes && this.focusIndex >= 0 && this.focusIndex < pose.players) {
            this.playerUpdate(dt, players, pose);
        } else {
            this.framedUpdate(dt, ball, players, pose, aspect, follow);
        }
        this.output(dt, players, pose);
        this.snap = false;
    }

    framedUpdate(dt, ball, players, pose, aspect, follow) {
        let fov = 30;
        let omega = 3.5;
        const b = this.ballSpring.x;
        if (this.mode === 'broadcast') {
            // A TV camera on a gantry of the main stand, panning with the play and leading it,
            // wider when play is fast and a little closer near the goals.
            const lead = this.leadSpring.x;
            const zoom = this.zoomSpring.x[0];
            const tx = THREE.MathUtils.clamp(b[0] + lead[0], -HALF_W + 4, HALF_W - 4);
            const tz = THREE.MathUtils.clamp(b[2] + lead[1], -26, 26);
            this.wantTarget.set(tx, 0, tz * 0.75);
            this.wantPos.set(tx * 0.8, 20 + zoom * 3, 63 + zoom * 4);
            fov = 26 + zoom * 6 - Math.max(0, Math.abs(tx) - 30) * 0.08;
            omega = 3.2;
        } else if (this.mode === 'tactical') {
            const halfFov = THREE.MathUtils.degToRad(17);
            const needW = (FULL_HALF_W + 2) / Math.max(0.5, aspect);
            const need = Math.max(needW, FULL_HALF_H + 2);
            const height = need / Math.tan(halfFov);
            this.wantTarget.set(b[0] * 0.15, 0, 0);
            this.wantPos.set(b[0] * 0.15, height, 6);
            fov = 34;
            omega = 2.4;
        } else if (this.mode === 'goal') {
            // Behind the goal the ball is nearer to, with hysteresis so it does not flip often.
            if (b[0] > 12) {
                this.goalSide = 1;
            } else if (b[0] < -12) {
                this.goalSide = -1;
            }
            const gx = this.goalSide * (HALF_W + 18);
            this.wantPos.set(gx, 8.5, b[2] * 0.25);
            this.wantTarget.set(b[0] * 0.6 + this.goalSide * HALF_W * 0.4, 0.5, b[2] * 0.6);
            fov = 46;
            omega = 3;
        } else if (this.mode === 'scripted' && this.scripted) {
            // A shot that a transition, a replay or a celebration sets each frame, with the wanted
            // position, target, field of view and easing rate, or snap for a hard cut.
            const s = this.scripted;
            if (s.snap) {
                this.snap = true;
                this.blend = null;
                s.snap = false;
            }
            const k = this.snap ? 1 : ease(s.rate || 3, dt);
            this.lerpShot(s.pos, s.target, s.fov || 40, k);
            this.finishShot();
            return;
        } else if (this.mode === 'lowside') {
            this.wantPos.set(b[0] * 0.92, 2.4, 44);
            this.wantTarget.set(ball.x, 1, ball.z);
            fov = 34;
            omega = 4.5;
        } else if (this.mode === 'orbit' && follow) {
            const a = performance.now() / 1000 * 0.35;
            this.wantPos.set(follow.x + Math.cos(a) * 9, 3.2, follow.z + Math.sin(a) * 9);
            this.wantTarget.set(follow.x, 1.1, follow.z);
            confine(pushOutOfPlayers(this.wantPos, pose, players, -1));
            fov = 40;
            omega = 5;
        } else if (this.mode === 'podium') {
            this.wantPos.set(0, 3.2, 14);
            this.wantTarget.set(0, 1.7, 0);
            fov = 42;
            omega = 3;
        } else if (this.mode === 'player' && players && pose.nodes) {
            const count = pose.players;
            const k = ((this.followIndex % count) + count) % count;
            const p = players.position(pose, k, new THREE.Vector3());
            const toBall = new THREE.Vector3().subVectors(ball, p).setY(0);
            if (toBall.lengthSq() < 1) {
                toBall.set(1, 0, 0);
            }
            toBall.normalize();
            this.wantPos.copy(p).addScaledVector(toBall, -9).setY(4.2);
            this.wantTarget.copy(p).addScaledVector(toBall, 6).setY(1);
            confine(this.wantPos);
            fov = 50;
            omega = 4.5;
        } else {
            // A mode without its subject keeps the last shot.
            this.wantPos.set(this.shotPos.x[0], this.shotPos.x[1], this.shotPos.x[2]);
            this.wantTarget.set(this.shotTarget.x[0], this.shotTarget.x[1], this.shotTarget.x[2]);
            fov = this.shotFov.x[0];
        }
        if (this.snap || this.shotCut) {
            this.shotCut = false;
            this.shotPos.reset(arr3(this.wantPos));
            this.shotTarget.reset(arr3(this.wantTarget));
            this.shotFov.reset([fov]);
        } else {
            this.shotPos.step(arr3(this.wantPos), omega, dt);
            this.shotTarget.step(arr3(this.wantTarget), omega * 1.15, dt);
            this.shotFov.step([fov], 2.5, dt);
        }
        this.finishShot();
    }

    // Eases the shot by a plain fraction k, as scripted shots ask.
    lerpShot(pos, target, fov, k) {
        for (let i = 0; i < 3; i++) {
            const axis = 'xyz'[i];
            this.shotPos.x[i] += (pos[axis] - this.shotPos.x[i]) * k;
            this.shotTarget.x[i] += (target[axis] - this.shotTarget.x[i]) * k;
            this.shotPos.v[i] = 0;
            this.shotTarget.v[i] = 0;
        }
        this.shotFov.x[0] += (fov - this.shotFov.x[0]) * k;
        this.shotFov.v[0] = 0;
    }

    finishShot() {
        const p = this.shotPos.x;
        const t = this.shotTarget.x;
        this.rawPos.set(p[0], p[1], p[2]);
        this.target.set(t[0], t[1], t[2]);
        this.rawQuat.setFromRotationMatrix(new THREE.Matrix4().lookAt(this.rawPos, this.target, UP));
        this.rawFov = this.shotFov.x[0];
    }

    // Player views. First person puts the camera at the eyes, looking where the head faces, the
    // others frame the player from behind, the side and the front. Position and turn are eased
    // to keep them comfortable, and the camera keeps out of the other players.
    playerUpdate(dt, players, pose) {
        const n = pose.nodes;
        const k = this.focusIndex;
        // Node 3 is the neck.
        const o = k * PLAYER_FLOATS + 3 * NODE_FLOATS;
        const q = new THREE.Quaternion(n[o + 3], n[o + 5], -n[o + 4], n[o + 6]);
        if (q.lengthSq() < 1e-8) {
            q.identity();
        }
        const neck = new THREE.Vector3(n[o], n[o + 2], -n[o + 1]);
        const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
        const flat = new THREE.Vector3(forward.x, 0, forward.z);
        if (flat.lengthSq() < 1e-6) {
            flat.set(0, 0, 1);
        }
        flat.normalize();
        const right = new THREE.Vector3(-flat.z, 0, flat.x);
        // The eyes of the model when the players can say where they are, else above the neck.
        let eyes = null;
        if (players && typeof players.eye === 'function') {
            eyes = players.eye(pose, k, new THREE.Vector3());
            if (eyes && !Number.isFinite(eyes.x + eyes.y + eyes.z)) {
                eyes = null;
            }
        }
        const head = eyes || neck.clone().setY(neck.y + 0.12);
        let eye;
        let look;
        let fovWant = 70;
        let rate = 6;
        if (this.playerView === 0) {
            eye = head.clone().addScaledVector(flat, -2.8).addScaledVector(right, 0.55);
            eye.y += 0.65;
            look = head.clone().addScaledVector(flat, 9);
            fovWant = 60;
        } else if (this.playerView === 2) {
            eye = head.clone().addScaledVector(right, 6);
            eye.y = 1.9;
            look = head.clone().setY(1.1);
            fovWant = 45;
        } else if (this.playerView === 3) {
            eye = head.clone().addScaledVector(flat, 2.6);
            eye.y += 0.05;
            look = head.clone().setY(head.y - 0.25);
            fovWant = 40;
        } else {
            forward.y = Math.max(-0.35, Math.min(0.25, forward.y));
            forward.normalize();
            if (eyes && typeof players.setHidden === 'function') {
                // The player is hidden, so the eyes themselves are the camera.
                eye = head.clone().addScaledVector(forward, 0.04);
            } else {
                // Far enough forward to clear the player's own head and hair.
                eye = neck.clone().addScaledVector(forward, 0.26);
                eye.y += 0.1;
            }
            look = eye.clone().add(forward);
            rate = 14;
        }
        pushOutOfPlayers(eye, pose, players, k);
        eye.y = Math.max(CAMERA_BOX.minY, eye.y);
        this.fppPos.lerp(eye, this.snap ? 1 : ease(rate, dt));
        const m = new THREE.Matrix4().lookAt(this.fppPos, look, UP);
        const want = new THREE.Quaternion().setFromRotationMatrix(m);
        this.fppQuat.slerp(want, this.snap ? 1 : ease(6, dt));
        this.fov += (fovWant - this.fov) * (this.snap ? 1 : ease(4, dt));
        this.rawPos.copy(this.fppPos);
        this.rawQuat.copy(this.fppQuat);
        this.rawFov = this.fov;
        this.target.copy(look);
    }

    // Hides the followed player's head over the shoulder and the whole player in first person,
    // until a blend out of those views ends. In the player views any other player whose trunk is
    // close enough to the camera to fill the screen with a hat or a shoulder is hidden too.
    hidePlayer(players, pose) {
        const want = new Map();
        const fpp = this.mode === 'fpp' || (this.blend && this.blend.from === 'fpp');
        const able = players && typeof players.setHidden === 'function';
        if (fpp && able) {
            if (this.playerView === 0 || this.playerView === 1) {
                want.set(this.focusIndex, this.playerView === 1 ? 'all' : 'head');
            }
            const p = this.camera.position;
            if (pose && pose.nodes && p.y < CAPSULE_TOP + 0.6) {
                const n = pose.nodes;
                for (let j = 0; j < pose.players; j++) {
                    if (j === this.focusIndex) {
                        continue;
                    }
                    const o = j * PLAYER_FLOATS + 2 * NODE_FLOATS;
                    if (Math.hypot(p.x - n[o], p.z + n[o + 1]) < NEAR_HIDE) {
                        want.set(j, 'all');
                    }
                }
            }
        }
        const h = this.hidden;
        if (h.players !== players) {
            h.modes.clear();
            h.players = players;
        }
        for (const k of h.modes.keys()) {
            if (!want.has(k)) {
                if (able) {
                    players.setHidden(k, 'none');
                }
                h.modes.delete(k);
            }
        }
        for (const [k, mode] of want) {
            if (h.modes.get(k) !== mode) {
                players.setHidden(k, mode);
                h.modes.set(k, mode);
            }
        }
    }

    output(dt, players, pose) {
        const cam = this.camera;
        const b = this.blend;
        let fov = this.rawFov;
        if (b) {
            if (!b.dur) {
                b.dur = THREE.MathUtils.clamp(0.8 + b.pos.distanceTo(this.rawPos) / 70, 0.9, 2.6);
            }
            b.t += dt;
            b.pos.addScaledVector(b.vel, dt);
            b.vel.multiplyScalar(Math.exp(-3 * dt));
            const s = smoother(b.t / b.dur);
            cam.position.lerpVectors(b.pos, this.rawPos, s);
            cam.quaternion.slerpQuaternions(b.quat, this.rawQuat, s);
            fov = b.fov + (this.rawFov - b.fov) * s;
            if (b.t >= b.dur) {
                this.blend = null;
            }
        } else {
            cam.position.copy(this.rawPos);
            cam.quaternion.copy(this.rawQuat);
        }
        if (this.snap) {
            this.camVel.set(0, 0, 0);
        } else {
            this.camVel.subVectors(cam.position, this.lastCamPos).divideScalar(dt).clampLength(0, 60);
        }
        this.lastCamPos.copy(cam.position);
        this.fov = this.mode === 'fpp' ? this.fov : fov;
        const near = this.mode === 'fpp' ? NEAR_PLAYER : NEAR_DEFAULT;
        if (Math.abs(cam.fov - fov) > 1e-3 || cam.near !== near) {
            cam.fov = fov;
            cam.near = near;
            cam.updateProjectionMatrix();
        }
        this.hidePlayer(players, pose);
    }
}
