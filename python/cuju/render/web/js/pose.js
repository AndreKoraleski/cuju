// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Interpolation of frames. Poses stay in the engine frame, z up in meters, and the scenes turn
// them into three.js, which is y up, as they draw.

import { BALL_FLOAT, NODE_COUNT, NODE_FLOATS, NODES_FLOAT, PLAYER_FLOATS } from './format.js';

// Interpolated pose of one moment, all in engine coordinates.
export class Pose {
    constructor(players) {
        this.players = players;
        this.nodes = new Float32Array(players * PLAYER_FLOATS);
        this.ball = new Float32Array(7);
        this.ballKnownRotation = false;
    }

    // Blends frame a and b with weight w of b, b may be null for no blending.
    compute(a, b, w) {
        const da = a.data;
        const db = b ? b.data : da;
        if (!b) {
            w = 0;
        }
        const out = this.nodes;
        const count = this.players * NODE_COUNT;
        for (let k = 0; k < count; k++) {
            const s = NODES_FLOAT + k * NODE_FLOATS;
            const o = k * NODE_FLOATS;
            blend(da, db, s, w, out, o);
        }
        blend(da, db, BALL_FLOAT, w, this.ball, 0);
        const r = BALL_FLOAT + 3;
        this.ballKnownRotation = !(da[r] === 0 && da[r + 1] === 0 && da[r + 2] === 0 && da[r + 3] === 0);
    }
}

// Lerps a position and slerps a quaternion, seven floats from offset s into out at o.
function blend(da, db, s, w, out, o) {
    out[o] = da[s] + (db[s] - da[s]) * w;
    out[o + 1] = da[s + 1] + (db[s + 1] - da[s + 1]) * w;
    out[o + 2] = da[s + 2] + (db[s + 2] - da[s + 2]) * w;
    let ax = da[s + 3];
    let ay = da[s + 4];
    let az = da[s + 5];
    let aw = da[s + 6];
    let bx = db[s + 3];
    let by = db[s + 4];
    let bz = db[s + 5];
    let bw = db[s + 6];
    if (ax === 0 && ay === 0 && az === 0 && aw === 0) {
        ax = bx; ay = by; az = bz; aw = bw;
    }
    if (bx === 0 && by === 0 && bz === 0 && bw === 0) {
        bx = ax; by = ay; bz = az; bw = aw;
    }
    if (ax === 0 && ay === 0 && az === 0 && aw === 0) {
        out[o + 3] = 0; out[o + 4] = 0; out[o + 5] = 0; out[o + 6] = 0;
        return;
    }
    let cos = ax * bx + ay * by + az * bz + aw * bw;
    if (cos < 0) {
        cos = -cos;
        bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    let ka;
    let kb;
    if (cos > 0.9995) {
        ka = 1 - w;
        kb = w;
    } else {
        const theta = Math.acos(cos);
        const sin = Math.sin(theta);
        ka = Math.sin((1 - w) * theta) / sin;
        kb = Math.sin(w * theta) / sin;
    }
    let x = ax * ka + bx * kb;
    let y = ay * ka + by * kb;
    let z = az * ka + bz * kb;
    let q = aw * ka + bw * kb;
    const n = Math.hypot(x, y, z, q) || 1;
    out[o + 3] = x / n;
    out[o + 4] = y / n;
    out[o + 5] = z / n;
    out[o + 6] = q / n;
}
