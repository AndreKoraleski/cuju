// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Reader of the cuju recording format, version 1, as the section Recording format of
// docs/rendering.md describes it.

export const MAGIC = 'CUJREC1\n';
export const NODE_COUNT = 14;
export const NODE_FLOATS = 7;
export const PLAYER_FLOATS = NODE_COUNT * NODE_FLOATS;
// The decoded floats of a frame keep the ball at index 8 and the nodes from index 15.
export const BALL_FLOAT = 8;
export const NODES_FLOAT = 15;
export const NODE_NAMES = [
    'player', 'body', 'middle', 'neck', 'left_shoulder', 'left_elbow', 'right_shoulder',
    'right_elbow', 'left_thigh', 'left_knee', 'left_ankle', 'right_thigh', 'right_knee',
    'right_ankle',
];
export const NODE_PARENTS = [-1, 0, 1, 2, 2, 4, 2, 6, 1, 8, 9, 1, 11, 12];

export const GAME_MODES = [
    'Play', 'Kick-off', 'Goal kick', 'Free kick', 'Corner', 'Throw-in', 'Penalty',
];

export class FormatError extends Error {}

// Each node is stored as seven int16, the position in steps of POSITION_UNIT meters and the
// quaternion in steps of 1 over 32767, which halves the bytes and lets gzip find more repetition.
export const POSITION_UNIT = 1 / 500;
const ROTATION_UNIT = 1 / 32767;

// GRF's e_FunctionType, the kind of animation a player runs, as the action bytes store it.
export const ACTION = {
    none: 0, movement: 1, ballControl: 2, trap: 3, shortPass: 4, longPass: 5, highPass: 6,
    header: 7, shot: 8, deflect: 9, catch: 10, interfere: 11, trip: 12, sliding: 13, special: 14,
};

// The touch bytes hold 0 when the player did not touch the ball during the step, else one more
// than GRF's e_TouchType.
export const TOUCH = { none: 0, kicked: 1, nonKicked: 2, accidental: 3 };

// Each of the three byte arrays that close a frame is padded to four bytes.
export function paddedPlayers(players) {
    return (players + 3) & ~3;
}

// The head of a frame, eight int32, the ball's seven float32 and four int32 of the referee.
export const HEAD_BYTES = 76;

export function frameBytes(players) {
    return HEAD_BYTES + players * 196 + 3 * paddedPlayers(players);
}

// GRF's kinds of foul, as the foul type of a frame stores them.
export const FOUL = { none: 0, foul: 1, yellow: 2, red: 3 };

export function isGzip(bytes) {
    return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

// A full match of 11 against 11 is about 13 MB once decompressed, far below this cap, which
// stops a hostile file from inflating into all of the browser's memory.
export const MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024;

export async function gunzip(bytes, limit = MAX_DECOMPRESSED_BYTES) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const reader = stream.getReader();
    const chunks = [];
    let total = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        total += value.length;
        if (total > limit) {
            await reader.cancel();
            throw new FormatError(`The recording decompresses to more than ${Math.round(limit / 1048576)} MB.`);
        }
        chunks.push(value);
    }
    const out = new Uint8Array(total);
    let at = 0;
    for (const chunk of chunks) {
        out.set(chunk, at);
        at += chunk.length;
    }
    return out;
}

// Returns the header and the offset of the frames once the magic and header are complete, or
// null when more bytes are needed. Throws FormatError on anything malformed.
export function parseHeader(bytes) {
    if (bytes.length < 12) {
        return null;
    }
    for (let i = 0; i < 8; i++) {
        if (bytes[i] !== MAGIC.charCodeAt(i)) {
            throw new FormatError('This is not a cuju recording, the file does not start with CUJREC1.');
        }
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const length = view.getUint32(8, true);
    // A header of 22 players takes a few kilobytes, a megabyte is plenty.
    if (length > 1024 * 1024) {
        throw new FormatError('The recording header is implausibly large.');
    }
    if (bytes.length < 12 + length) {
        return null;
    }
    let header;
    try {
        header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + length)));
    } catch (err) {
        throw new FormatError(`The recording header is not valid JSON, ${err.message}.`);
    }
    validateHeader(header);
    return { header, offset: 12 + length };
}

export function validateHeader(header) {
    if (header.version !== 1) {
        throw new FormatError(`Recording version ${header.version} is not supported, this viewer reads version 1.`);
    }
    if (!Array.isArray(header.players) || header.players.length === 0) {
        throw new FormatError('The recording header lists no players.');
    }
    if (Array.isArray(header.nodes) && header.nodes.length !== NODE_COUNT) {
        throw new FormatError(`The recording has ${header.nodes.length} skeleton nodes, the viewer expects ${NODE_COUNT}.`);
    }
    const expected = frameBytes(header.players.length);
    const n = header.players.length;
    if (header.frame_bytes === 60 + n * 196 + paddedPlayers(n) || header.frame_bytes === 60 + n * 196 + 3 * paddedPlayers(n)) {
        throw new FormatError('This recording comes from an older cuju whose frames carry no actions, touches or referee calls, record it again to watch it here.');
    }
    if (header.frame_bytes !== expected) {
        throw new FormatError(`The header says frames are ${header.frame_bytes} bytes, ${header.players.length} players need ${expected}.`);
    }
    if (!Array.isArray(header.teams) || header.teams.length < 2) {
        header.teams = [
            { name: 'Left', color: [200, 40, 40], secondary: [240, 240, 240] },
            { name: 'Right', color: [40, 80, 200], secondary: [30, 30, 40] },
        ];
    }
    header.steps_per_second = header.steps_per_second || 10;
}

// Wraps one frame. The bytes must be aligned to four bytes, the caller copies when they are not.
// The frame keeps copies of what it needs, so the raw bytes can be freed once it is made.
export function makeFrame(buffer, byteOffset, header) {
    const ints = new Int32Array(buffer, byteOffset, 8);
    const players = header.players.length;
    // The nodes are decoded once into floats, laid out as the readers of data expect them.
    const data = new Float32Array(NODES_FLOAT + players * PLAYER_FLOATS);
    data.set(new Float32Array(buffer, byteOffset + BALL_FLOAT * 4, 7), BALL_FLOAT);
    const referee = new Int32Array(buffer, byteOffset + 60, 4);
    const packed = new Int16Array(buffer, byteOffset + HEAD_BYTES, players * PLAYER_FLOATS);
    for (let i = 0; i < packed.length; i++) {
        const unit = i % NODE_FLOATS < 3 ? POSITION_UNIT : ROTATION_UNIT;
        data[NODES_FLOAT + i] = packed[i] * unit;
    }
    // Flags, actions and touches close the frame, one padded byte array each.
    const padded = paddedPlayers(players);
    const bytesAt = byteOffset + HEAD_BYTES + players * PLAYER_FLOATS * 2;
    return {
        step: ints[0],
        gameMode: ints[1],
        scoreLeft: ints[2],
        scoreRight: ints[3],
        ownedTeam: ints[4],
        ownedPlayer: ints[5],
        timeMs: ints[6],
        flags: ints[7],
        // The referee's decision on this step, the offender and victim as header player indices.
        foul: { player: referee[0], victim: referee[1], type: referee[2] },
        // The first player in an offside position at the last pass, a call only when play stops.
        offside: referee[3],
        data,
        playerFlags: new Uint8Array(buffer, bytesAt, players).slice(),
        actions: new Uint8Array(buffer, bytesAt + padded, players).slice(),
        touches: new Uint8Array(buffer, bytesAt + 2 * padded, players).slice(),
    };
}

// Parses a whole recording, compressed with gzip or not, into its header and frames.
export async function parseRecording(input) {
    let bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    if (isGzip(bytes)) {
        try {
            bytes = await gunzip(bytes);
        } catch (err) {
            throw new FormatError(`The recording could not be decompressed, ${err.message}.`);
        }
    }
    const parsed = parseHeader(bytes);
    if (!parsed) {
        throw new FormatError('The recording is truncated inside its header.');
    }
    const { header, offset } = parsed;
    const size = header.frame_bytes;
    const count = Math.floor((bytes.length - offset) / size);
    // The writer pads the header to four bytes, and a copy keeps older or foreign files readable.
    const aligned = bytes.slice(offset, offset + count * size).buffer;
    const frames = new Array(count);
    for (let i = 0; i < count; i++) {
        frames[i] = makeFrame(aligned, i * size, header);
    }
    if (header.frame_count != null && header.frame_count !== count) {
        console.warn(`header says ${header.frame_count} frames, file holds ${count}`);
    }
    return { header, frames };
}

// Incremental parser for the live stream, fed with chunks as they arrive.
export class StreamParser {
    constructor(onHeader, onFrame) {
        this.onHeader = onHeader;
        this.onFrame = onFrame;
        this.buffer = new Uint8Array(1 << 16);
        this.length = 0;
        this.header = null;
    }

    push(chunk) {
        if (this.length + chunk.length > this.buffer.length) {
            let size = this.buffer.length;
            while (size < this.length + chunk.length) {
                size *= 2;
            }
            const grown = new Uint8Array(size);
            grown.set(this.buffer.subarray(0, this.length));
            this.buffer = grown;
        }
        this.buffer.set(chunk, this.length);
        this.length += chunk.length;
        let start = 0;
        if (!this.header) {
            const parsed = parseHeader(this.buffer.subarray(0, this.length));
            if (!parsed) {
                return;
            }
            this.header = parsed.header;
            start = parsed.offset;
            this.onHeader(this.header);
        }
        const size = this.header.frame_bytes;
        while (this.length - start >= size) {
            const copy = this.buffer.slice(start, start + size).buffer;
            this.onFrame(makeFrame(copy, 0, this.header));
            start += size;
        }
        this.buffer.copyWithin(0, start, this.length);
        this.length -= start;
    }
}

// Finds frames where the score changes, the first frame of each goal.
export function findGoals(frames, from = 1, goals = []) {
    for (let i = Math.max(1, from); i < frames.length; i++) {
        const a = frames[i - 1];
        const b = frames[i];
        if (b.scoreLeft > a.scoreLeft || b.scoreRight > a.scoreRight) {
            goals.push({ index: i, team: b.scoreLeft > a.scoreLeft ? 0 : 1, step: b.step });
        }
    }
    return goals;
}

// Bits of a frame's flags. EPISODE_END marks the frame that ends an episode, EPISODE_START a
// frame that starts the next one when the step that ended the last was never stored.
export const EPISODE_END = 1;
export const EPISODE_START = 2;

// A live source that runs faster than real time drops steps, gaps up to this long still blend.
const MAX_BLEND_GAP = 10;

// Whether the episode ends between frames a and b.
export function episodeBoundary(a, b) {
    return !!(a.flags & EPISODE_END) || !!(b.flags & EPISODE_START) || b.step <= a.step;
}

// A cut is an episode boundary or a gap of more than a second, never interpolate across it.
export function isCut(a, b) {
    return episodeBoundary(a, b) || b.step - a.step > MAX_BLEND_GAP;
}
