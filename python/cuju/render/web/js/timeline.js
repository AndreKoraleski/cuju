// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Frame storage and the playback clock, shared by recordings and live streams.
// Positions on the timeline are absolute frame numbers, so live frames dropped from the front
// do not move anything, the first kept frame is just no longer zero.

import { findGoals, isCut } from './format.js';

// A recording keeps all its frames. A live store is a ring of at most limit frames, so a long
// stream drops its oldest frames in constant time.
export class FrameStore {
    constructor(header, frames = [], limit = Infinity) {
        this.header = header;
        this.limit = limit;
        this.ring = Number.isFinite(limit) ? new Array(limit) : frames;
        this.head = 0;
        this.count = frames.length;
        this.base = 0;
        if (this.ring !== frames) {
            frames.slice(-limit).forEach((f, i) => {
                this.ring[i] = f;
            });
            this.count = Math.min(frames.length, limit);
            this.base = frames.length - this.count;
        }
        this.goals = findGoals(frames).filter((g) => g.index >= this.base);
    }

    get first() {
        return this.base;
    }

    get last() {
        return this.base + this.count - 1;
    }

    get size() {
        return this.count;
    }

    get(index) {
        const i = index - this.base;
        if (i < 0 || i >= this.count) {
            return undefined;
        }
        return Number.isFinite(this.limit) ? this.ring[(this.head + i) % this.limit] : this.ring[i];
    }

    push(frame) {
        const previous = this.count ? this.get(this.last) : null;
        if (!Number.isFinite(this.limit)) {
            this.ring.push(frame);
            this.count++;
        } else if (this.count < this.limit) {
            this.ring[(this.head + this.count) % this.limit] = frame;
            this.count++;
        } else {
            this.ring[this.head] = frame;
            this.head = (this.head + 1) % this.limit;
            this.base++;
            if (this.goals.length && this.goals[0].index < this.base) {
                this.goals = this.goals.filter((g) => g.index >= this.base);
            }
        }
        if (previous && (frame.scoreLeft > previous.scoreLeft || frame.scoreRight > previous.scoreRight)) {
            this.goals.push({
                index: this.last,
                team: frame.scoreLeft > previous.scoreLeft ? 0 : 1,
                step: frame.step,
            });
        }
    }
}

// A live view plays three seconds behind the newest frame, which absorbs network jitter and
// leaves the restart transitions frames to look ahead into, the taker and the first step.
export const LIVE_DELAY_FRAMES = 30;

export class Timeline {
    constructor(store, live) {
        this.store = store;
        this.live = live;
        this.sps = store.header.steps_per_second || 10;
        this.t = store.first;
        this.playing = false;
        this.direction = 1;
        this.speed = 1;
        this.following = live;
        this.onEnd = null;
        // Frames the view keeps behind the newest one, how strongly it leans towards that, and
        // the gap past which it jumps. A game that a person plays keeps a small delay.
        this.delay = LIVE_DELAY_FRAMES;
        this.lean = 0.12;
        this.jump = 90;
    }

    get first() {
        return this.store.first;
    }

    get last() {
        return Math.max(this.store.first, this.store.last);
    }

    get index() {
        return Math.floor(this.t);
    }

    // Seconds between the newest frame and the frame on screen.
    get behindLive() {
        return Math.max(0, (this.store.last - this.t) / this.sps);
    }

    update(dt) {
        if (this.store.size === 0) {
            return;
        }
        // Up to a second per update, so playback keeps real time even at a few frames a second.
        dt = Math.min(Math.max(0, dt), 1);
        if (this.live && this.following) {
            this.followLive(dt);
            return;
        }
        if (!this.playing) {
            this.clamp();
            return;
        }
        this.t += this.direction * this.speed * this.sps * dt;
        if (this.t >= this.last) {
            this.t = this.last;
            if (this.live) {
                this.following = true;
                this.speed = 1;
            } else {
                this.playing = false;
                if (this.onEnd) {
                    this.onEnd();
                }
            }
        } else if (this.t <= this.first) {
            this.t = this.first;
            this.playing = false;
        }
    }

    // Plays at real time a few frames behind the newest one, which absorbs network jitter.
    // The rate leans towards the target so bursts and gaps even out without visible jumps.
    followLive(dt) {
        this.playing = true;
        this.direction = 1;
        const target = this.last - this.delay;
        // A restart's transition and pre-roll hold play for a few seconds, which the lean below
        // makes up for, only a longer gap jumps.
        if (target - this.t > this.jump || this.t < this.first) {
            this.t = Math.max(this.first, target);
            return;
        }
        const rate = Math.min(1.6, Math.max(0.6, 1 + (target - this.t) * this.lean));
        this.t = Math.min(this.last, this.t + rate * this.sps * dt);
    }

    clamp() {
        this.t = Math.min(this.last, Math.max(this.first, this.t));
    }

    // Returns the two frames around t and the blend weight, zero across a cut.
    sample() {
        this.clamp();
        const i = Math.floor(this.t);
        const a = this.store.get(i);
        const b = i + 1 <= this.store.last ? this.store.get(i + 1) : null;
        let w = this.t - i;
        if (!b || isCut(a, b)) {
            w = 0;
        }
        return { index: i, a, b: w > 0 ? b : null, w };
    }

    play(direction = 1) {
        if (this.store.size === 0) {
            return;
        }
        this.direction = direction;
        if (!this.live && direction > 0 && this.t >= this.last) {
            this.t = this.first;
        }
        if (direction < 0 && this.t <= this.first) {
            this.t = this.last;
        }
        if (this.live && direction < 0) {
            this.following = false;
        }
        this.playing = true;
    }

    pause() {
        this.playing = false;
        this.following = false;
    }

    toggle() {
        if (this.playing) {
            this.pause();
        } else {
            this.play(this.direction);
        }
    }

    seek(t) {
        this.t = t;
        this.clamp();
        if (this.live && this.t < this.last - this.delay - 1) {
            this.following = false;
        }
    }

    // Steps whole frames from where the view is. Forwards counts from the frame shown and
    // backwards from the next one, so a position between two frames lands on the nearer one in
    // the direction of the step and never skips a frame.
    stepFrames(n) {
        this.pause();
        const from = n > 0 ? Math.floor(this.t + 1e-6) : Math.ceil(this.t - 1e-6);
        this.seek(from + n);
    }

    skipSeconds(seconds) {
        const wasFollowing = this.following;
        this.seek(this.t + seconds * this.sps);
        if (this.live && !wasFollowing && this.t >= this.last - this.delay) {
            this.goLive();
        } else if (wasFollowing && seconds < 0) {
            this.following = false;
            this.playing = true;
            this.direction = 1;
        }
    }

    goLive() {
        this.following = true;
        this.playing = true;
        this.direction = 1;
        this.speed = 1;
        this.t = Math.max(this.first, this.last - this.delay);
    }

    restart() {
        this.seek(this.first);
    }

    // Goal before or after the current position, by frame number. Lead is how far before a goal
    // the jumps land, so a press right after a jump moves on to the next goal.
    goalNear(direction, lead = 0) {
        const goals = this.store.goals;
        const here = this.t + lead;
        if (direction > 0) {
            return goals.find((g) => g.index > here + 1) || null;
        }
        for (let k = goals.length - 1; k >= 0; k--) {
            if (goals[k].index < here - 1) {
                return goals[k];
            }
        }
        return null;
    }
}
