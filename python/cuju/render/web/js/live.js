// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Reads a live stream into a growing frame store, like a DVR, keeping the newest frames. A source
// opens the stream with up to ten seconds of frames it captured before the viewer joined, which
// arrive at once and land in the store like any other frames.

import { api } from './api.js';
import { NODES_FLOAT, PLAYER_FLOATS, StreamParser } from './format.js';
import { FrameStore } from './timeline.js';

// A stream keeps its newest frames within this memory, which holds 15 minutes of 11 against 11
// and longer for smaller teams, up to LIVE_KEEP_MINUTES.
export const LIVE_KEEP_MINUTES = 30;
const LIVE_BUDGET_BYTES = 80 * 1024 * 1024;

// Frames a live store keeps, from the size of one decoded frame.
export function liveLimit(header) {
    const players = header.players.length;
    const frameBytes = (NODES_FLOAT + players * PLAYER_FLOATS) * 4 + 3 * players + 160;
    const sps = header.steps_per_second || 10;
    return Math.max(60 * sps, Math.min(LIVE_KEEP_MINUTES * 60 * sps, Math.floor(LIVE_BUDGET_BYTES / frameBytes)));
}
// The backlog arrives within this long after the header, frames after it come at real time.
const BACKLOG_SETTLE_MS = 700;
// Frames that may run ahead of real time from network jitter alone.
const JITTER_FRAMES = 2;

export class LiveSource {
    // A full URL reaches a stream on another server, a tunnel for example. The caller only passes
    // one that the user typed, never one from a shared link.
    // open, when given, opens the stream in place of the live source's own, as a game does.
    constructor(id, { onHeader, onFrame, onEnd, onError, open = null }) {
        this.id = id;
        this.open = open;
        this.url = /^https?:\/\//i.test(id) ? id : null;
        this.abort = new AbortController();
        this.store = null;
        this.ended = false;
        this.callbacks = { onHeader, onFrame, onEnd, onError };
        this.received = 0;
        this.headerAt = 0;
        this.backlog = 0;
    }

    // Resolves with the whole seconds of history the stream opened with, once they have arrived.
    async backlogSeconds() {
        if (!this.headerAt) {
            return 0;
        }
        const wait = this.headerAt + BACKLOG_SETTLE_MS - performance.now();
        if (wait > 0) {
            await new Promise((resolve) => setTimeout(resolve, wait));
        }
        const sps = (this.store && this.store.header.steps_per_second) || 10;
        return Math.round(this.backlog / sps);
    }

    async start() {
        let res;
        try {
            res = this.open
                ? await this.open(this.abort.signal)
                : this.url
                    ? await fetch(this.url, { signal: this.abort.signal, cache: 'no-store' })
                    : await api.liveStream(this.id, this.abort.signal);
        } catch (err) {
            if (err.name !== 'AbortError') {
                this.callbacks.onError(new Error(`Cannot reach the live stream, ${err.message}.${this.url ? ' A stream on another server must allow this page, through CORS.' : ''}`));
            }
            return;
        }
        if (!res.ok || !res.body) {
            this.stop();
            this.callbacks.onError(new Error(`The live stream answered ${res.status} ${res.statusText}.`));
            return;
        }
        const parser = new StreamParser(
            (header) => {
                this.store = new FrameStore(header, [], liveLimit(header));
                this.headerAt = performance.now();
                this.callbacks.onHeader(header, this.store);
            },
            (frame) => {
                this.received++;
                // The backlog is how far the frames ran ahead of real time just after the header.
                const elapsed = (performance.now() - this.headerAt) / 1000;
                if (elapsed < 3) {
                    const ahead = this.received - elapsed * (this.store.header.steps_per_second || 10) - JITTER_FRAMES;
                    this.backlog = Math.max(this.backlog, ahead);
                }
                this.store.push(frame);
                this.callbacks.onFrame(frame);
            },
        );
        const reader = res.body.getReader();
        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }
                parser.push(value);
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                return;
            }
            // A malformed stream stops here, so the connection does not keep the source busy.
            this.stop();
            this.callbacks.onError(err);
            return;
        }
        this.ended = true;
        this.callbacks.onEnd();
    }

    stop() {
        this.abort.abort();
    }
}
