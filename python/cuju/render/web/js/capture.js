// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A frame stepper for capturing video of the page, active only when the address carries the query
// parameter capture. While stepping, the clocks, timers, animation frames and CSS animations of the
// page advance by a fixed step per captured frame, whatever the speed of drawing.

(() => {
    if (!new URLSearchParams(location.search).has('capture')) {
        return;
    }
    const realNow = performance.now.bind(performance);
    const realFrame = window.requestAnimationFrame.bind(window);
    const realCancel = window.cancelAnimationFrame.bind(window);
    const realTimeout = window.setTimeout.bind(window);
    const realClear = window.clearTimeout.bind(window);
    const epoch = Date.now() - realNow();
    let offset = 0;
    let stepped = false;
    let clock = 0;
    let next = 1e9;
    const frames = new Map();
    const timers = new Map();
    const now = () => (stepped ? clock : realNow() + offset);
    performance.now = now;
    Date.now = () => Math.round(epoch + now());
    window.requestAnimationFrame = (fn) => {
        if (!stepped) {
            return realFrame(fn);
        }
        frames.set(++next, fn);
        return next;
    };
    window.cancelAnimationFrame = (id) => (frames.delete(id) ? undefined : realCancel(id));
    window.setTimeout = (fn, ms = 0, ...args) => {
        if (!stepped || typeof fn !== 'function') {
            return realTimeout(fn, ms, ...args);
        }
        timers.set(++next, { at: clock + Math.max(0, ms), fn, args });
        return next;
    };
    window.clearTimeout = (id) => (timers.delete(id) ? undefined : realClear(id));
    const advanceAnimations = (ms) => {
        for (const a of document.getAnimations()) {
            if (a.playState === 'finished') {
                continue;
            }
            if (a.playState === 'running') {
                a.pause();
            }
            const end = a.effect ? a.effect.getComputedTiming().endTime : Infinity;
            const t = (a.currentTime || 0) + ms * (a.playbackRate || 1);
            if (t >= end) {
                a.finish();
            } else {
                a.currentTime = t;
            }
        }
    };
    window.cujuCapture = {
        // Freezes the clocks, the page then moves only by step.
        start() {
            if (!stepped) {
                clock = now();
                stepped = true;
            }
        },
        // Moves the page by ms and draws one frame.
        step(ms) {
            clock += ms;
            for (const [id, t] of [...timers].sort((a, b) => a[1].at - b[1].at)) {
                if (t.at <= clock && timers.delete(id)) {
                    t.fn(...t.args);
                }
            }
            advanceAnimations(ms);
            const due = [...frames.values()];
            frames.clear();
            for (const fn of due) {
                fn(clock);
            }
        },
        // Returns the page to real time, from the stepped clock onwards.
        stop() {
            if (stepped) {
                stepped = false;
                offset = clock - realNow();
                for (const fn of frames.values()) {
                    realFrame(fn);
                }
                frames.clear();
                for (const t of timers.values()) {
                    realTimeout(t.fn, Math.max(0, t.at - clock), ...t.args);
                }
                timers.clear();
            }
        },
    };
})();
