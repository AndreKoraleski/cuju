// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The broadcast graphics over the view, the scorebug, restart banners, the goal graphic, lower
// thirds for players, the replay badge, the stinger between live play and replays, and notices.

import { GAME_MODES } from './format.js';
import { css } from './kits.js';

const $ = (id) => document.getElementById(id);

export function formatClock(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function formatSeconds(seconds) {
    // Rounding first keeps a fraction below zero from showing as minus zero.
    const rounded = Math.round(seconds) || 0;
    const sign = rounded < 0 ? '-' : '';
    const t = Math.abs(rounded);
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = t % 60;
    const mm = h ? String(m).padStart(2, '0') : String(m);
    return `${sign}${h ? `${h}:` : ''}${mm}:${String(s).padStart(2, '0')}`;
}

// The names of the two teams as the page shows them, two teams of the same name told apart by
// their side.
export function teamNames(header) {
    const raw = [0, 1].map((t) => ((header.teams[t] && header.teams[t].name) || (t ? 'Right' : 'Left')).trim());
    if (raw[0].toLowerCase() === raw[1].toLowerCase()) {
        return [`${raw[0]} (left)`, `${raw[1]} (right)`];
    }
    return raw;
}

// The bare name of a team, without the side added to tell two of the same name apart.
export function bareTeamName(header, t) {
    return ((header.teams[t] && header.teams[t].name) || (t ? 'Right' : 'Left')).trim();
}

// A swatch of a team's kit, its shirt color, split with the second color for patterned kits.
export function kitSwatch(style) {
    if (!style || !style.kit) {
        return 'var(--accent)';
    }
    const a = css(style.kit.primary);
    const b = css(style.kit.secondary);
    if (style.kit.pattern === 'plain') {
        return a;
    }
    return `linear-gradient(90deg, ${a} 0 50%, ${b} 50% 100%)`;
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) {
        e.className = cls;
    }
    if (text != null) {
        e.textContent = text;
    }
    return e;
}

export class Hud {
    constructor() {
        this.el = {
            board: $('scoreboard'),
            leftName: document.querySelector('#scoreboard .left .name'),
            rightName: document.querySelector('#scoreboard .right .name'),
            leftChip: document.querySelector('#scoreboard .left .chip'),
            rightChip: document.querySelector('#scoreboard .right .chip'),
            leftBadge: document.querySelector('#scoreboard .left .badge'),
            rightBadge: document.querySelector('#scoreboard .right .badge'),
            leftScore: $('score-left'),
            rightScore: $('score-right'),
            clock: $('clock'),
            mode: $('mode-banner'),
            goal: $('goal-banner'),
            goalTeam: $('goal-team'),
            replay: $('replay-badge'),
            toast: $('toast'),
            follow: $('follow-label'),
        };
        this.lastMode = -1;
        this.modeTimer = null;
        this.goalTimer = null;
        this.toastTimer = null;
        this.stingerTimers = [];
        this.score = [-1, -1];
        this.frozen = false;
        this.scoreFrom = null;
        this.styles = [null, null];
    }

    setTeams(header, styles) {
        const names = teamNames(header);
        this.el.leftName.textContent = names[0];
        this.el.rightName.textContent = names[1];
        this.el.leftName.title = names[0];
        this.el.rightName.title = names[1];
        this.el.leftChip.style.background = kitSwatch(styles[0]);
        this.el.rightChip.style.background = kitSwatch(styles[1]);
        this.el.leftBadge.textContent = styles[0].badge || '';
        this.el.rightBadge.textContent = styles[1].badge || '';
        this.el.leftBadge.hidden = !styles[0].badge;
        this.el.rightBadge.hidden = !styles[1].badge;
        this.el.board.hidden = false;
        this.teamNames = names;
        this.styles = styles;
        this.lastMode = -1;
        this.score = [-1, -1];
    }

    chip(style) {
        return kitSwatch(style);
    }

    hide() {
        this.el.board.hidden = true;
        this.el.mode.classList.remove('show');
        this.el.goal.classList.remove('show');
    }

    update(frame) {
        if (this.frozen) {
            return;
        }
        const scored = this.scoreFrom || frame;
        this.setScore(scored.scoreLeft, scored.scoreRight);
        this.el.clock.textContent = formatClock(frame.timeMs);
        if (frame.gameMode !== this.lastMode) {
            this.lastMode = frame.gameMode;
            if (frame.gameMode > 0) {
                this.showMode(GAME_MODES[frame.gameMode] || `Mode ${frame.gameMode}`);
            } else {
                this.el.mode.classList.remove('show');
            }
        }
    }

    // A score that goes up by one while the bug shows it pulses, other changes, from a seek or
    // a new match, just change.
    setScore(left, right) {
        const [l0, r0] = this.score;
        if (left === l0 && right === r0) {
            return;
        }
        this.score = [left, right];
        this.el.leftScore.textContent = left;
        this.el.rightScore.textContent = right;
        const bump = (node) => {
            node.classList.remove('bump');
            void node.offsetWidth;
            node.classList.add('bump');
        };
        if (l0 >= 0 && left === l0 + 1 && right === r0) {
            bump(this.el.leftScore);
        } else if (r0 >= 0 && right === r0 + 1 && left === l0) {
            bump(this.el.rightScore);
        }
    }

    // Holds the bug at a frame's score and clock, as the show after the match asks.
    freeze(frame) {
        this.frozen = false;
        if (frame) {
            this.update(frame);
        }
        this.frozen = true;
        this.el.mode.classList.remove('show');
    }

    unfreeze() {
        this.frozen = false;
        this.lastMode = -1;
    }

    // Shows the score of another frame than the one on screen, the goal's own while the frame
    // before it holds for the celebration, or null for the frame on screen.
    scoreOf(frame) {
        this.scoreFrom = frame || null;
    }

    showMode(text) {
        this.el.mode.textContent = text;
        this.el.mode.classList.add('show');
        clearTimeout(this.modeTimer);
        this.modeTimer = setTimeout(() => this.el.mode.classList.remove('show'), 2600);
    }

    // The goal graphic, a lower third with the team, the new score, the scorer with his number
    // and minute, and the assist. info may come a moment later than the goal, from the director.
    showGoal(team, canReplay, info = null, ms = 7000) {
        const g = this.el.goal;
        this.el.goalTeam.textContent = this.teamNames ? this.teamNames[team] : '';
        const chip = g.querySelector('.goal-chip');
        if (chip) {
            chip.style.background = kitSwatch(this.styles[team]);
        }
        const set = (id, text) => {
            const node = $(id);
            if (node) {
                node.textContent = text || '';
                node.hidden = !text;
            }
        };
        if (info) {
            set('goal-score', info.score);
            set('goal-number', info.number != null ? String(info.number) : '');
            set('goal-scorer', info.name);
            set('goal-minute', info.minute ? `${info.minute}′` : '');
            set('goal-assist', info.assist);
            const num = $('goal-number');
            if (num && info.color) {
                num.style.background = info.color;
            }
            g.classList.toggle('has-scorer', !!info.name);
        } else if (!g.classList.contains('show')) {
            for (const id of ['goal-score', 'goal-number', 'goal-scorer', 'goal-minute', 'goal-assist']) {
                set(id, '');
            }
            g.classList.remove('has-scorer');
        }
        g.dataset.team = String(team);
        g.classList.toggle('can-replay', canReplay);
        // Its replay button asks for a click, so the page gives the mouse back while it shows.
        if (canReplay) {
            g.dataset.prompt = '';
        }
        g.classList.add('show');
        clearTimeout(this.goalTimer);
        this.goalTimer = setTimeout(() => this.hideGoal(), ms);
    }

    hideGoal() {
        delete this.el.goal.dataset.prompt;
        this.el.goal.classList.remove('show');
    }

    // The replay badge, with a hint of the keys while a replay runs.
    setReplay(on, hint = '') {
        this.el.replay.hidden = !on;
        const h = $('replay-hint');
        if (h) {
            h.textContent = hint;
            h.hidden = !hint;
        }
    }

    // The lower third of a followed player, his number on his team's color, his name, and the
    // team and view below. info is null to hide it.
    followLabel(info, view = '') {
        const f = this.el.follow;
        if (!info) {
            f.hidden = true;
            return;
        }
        const num = el('span', 'lt-number', String(info.number));
        num.style.background = info.color || 'var(--accent)';
        num.style.color = info.ink || '#fff';
        const body = el('span', 'lt-body');
        body.append(el('span', 'lt-name', info.name), el('span', 'lt-sub', view ? `${info.teamName}, ${view}` : info.teamName));
        f.replaceChildren(num, body);
        f.hidden = false;
    }

    // A quick branded wipe. onCovered runs while the wipe hides the view, where a cut belongs,
    // and onDone once it has cleared.
    stinger(onCovered, onDone = null) {
        const s = $('stinger');
        for (const t of this.stingerTimers) {
            clearTimeout(t);
        }
        this.stingerTimers = [];
        if (!s) {
            if (onCovered) {
                onCovered();
            }
            if (onDone) {
                onDone();
            }
            return;
        }
        s.hidden = false;
        s.classList.remove('in', 'out');
        void s.offsetWidth;
        s.classList.add('in');
        this.stingerTimers.push(setTimeout(() => {
            if (onCovered) {
                onCovered();
            }
            this.stingerTimers.push(setTimeout(() => {
                s.classList.remove('in');
                s.classList.add('out');
                this.stingerTimers.push(setTimeout(() => {
                    s.classList.remove('out');
                    s.hidden = true;
                    if (onDone) {
                        onDone();
                    }
                }, 380));
            }, 160));
        }, 360));
    }

    // Clears a wipe at once, for a stop in the middle of one.
    cancelStinger() {
        for (const t of this.stingerTimers) {
            clearTimeout(t);
        }
        this.stingerTimers = [];
        const s = $('stinger');
        if (s) {
            s.classList.remove('in', 'out');
            s.hidden = true;
        }
    }

    toast(text, kind = 'info', ms = 4000) {
        const t = this.el.toast;
        t.textContent = text;
        t.dataset.kind = kind;
        t.classList.add('show');
        clearTimeout(this.toastTimer);
        this.toastTimer = setTimeout(() => t.classList.remove('show'), ms);
    }
}
