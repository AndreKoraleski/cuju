// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The broadcast, match statistics and the MVP, lineups before kick-off, the celebration camera
// and a chosen replay after goals, and the show with the podium at full time.

import * as THREE from 'three';
import * as cameraModule from './cameras.js';
import * as transitionModule from './transition.js';
import {
    Celebration, Reactions, ballIntoNet, choseCelebration, danceNames, figureSize, kaykitCelebration, posePlayer, roleName,
    seeded, showDance,
} from './celebrate.js';
import { ACTION, NODES_FLOAT, PLAYER_FLOATS, TOUCH, episodeBoundary, isCut } from './format.js';
import { bareTeamName, kitSwatch, teamNames } from './hud.js';
import { HALF_H, HALF_W } from './pitch.js';
import { Players } from './players.js';
import { h } from './library.js';
import { css } from './kits.js';

// GRF's goal, the posts at 3.66 meters either side of the center of the goal line and the bar at
// 2.44 meters.
const POST_Y = 3.66;
const BAR_Z = 2.44;
const PASSES = new Set([ACTION.shortPass, ACTION.longPass, ACTION.highPass]);
const SAVES = new Set([ACTION.catch, ACTION.deflect, ACTION.interfere, ACTION.ballControl, ACTION.trap]);
const CATCHES = new Set([ACTION.catch, ACTION.ballControl, ACTION.trap]);
const TACKLES = new Set([ACTION.sliding, ACTION.interfere]);

const ballX = (f) => f.data[8];
const ballY = (f) => f.data[9];
const ballZ = (f) => f.data[10];

// The goal line a team attacks, the left team defends negative x.
const attackX = (team) => (team === 0 ? HALF_W : -HALF_W);
const scoreOf = (f, team) => (team === 0 ? f.scoreLeft : f.scoreRight);

// Whether the ball, flying from a to b over one step, would cross the goal line at goalX between
// the posts and under the bar. It flies straight with gravity and no drag, close enough over the
// two seconds that a save allows.
function headsInGoal(a, b, goalX, sps, margin) {
    const vx = (ballX(b) - ballX(a)) * sps;
    if (Math.sign(vx) !== Math.sign(goalX) || Math.abs(vx) < 1) {
        return false;
    }
    const t = (goalX - ballX(b)) / vx;
    if (t < 0 || t > 2) {
        return false;
    }
    const y = ballY(b) + (ballY(b) - ballY(a)) * sps * t;
    const z = ballZ(b) + (ballZ(b) - ballZ(a)) * sps * t - 4.9 * t * t;
    return Math.abs(y) < POST_Y + margin && z < BAR_Z + margin;
}

// Follows the ball after a shot until it crosses the goal line, a player touches it, play stops
// or four seconds pass. A shot is on target when the ball crosses between the posts and under the
// bar, when it goes in, or when the keeper touches a ball that was heading in. Another player's
// touch blocks it.
function followShot(store, shot, sps, n, header) {
    const goalX = attackX(shot.team);
    const end = Math.min(store.last, shot.i + 4 * sps);
    const out = { onTarget: false, goal: false, keeper: null, end };
    let prev = store.get(shot.i);
    for (let j = shot.i + 1; j <= end; j++) {
        const f = store.get(j);
        if (!prev || !f || isCut(prev, f)) {
            out.end = j;
            return out;
        }
        if (scoreOf(f, shot.team) > scoreOf(prev, shot.team)) {
            Object.assign(out, { onTarget: true, goal: true, goalIndex: j, end: j });
            return out;
        }
        const x0 = ballX(prev);
        const x1 = ballX(f);
        if (Math.abs(x1) >= HALF_W && Math.sign(x1) === Math.sign(goalX) && Math.abs(x0) < HALF_W) {
            const w = (Math.abs(goalX) - Math.abs(x0)) / Math.max(1e-6, Math.abs(x1 - x0));
            const y = ballY(prev) + (ballY(f) - ballY(prev)) * w;
            const z = ballZ(prev) + (ballZ(f) - ballZ(prev)) * w;
            out.onTarget = Math.abs(y) < POST_Y && z < BAR_Z;
            out.end = j;
            // A ball over the line between the posts goes in a step or two later.
            for (let g = j; g <= Math.min(store.last, j + 3); g++) {
                if (store.get(g) && scoreOf(store.get(g), shot.team) > scoreOf(prev, shot.team)) {
                    Object.assign(out, { onTarget: true, goal: true, goalIndex: g });
                }
            }
            return out;
        }
        for (let k = 0; k < n; k++) {
            if (!f.touches[k] || (k === shot.k && prev.touches[k] && prev.actions[k] === f.actions[k])) {
                continue;
            }
            const p = header.players[k];
            out.end = j;
            if (p.goalkeeper && p.team !== shot.team) {
                const a = store.get(Math.max(shot.i, j - 2));
                const b = j - 1 > shot.i ? prev : f;
                out.onTarget = headsInGoal(a, b, goalX, sps, 0.5);
                out.keeper = { i: j, k, action: f.actions[k] };
            }
            return out;
        }
        if (f.gameMode !== 0 && prev.gameMode === 0) {
            out.end = j;
            return out;
        }
        prev = f;
    }
    return out;
}

// Whether the shooting team scores after the keeper's touch before anyone else touches the ball.
function goalAfterKeeper(store, shot, keeper, sps, n) {
    const before = scoreOf(store.get(keeper.i - 1), shot.team);
    let prev = store.get(keeper.i);
    for (let j = keeper.i + 1; j <= Math.min(store.last, keeper.i + 3 * sps); j++) {
        const f = store.get(j);
        if (!prev || !f || isCut(prev, f)) {
            return -1;
        }
        if (scoreOf(f, shot.team) > before) {
            return j;
        }
        for (let k = 0; k < n; k++) {
            if (f.touches[k] && k !== keeper.k) {
                return -1;
            }
        }
        prev = f;
    }
    return -1;
}

// Whether a tackle at frame i won the ball, the next owner or the next player to touch it within
// three seconds plays for the tackler's team.
function wonBall(store, i, k, team, sps, n, header) {
    let prev = store.get(i);
    for (let j = i; j <= Math.min(store.last, i + 3 * sps); j++) {
        const f = store.get(j);
        if (j > i && isCut(prev, f)) {
            return false;
        }
        if (f.ownedTeam >= 0) {
            return f.ownedTeam === team;
        }
        if (j > i) {
            for (let m = 0; m < n; m++) {
                if (f.touches[m] && m !== k) {
                    return header.players[m].team === team;
                }
            }
        }
        prev = f;
    }
    return false;
}

// Whether play runs on between frames from and to, no restart, cut or goal in between.
function openPlay(store, from, to) {
    let prev = store.get(from);
    if (!prev || prev.gameMode !== 0) {
        return false;
    }
    for (let j = from + 1; j <= to; j++) {
        const f = store.get(j);
        if (!f || f.foul.type > 0 || f.gameMode !== 0 || isCut(prev, f) || f.scoreLeft !== prev.scoreLeft || f.scoreRight !== prev.scoreRight) {
            return false;
        }
        prev = f;
    }
    return true;
}

// Touch events from the touch bytes of frames from to to. A touch that lasts several steps with
// the same action is one event, at its first step.
function touchEvents(store, header, from, to) {
    const n = header.players.length;
    const events = [];
    let prev = null;
    for (let i = from; i <= to; i++) {
        const f = store.get(i);
        if (!f) {
            prev = null;
            continue;
        }
        const joined = prev && !isCut(prev, f);
        for (let k = 0; k < n; k++) {
            const touch = f.touches[k];
            if (!touch || (joined && prev.touches[k] && prev.actions[k] === f.actions[k])) {
                continue;
            }
            events.push({ i, k, team: header.players[k].team, action: f.actions[k], touch });
        }
        prev = f;
    }
    return events;
}

// The scorer and assist of the goal that shows at frame i. The scorer is the shooter of a shot
// that went in, else the last player to touch the ball before the goal, else its last owner, and
// a goal by a player of the conceding team is his own goal. The assist is the previous touch by
// a teammate of the scorer within ten seconds, an opponent's intentional touch in between
// cancels it.
function attribute(store, header, events, shots, i, team) {
    const n = header.players.length;
    const sps = header.steps_per_second || 10;
    let scorer = -1;
    let from = -1;
    const shot = shots.find((s) => s.goal && s.goalIndex === i);
    if (shot) {
        scorer = shot.k;
        from = events.indexOf(shot.event);
    } else {
        let last = events.length - 1;
        while (last >= 0 && events[last].i > i) {
            last--;
        }
        while (last >= 0 && events[last].touch === TOUCH.accidental) {
            last--;
        }
        if (last >= 0 && events[last].i >= i - 15 * sps) {
            // Of several touches in the same step, the scoring team's counts.
            from = last;
            for (let e = last; e >= 0 && events[e].i === events[last].i; e--) {
                if (events[e].team === team && events[e].touch !== TOUCH.accidental) {
                    from = e;
                    break;
                }
            }
            scorer = events[from].k;
        }
    }
    if (scorer < 0) {
        for (let j = i - 1; j >= Math.max(store.first, i - 30 * sps); j--) {
            const f = store.get(j);
            if (f.ownedPlayer >= 0 && f.ownedPlayer < n) {
                scorer = f.ownedPlayer;
                break;
            }
        }
    }
    if (header.engine_scorer && ((store.get(i).flags >>> 8) & 31)) {
        scorer = ((store.get(i).flags >>> 8) & 31) - 1;
        from = events.findLastIndex((e) => e.k === scorer && e.i <= i);
    }
    const own = scorer >= 0 && header.players[scorer].team !== team;
    let assist = -1;
    if (scorer >= 0 && !own && from >= 0) {
        const at = events[from].i;
        for (let e = from - 1; e >= 0 && events[e].i >= at - 10 * sps; e--) {
            const ev = events[e];
            if (ev.k === scorer) {
                continue;
            }
            if (ev.team !== team) {
                if (ev.touch === TOUCH.accidental) {
                    continue;
                }
                break;
            }
            assist = ev.k;
            break;
        }
    }
    return { scorer, assist, ownGoal: own };
}

// Shots, saves, passes and tackles of frames from to to, from the actions and touches.
// A shot is an intentional touch with the Shot action, or with the Header action when the ball
// then flies towards the opponent goal from within 30 meters of it. A save is the defending
// keeper touching a shot on target within two seconds with the Catch, Deflect, Interfere,
// BallControl or Trap action, when no goal follows before someone else touches the ball. A pass
// is an intentional touch with a pass action, completed when the next player to touch the ball
// is a teammate. A tackle is a Sliding or Interfere touch after an opponent's touch that wins the
// ball for the tackler's team, or a steal, an outfield player's intentional touch in open play
// right after an opponent's intentional one that wins the ball the same way.
function analyse(store, header, from = store.first, to = store.last) {
    const n = header.players.length;
    const sps = header.steps_per_second || 10;
    const events = touchEvents(store, header, from, to);
    const shots = [];
    const saveTouches = new Set();
    for (let e = 0; e < events.length; e++) {
        const ev = events[e];
        if (ev.touch !== TOUCH.kicked && ev.touch !== TOUCH.nonKicked) {
            continue;
        }
        let shoots = ev.action === ACTION.shot;
        if (ev.action === ACTION.header && ev.i + 2 <= to) {
            const a = store.get(ev.i);
            const b = store.get(ev.i + 2);
            const goalX = attackX(ev.team);
            const vx = (ballX(b) - ballX(a)) * sps / 2;
            shoots = !isCut(a, store.get(ev.i + 1)) && !isCut(store.get(ev.i + 1), b)
                && Math.sign(vx) === Math.sign(goalX) && Math.abs(vx) > 3
                && Math.abs(goalX - ballX(a)) < 30;
        }
        if (!shoots) {
            continue;
        }
        const shot = { i: ev.i, k: ev.k, team: ev.team, event: ev, ...followShot(store, ev, sps, n, header) };
        if (shot.keeper) {
            const scored = goalAfterKeeper(store, shot, shot.keeper, sps, n);
            if (scored >= 0) {
                Object.assign(shot, { onTarget: true, goal: true, goalIndex: scored });
            } else if (shot.onTarget && shot.keeper.i - shot.i <= 2 * sps && SAVES.has(shot.keeper.action)) {
                shot.saved = true;
                saveTouches.add(`${shot.keeper.i}:${shot.keeper.k}`);
            }
        }
        shots.push(shot);
    }
    const passes = [];
    const tackles = [];
    for (let e = 0; e < events.length; e++) {
        const ev = events[e];
        const intentional = ev.touch === TOUCH.kicked || ev.touch === TOUCH.nonKicked;
        if (intentional && PASSES.has(ev.action)) {
            let next = null;
            for (let m = e + 1; m < events.length && events[m].i <= ev.i + 10 * sps; m++) {
                if (!openPlay(store, ev.i, events[m].i)) {
                    break;
                }
                if (events[m].k !== ev.k) {
                    next = events[m];
                    break;
                }
            }
            passes.push({ i: ev.i, k: ev.k, team: ev.team, receiver: next ? next.k : -1, completed: !!next && next.team === ev.team });
        }
        if (TACKLES.has(ev.action) && !saveTouches.has(`${ev.i}:${ev.k}`)) {
            let before = null;
            for (let m = e - 1; m >= 0 && events[m].i >= ev.i - 5 * sps; m--) {
                if (events[m].k !== ev.k) {
                    before = events[m];
                    break;
                }
            }
            if (before && before.team !== ev.team && wonBall(store, ev.i, ev.k, ev.team, sps, n, header)) {
                tackles.push({ i: ev.i, k: ev.k, team: ev.team });
            }
        } else if (intentional && !header.players[ev.k].goalkeeper && !saveTouches.has(`${ev.i}:${ev.k}`)) {
            // A steal, an outfield player's intentional touch that takes the ball from an
            // opponent who played it on purpose in open play, and keeps it for his team.
            let before = null;
            for (let m = e - 1; m >= 0 && events[m].i >= ev.i - 3 * sps; m--) {
                if (events[m].k !== ev.k) {
                    before = events[m];
                    break;
                }
            }
            const opponent = before && before.team !== ev.team
                && (before.touch === TOUCH.kicked || before.touch === TOUCH.nonKicked);
            if (opponent && openPlay(store, before.i, ev.i) && wonBall(store, ev.i, ev.k, ev.team, sps, n, header)) {
                tackles.push({ i: ev.i, k: ev.k, team: ev.team, steal: true });
            }
        }
    }
    const goals = [];
    let prev = null;
    for (let i = Math.max(from, store.first + 1); i <= to; i++) {
        const f = store.get(i);
        const p = prev || store.get(i - 1);
        prev = f;
        if (isCut(p, f) || !(f.scoreLeft > p.scoreLeft || f.scoreRight > p.scoreRight)) {
            continue;
        }
        const team = f.scoreLeft > p.scoreLeft ? 0 : 1;
        goals.push({ index: i, team, ...attribute(store, header, events, shots, i, team) });
    }
    return { events, shots, passes, tackles, goals };
}

// Statistics from the frames alone, see analyse for the events and attribute for the goals.
// The MVP score adds 5 per goal, 3 per assist, 2 per save, 1 per tackle, 0.1 per completed pass
// and 0.05 per touch, and takes 5 per own goal.
const statsCache = new WeakMap();

export function matchStats(store, header) {
    const cached = statsCache.get(store);
    if (cached && cached.first === store.first && cached.last === store.last) {
        return cached.result;
    }
    const n = header.players.length;
    const stats = header.players.map((p, k) => ({
        k, team: p.team, index: p.index, goalkeeper: p.goalkeeper,
        goals: 0, assists: 0, touches: 0, passes: 0, passesCompleted: 0, tackles: 0, shots: 0,
        shotsOnTarget: 0, saves: 0, catches: 0, parries: 0, ownGoals: 0, score: 0,
        goalFrames: [], assistFrames: [], saveFrames: [], tackleFrames: [], run: null,
    }));
    const a = analyse(store, header);
    for (const ev of a.events) {
        stats[ev.k].touches++;
    }
    for (const s of a.shots) {
        stats[s.k].shots++;
        stats[s.k].shotsOnTarget += s.onTarget ? 1 : 0;
        if (s.saved) {
            const keeper = stats[s.keeper.k];
            keeper.saves++;
            keeper.saveFrames.push(s.keeper.i);
            if (CATCHES.has(s.keeper.action)) {
                keeper.catches++;
            } else {
                keeper.parries++;
            }
        }
    }
    for (const p of a.passes) {
        stats[p.k].passes++;
        stats[p.k].passesCompleted += p.completed ? 1 : 0;
    }
    for (const t of a.tackles) {
        stats[t.k].tackles++;
        stats[t.k].tackleFrames.push(t.i);
    }
    for (const g of a.goals) {
        if (g.scorer >= 0) {
            if (g.ownGoal) {
                stats[g.scorer].ownGoals++;
            } else {
                stats[g.scorer].goals++;
                stats[g.scorer].goalFrames.push(g.index);
            }
        }
        if (g.assist >= 0) {
            stats[g.assist].assists++;
            stats[g.assist].assistFrames.push(g.index);
        }
    }
    // The longest spell on the ball, for players with nothing better to show. A spell ends when
    // another player takes the ball, at a cut, or with the last frame.
    let owner = -1;
    let runStart = -1;
    const closeRun = (end) => {
        if (owner >= 0 && runStart >= 0) {
            const r = stats[owner].run;
            if (!r || end - runStart > r.end - r.start) {
                stats[owner].run = { start: runStart, end };
            }
        }
        owner = -1;
    };
    for (let i = store.first; i <= store.last; i++) {
        const f = store.get(i);
        if (i > store.first && isCut(store.get(i - 1), f)) {
            closeRun(i - 1);
        }
        if (f.ownedPlayer >= 0 && f.ownedPlayer < n && f.ownedPlayer !== owner) {
            closeRun(i);
            runStart = i;
            owner = f.ownedPlayer;
        }
    }
    closeRun(store.last);
    for (const s of stats) {
        s.score = 5 * s.goals + 3 * s.assists + 2 * s.saves + s.tackles + 0.1 * s.passesCompleted
            + 0.05 * s.touches - 5 * s.ownGoals;
    }
    const result = { stats, goals: a.goals, shots: a.shots, passes: a.passes, tackles: a.tackles };
    statsCache.set(store, { first: store.first, last: store.last, result });
    return result;
}

// The scorer and assist of the goal that shows at frame i, from the twenty seconds before it.
export function scorerAt(store, i, team, header) {
    const sps = header.steps_per_second || 10;
    const a = analyse(store, header, Math.max(store.first, i - 20 * sps), Math.min(store.last, i));
    const g = a.goals.find((goal) => goal.index === i);
    return g ? { scorer: g.scorer, assist: g.assist, ownGoal: g.ownGoal } : { scorer: -1, assist: -1, ownGoal: false };
}

// Whether player k of these figures plays a clip, a test that is false for figures without clips.
function clipTest(players) {
    return (k, name) => {
        if (!players || !name) {
            return false;
        }
        if (typeof players.hasClip === 'function') {
            return players.hasClip(k, name);
        }
        return typeof players.clipNames === 'function' && players.clipNames(k).includes(name);
    };
}

function threeColorOf(c) {
    return new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
}

// The few rules the show's panels need beyond the page's own sheet, two compact cards stacked in
// the panel after a draw.
function injectShowStyle() {
    if (document.getElementById('cuju-show-style')) {
        return;
    }
    const style = document.createElement('style');
    style.id = 'cuju-show-style';
    style.textContent = `
#victory.draw .victory-cards { flex-direction: column; gap: 8px; }
#victory.draw .mvp-card { min-width: 0; padding: 8px 12px; }
#victory { max-height: calc(100% - 32px); overflow-y: auto; }
`;
    document.head.appendChild(style);
}

// Gives figures drawn apart from the match, on the podium or the stage, the colors that the
// match's figures show, the away colors of a clash among them.
function shareColors(target, source) {
    if (!target || !source || typeof source.colorsOf !== 'function' || typeof target.setColors !== 'function' || !target.used) {
        return;
    }
    try {
        [0, 1].forEach((team) => {
            const used = target.used[team];
            if (used) {
                for (const model of used.keys()) {
                    target.setColors(team, model, source.colorsOf(team, model));
                }
            }
        });
    } catch (err) {
        console.warn('show colors not shared', err);
    }
}

// A beam of light, brightest where the surface faces the eye and near the lamp, so a cone of it
// reads as light in the air rather than as a tinted shell.
function beamMaterial(color, opacity) {
    return new THREE.ShaderMaterial({
        uniforms: { color: { value: new THREE.Color(color) }, opacity: { value: opacity } },
        vertexShader: `
            varying vec3 vNormal;
            varying vec3 vView;
            varying float vHeight;
            void main() {
                vec4 mv = modelViewMatrix * vec4(position, 1.0);
                vView = -mv.xyz;
                vNormal = normalMatrix * normal;
                vHeight = uv.y;
                gl_Position = projectionMatrix * mv;
            }`,
        fragmentShader: `
            uniform vec3 color;
            uniform float opacity;
            varying vec3 vNormal;
            varying vec3 vView;
            varying float vHeight;
            void main() {
                float facing = abs(dot(normalize(vNormal), normalize(vView)));
                float a = opacity * pow(facing, 2.5) * mix(0.2, 1.0, vHeight);
                gl_FragColor = vec4(color * a, a);
            }`,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
    });
}

// A dance player k of these figures can perform, a clip only where his model carries it.
function usableDance(players, k, name, random = Math.random) {
    if (!name || !name.startsWith('clip:')) {
        return name;
    }
    if (clipTest(players)(k, name)) {
        return name;
    }
    const list = danceNames();
    return list.length ? list[Math.floor(random() * list.length)] : 'dance';
}

// What the page shows of a player, his number, name, team and colors.
export function playerInfo(app, k) {
    const header = app.source.header;
    const p = header.players[k];
    const style = app.styles ? app.styles[p.team] : null;
    const roster = style && style.players ? style.players[p.index] : null;
    const number = roster ? roster.number : (p.goalkeeper ? 1 : p.index + 1);
    const name = roster && roster.name ? roster.name : (p.goalkeeper ? 'Goalkeeper' : `Player ${number}`);
    const kit = style ? (p.goalkeeper && style.goalkeeper ? style.goalkeeper : style.kit) : null;
    const rgb = kit ? kit.primary : [255, 111, 0];
    const light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 150;
    return {
        k, team: p.team, index: p.index, goalkeeper: p.goalkeeper, number, name, named: !!(roster && roster.name),
        teamName: teamNames(header)[p.team], color: css(rgb), ink: light ? '#202124' : '#ffffff',
        badge: style && style.badge ? style.badge : '',
    };
}

// The goal graphic's lines for the goal at frame index.
function goalInfo(app, index, team, who) {
    const store = app.timeline.store;
    const f = store.get(index);
    const info = {
        score: `${f.scoreLeft} – ${f.scoreRight}`,
        minute: Math.max(1, Math.ceil(f.timeMs / 60000)),
        name: '', number: null, assist: '', color: null,
    };
    if (who.scorer >= 0) {
        const s = playerInfo(app, who.scorer);
        info.name = s.name;
        info.number = s.number;
        info.color = s.color;
        if (who.ownGoal) {
            info.assist = `Own goal, ${s.teamName}`;
        } else if (who.assist >= 0) {
            const a = playerInfo(app, who.assist);
            info.assist = a.named ? `Assist, ${a.number} ${a.name}` : `Assist, ${a.name}`;
        }
    }
    return info;
}

// The engine position of the ball at timeline position t, between frames.
function ballAt(store, t, out) {
    const i = Math.floor(t);
    const a = store.get(Math.max(store.first, Math.min(store.last, i)));
    const b = store.get(Math.max(store.first, Math.min(store.last, i + 1))) || a;
    const w = isCut(a, b) ? 0 : t - i;
    for (let c = 0; c < 3; c++) {
        out[c] = a.data[8 + c] + (b.data[8 + c] - a.data[8 + c]) * w;
    }
    return out;
}

function playerAt(store, k, t, out) {
    const i = Math.floor(t);
    const a = store.get(Math.max(store.first, Math.min(store.last, i)));
    const b = store.get(Math.max(store.first, Math.min(store.last, i + 1))) || a;
    const w = isCut(a, b) ? 0 : t - i;
    const o = NODES_FLOAT + k * PLAYER_FLOATS;
    out[0] = a.data[o] + (b.data[o] - a.data[o]) * w;
    out[1] = a.data[o + 1] + (b.data[o + 1] - a.data[o + 1]) * w;
    return out;
}

// Replay angles. high rides above and behind a team move, reverse stands behind the scorer and
// watches the ball fly, netcam looks out from inside the goal at a close finish, the rest are
// the camera rig's own modes.
export const REPLAY_ANGLES = ['broadcast', 'reverse', 'netcam', 'high'];
const ANGLE_NAMES = {
    broadcast: 'Wide', reverse: 'Reverse angle', netcam: 'Goal camera', high: 'High angle', lowside: 'Pitch side',
};

// How a goal's replay plays, the angle that suits it best unless one is asked for, where the
// clip starts, the shot to slow down at and the frame before the goal where it ends.
export function replayPlan(store, header, goal, angle = 'auto') {
    const sps = header.steps_per_second || 10;
    const g = goal.index;
    const from = Math.max(store.first, g - 25 * sps);
    const a = analyse(store, header, from, g);
    const scored = a.goals.find((x) => x.index === g) || { scorer: -1, assist: -1, ownGoal: false };
    const shot = a.shots.find((s) => s.goal && s.goalIndex === g);
    let shotIndex = shot ? shot.i : -1;
    if (shotIndex < 0) {
        for (let e = a.events.length - 1; e >= 0; e--) {
            if (a.events[e].i < g && (scored.scorer < 0 || a.events[e].k === scored.scorer)) {
                shotIndex = a.events[e].i;
                break;
            }
        }
    }
    if (shotIndex < 0) {
        shotIndex = Math.max(store.first, g - 8);
    }
    const sf = store.get(shotIndex);
    const distance = sf ? Math.hypot(ballX(sf) - attackX(goal.team), ballY(sf)) : 0;
    // The move, the team's completed passes since the other team last played the ball.
    let lastLoss = from;
    for (const ev of a.events) {
        const intentional = ev.touch === TOUCH.kicked || ev.touch === TOUCH.nonKicked;
        if (ev.i < shotIndex && ev.team !== goal.team && intentional) {
            lastLoss = ev.i;
        }
    }
    const passes = a.passes.filter((p) => p.team === goal.team && p.completed && p.i > lastLoss && p.i < shotIndex
        && p.i >= shotIndex - 12 * sps);
    let chosen = angle;
    if (chosen === 'auto') {
        if (scored.ownGoal || passes.length >= 3) {
            chosen = 'high';
        } else if (distance >= 18) {
            chosen = 'reverse';
        } else {
            chosen = 'netcam';
        }
    }
    let begin = passes.length >= 2 ? passes[0].i - 12 : shotIndex - 45;
    begin = Math.max(store.first, g - 130, Math.min(begin, g - 35));
    // Never across a cut, the replay starts after the last one.
    for (let i = g - 1; i > begin; i--) {
        if (isCut(store.get(i - 1), store.get(i))) {
            begin = i;
            break;
        }
    }
    const keeper = header.players.findIndex((p) => p.goalkeeper && p.team !== goal.team);
    return {
        goal, angle: chosen, begin, end: g - 1, shotIndex, slowFrom: shotIndex - 4, scorer: scored.scorer,
        keeper, distance, passes: passes.length,
    };
}

// The stages of the celebration camera in seconds, the live shot while the ball hits the net,
// a close tracking shot of the scorer, a slow orbit of the group and a low hero angle.
const CELEBRATION_SHOTS = [
    { at: 0, kind: 'live' },
    { at: 1.3, kind: 'close', snap: true },
    { at: 3.9, kind: 'orbit' },
    { at: 6.8, kind: 'hero' },
];
const CELEBRATION_LENGTH = 9.4;
// Seconds the end of a replay holds on the goal while the ball settles in the net.
const REPLAY_HOLD = 0.75;
const LINEUP_LENGTH = 9;

const tmpBall = new Float32Array(3);
const tmpA = new Float32Array(3);
const tmpB = new Float32Array(3);

// Gives the mouse back for a part that asks for a click or a key.
function release(app) {
    if (typeof app.releasePointer === 'function') {
        app.releasePointer();
    }
}

function confineShot(v) {
    if (typeof cameraModule.confine === 'function') {
        cameraModule.confine(v);
    }
    return v;
}

// The director of the broadcast. Lineups before kick-off, the celebration camera and one chosen
// replay after each goal, more angles on request, and the reel of goals after the match. Every
// part ends with Enter, Escape or a click, and the cuts between parts hide behind a stinger.
export class Director {
    constructor(app) {
        this.app = app;
        this.active = null;
        this.cel = null;
        this.lineup = null;
        this.hold = null;
        this.wiping = false;
        this.handled = null;
        this.recent = null;
        this.lastT = null;
        this.lastNow = performance.now();
        this.pending = null;
        this.shot = { pos: new THREE.Vector3(), target: new THREE.Vector3(), fov: 40, rate: 3, snap: false };
        this.hook = (pose, sample) => this.dress(pose, sample);
        this.bindSettings();
        const stage = document.getElementById('stage');
        if (stage) {
            // A click skips, a drag that turns a camera does not.
            let down = null;
            stage.addEventListener('pointerdown', (e) => {
                down = { x: e.clientX, y: e.clientY };
            });
            stage.addEventListener('click', (e) => {
                const still = down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6;
                if (still && this.running && !this.app.postMatch.running) {
                    this.skip();
                }
            });
        }
    }

    // The settings this director adds, kept like the page's own.
    bindSettings() {
        const app = this.app;
        const load = (key, fallback) => {
            try {
                const v = localStorage.getItem(`cuju.${key}`);
                return v === null ? fallback : JSON.parse(v);
            } catch {
                return fallback;
            }
        };
        for (const [id, key] of [['lineups-on', 'lineups'], ['celebrations-on', 'celebrations']]) {
            if (app.settings[key] === undefined) {
                app.settings[key] = load(key, true);
            }
            const box = document.getElementById(id);
            if (!box) {
                continue;
            }
            box.checked = !!app.settings[key];
            box.addEventListener('change', () => {
                app.settings[key] = box.checked;
                try {
                    localStorage.setItem(`cuju.${key}`, JSON.stringify(box.checked));
                } catch {
                    // Storage can be unavailable, the setting then lasts for this visit only.
                }
            });
        }
    }

    get running() {
        return !!this.active || !!this.cel || !!this.lineup || this.wiping;
    }

    get celebrating() {
        return this.cel;
    }

    savedState() {
        const app = this.app;
        const tl = app.timeline;
        const cams = app.cameras;
        let camera = cams.mode;
        if (camera === 'scripted' || camera === 'podium') {
            camera = app.settings.camera || 'broadcast';
        }
        return {
            t: tl.t, playing: tl.playing, following: tl.following, speed: tl.speed || 1, direction: tl.direction,
            camera, focus: cams.focusIndex, view: cams.playerView,
        };
    }

    useScripted() {
        const cams = this.app.cameras;
        cams.scripted = this.shot;
        if (cams.mode !== 'scripted') {
            cams.setMode('scripted');
        }
    }

    // The wipe that hides a cut, the restarts' own logo wipe when the page has it, so every
    // transition of the broadcast looks alike, else the stinger of the HUD. onCovered runs while
    // the view is hidden. The wipe runs on the frame clock, see update.
    sting(label, onCovered) {
        const app = this.app;
        const wipe = app.transition && app.transition.wipe;
        if (!wipe || typeof wipe.show !== 'function') {
            app.hud.stinger(onCovered);
            return;
        }
        if (typeof wipe.setTeams === 'function') {
            wipe.setTeams(app.styles);
        }
        const seconds = typeof transitionModule.WIPE === 'number' ? transitionModule.WIPE : 0.72;
        this.stingState = { clock: 0, seconds, onCovered, label, covered: false };
        wipe.show(0.01, label);
    }

    stepSting(dt) {
        const st = this.stingState;
        if (!st) {
            return;
        }
        const wipe = this.app.transition.wipe;
        st.clock += dt;
        const p = st.clock / st.seconds;
        if (!st.covered && p >= 0.5) {
            st.covered = true;
            if (st.onCovered) {
                st.onCovered();
            }
        }
        if (this.stingState !== st) {
            return;
        }
        if (p >= 1) {
            this.stingState = null;
            wipe.show(null);
        } else {
            wipe.show(p, st.label);
        }
    }

    cancelSting() {
        const app = this.app;
        app.hud.cancelStinger();
        if (this.stingState) {
            this.stingState = null;
            if (app.transition && app.transition.wipe) {
                app.transition.wipe.show(null);
            }
        }
    }

    restoreCamera(s) {
        const cams = this.app.cameras;
        cams.focusIndex = s.focus;
        cams.playerView = s.view;
        cams.setMode(s.camera);
    }

    // Back to play at frame t with the state saved before a sequence, under a cut.
    resume(s, t) {
        const app = this.app;
        const tl = app.timeline;
        if (!tl) {
            return;
        }
        tl.speed = s.speed;
        tl.direction = s.direction;
        this.restoreCamera(s);
        if (s.following && tl.live) {
            tl.goLive();
        } else {
            tl.seek(t);
            tl.playing = s.playing;
        }
        app.lastIndex = tl.index;
        this.lastT = tl.t;
        // The restarts and the ball take this as a seek, so the goal is not played again.
        app.afterSeek();
        app.updatePlayUi();
    }

    // Called by the page when play crosses a goal, and by update a moment before, so the frame
    // before the goal can hold while the scorer celebrates.
    goalScored(index, team, early = false) {
        const app = this.app;
        const tl = app.timeline;
        if (!tl || this.running || this.handled === index || app.postMatch.running) {
            return;
        }
        this.handled = index;
        const who = scorerAt(tl.store, index, team, app.source.header);
        const info = goalInfo(app, index, team, who);
        if (early) {
            app.sound.goal(team);
            app.goalBoost = 1;
        }
        app.lastGoal = { index, team };
        app.hud.showGoal(team, true, info, 8000);
        release(app);
        this.recent = null;
        clearTimeout(this.pending);
        if (app.settings.celebrations && index - 1 >= tl.first && tl.direction > 0) {
            this.startCelebration(index, team, who);
        } else if (app.settings.replays) {
            this.pending = setTimeout(() => {
                // Only while play is still just after the goal, a seek away cancels it.
                const near = tl.t >= index - 1 && tl.t < index + 8 * tl.sps;
                if (!this.running && app.timeline === tl && tl.playing && near && !app.postMatch.running) {
                    this.play([{ goal: { index, team }, shot: 'auto' }]);
                }
            }, 2500);
        }
    }

    startCelebration(index, team, who) {
        const app = this.app;
        // The team's own celebration sound follows the roar, once the scorer is off running.
        setTimeout(() => {
            if (this.running && typeof app.sound.teamSound === 'function') {
                app.sound.teamSound(team, 'celebration');
            }
        }, 1400);
        const tl = app.timeline;
        const store = tl.store;
        const header = app.source.header;
        const saved = { ...this.savedState(), t: index };
        tl.following = false;
        tl.seek(index - 1);
        tl.t = index - 1;
        tl.playing = true;
        tl.direction = 1;
        tl.speed = 0;
        app.lastIndex = index - 1;
        const frame = store.get(index - 1);
        const before = store.get(index - 2) || frame;
        // An own goal is celebrated by the scoring team's player nearest the ball.
        let celebrant = who.scorer;
        if (celebrant < 0 || who.ownGoal) {
            let best = Infinity;
            celebrant = -1;
            header.players.forEach((p, k) => {
                if (p.team !== team || p.goalkeeper || !(frame.playerFlags[k] & 1)) {
                    return;
                }
                const o = NODES_FLOAT + k * PLAYER_FLOATS;
                const d = Math.hypot(frame.data[o] - ballX(frame), frame.data[o + 1] - ballY(frame));
                if (d < best) {
                    best = d;
                    celebrant = k;
                }
            });
        }
        const players = app.players;
        const hasClip = clipTest(players);
        const choreo = new Celebration({
            frame, before, header, styles: app.styles, team, scorer: celebrant, sps: tl.sps, seed: index * 31 + team, hasClip,
            radius: (k) => figureSize(players, k).radius,
        });
        this.cel = { goal: { index, team }, who, celebrant, clock: 0, choreo, saved, stage: -1, orbit: 0 };
        app.hud.scoreOf(store.get(index));
    }

    // Each frame, the parts that run, or a look for a goal just ahead.
    update() {
        const app = this.app;
        const now = performance.now();
        const dt = Math.min(0.1, Math.max(0, (now - this.lastNow) / 1000));
        this.lastNow = now;
        if (app.poseHooks && !app.poseHooks.includes(this.hook)) {
            app.poseHooks.push(this.hook);
        }
        this.stepSting(dt);
        if (app.postMatch && typeof app.postMatch.update === 'function') {
            app.postMatch.update(dt);
        }
        const tl = app.timeline;
        if (!tl) {
            return;
        }
        if (this.handled != null && (tl.t < this.handled - 2 || tl.t > this.handled + 40)) {
            this.handled = null;
        }
        if (this.lineup) {
            this.updateLineup(dt);
        } else if (this.cel) {
            this.updateCelebration(dt);
        } else if (this.active) {
            this.updateReplay(dt);
        } else if (!this.wiping) {
            this.lookAhead();
        }
        this.lastT = tl.t;
    }

    // Stops play on the frame before a goal that normal play is about to cross, so the frames of
    // the kick-off after it stay unseen until the celebration and replay are over.
    lookAhead() {
        const app = this.app;
        const tl = app.timeline;
        if (!app.settings.celebrations || !tl.playing || tl.direction <= 0 || tl.speed > 2.5 || tl.speed <= 0
            || this.lastT == null || app.postMatch.running) {
            return;
        }
        const from = this.lastT;
        if (tl.t - from > 1.5 || tl.t <= from) {
            return;
        }
        for (const g of tl.store.goals) {
            if (from <= g.index - 1 && tl.t > g.index - 1 && g.index - 1 >= tl.first && this.handled !== g.index) {
                const a = tl.store.get(g.index - 1);
                const b = tl.store.get(g.index);
                if (!a || !b || isCut(a, b)) {
                    return;
                }
                this.goalScored(g.index, g.team, true);
                return;
            }
        }
    }

    updateCelebration(dt) {
        const app = this.app;
        const tl = app.timeline;
        const c = this.cel;
        if (!this.wiping && (tl.index !== c.goal.index - 1 || !tl.playing || tl.speed !== 0)) {
            // Someone moved the timeline, the celebration gives way at once.
            this.endCelebration();
            return;
        }
        c.clock += dt;
        let stage = 0;
        for (let i = 0; i < CELEBRATION_SHOTS.length; i++) {
            if (c.clock >= CELEBRATION_SHOTS[i].at) {
                stage = i;
            }
        }
        const shot = CELEBRATION_SHOTS[stage];
        if (shot.kind !== 'live') {
            const s = c.choreo.focus(c.clock);
            const p = new THREE.Vector3(s.x, 0, -s.y);
            // The scorer's neck as drawn, so slides and moves on the ground stay in the frame.
            let head = 1.5;
            const pose = app.pose;
            if (pose && c.celebrant >= 0 && c.celebrant < pose.players) {
                const o = c.celebrant * PLAYER_FLOATS + 3 * 7;
                p.set(pose.nodes[o], 0, -pose.nodes[o + 1]);
                head = Math.max(0.35, Math.min(1.6, pose.nodes[o + 2]));
            }
            const face = new THREE.Vector3(Math.sin(s.heading), 0, Math.cos(s.heading));
            const right = new THREE.Vector3(-face.z, 0, face.x);
            const sh = this.shot;
            if (stage !== c.stage) {
                sh.snap = !!shot.snap;
                if (shot.kind === 'orbit') {
                    const from = app.cameras.camera ? app.cameras.camera.position : p;
                    c.orbit = Math.atan2(from.z - p.z, from.x - p.x);
                }
                this.useScripted();
            }
            if (shot.kind === 'close') {
                sh.pos.copy(p).addScaledVector(face, 5.2).addScaledVector(right, 1.6);
                sh.pos.y = 1.55;
                sh.target.copy(p).setY(Math.min(1.15, head * 0.8));
                sh.fov = 32;
                sh.rate = 5;
            } else if (shot.kind === 'orbit') {
                // Around the scorer and the teammates close to him.
                const center = p.clone();
                let n = 1;
                const header = app.source.header;
                for (let k = 0; k < header.players.length; k++) {
                    if (k === c.celebrant || header.players[k].team !== c.goal.team) {
                        continue;
                    }
                    const m = c.choreo.state(k, c.clock);
                    if (m && Math.hypot(m.x - s.x, m.y - s.y) < 3) {
                        center.x += m.x;
                        center.z -= m.y;
                        n++;
                    }
                }
                // The scorer stays near the middle of the frame, the group around him.
                center.multiplyScalar(1 / n).lerp(p, 0.5);
                const a = c.orbit + 0.26 * (c.clock - shot.at);
                sh.pos.set(center.x + Math.cos(a) * 9.5, 3.9, center.z + Math.sin(a) * 9.5);
                sh.target.copy(center).setY(1.0);
                sh.fov = 36;
                sh.rate = 1.8;
            } else {
                sh.pos.copy(p).addScaledVector(face, 4.0).addScaledVector(right, 0.8);
                sh.pos.y = 0.45;
                sh.target.copy(p).setY(head);
                sh.fov = 46;
                sh.rate = 2.2;
            }
            sh.pos.y += this.clearance(c, sh.pos, sh.target);
            confineShot(sh.pos);
        }
        c.stage = stage;
        if (c.clock >= CELEBRATION_LENGTH && !this.wiping) {
            this.afterCelebration();
        }
    }

    // How far to raise a celebration shot so nobody stands between the camera and the scorer,
    // from the players near the line of sight on the ground.
    clearance(c, pos, target) {
        const header = this.app.source.header;
        const dx = target.x - pos.x;
        const dz = target.z - pos.z;
        const len2 = dx * dx + dz * dz || 1;
        let lift = 0;
        for (let k = 0; k < header.players.length; k++) {
            if (k === c.celebrant) {
                continue;
            }
            const m = c.choreo.state(k, c.clock);
            if (!m) {
                continue;
            }
            const px = m.x - pos.x;
            const pz = -m.y - pos.z;
            const u = (px * dx + pz * dz) / len2;
            if (u <= 0 || u >= 0.85) {
                continue;
            }
            const d = Math.hypot(px - dx * u, pz - dz * u);
            if (d < 1.1) {
                lift = Math.max(lift, ((1.1 - d) / 1.1) * 3 * (1 - u));
            }
        }
        return lift;
    }

    // The stinger, and behind it the chosen replay, or play again.
    afterCelebration() {
        const app = this.app;
        const c = this.cel;
        if (!c || this.wiping) {
            return;
        }
        this.wiping = true;
        this.sting(app.settings.replays ? 'Replay' : '', () => {
            this.wiping = false;
            this.cel = null;
            this.releaseClips(c);
            app.hud.scoreOf(null);
            app.hud.hideGoal();
            if (app.settings.replays) {
                this.play([{ goal: c.goal, shot: 'auto' }], { saved: c.saved, returnT: c.goal.index, wipe: false });
            } else {
                this.resume(c.saved, c.goal.index);
            }
        });
    }

    releaseClips(c) {
        const players = this.app.players;
        if (players && typeof players.stopClip === 'function') {
            c.choreo.release((k) => players.stopClip(k));
        }
    }

    // Ends a celebration that something else interrupted, without a stinger.
    endCelebration() {
        const app = this.app;
        const c = this.cel;
        this.cel = null;
        if (!c) {
            return;
        }
        this.releaseClips(c);
        app.hud.scoreOf(null);
        const tl = app.timeline;
        this.restoreCamera(c.saved);
        if (tl) {
            if (tl.speed === 0) {
                tl.speed = c.saved.speed;
            }
            // Still on the frame before the goal, play moves on to the kick-off after it.
            if (tl.index === c.goal.index - 1 && Math.abs(tl.t - (c.goal.index - 1)) < 1e-6) {
                tl.seek(c.goal.index);
                app.lastIndex = tl.index;
                this.lastT = tl.t;
                app.afterSeek();
            }
        }
        app.ball.reset();
        app.cameras.jump();
    }

    // Dresses the frame on screen, the celebration on the frame before the goal, the ball
    // settling in the net at the end of a replay.
    dress(pose, sample) {
        const c = this.cel;
        if (c && sample.index === c.goal.index - 1) {
            const players = this.app.players;
            const clip = players && typeof players.playClip === 'function' ? (k, name, t, w) => players.playClip(k, name, t, w) : null;
            c.choreo.apply(pose, c.clock, 1, clip);
            return;
        }
        const h = this.hold;
        if (h && sample.index === h.index && pose.ball) {
            ballIntoNet(h.a, h.b, h.team, h.sps, h.clock, tmpBall);
            pose.ball[0] = tmpBall[0];
            pose.ball[1] = tmpBall[1];
            pose.ball[2] = tmpBall[2];
        }
    }

    // A single replay on request, the chosen angle.
    start(goal) {
        this.recent = null;
        this.play([{ goal, shot: 'auto' }]);
    }

    // The angles not shown yet of the last replayed goal, or of the last goal.
    moreAngles() {
        const app = this.app;
        const tl = app.timeline;
        if (!tl) {
            return;
        }
        const goal = (this.active && this.active.plan && this.active.plan.goal) || (this.recent && this.recent.goal)
            || app.lastGoal || tl.goalNear(-1);
        if (!goal) {
            app.hud.toast('No goal to show again yet.');
            return;
        }
        const shown = new Set(this.recent && this.recent.goal.index === goal.index ? this.recent.angles : []);
        if (this.active && this.active.plan) {
            shown.add(this.active.plan.angle);
        }
        if (!shown.size) {
            shown.add(replayPlan(tl.store, app.source.header, goal).angle);
        }
        let left = REPLAY_ANGLES.filter((a) => !shown.has(a));
        if (!left.length) {
            left = REPLAY_ANGLES.slice();
            shown.clear();
        }
        const queue = left.slice(0, 2).map((shot) => ({ goal, shot }));
        if (this.active) {
            // A replay in progress plays the new angles next, keeping where it returns to.
            this.active.queue = this.active.queue.slice(0, this.active.item + 1).concat(queue);
            app.hud.toast(`More angles next, ${queue.map((q) => ANGLE_NAMES[q.shot].toLowerCase()).join(' and ')}.`, 'info', 2200);
            return;
        }
        this.recent = { goal, angles: [...shown] };
        this.play(queue);
    }

    // Plays a queue of goals and angles. The options are saved and returnT for where play goes
    // back to, onDone called at the end, restore false to stay where the queue ended, slow false
    // for no slow motion, wipe false when a stinger already covers the start.
    play(queue, options = {}) {
        const app = this.app;
        const tl = app.timeline;
        if (!tl || !queue.length) {
            if (options.onDone) {
                options.onDone();
            }
            return;
        }
        clearTimeout(this.pending);
        // A replay asked for during a celebration returns to the kick-off after the goal.
        const cel = this.cel;
        if (!this.active) {
            const saved = options.saved || (cel ? cel.saved : this.savedState());
            const back = options.returnT != null ? options.returnT : (cel ? cel.goal.index : saved.t);
            this.active = { saved, returnT: back };
            // The bug holds the live score and clock while a replay shows earlier frames.
            if (!app.postMatch.running) {
                const store = tl.store;
                app.hud.freeze(store.get(Math.max(store.first, Math.min(store.last, Math.floor(back)))));
            }
        }
        Object.assign(this.active, { queue, item: -1, options, plan: null, phase: 'wait' });
        app.hud.hideGoal();
        release(app);
        const begin = () => {
            if (this.cel) {
                const c = this.cel;
                this.cel = null;
                this.releaseClips(c);
                app.hud.scoreOf(null);
            }
            this.nextShot();
        };
        if (options.wipe === false) {
            begin();
            return;
        }
        this.wiping = true;
        this.sting('Replay', () => {
            this.wiping = false;
            if (this.active) {
                begin();
            }
        });
    }

    nextShot() {
        const a = this.active;
        const app = this.app;
        const tl = app.timeline;
        a.item++;
        this.hold = null;
        if (a.item >= a.queue.length) {
            const done = a.options.onDone;
            this.stop(a.options.restore !== false, true);
            if (done) {
                done();
            }
            return;
        }
        const { goal, shot } = a.queue[a.item];
        const plan = replayPlan(tl.store, app.source.header, goal, shot || 'auto');
        a.plan = plan;
        a.phase = 'play';
        a.holdClock = 0;
        a.slow = a.options.slow !== false;
        const shownBefore = this.recent && this.recent.goal.index === goal.index ? this.recent.angles : [];
        this.recent = { goal, angles: [...new Set([...shownBefore, plan.angle])] };
        tl.following = false;
        tl.seek(plan.begin);
        tl.direction = 1;
        tl.playing = true;
        tl.speed = 1;
        app.lastIndex = tl.index;
        const cams = app.cameras;
        if (plan.angle === 'broadcast' || plan.angle === 'lowside' || plan.angle === 'goal') {
            cams.focusIndex = plan.scorer;
            cams.setMode(plan.angle);
        } else {
            this.replayShot(plan, tl.t);
            this.shot.snap = true;
            this.useScripted();
        }
        app.afterSeek();
        const more = a.item < a.queue.length - 1 ? 'Enter next angle' : 'Enter skip';
        app.hud.setReplay(true, `${ANGLE_NAMES[plan.angle] || 'Replay'}, Shift R more angles, ${more}`);
    }

    // The scripted replay angles, from the ball and the scorer at timeline position t.
    replayShot(plan, t) {
        const app = this.app;
        const store = app.timeline.store;
        const sh = this.shot;
        const b = ballAt(store, t, tmpBall);
        const ball = new THREE.Vector3(b[0], Math.max(0.2, b[2]), -b[1]);
        const dir = plan.goal.team === 0 ? 1 : -1;
        const goal = new THREE.Vector3(attackX(plan.goal.team), 1, 0);
        if (plan.angle === 'high') {
            sh.pos.set(b[0] - dir * 24, 24, -b[1] * 0.55 + 14);
            sh.target.set(b[0] + dir * 8, 0, -b[1] * 0.85);
            sh.fov = 44;
            sh.rate = 2.2;
        } else if (plan.angle === 'reverse') {
            const k = plan.scorer;
            const at = Math.min(t, plan.shotIndex);
            const s = k >= 0 ? playerAt(store, k, at, tmpA) : ballAt(store, at, tmpA);
            const anchor = new THREE.Vector3(s[0], 0, -s[1]);
            const d = new THREE.Vector3().subVectors(goal, anchor).setY(0);
            if (d.lengthSq() < 1) {
                d.set(dir, 0, 0);
            }
            d.normalize();
            const side = new THREE.Vector3(-d.z, 0, d.x);
            // High enough to see over the shooter, the goal mouth always in the frame.
            sh.pos.copy(anchor).addScaledVector(d, -10).addScaledVector(side, 1.4);
            sh.pos.y = 4.2;
            sh.target.copy(ball).lerp(goal, t >= plan.shotIndex ? 0.6 : 0.45);
            sh.target.y = Math.min(1.4, sh.target.y);
            sh.fov = 34;
            sh.rate = 3;
        } else {
            // From behind the net, a little off center, looking out at the ball.
            const z = Math.max(-1.2, Math.min(1.2, -b[1] * 0.15));
            sh.pos.set(dir * (HALF_W + 3.6), 1.35, z);
            sh.target.copy(ball);
            sh.target.y = Math.min(ball.y, 1.6);
            sh.fov = 42;
            sh.rate = 4;
        }
        confineShot(sh.pos);
    }

    // Each frame of a replay, slow motion from the shot on, the hold on the goal, the next angle.
    updateReplay(dt) {
        const a = this.active;
        const app = this.app;
        const tl = app.timeline;
        const p = a.plan;
        if (!p || this.wiping) {
            return;
        }
        const slowmo = Number(app.settings.slowmo) || 1;
        if (a.phase === 'play') {
            if (!tl.playing || tl.direction < 0) {
                this.stop(false);
                return;
            }
            tl.speed = a.slow && tl.t >= p.slowFrom ? slowmo : 1;
            if (tl.t >= p.end) {
                tl.t = p.end;
                tl.speed = 0;
                a.phase = 'hold';
                const fa = tl.store.get(p.end - 1) || tl.store.get(p.end);
                const fb = tl.store.get(p.end);
                this.hold = {
                    index: p.end, clock: 0, team: p.goal.team, sps: tl.sps,
                    a: fa.data.slice(8, 11), b: fb.data.slice(8, 11),
                };
            }
        } else if (a.phase === 'hold') {
            a.holdClock += dt * (a.slow ? slowmo : 1);
            this.hold.clock = a.holdClock;
            if (a.holdClock >= REPLAY_HOLD) {
                a.phase = 'wait';
                this.wiping = true;
                this.sting(a.item < a.queue.length - 1 ? 'Replay' : '', () => {
                    this.wiping = false;
                    if (this.active === a) {
                        this.nextShot();
                    }
                });
            }
        }
        if (app.cameras.mode === 'scripted') {
            const t = a.phase === 'hold' ? p.end + Math.min(1, a.holdClock * 1.5) * 0.9 : tl.t;
            if (a.phase === 'hold') {
                // The camera follows the ball on into the net.
                const bx = tmpB;
                ballIntoNet(this.hold.a, this.hold.b, p.goal.team, tl.sps, a.holdClock, bx);
                this.replayShot(p, p.end);
                this.shot.target.set(bx[0], Math.min(1.6, Math.max(0.2, bx[2])), -bx[1]);
            } else {
                this.replayShot(p, t);
            }
        }
    }

    // The orbit shot circles the player in focus.
    focusPoint(pose) {
        const k = this.app.cameras.focusIndex;
        if (k == null || k < 0 || k >= pose.players) {
            return null;
        }
        const o = k * PLAYER_FLOATS;
        return new THREE.Vector3(pose.nodes[o], 0, -pose.nodes[o + 1]);
    }

    // Enter or a click moves on, the lineups to kick-off, a celebration to its replay, a replay
    // to its next angle or back to play.
    skip() {
        if (this.wiping) {
            return;
        }
        if (this.lineup) {
            this.endLineups(true);
        } else if (this.cel) {
            this.afterCelebration();
        } else if (this.active && this.active.plan) {
            const a = this.active;
            a.phase = 'wait';
            this.wiping = true;
            this.sting(a.item < a.queue.length - 1 ? 'Replay' : '', () => {
                this.wiping = false;
                if (this.active === a) {
                    this.nextShot();
                }
            });
        }
    }

    // Ends whatever runs. restore puts play back where it was, else play stays where it is,
    // as when the user seeks or pauses. covered says a stinger already hides the cut.
    stop(restore, covered = false) {
        const app = this.app;
        clearTimeout(this.pending);
        if (this.wiping) {
            this.cancelSting();
            this.wiping = false;
        }
        this.hold = null;
        if (this.lineup) {
            this.closeLineups(restore);
        }
        if (this.cel) {
            const c = this.cel;
            if (restore) {
                this.cel = null;
                this.releaseClips(c);
                app.hud.scoreOf(null);
                this.resume(c.saved, c.goal.index);
            } else {
                this.endCelebration();
            }
        }
        const a = this.active;
        if (!a) {
            return;
        }
        this.active = null;
        app.hud.setReplay(false);
        if (!app.postMatch.running) {
            app.hud.unfreeze();
        }
        if (!app.timeline) {
            return;
        }
        const tl = app.timeline;
        tl.speed = a.saved.speed;
        if (!restore) {
            if (!covered) {
                this.restoreCamera(a.saved);
            }
            return;
        }
        this.resume(a.saved, a.returnT);
        app.afterSeek();
    }

    // Team sheets before kick-off over a slow pan of the ground, while play waits.
    showLineups() {
        const app = this.app;
        const tl = app.timeline;
        if (!tl || !app.source || app.source.kind === 'live' || !app.settings.lineups || this.running
            || tl.t > tl.first + 1 || app.postMatch.running) {
            return;
        }
        const saved = this.savedState();
        tl.pause();
        const overlay = this.lineupCard();
        document.getElementById('watch').appendChild(overlay);
        release(app);
        this.lineup = { clock: 0, overlay, saved };
        this.shot.snap = true;
        this.lineupShot(0);
        this.useScripted();
        app.updatePlayUi();
    }

    lineupShot(clock) {
        const u = clock / LINEUP_LENGTH;
        const sh = this.shot;
        sh.pos.set(-38 + 76 * u, 26 - 6 * u, 58);
        sh.target.set(-14 + 28 * u, 0, 0);
        sh.fov = 40;
        sh.rate = 2;
        confineShot(sh.pos);
    }

    updateLineup(dt) {
        const l = this.lineup;
        l.clock += dt;
        this.lineupShot(l.clock);
        if (l.clock >= LINEUP_LENGTH && !this.wiping) {
            this.endLineups(true);
        }
    }

    endLineups(playAfter) {
        if (this.wiping || !this.lineup) {
            return;
        }
        this.wiping = true;
        this.lineup.overlay.classList.add('leaving');
        this.sting('', () => {
            this.wiping = false;
            this.closeLineups(playAfter);
        });
    }

    closeLineups(playAfter) {
        const app = this.app;
        const l = this.lineup;
        this.lineup = null;
        if (!l) {
            return;
        }
        l.overlay.remove();
        this.restoreCamera(l.saved);
        app.cameras.jump();
        if (playAfter && app.timeline) {
            app.timeline.play(1);
        }
        app.updatePlayUi();
    }

    // The team sheets, keeper first, then the lines from the back, each from the team's left to
    // its right, with the formation read from the first frame.
    lineupCard() {
        const app = this.app;
        const header = app.source.header;
        const store = app.timeline.store;
        const names = teamNames(header);
        const sheets = [0, 1].map((team) => {
            const style = app.styles ? app.styles[team] : null;
            const { keeper, lines } = formation(store, header, team);
            const rows = [];
            const row = (k, role) => {
                const info = playerInfo(app, k);
                const num = h('span', { class: 'ls-num' }, String(info.number));
                num.style.background = info.color;
                num.style.color = info.ink;
                return h('li', { style: `--i:${rows.length}` }, num, h('span', { class: 'ls-name' }, info.name), h('span', { class: 'ls-role' }, role));
            };
            if (keeper >= 0) {
                rows.push(row(keeper, 'GK'));
            }
            lines.forEach((line, i) => {
                const role = i === 0 && lines.length > 1 ? 'DF' : (i === lines.length - 1 && lines.length > 1 ? 'FW' : 'MF');
                for (const k of line) {
                    rows.push(row(k, role));
                }
            });
            const chip = h('i', { class: 'ls-chip' });
            chip.style.background = kitSwatch(style);
            return h('section', { class: `ls-sheet side-${team}` },
                h('header', {}, chip, style && style.badge ? h('b', { class: 'ls-badge' }, style.badge) : null,
                    h('span', { class: 'ls-team' }, names[team]),
                    lines.length > 1 ? h('span', { class: 'ls-formation' }, lines.map((l) => l.length).join('-')) : null),
                h('ol', { class: 'ls-list' }, rows));
        });
        const card = h('div', { id: 'lineups', class: 'lineups' },
            h('div', { class: 'post-kicker' }, 'Lineups'),
            h('div', { class: 'ls-sheets' }, sheets),
            h('div', { class: 'post-hint muted small' }, 'Enter or a click to kick off, Esc to skip'));
        card.addEventListener('click', () => this.skip());
        return card;
    }
}

// A team's keeper and its outfield lines from the back, read from the first frame. Players
// within 5 meters of depth of the one before share a line, each line from the team's left.
export function formation(store, header, team) {
    const f = store.get(store.first);
    const attack = team === 0 ? 1 : -1;
    let keeper = -1;
    const list = [];
    header.players.forEach((p, k) => {
        if (p.team !== team) {
            return;
        }
        if (p.goalkeeper) {
            keeper = k;
            return;
        }
        const o = NODES_FLOAT + k * PLAYER_FLOATS;
        list.push({ k, depth: f ? f.data[o] * attack : p.index, side: f ? f.data[o + 1] * attack : 0 });
    });
    list.sort((a, b) => a.depth - b.depth);
    const lines = [];
    for (const x of list) {
        const line = lines[lines.length - 1];
        if (line && x.depth - line[line.length - 1].depth < 5) {
            line.push(x);
        } else {
            lines.push([x]);
        }
    }
    return { keeper, lines: lines.map((line) => line.sort((a, b) => b.side - a.side).map((x) => x.k)) };
}

// Match facts, scorers with minutes, possession, shots, passes, tackles, saves and cards, all
// from the frames. Possession counts the steps each team owns the ball.
export function matchFacts(store, header) {
    const { stats, goals, shots, passes, tackles } = matchStats(store, header);
    const owned = [0, 0];
    for (let i = store.first; i <= store.last; i++) {
        const f = store.get(i);
        if (f.ownedTeam === 0 || f.ownedTeam === 1) {
            owned[f.ownedTeam]++;
        }
    }
    const perTeam = (list, keep = () => true) => [0, 1].map((team) => list.filter((x) => x.team === team && keep(x)).length);
    const saves = [0, 1].map((team) => stats.filter((s) => s.team === team).reduce((sum, s) => sum + s.saves, 0));
    const final = store.get(store.last);
    const cards = header.players.map((p, k) => ({ k, p, flags: final.playerFlags[k] }))
        .filter((c) => c.flags & 6)
        .map((c) => ({ k: c.k, team: c.p.team, red: !!(c.flags & 4) }));
    const total = owned[0] + owned[1] || 1;
    return {
        stats,
        shotEvents: shots,
        passEvents: passes,
        score: [final.scoreLeft, final.scoreRight],
        scorers: goals.map((g) => ({ ...g, minute: Math.max(1, Math.ceil(store.get(g.index).timeMs / 60000)) })),
        possession: [Math.round((owned[0] / total) * 100), Math.round((owned[1] / total) * 100)],
        shots: perTeam(shots),
        onTarget: perTeam(shots, (s) => s.onTarget),
        passes: perTeam(passes),
        passesCompleted: perTeam(passes, (p) => p.completed),
        tackles: perTeam(tackles),
        saves,
        cards,
    };
}

// Player contribution grades, achievements and the statistics screen.
export const GRADES =
    [ 'F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+', 'S' ];
export const CUTOFFS = [ 0, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9.5 ];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const pos = (f, k) => f.data[NODES_FLOAT + k * PLAYER_FLOATS];
const medal = (s) => h('span', {
    class: `grade-medal tier-${s.tier}`,
    title: s.grade ? `Rating ${s.rating.toFixed(2)}` : 'Too little time'
},
    s.grade || '—');

export function summaryStore(store, live) {
    let first = store.first;
    let last = store.last;
    if (live) {
        for (let i = last; i >= first; i--) {
            if (store.get(i).flags & 1) {
                last = i;
                break;
            }
        }
        for (let i = last; i > first; i--) {
            if (episodeBoundary(store.get(i - 1), store.get(i))) {
                first = i;
                break;
            }
        }
    }
    const frames = [];
    for (let i = first; i <= last; i++) frames.push(store.get(i));
    return {first, last, get: (i) => frames[i - first]};
}

export function contribution(store, header, facts) {
    const first = store.get(store.first);
    const last = store.get(store.last);
    const duration = Math.max(0, (last.timeMs - first.timeMs) / 60000);
    const players = facts.stats.map((s) => {
        const x = pos(first, s.k) * (s.team === 0 ? 1 : -1);
        const role = s.goalkeeper ? 'GK' : x < -15 ? 'DF' : x > 10 ? 'FW' : 'MF';
        return {
            ...s,
            role,
            minutes: 0,
            elapsedMs: 0,
            keyPasses: 0,
            yellow: 0,
            red: 0,
            clean: 0,
            achievements: []
        };
    });
    const bins = Array.from({length: 12},
        () => ({owned: [ 0, 0 ], pressure: 0, count: 0, shots: [ 0, 0 ], goals: [ 0, 0 ]}));
    const marks = [];
    const binOf = (i) =>
        clamp(Math.floor((i - store.first) / Math.max(1, store.last - store.first) * 12), 0, 11);
    for (let i = store.first + 1; i <= store.last; i++) {
        const a = store.get(i - 1);
        const b = store.get(i);
        if (!a || !b || isCut(a, b)) continue;
        const dt = clamp(b.timeMs - a.timeMs, 0, 60000);
        const bin = bins[binOf(i)];
        if (b.ownedTeam >= 0 && b.ownedTeam <= 1) {
            bin.owned[b.ownedTeam]++;
            bin.pressure += 0.5 * clamp(b.data[8] / 52.5, -1, 1) + (b.ownedTeam === 0 ? 0.5 : -0.5);
            bin.count++;
        }
        for (const s of players) {
            if (a.playerFlags[s.k] & 1) s.elapsedMs += dt;
            const added = b.playerFlags[s.k] & ~a.playerFlags[s.k];
            if (added & 2) s.yellow++;
            if (added & 4) s.red++;
            if (added & 6)
                marks.push({
                    i,
                    team: s.team,
                    label: added & 4 ? 'Red card' : 'Yellow card',
                    icon: added & 4 ? '🟥' : '🟨'
                });
        }
    }
    for (const shot of facts.shotEvents) {
        bins[binOf(shot.i)].shots[shot.team]++;
        const pass = facts.passEvents.findLast((p) => p.completed && p.team === shot.team &&
                p.receiver === shot.k && p.i < shot.i &&
                shot.i - p.i <= 10 * (header.steps_per_second || 10));
        if (pass) {
            let clear = true;
            for (let i = pass.i + 1; i <= shot.i; i++) {
                const f = store.get(i);
                if (!f || isCut(store.get(i - 1), f) || f.gameMode !== 0 ||
                    (f.ownedTeam >= 0 && f.ownedTeam !== shot.team) ||
                    f.touches.some((touch, k) => (touch === 1 || touch === 2) &&
                            header.players[k].team !== shot.team)) {
                    clear = false;
                    break;
                }
            }
            if (clear) players[pass.k].keyPasses++;
        }
    }
    for (const g of facts.scorers) {
        bins[binOf(g.index)].goals[g.team]++;
        marks.push({i: g.index, team: g.team, label: 'Goal', icon: '●'});
    }
    const award = (s, name, icon, reason) => s.achievements.push({name, icon, reason});
    const max = (key) => Math.max(0, ...players.map((s) => s[key]));
    for (const s of players) {
        s.minutes = s.elapsedMs / 60000;
        const conceded =
            facts.scorers
                .filter((g) => g.team !== s.team && (store.get(g.index).playerFlags[s.k] & 1))
                .length;
        s.clean = duration >= 1 && s.minutes >= duration * 0.8 && conceded === 0 ? 1 : 0;
        const defense = s.role === 'GK' || s.role === 'DF';
        const scale = Math.min(3, 30 / Math.max(10, s.minutes));
        const accuracy = s.passes ? s.passesCompleted / s.passes : 0;
        s.accuracy = s.passes ? Math.round(100 * accuracy) : null;
        s.shotAccuracy = s.shots ? Math.round(100 * s.shotsOnTarget / s.shots) : null;
        s.positive = scale *
            (1.4 * s.goals + s.assists + 0.25 * s.shotsOnTarget + 0.35 * s.keyPasses +
                (defense ? 0.6 : 0.35) * s.tackles + (s.goalkeeper ? 0.65 : 0) * s.saves);
        s.passing = s.passes >= 3 ? (accuracy - 0.65) * 1.5 : 0;
        s.sheet = s.clean ? (defense ? 1 : 0.3) : 0;
        s.penalty = 2 * s.ownGoals + 0.4 * s.yellow + 1.5 * s.red;
        s.rating = clamp(5 + s.positive + s.passing + s.sheet - s.penalty, 0, 10);
        const eligible = s.minutes >= Math.max(0.5, Math.min(10, duration * 0.2));
        let gradeIndex = CUTOFFS.findLastIndex((v) => s.rating >= v);
        if (gradeIndex === 13 &&
            (s.minutes < Math.max(1, duration * 0.6) || s.positive < 4.5 || s.penalty > 0))
            gradeIndex = 12;
        s.gradeRank = eligible ? gradeIndex : -1;
        s.grade = eligible ? GRADES[gradeIndex] : null;
        const tiers = ['plain', 'plain', 'plain', 'plain', 'bronze', 'bronze', 'bronze',
            'silver', 'silver', 'silver', 'gold', 'gold', 'gold', 's'];
        s.tier = eligible ? tiers[gradeIndex] : 'plain';
        if (s.goals >= 3)
            award(s, 'Hat trick', '⚽', `${s.goals} goals`);
        else if (s.goals === 2)
            award(s, 'Brace', '⚽', '2 goals');
        if (s.assists > 0 && s.assists === max('assists'))
            award(s, 'Playmaker', '✦', `${s.assists} assists, joint leaders included`);
        else if (s.keyPasses >= 2 && s.keyPasses === max('keyPasses'))
            award(s, 'Creator', '✦', `${s.keyPasses} key passes, joint leaders included`);
        if (s.saves > 0 && s.saves === max('saves'))
            award(s, 'Wall', '▣', `${s.saves} saves, joint leaders included`);
        if (s.clean)
            award(s, 'Clean sheet', '◇', `${s.minutes.toFixed(1)} minutes, 0 conceded on pitch`);
        if (s.touches >= 5 && s.touches === max('touches'))
            award(s, 'Engine', '↯', `${s.touches} touches, joint leaders included`);
        if (s.tackles >= 2 && s.tackles === max('tackles'))
            award(s, 'Ball winner', '◆',
                `${s.tackles} tackles and recoveries, joint leaders included`);
        if (s.shots >= 3 && s.shotsOnTarget / s.shots >= 0.75)
            award(s, 'Sharpshooter', '◎', `${s.shotsOnTarget} of ${s.shots} shots on target`);
        if (s.ownGoals) award(s, 'Own goal', '↶', `${s.ownGoals} own goals`);
        if (s.red) award(s, 'Sent off', '▮', `${s.red} red card`);
    }
    const teams = [ [], [] ];
    for (const team of [0, 1]) {
        const opponent = 1 - team;
        if (duration >= 1 && facts.score[opponent] === 0 &&
            (team === 0 ? first.scoreRight : first.scoreLeft) === 0)
            teams[team].push({
                name: 'Clean sheet',
                icon: '◇',
                reason: '0 goals conceded in the available frames'
            });
        if (facts.score[team] > facts.score[opponent]) {
            const trailed = facts.scorers.some((g) => {
                const f = store.get(g.index);
                return (team === 0 ? f.scoreLeft - f.scoreRight : f.scoreRight - f.scoreLeft) < 0;
            });
            if (trailed)
                teams[team].push({
                    name: 'Comeback',
                    icon: '↗',
                    reason: `Won ${facts.score[team]} to ${facts.score[opponent]} after trailing`
                });
            const winner = facts.scorers.find((g) => g.team === team &&
                    (team === 0 ? store.get(g.index).scoreLeft : store.get(g.index).scoreRight) ===
                        facts.score[opponent] + 1);
            if (winner && duration >= 1 &&
                store.get(winner.index).timeMs >= first.timeMs + duration * 60000 * 0.9)
                teams[team].push({
                    name: 'Late winner',
                    icon: '◷',
                    reason:
                        `Decisive goal at ${winner.minute} minutes, final tenth of this episode`
                });
        }
    }
    return {players, bins, marks, teams, duration};
}

// The comparison of the two teams, a row per fact with both values and a bar of their shares.
export function comparisonTable(f, styles = [null, null]) {
    const row = (label, a, b, va = Number(a), vb = Number(b)) => {
        const sum = (va || 0) + (vb || 0);
        const share = sum > 0 ? (va / sum) * 100 : 50;
        const bar = h('div', { class: 'post-bar', 'aria-hidden': 'true' }, h('i', { style: `width:${share}%` }));
        return h('tr', {}, h('td', { class: 'num' }, String(a)),
            h('th', { scope: 'row' }, label, sum > 0 ? bar : null), h('td', { class: 'num' }, String(b)));
    };
    const cards = (team, red) => f.cards.filter((c) => c.team === team && c.red === red).length;
    const accuracy = (team) => (f.passes[team] ? Math.round((100 * f.passesCompleted[team]) / f.passes[team]) : null);
    const pct = (v) => (v == null ? '–' : `${v}%`);
    const table = h('table', { class: 'post-table' }, h('tbody', {},
        row('Possession', `${f.possession[0]}%`, `${f.possession[1]}%`, f.possession[0], f.possession[1]),
        row('Shots', f.shots[0], f.shots[1]),
        row('On target', f.onTarget[0], f.onTarget[1]),
        row('Passes', f.passes[0], f.passes[1]),
        row('Pass accuracy', pct(accuracy(0)), pct(accuracy(1)), accuracy(0) || 0, accuracy(1) || 0),
        row('Tackles', f.tackles[0], f.tackles[1]),
        row('Saves', f.saves[0], f.saves[1]),
        row('Yellow cards', cards(0, false), cards(1, false)),
        row('Red cards', cards(0, true), cards(1, true))));
    table.style.setProperty('--left', styles && styles[0] ? css(styles[0].kit.primary) : 'var(--accent)');
    table.style.setProperty('--right', styles && styles[1] ? css(styles[1].kit.primary) : 'var(--muted)');
    return table;
}

// The player of the match, the winners' best by score, or the best of both teams after a draw.
export function playerOfMatch(stats, score) {
    const winner = score[0] === score[1] ? -1 : score[0] > score[1] ? 0 : 1;
    const best = stats.slice().sort((a, b) => b.score - a.score);
    return winner >= 0 ? best.find((s) => s.team === winner) : best[0];
}

const count = (value, one, many) => `${value} ${value === 1 ? one : many}`;

// The comparison of the statistics screen draws its bars in the colors of the charts.
function chartColors(table) {
    table.style.setProperty('--left', 'var(--chart-0)');
    table.style.setProperty('--right', 'var(--chart-1)');
    return table;
}

let medalCount = 0;

// The MVP medal, a gold disc with an embossed rim, a laurel, a star and MVP lettering, hung from a
// ribbon in the two colors of the player's team. It is inline SVG, crisp at any size, and a band of
// light sweeps across the gold now and then. Each color comes as its red, green and blue channels,
// and the SVG takes them as numbers from 0 to 255 alone.
export function mvpMedal(channels) {
    const id = `mvp-medal-${++medalCount}`;
    const colors = channels.map((c) => `rgb(${[0, 1, 2].map((i) =>
        Math.round(Math.min(255, Math.max(0, Number(c[i]) || 0)))).join(', ')})`);
    const f = (v) => v.toFixed(2);
    const leaves = [];
    for (const side of [-1, 1]) {
        for (let i = 0; i < 7; i++) {
            const phi = 26 + i * 17;
            const r = (phi * Math.PI) / 180;
            const x = 60 + side * 29 * Math.sin(r);
            const y = 112 + 29 * Math.cos(r);
            const angle = side * (90 - phi) + side * (i % 2 ? -28 : 28);
            leaves.push(`<ellipse cx="${f(x)}" cy="${f(y)}" rx="2.4" ry="5.6" transform="rotate(${f(angle)} ${f(x)} ${f(y)})"/>`);
        }
    }
    const star = Array.from({ length: 10 }, (_, i) => {
        const r = i % 2 ? 4.6 : 11;
        const a = (Math.PI / 5) * i - Math.PI / 2;
        return `${f(60 + r * Math.cos(a))},${f(98 + r * Math.sin(a))}`;
    }).join(' ');
    const node = h('div', { class: 'mvp-medal', role: 'img', 'aria-label': 'MVP medal' });
    node.innerHTML = `<svg viewBox="0 0 120 170" aria-hidden="true" focusable="false">
<defs>
<linearGradient id="${id}-rim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fff4c2"/><stop offset="0.35" stop-color="#e7b94c"/><stop offset="0.7" stop-color="#a8701a"/><stop offset="1" stop-color="#6e4608"/></linearGradient>
<linearGradient id="${id}-bevel" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8a5a10"/><stop offset="0.5" stop-color="#c8922e"/><stop offset="1" stop-color="#fff0b0"/></linearGradient>
<radialGradient id="${id}-face" cx="0.38" cy="0.32" r="0.75"><stop offset="0" stop-color="#fff6cf"/><stop offset="0.45" stop-color="#f0c75a"/><stop offset="1" stop-color="#b8801f"/></radialGradient>
<linearGradient id="${id}-shade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#000" stop-opacity="0.35"/><stop offset="0.5" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.35"/></linearGradient>
<linearGradient id="${id}-shine" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="0.5" stop-color="#fff" stop-opacity="0.75"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<clipPath id="${id}-clip"><circle cx="60" cy="112" r="46"/></clipPath>
</defs>
<path d="M36 0H60V64L48 74L36 64Z" fill="${colors[0]}"/>
<path d="M60 0H84V64L72 74L60 64Z" fill="${colors[1]}"/>
<path d="M36 0H84V64L72 74L60 64L48 74L36 64Z" fill="url(#${id}-shade)" stroke="#00000066" stroke-width="1"/>
<rect x="49" y="57" width="22" height="10" rx="2" fill="url(#${id}-rim)" stroke="#6e4608" stroke-width="0.8"/>
<circle cx="60" cy="112" r="46" fill="url(#${id}-rim)"/>
<circle cx="60" cy="112" r="40.5" fill="url(#${id}-bevel)"/>
<circle cx="60" cy="112" r="37.5" fill="url(#${id}-face)"/>
<circle cx="60" cy="112" r="43.2" fill="none" stroke="#fff4c8" stroke-opacity="0.75" stroke-width="1.5" stroke-dasharray="0.1 3.4" stroke-linecap="round"/>
<g fill="#9a6a14" stroke="#fff0b8" stroke-opacity="0.55" stroke-width="0.5">${leaves.join('')}</g>
<polygon points="${star}" fill="#fff1b8" stroke="#8a5a10" stroke-width="1.2" stroke-linejoin="round"/>
<text x="60" y="131" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-size="17" font-weight="700" letter-spacing="1.5" fill="#fff4cc" opacity="0.8">MVP</text>
<text x="60" y="130" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-size="17" font-weight="700" letter-spacing="1.5" fill="#6b440a">MVP</text>
<g clip-path="url(#${id}-clip)"><g transform="rotate(24 60 112)"><rect class="medal-shine" x="-10" y="40" width="22" height="150" fill="url(#${id}-shine)"/></g></g>
</svg>`;
    return node;
}

// The kit colors of the two teams for the charts, or null without kits. A second team whose shirt
// is close to the first team's takes its trim color, so the two sides stay apart.
export function teamChartColors(styles) {
    if (!styles || !styles[0] || !styles[1] || !styles[0].kit || !styles[1].kit) {
        return null;
    }
    const a = styles[0].kit.primary;
    let b = styles[1].kit.primary;
    const gap = (x, y) => Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
    if (gap(a, b) < 90 && styles[1].kit.secondary && gap(a, styles[1].kit.secondary) > gap(a, b)) {
        b = styles[1].kit.secondary;
    }
    return [css(a), css(b)];
}

// Numbers the clip paths of the possession charts, so each chart on the page has its own.
let chartCount = 0;

// One chart over the periods of the match, drawn in the colors of the two sides.
function chart(report, kind, names, store) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 320 106');
    svg.setAttribute('role', 'img');
    const draw = (tag, attrs, text) => {
        const node = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
        if (text != null) node.textContent = text;
        svg.append(node);
        return node;
    };
    const tip = (node, text) => {
        const title = document.createElementNS(ns, 'title');
        title.textContent = text;
        node.append(title);
    };
    const left = 6;
    const width = 308;
    const top = 6;
    const bottom = 86;
    const step = width / 12;
    const bins = report.bins;
    const legend = [];
    if (kind === 'possession') {
        // The share of the first team in each period around an even split. The line and the
        // area under it take the first team's color above the middle and the second's below,
        // so the color changes where the line crosses the middle.
        const mid = (top + bottom) / 2;
        const id = `possession-${++chartCount}`;
        const defs = draw('defs', {});
        for (const [side, y, height] of [[0, top - 4, mid - top + 4], [1, mid, bottom - mid + 4]]) {
            const clip = document.createElementNS(ns, 'clipPath');
            clip.setAttribute('id', `${id}-${side}`);
            const rect = document.createElementNS(ns, 'rect');
            for (const [k, v] of Object.entries({ x: 0, y, width: 320, height })) rect.setAttribute(k, v);
            clip.append(rect);
            defs.append(clip);
        }
        let last = 0.5;
        const points = bins.map((b, i) => {
            const total = b.owned[0] + b.owned[1];
            const share = total ? b.owned[0] / total : last;
            last = share;
            return { x: left + (i + 0.5) * step, y: mid - (share - 0.5) * 2 * (mid - top), share, total };
        });
        const line = points.map((p) => `${p.x},${p.y}`);
        const first = points[0] || { y: mid };
        const end = points[points.length - 1] || first;
        const edge = [`${left},${first.y}`, ...line, `${left + width},${end.y}`];
        const area = `M${left} ${mid}L${edge.join('L')}L${left + width} ${mid}Z`;
        for (const side of [0, 1]) {
            draw('path', { d: area, class: `chart-side-${side} chart-area`, 'clip-path': `url(#${id}-${side})` });
            draw('polyline', { points: edge.join(' '), class: `chart-line side-${side}`, 'clip-path': `url(#${id}-${side})` });
        }
        for (const p of points) {
            if (!p.total) continue;
            const text = `${names[0]} ${Math.round(100 * p.share)}%, ${names[1]} ${Math.round(100 * (1 - p.share))}%`;
            tip(draw('circle', { cx: p.x, cy: p.y, r: 2.5, class: `chart-side-${p.share >= 0.5 ? 0 : 1}` }), text);
        }
        draw('path', { d: `M${left} ${mid}H${left + width}`, class: 'chart-rule half' });
        draw('text', { x: left + width - 3, y: mid - 3, class: 'chart-note end' }, '50%');
        draw('text', { x: left + 3, y: top + 11, class: 'chart-note' }, `${names[0]} on the ball more`);
        draw('text', { x: left + 3, y: bottom - 5, class: 'chart-note' }, `${names[1]} on the ball more`);
        legend.push(['side-0', `${names[0]} above 50%`], ['side-1', `${names[1]} below 50%`]);
    } else if (kind === 'shots') {
        const sums = (team, key) => bins.reduce((v, b) => v + b[key][team], 0);
        const ceiling = Math.max(1, sums(0, 'shots'), sums(1, 'shots'), sums(0, 'goals'), sums(1, 'goals'));
        draw('path', { d: `M${left} ${top}H${left + width}`, class: 'chart-rule faint' });
        for (const team of [0, 1]) {
            for (const key of ['shots', 'goals']) {
                let sum = 0;
                const points = [`${left},${bottom}`];
                bins.forEach((b, i) => {
                    sum += b[key][team];
                    points.push(`${left + (i + 1) * step},${bottom - (sum / ceiling) * (bottom - top)}`);
                });
                draw('polyline', { points: points.join(' '), class: `chart-line side-${team} ${key}` });
            }
        }
        draw('text', { x: left + 3, y: top + 11, class: 'chart-note' }, `Top line, ${count(ceiling, 'shot', 'shots')}`);
        legend.push(['side-0', names[0]], ['side-1', names[1]], ['solid', 'Goals'], ['dashed', 'Shots']);
    } else {
        const mid = (top + bottom) / 2;
        draw('path', { d: `M${left} ${mid}H${left + width}`, class: 'chart-rule half' });
        const points = bins.map((b, i) => {
            const pressure = b.count ? b.pressure / b.count : 0;
            return `${left + (i + 0.5) * step},${mid - pressure * (mid - top)}`;
        });
        draw('polyline', { points: points.join(' '), class: 'chart-line neutral' });
        draw('text', { x: left + 3, y: top + 11, class: 'chart-note' }, `${names[0]} on top`);
        draw('text', { x: left + 3, y: bottom - 5, class: 'chart-note' }, `${names[1]} on top`);
    }
    draw('path', { d: `M${left} ${bottom}H${left + width}`, class: 'chart-rule' });
    // Goals and cards on a strip under the chart, at their moment.
    let cards = false;
    for (const m of report.marks) {
        const x = left + ((m.i - store.first) / Math.max(1, store.last - store.first)) * width;
        const y = 97;
        const node = m.label === 'Goal'
            ? draw('circle', { cx: x, cy: y, r: 3.5, class: `chart-side-${m.team}` })
            : draw('rect', { x: x - 2.5, y: y - 4.5, width: 5, height: 9, class: m.label === 'Red card' ? 'chart-red' : 'chart-yellow' });
        cards ||= m.label !== 'Goal';
        tip(node, `${m.label}, ${names[m.team]}, ${(store.get(m.i).timeMs / 60000).toFixed(1)} minutes`);
    }
    if (report.marks.some((m) => m.label === 'Goal')) legend.push(['dot', 'Goal']);
    if (cards) legend.push(['card', 'Card']);
    const values = bins.map((b, i) => {
        const total = b.owned[0] + b.owned[1];
        const value = kind === 'possession'
            ? total ? `${Math.round((100 * b.owned[0]) / total)} percent for ${names[0]}` : 'no possession'
            : kind === 'pressure'
                ? (b.count ? b.pressure / b.count : 0).toFixed(2)
                : `${b.shots.join(' to ')} shots and ${b.goals.join(' to ')} goals`;
        return `Period ${i + 1}, ${value}`;
    });
    const title = {
        possession: 'Possession against an even split, in each period',
        shots: 'Shots and goals, running total',
        pressure: 'Territory and possession',
    }[kind];
    svg.setAttribute('aria-label', `${title}. ${names[0]} first. ${values.join('. ')}`);
    const minute = (i) => `${Math.round(store.get(i).timeMs / 60000)} min`;
    return h('figure', { class: 'stats-chart' },
        h('figcaption', {}, title),
        svg,
        h('div', { class: 'chart-axis', 'aria-hidden': 'true' }, h('span', {}, minute(store.first)), h('span', {}, minute(store.last))),
        h('div', { class: 'chart-legend', 'aria-hidden': 'true' },
            legend.map(([key, text]) => h('span', { class: `chart-key ${key}` }, text))));
}

// Awards, in the order the headline shows them, the rarest first.
const AWARD_ORDER = ['Hat trick', 'Brace', 'Playmaker', 'Wall', 'Sharpshooter', 'Creator', 'Ball winner', 'Clean sheet', 'Engine'];

// The columns of the player table, the first set always shown and the rest on demand.
const COLUMNS = [
    ['gradeRank', 'Grade'], ['minutes', 'Minutes'], ['goals', 'Goals'], ['assists', 'Assists'],
    ['shots', 'Shots'], ['passesCompleted', 'Passes'], ['tackles', 'Tackles'], ['saves', 'Saves'],
];
const MORE_COLUMNS = [
    ['shotsOnTarget', 'On target'], ['keyPasses', 'Key passes'], ['accuracy', 'Pass accuracy'],
    ['shotAccuracy', 'Shot accuracy'], ['touches', 'Touches'], ['clean', 'Clean sheet'],
    ['ownGoals', 'Own goals'], ['yellow', 'Yellow cards'], ['red', 'Red cards'],
];
const ROLE_NAMES = { GK: 'Goalkeeper', DF: 'Defender', MF: 'Midfielder', FW: 'Forward' };

// The statistics screen of the show after the match. It opens on the result and the headline
// awards, then compares the teams, lists every player of each team with a grade medal, and ends
// with charts over the match. Extra columns, the rating inputs of a player and the grade guide
// open on demand.
export function statsScreen(report, names, info, store, advance, extra = {}) {
    const facts = extra.facts;
    let team = 0;
    let sort = 'gradeRank';
    let descending = true;
    let more = false;
    const head = h('thead', {});
    const body = h('tbody', {});
    const cell = (s, key) => {
        if (key === 'gradeRank') return medal(s);
        if (key === 'minutes') return s.minutes.toFixed(1);
        if (key === 'passesCompleted') return s.passes ? `${s.passesCompleted} of ${s.passes}` : '0';
        if (key === 'accuracy' || key === 'shotAccuracy') return s[key] == null ? '–' : `${s[key]}%`;
        if (key === 'clean') return s.clean ? 'Yes' : '–';
        return String(s[key] ?? '–');
    };
    const columns = () => (more ? [...COLUMNS, ...MORE_COLUMNS] : COLUMNS);
    const renderHead = () => {
        head.replaceChildren(h('tr', {}, h('th', { scope: 'col', class: 'player-col' }, 'Player'),
            ...columns().map(([key, label]) => h('th', {
                scope: 'col',
                'aria-sort': key === sort ? (descending ? 'descending' : 'ascending') : 'none',
            }, h('button', {
                type: 'button',
                title: `Sort by ${label.toLowerCase()}`,
                onclick: () => {
                    descending = sort === key ? !descending : true;
                    sort = key;
                    render();
                    head.querySelector(`th[aria-sort]:not([aria-sort="none"]) button`)?.focus();
                },
            }, label)))));
    };
    const tabs = h('div', { class: 'stats-tabs', role: 'tablist', 'aria-label': 'Team' });
    const moreButton = h('button', {
        type: 'button', class: 'btn ghost', 'aria-controls': 'stats-roster',
        onclick: () => {
            more = !more;
            if (!more && MORE_COLUMNS.some(([key]) => key === sort)) sort = 'gradeRank';
            render();
        },
    });
    const roster = h('div', { class: 'stats-table', id: 'stats-roster', role: 'tabpanel', tabindex: '0' },
        h('table', {}, head, body));
    const render = () => {
        renderHead();
        const ordered = report.players.filter((p) => p.team === team).sort((a, b) => {
            const difference = (b[sort] ?? -1) - (a[sort] ?? -1);
            return difference * (descending ? 1 : -1) || a.k - b.k;
        });
        body.replaceChildren(...ordered.map((s) => {
            const who = info(s.k);
            const details = h('details', {},
                h('summary', {}, h('span', { class: 'shirt-number' }, String(who.number)), who.name,
                    h('small', {}, ROLE_NAMES[s.role] || s.role)),
                h('div', { class: 'rating-inputs' },
                    h('p', {}, `Rating ${s.rating.toFixed(2)} of 10, ${s.grade ? `grade ${s.grade}` : 'too little time for a grade'}.`),
                    h('p', {}, `Base 5, contribution ${s.positive.toFixed(2)}, passing ${s.passing.toFixed(2)}, `
                        + `clean sheet ${s.sheet.toFixed(2)}, discipline less ${s.penalty.toFixed(2)}.`),
                    ...s.achievements.map((a) => h('p', {}, `${a.name}, ${a.reason}.`))));
            const row = h('tr', {}, h('th', { scope: 'row', class: 'player-cell' }, details,
                s.achievements.length ? h('div', { class: 'player-awards' }, s.achievements.map((a) =>
                    h('span', { class: 'award-chip', 'data-award': a.name, title: a.reason }, a.name))) : null));
            for (const [key] of columns()) row.append(h('td', {}, cell(s, key)));
            return row;
        }));
        for (const button of tabs.children) {
            const selected = Number(button.dataset.team) === team;
            button.setAttribute('aria-selected', String(selected));
            button.tabIndex = selected ? 0 : -1;
        }
        roster.setAttribute('aria-labelledby', `stats-tab-${team}`);
        moreButton.setAttribute('aria-pressed', String(more));
        moreButton.textContent = more ? 'Fewer columns' : 'More columns';
    };
    for (const t of [0, 1]) {
        const players = report.players.filter((p) => p.team === t).length;
        const button = h('button', {
            type: 'button', role: 'tab', 'data-team': String(t), 'aria-controls': 'stats-roster', id: `stats-tab-${t}`,
            onclick: () => {
                team = t;
                render();
            },
        }, h('span', { class: `chart-key side-${t}`, 'aria-hidden': 'true' }), names[t],
        h('span', { class: 'tab-count' }, count(players, 'player', 'players')));
        button.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                e.preventDefault();
                team = 1 - team;
                render();
                tabs.children[team].focus();
            }
        });
        tabs.append(button);
    }

    // The headline, the player of the match first, then the rarest achievements and team awards.
    const award = (name, who, side, why, cls = 'award') => h('li', { class: cls },
        h('span', { class: 'award-name' }, name),
        h('span', { class: 'award-who' }, who, side != null ? h('small', {}, names[side]) : null),
        h('span', { class: 'award-why' }, why));
    const headline = [];
    const mvp = facts ? playerOfMatch(report.players, facts.score) : null;
    if (mvp) {
        const who = info(mvp.k);
        const line = [count(mvp.goals, 'goal', 'goals'), count(mvp.assists, 'assist', 'assists'),
            mvp.goalkeeper || mvp.saves ? count(mvp.saves, 'save', 'saves') : count(mvp.tackles, 'tackle', 'tackles')];
        headline.push(award('Player of the match', `${who.number} ${who.name}`, mvp.team, line.join(', '), 'award mvp'));
    }
    const top = report.players
        .flatMap((s) => s.achievements.filter((a) => AWARD_ORDER.includes(a.name)).map((a) => ({ s, a })))
        .sort((x, y) => AWARD_ORDER.indexOf(x.a.name) - AWARD_ORDER.indexOf(y.a.name) || y.s.score - x.s.score)
        .slice(0, 4);
    for (const { s, a } of top) {
        const who = info(s.k);
        headline.push(award(a.name, `${who.number} ${who.name}`, s.team, a.reason.replace(/, joint leaders included$/, '')));
    }
    for (const t of [0, 1]) {
        for (const a of report.teams[t]) headline.push(award(a.name, names[t], null, a.reason));
    }

    const section = (id, title, ...content) => h('section', { class: `stats-section stats-${id}`, 'aria-labelledby': `stats-${id}-title` },
        h('h3', { id: `stats-${id}-title` }, title), ...content);
    const root = h('section', { class: 'match-lab', 'aria-labelledby': 'stats-title' },
        h('header', { class: 'stats-heading' },
            h('div', { class: 'stats-result' }, h('h2', { id: 'stats-title' }, 'Match statistics'), extra.scoreLine || null),
            h('button', { type: 'button', class: 'btn primary', onclick: advance }, extra.closeLabel || 'Continue')),
        headline.length ? section('awards', 'Awards', h('ol', { class: 'award-list' }, headline)) : null,
        facts ? section('teams', 'Teams',
            h('div', { class: 'compare-names', 'aria-hidden': 'true' }, h('span', {}, names[0]), h('span', {}, names[1])),
            chartColors(comparisonTable(facts))) : null,
        section('players', 'Players', h('div', { class: 'stats-toolbar' }, tabs, moreButton), roster),
        section('time', 'Over the match',
            h('div', { class: 'stats-charts' }, ['possession', 'shots', 'pressure'].map((kind) => chart(report, kind, names, store)))),
        h('details', { class: 'rating-guide' }, h('summary', {}, 'How grades work'),
            h('p', {}, 'Grades run from F through D, C, B and A to S, with minus and plus steps between. '
                + 'A player with too little time shows a dash. Roles come from the starting position.'),
            h('p', {}, 'A rating starts at 5. Contribution per 30 minutes, capped at three times, counts goals 1.4, '
                + 'assists 1, shots on target 0.25, key passes 0.35, tackles 0.6 for a goalkeeper or defender '
                + 'and 0.35 otherwise, and goalkeeper saves 0.65. Passing adds accuracy less 0.65, times 1.5, '
                + 'after three passes. A clean sheet adds 1 for a goalkeeper or defender and 0.3 otherwise. '
                + 'Own goals cost 2, yellow cards 0.4 and red cards 1.5.'),
            h('p', {}, 'The charts split the match into twelve equal periods. Possession is a line around an '
                + 'even split, in the color of the first team above the middle and of the second below, and '
                + 'leaves out time when neither team holds the ball. Territory and possession averages the ball position and the '
                + 'side in possession, and a line above the middle favours the first team.')));
    root.addEventListener('click', (e) => e.stopPropagation());
    root.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') e.stopPropagation();
        if (e.key === 'Enter') {
            e.preventDefault();
            const control = e.target.closest('button, summary');
            if (control)
                control.click();
            else
                advance();
        }
    });
    render();
    return root;
}

// Whether a team's name reads as plural, so the podium says win rather than wins.
function plural(name) {
    const n = name.trim();
    return /[^s']s$/i.test(n) && !/ss$/i.test(n);
}

// The stages of the scene at the final whistle in seconds. The camera of play stays on while the
// play winds down under the full time card, then a wide shot, a close-up of the winners' hero,
// one of the beaten keeper, and a wide shot as the players walk to the stands applauding the fans.
// Times after the full time card comes up, which is once the play has wound down.
const REACTION_SHOTS = [
    { at: -Infinity, kind: 'live' },
    { at: 0.6, kind: 'wide', snap: true },
    { at: 3.8, kind: 'hero', snap: true },
    { at: 7.0, kind: 'beaten', snap: true },
    { at: 10.2, kind: 'stands', snap: true },
];
const REACTION_LENGTH = 16;
const TITLE_SECONDS = 3.2;
// The players react from the final whistle on, the wind down of play comes first.
const REACTION_LEAD = 0;
// The fade into the Home page after the show.
const HOME_FADE_MS = 420;

// The show after the match, as on television. The players react at the final whistle under
// the full time card, then the match facts, a reel of the goals, the player of the match and
// the trophy with the podium, each part joined by the wipe. Enter or a click moves on, a goal
// at a time in the reel, Esc ends the show.
export class PostMatch {
    constructor(app) {
        this.app = app;
        this.step = -1;
        this.card = null;
        this.timer = null;
        this.react = null;
        this.wiping = false;
        this.hook = (pose, sample) => this.dress(pose, sample);
    }

    get running() {
        return this.step >= 0;
    }

    start() {
        const app = this.app;
        if (this.running || !app.source || !app.settings.podium || !app.timeline) {
            return;
        }
        const tl = app.timeline;
        if (this.closeStats) {
            this.closeStats();
        }
        if (app.director.running) {
            app.director.stop(false);
        }
        if (app.poseHooks && !app.poseHooks.includes(this.hook)) {
            app.poseHooks.push(this.hook);
        }
        this.saved = { t: tl.t, camera: app.cameras.mode, focus: app.cameras.focusIndex, following: tl.following };
        this.statsStore = summaryStore(tl.store, app.source.kind === 'live');
        this.facts = matchFacts(this.statsStore, app.source.header);
        this.steps = ['reactions', 'facts', 'reel', 'spotlight', 'podium'];
        this.step = -1;
        this.wiping = false;
        document.getElementById('app').classList.add('post-show');
        // The bug holds the final score and clock while the show replays earlier moments.
        app.hud.freeze(tl.store.get(tl.last));
        tl.pause();
        this.next();
    }

    teamName(t) {
        return teamNames(this.app.source.header)[t];
    }

    playerName(k) {
        return playerInfo(this.app, k).name;
    }

    winner() {
        const f = this.facts;
        return f.score[0] === f.score[1] ? -1 : (f.score[0] > f.score[1] ? 0 : 1);
    }

    showCard(node) {
        this.clearCard();
        this.card = h('div', { class: 'post-card', 'data-prompt': '' }, node,
            h('div', { class: 'post-hint muted small' }, 'Enter or a click for the next part, Esc to finish'));
        this.card.addEventListener('click', () => this.advance());
        document.getElementById('watch').appendChild(this.card);
        release(this.app);
    }

    clearCard() {
        if (this.card) {
            this.card.remove();
            this.card = null;
        }
    }

    // The heading of the cards, the two teams with their kit colors and badges and the score.
    scoreLine(f) {
        const app = this.app;
        const side = (t) => {
            const style = app.styles ? app.styles[t] : null;
            const chip = h('i', { class: 'post-chip' });
            chip.style.background = kitSwatch(style);
            return h('span', { class: `post-team side-${t}` }, t === 0 ? chip : null,
                style && style.badge ? h('b', { class: 'post-badge' }, style.badge) : null,
                this.teamName(t), t === 1 ? chip : null);
        };
        return h('div', { class: 'post-line' }, side(0), h('b', {}, `${f.score[0]} – ${f.score[1]}`), side(1));
    }

    // The next part behind the wipe, so no part cuts hard into the next.
    advance() {
        if (!this.running || this.wiping) {
            return;
        }
        this.wiping = true;
        this.app.director.sting('', () => {
            this.wiping = false;
            if (this.running) {
                this.next();
            }
        });
    }

    // The players at the final whistle, on the held last frame, with the cameras of the scene.
    startReactions() {
        const app = this.app;
        const tl = app.timeline;
        const store = tl.store;
        const header = app.source.header;
        let index = Math.max(store.first, Math.min(store.last, tl.index));
        // A live stream may be past the whistle already, the show starts from its frame.
        for (let i = index, n = 0; i >= store.first && n < 1200; i--, n++) {
            const f = store.get(i);
            if (f && (f.flags & 1)) {
                index = i;
                break;
            }
        }
        tl.seek(index);
        tl.t = index;
        tl.pause();
        app.lastIndex = index;
        const frame = store.get(index);
        const winner = this.winner();
        const players = app.players;
        // The hero is the best player of the winners, or of the match on a draw, the beaten one
        // the keeper of the losers, or the best of the other team on a draw.
        const ranked = this.facts.stats.slice().sort((a, b) => b.score - a.score);
        const on = (s) => s && (frame.playerFlags[s.k] & 1);
        const hero = (winner >= 0 ? ranked.find((s) => s.team === winner && on(s)) : ranked.find(on)) || null;
        let beaten = null;
        if (winner >= 0) {
            const k = header.players.findIndex((p, i) => p.goalkeeper && p.team !== winner && (frame.playerFlags[i] & 1));
            beaten = k >= 0 ? k : (ranked.find((s) => s.team !== winner && on(s)) || {}).k;
        } else if (hero) {
            beaten = (ranked.find((s) => s.team !== hero.team && on(s)) || {}).k;
        }
        // The hero celebrates on his own, clear of the huddle, so his close-up shows him.
        const prev = store.get(index - 1);
        const choreo = new Reactions({
            frame, header, winner, seed: index * 17 + 3,
            prev: prev && !isCut(prev, frame) ? prev : null,
            sps: tl.sps,
            card: TITLE_SECONDS,
            radius: (k) => figureSize(players, k).radius,
            hasClip: clipTest(players),
            hero: winner >= 0 && hero ? hero.k : -1,
        });
        // The full time card waits for the play to wind down, the cut to the show hides under it.
        this.react = {
            index, clock: 0, choreo, hero: hero ? hero.k : -1, beaten: beaten != null ? beaten : -1, stage: -1,
            titled: false, titleAt: choreo.titleAt, titleDone: false, frame: store.get(index), winner,
        };
        this.hint = h('div', { class: 'post-skip', 'data-prompt': '' }, 'Enter or a click to go on, Esc to finish');
        this.hint.addEventListener('click', () => this.advance());
        document.getElementById('watch').appendChild(this.hint);
        release(app);
    }

    stopReactions() {
        const app = this.app;
        const r = this.react;
        this.react = null;
        if (this.hint) {
            this.hint.remove();
            this.hint = null;
        }
        if (!r) {
            return;
        }
        if (r.titled && app.transition && app.transition.title) {
            app.transition.title.hide();
        }
        const players = app.players;
        if (players && typeof players.stopClip === 'function') {
            r.choreo.release((k) => players.stopClip(k));
        }
    }

    // Each frame, from the director's update, the clock and the cameras of the scene.
    update(dt) {
        const r = this.react;
        if (!r || !this.running) {
            return;
        }
        const app = this.app;
        r.clock += dt;
        const title = app.transition && app.transition.title;
        if (!r.titleDone && r.clock >= r.titleAt) {
            r.titleDone = true;
            const header = app.source.header;
            if (title && typeof title.show === 'function') {
                const w = r.winner;
                const foot = w >= 0 ? `${this.teamName(w)} ${plural(bareTeamName(header, w)) ? 'win' : 'wins'}` : 'A draw, the points are shared';
                title.show(header.full_match === false ? 'Episode complete' : 'Full time', header, app.styles, r.frame, foot);
                r.titled = true;
            } else {
                this.showCard(h('div', { class: 'post-score' }, h('div', { class: 'post-kicker' }, header.full_match === false ? 'Episode complete' : 'Full time'), this.scoreLine(this.facts)));
            }
        }
        if (r.titled && r.clock >= r.titleAt + TITLE_SECONDS && title) {
            title.hide();
            r.titled = false;
        }
        let stage = 0;
        for (let i = 0; i < REACTION_SHOTS.length; i++) {
            if (r.clock >= r.titleAt + REACTION_SHOTS[i].at) {
                stage = i;
            }
        }
        const shot = REACTION_SHOTS[stage];
        const director = app.director;
        const sh = director.shot;
        // Seeking or playing away from the whistle cancels the show where the viewer went.
        const tl = app.timeline;
        if (tl && !this.wiping && (tl.playing || Math.abs(tl.t - r.index) > 0.5)) {
            this.end({ home: false, restore: false });
            return;
        }
        if (stage !== r.stage) {
            sh.snap = !!shot.snap && r.stage >= 0;
            if (shot.kind !== 'live') {
                director.useScripted();
            }
            r.stage = stage;
        }
        if (shot.kind === 'live') {
            // The cameras of play follow the ball as it rolls to rest.
            return;
        }
        const pose = app.pose;
        // The head as drawn, which a clip may have taken down to the grass, else the neck node.
        const head = (k) => {
            const bone = app.players && typeof app.players.headBone === 'function' ? app.players.headBone(k) : null;
            if (bone) {
                const v = new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld);
                if (Number.isFinite(v.y) && v.lengthSq() > 0) {
                    return v.setY(Math.max(0.1, Math.min(2.6, v.y)));
                }
            }
            const o = k * PLAYER_FLOATS + 3 * 7;
            return new THREE.Vector3(pose.nodes[o], Math.max(0.4, Math.min(2.4, pose.nodes[o + 2])), -pose.nodes[o + 1]);
        };
        const close = (k, dist, side) => {
            const s = r.choreo.state(k, Math.max(0, r.clock - REACTION_LEAD));
            const p = head(k);
            // Waist up on any model, nearer a short one, so a big hat does not fill the frame.
            const tall = figureSize(app.players, k).height;
            const d = dist * Math.max(0.95, Math.min(1.35, tall / 1.85));
            const face = new THREE.Vector3(Math.sin(s.heading), 0, Math.cos(s.heading));
            const right = new THREE.Vector3(-face.z, 0, face.x);
            sh.pos.copy(p).addScaledVector(face, d).addScaledVector(right, side);
            // A standing figure is framed by its middle, hat and all, one on the grass by its head.
            const standing = p.y > 0.55 * tall;
            const aim = standing ? Math.max(p.y * 0.8, tall * 0.55) : Math.max(0.15, p.y * 0.8);
            sh.pos.y = Math.max(0.9, aim + 0.3);
            sh.target.copy(p).setY(aim);
            sh.fov = 30;
            sh.rate = 3;
        };
        if (shot.kind === 'hero' && r.hero >= 0 && pose) {
            close(r.hero, 4.6, 1.2);
        } else if (shot.kind === 'beaten' && r.beaten >= 0 && pose) {
            close(r.beaten, 5.6, -1.3);
        } else if (shot.kind === 'stands' && pose) {
            // From the far touchline back at the players walking up to applaud the fans.
            let cx = 0;
            let n = 0;
            for (let k = 0; k < pose.players; k++) {
                if (r.choreo.plans[k]) {
                    cx += pose.nodes[k * PLAYER_FLOATS];
                    n++;
                }
            }
            cx = n ? cx / n : 0;
            const drift = (r.clock - r.titleAt - shot.at) * 1.2;
            sh.pos.set(cx * 0.6 - 10 + drift, 4.2, -(HALF_H + 2.5));
            sh.target.set(cx * 0.6 + drift * 0.5, 1.4, -22);
            sh.fov = 44;
            sh.rate = 1.6;
        } else {
            sh.pos.set(0, 30, 60);
            sh.target.set(0, 0, 0);
            sh.fov = 44;
            sh.rate = 1.5;
        }
        confineShot(sh.pos);
        if (r.clock >= r.titleAt + REACTION_LENGTH && !this.wiping) {
            this.advance();
        }
    }

    // Dresses the held last frame while the players react.
    dress(pose, sample) {
        const r = this.react;
        if (r && sample.index === r.index) {
            const players = this.app.players;
            const clip = players && typeof players.playClip === 'function' ? (k, name, t, w) => players.playClip(k, name, t, w) : null;
            r.choreo.apply(pose, Math.max(0, r.clock - REACTION_LEAD), 1, clip);
        }
    }

    next() {
        const app = this.app;
        clearTimeout(this.timer);
        this.clearCard();
        this.stopReactions();
        if (app.director.running) {
            app.director.stop(false);
        }
        if (app.spotlight.showing) {
            app.spotlight.hide(false);
        }
        if (app.podium.showing) {
            app.podium.hide();
        }
        this.step++;
        const name = this.steps[this.step];
        const f = this.facts;
        // A goal graphic left from the reel or the play of the match stays with its part.
        app.hud.hideGoal();
        if (!name) {
            this.end();
            return;
        }
        if (name === 'reactions') {
            this.startReactions();
        } else if (name === 'facts') {
            app.cameras.setMode('broadcast');
            app.cameras.jump();
            const header = app.source.header;
            const scorers = (team) => f.scorers.filter((g) => g.team === team).map((g) =>
                h('div', {}, `${g.scorer >= 0 ? this.playerName(g.scorer) : 'Unknown'}${g.scorer >= 0 && header.players[g.scorer].team !== team ? ' (own goal)' : ''}  ${g.minute}′`));
            const table = comparisonTable(f, app.styles || [null, null]);
            this.showCard(h('div', { class: 'post-facts' },
                h('div', { class: 'post-kicker' }, 'Match facts'),
                this.scoreLine(f),
                h('div', { class: 'post-scorers' }, h('div', {}, scorers(0)), h('div', {}, scorers(1))),
                table));
            this.timer = setTimeout(() => this.advance(), 12000);
        } else if (name === 'reel') {
            const queue = f.scorers.map((g) => ({ goal: { index: g.index, team: g.team }, shot: 'auto' }));
            if (!queue.length) {
                this.next();
                return;
            }
            // The wipe into the reel already ran, and its own last wipe leads to the spotlight.
            release(app);
            app.director.play(queue, { restore: false, wipe: false, onDone: () => this.next() });
        } else if (name === 'spotlight') {
            const mvp = playerOfMatch(f.stats, f.score);
            if (!mvp) {
                this.next();
                return;
            }
            app.spotlight.show(mvp);
        } else if (name === 'podium') {
            app.podium.show(f);
        }
    }

    // The statistics screen on demand, from the last panel of the show, the card at the end of a
    // recording or the Match menu. It opens over what is on screen and closing it returns there.
    openStats() {
        const app = this.app;
        if (this.statsCard || !app.source || !app.timeline || !app.timeline.store.size) {
            return;
        }
        const header = app.source.header;
        const store = this.statsStore || summaryStore(app.timeline.store, app.source.kind === 'live');
        const f = this.statsStore && this.facts ? this.facts : matchFacts(store, header);
        const report = contribution(store, header, f);
        const back = document.activeElement;
        const hidden = [this.card, app.podium.overlay].filter((node) => node && !node.hidden);
        const timerWas = !!this.timer;
        clearTimeout(this.timer);
        for (const node of hidden) {
            node.hidden = true;
        }
        const close = () => {
            if (!this.statsCard) {
                return;
            }
            this.statsCard.remove();
            this.statsCard = null;
            for (const node of hidden) {
                node.hidden = false;
            }
            if (timerWas && this.running && this.steps[this.step] === 'facts') {
                this.timer = setTimeout(() => this.advance(), 6000);
            }
            if (back && back.isConnected && typeof back.focus === 'function') {
                back.focus();
            }
        };
        const names = [this.teamName(0), this.teamName(1)];
        const screen = statsScreen(report, names, (k) => playerInfo(app, k), store, close,
            { facts: f, scoreLine: this.scoreLine(f), closeLabel: 'Close' });
        this.statsCard = h('div', { class: 'post-card stats-card', tabindex: '-1', role: 'dialog', 'aria-labelledby': 'stats-title' }, screen);
        const colors = teamChartColors(app.styles);
        if (colors) {
            this.statsCard.style.setProperty('--chart-0', colors[0]);
            this.statsCard.style.setProperty('--chart-1', colors[1]);
        }
        this.statsCard.addEventListener('click', (e) => e.stopPropagation());
        this.statsCard.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                close();
            }
        });
        this.closeStats = close;
        document.getElementById('watch').appendChild(this.statsCard);
        release(app);
        // A frame later, after a menu that opened it has handed focus back.
        requestAnimationFrame(() => this.statsCard && this.statsCard.focus());
    }

    // A button that opens the statistics screen, for the last panel and the end card.
    statsButton(className = 'btn') {
        return h('button', { type: 'button', class: className, onclick: (e) => {
            e.stopPropagation();
            this.openStats();
        } }, 'Match statistics');
    }

    // Enter shows the next part, the next goal while the reel plays, or ends the show after
    // the podium.
    skip() {
        if (!this.running) {
            return;
        }
        const d = this.app.director;
        if (this.steps[this.step] === 'reel' && d.active && d.active.queue) {
            d.skip();
            return;
        }
        this.advance();
    }

    // Ends the show. After the show the page fades to Home with the match still selected, a
    // cancel by seeking stays where the viewer went, and restore false keeps the timeline there.
    end({ home = true, restore = true } = {}) {
        const app = this.app;
        const wasRunning = this.running;
        clearTimeout(this.timer);
        if (this.closeStats) {
            this.closeStats();
        }
        this.clearCard();
        this.stopReactions();
        this.step = -1;
        this.statsStore = null;
        if (this.wiping) {
            this.wiping = false;
            app.director.cancelSting();
        }
        document.getElementById('app').classList.remove('post-show');
        if (app.director.running) {
            app.director.stop(false);
        }
        if (app.spotlight.showing) {
            app.spotlight.hide(false);
        }
        if (app.podium.showing) {
            app.podium.hide();
        }
        app.hud.unfreeze();
        const s = this.saved;
        const tl = app.timeline;
        if (s && tl) {
            app.cameras.focusIndex = s.focus;
            app.cameras.setMode(s.camera === 'scripted' ? (app.settings.camera || 'broadcast') : s.camera);
            if (restore) {
                if (s.following) {
                    tl.goLive();
                } else {
                    tl.seek(s.t);
                    tl.pause();
                }
                app.lastIndex = tl.index;
                app.afterSeek();
            }
            app.updatePlayUi();
        }
        if (home && wasRunning) {
            this.fadeHome(app.source);
        }
    }

    // A soft fade to the Home page, once the show is over, unless the viewer has opened another
    // match or another page in the meantime, or the page has no Home.
    fadeHome(source) {
        const app = this.app;
        if (typeof app.showPage !== 'function' || !document.getElementById('page-home')) {
            return;
        }
        let veil = document.getElementById('cuju-home-veil');
        if (!veil) {
            veil = document.createElement('div');
            veil.id = 'cuju-home-veil';
            veil.style.cssText = `position:fixed;inset:0;z-index:50;pointer-events:none;background:#0b0c0e;opacity:0;transition:opacity ${HOME_FADE_MS}ms ease`;
            document.body.appendChild(veil);
        }
        clearTimeout(this.homeTimer);
        const still = () => app.source === source && app.page === 'watch' && !this.running;
        // A frame later, so a show ended by opening another match does not fade at all.
        requestAnimationFrame(() => {
            if (!still()) {
                return;
            }
            veil.style.opacity = '1';
            this.homeTimer = setTimeout(() => {
                if (still()) {
                    app.showPage('home');
                }
                veil.style.opacity = '0';
            }, HOME_FADE_MS + 40);
        });
    }
}

// Full time, the winners on a podium in the center circle, dancing, confetti and the MVP cards.
export class Podium {
    constructor(app) {
        this.app = app;
        this.group = null;
        this.overlay = null;
        this.clock = 0;
    }

    get showing() {
        return !!this.overlay;
    }

    show() {
        const app = this.app;
        if (this.showing || !app.source) {
            return;
        }
        const header = app.source.header;
        const { stats } = matchStats(app.timeline.store, header);
        const last = app.timeline.store.get(app.timeline.last);
        app.hud.freeze(last);
        const winner = last.scoreLeft === last.scoreRight ? -1 : (last.scoreLeft > last.scoreRight ? 0 : 1);
        const best = (team) => stats.filter((s) => s.team === team).sort((a, b) => b.score - a.score);
        app.sound.fanfare();
        this.previousCamera = app.cameras.mode === 'scripted' ? (app.settings.camera || 'broadcast') : app.cameras.mode;
        if (winner >= 0) {
            this.buildScene(header, best(winner), winner);
        } else {
            this.buildDraw(header, [best(0), best(1)]);
        }
        injectShowStyle();
        const names = teamNames(header);
        const cards = winner >= 0 ? [this.card(best(winner)[0], header)] : [this.card(best(0)[0], header, true), this.card(best(1)[0], header, true)];
        const all = stats.slice().sort((a, b) => b.score - a.score);
        this.mvp = winner >= 0 ? best(winner)[0] : all[0];
        const title = winner >= 0 ? `${names[winner]} ${plural(bareTeamName(header, winner)) ? 'win' : 'wins'}` : 'A draw';
        this.overlay = h('div', { id: 'victory', class: winner >= 0 ? '' : 'draw' },
            h('div', { class: 'victory-head' },
                h('div', { class: 'victory-title' }, title),
                h('div', { class: 'victory-score' }, `${names[0]}  ${last.scoreLeft} – ${last.scoreRight}  ${names[1]}`)),
            h('div', { class: 'victory-cards' }, cards),
            h('div', { class: 'victory-actions' }, app.postMatch.statsButton('btn ghost'),
                h('button', { class: 'btn', onclick: () => app.postMatch.end() }, 'Close')));
        document.getElementById('watch').appendChild(this.overlay);
    }

    // The card of a team's best player, compact when two share the panel after a draw.
    card(s, header, compact = false) {
        if (!s) {
            return h('div', { class: 'mvp-card' }, 'No player data.');
        }
        const app = this.app;
        const info = playerInfo(app, s.k);
        const style = app.styles ? app.styles[s.team] : null;
        const row = (label, value) => h('div', { class: 'stat' }, h('span', {}, label), h('b', {}, String(value)));
        void header;
        return h('div', { class: 'mvp-card', style: `border-top-color:${style ? css(style.kit.primary) : 'var(--accent)'}` },
            h('div', { class: 'mvp-label' }, 'Most valuable player'),
            h('div', { class: 'mvp-name' }, h('span', { class: 'mvp-number' }, String(info.number)), info.name),
            h('div', { class: 'mvp-team' }, info.badge ? `${info.badge}  ` : '', info.teamName),
            row('Goals', s.goals), row('Assists', s.assists), compact ? null : row('Touches', s.touches),
            row('Passes', s.passes ? `${s.passesCompleted} of ${s.passes}` : 0), row('Tackles', s.tackles),
            s.goalkeeper || s.saves ? row('Saves', s.catches || s.parries ? `${s.saves}, ${s.catches} caught` : s.saves) : null,
            s.ownGoals ? row('Own goals', s.ownGoals) : null);
    }

    buildScene(header, ranked, team) {
        const app = this.app;
        const g = new THREE.Group();
        app.view.scene.add(g);
        this.group = g;
        // Steps and places sized for the team's biggest model, so big figures stand clear.
        const size = ranked.reduce((m, st) => {
            const f = figureSize(app.players, st.k);
            return { radius: Math.max(m.radius, f.radius), height: Math.max(m.height, f.height) };
        }, { radius: 0.35, height: 1.85 });
        const R = size.radius;
        const lift = Math.max(1, Math.min(1.7, size.height / 1.85));
        const stepW = Math.max(1.5, 2 * R + 0.5);
        const apart = stepW + 0.15;
        const stone = new THREE.MeshStandardMaterial({ color: 0xe8eaed, roughness: 0.6 });
        const steps = [
            { x: 0, h: 0.9 * lift, label: 1 }, { x: -apart, h: 0.6 * lift, label: 2 }, { x: apart, h: 0.4 * lift, label: 3 },
        ];
        for (const st of steps) {
            const box = new THREE.Mesh(new THREE.BoxGeometry(stepW, st.h, stepW), stone);
            box.position.set(st.x, st.h / 2, 0);
            box.castShadow = true;
            box.receiveShadow = true;
            g.add(box);
        }
        // Everyone of the team on a pose buffer, the best three dancing on the steps, the rest
        // in staggered rows behind them a body apart.
        const players = ranked.map((s, i) => ({ ...header.players[s.k], slot: i }));
        const pose = { players: players.length, nodes: new Float32Array(players.length * PLAYER_FLOATS) };
        const style = app.styles ? app.styles[team] : null;
        const random = seeded(team * 101 + players.length);
        const rest = Math.max(0, players.length - 3);
        const perRow = Math.max(4, Math.ceil(rest / 2));
        // Arms raised in a cheer reach past the resting radius, so neighbours stand a little wider.
        const spacing = 2 * R + 0.55;
        let width = 3 * stepW + 0.3;
        this.slots = players.map((p, i) => {
            if (i < 3) {
                return { x: steps[i].x, y: 0, z: steps[i].h, dance: showDance(style, p.index, random), phase: i * 0.37 };
            }
            const r = i - 3;
            const row = Math.floor(r / perRow);
            const inRow = Math.min(perRow, rest - row * perRow);
            const col = r - row * perRow;
            const x = (col - (inRow - 1) / 2) * spacing + (row % 2 ? spacing / 2 : 0);
            width = Math.max(width, inRow * spacing + spacing);
            const clap = roleName('clap');
            return {
                x, y: stepW / 2 + R + 0.6 + row * (2 * R + 0.5), z: 0,
                dance: i % 2 && clap ? clap : showDance(style, p.index, random), phase: random() * 2,
            };
        });
        this.frameShot(width, size.height, steps[0].h);
        this.pose = pose;
        this.clock = 0;
        const podiumHeader = { players: players.map((p) => ({ team: p.team, index: p.index, goalkeeper: p.goalkeeper, controlled: false })), teams: header.teams };
        const rigs = app.teamRigs || [null, null];
        this.players = new Players(g, podiumHeader, app.styles, app.assets, rigs);
        shareColors(this.players, app.players);
        this.frame = { playerFlags: new Uint8Array(players.length).fill(1), ownedPlayer: -1 };
        // KayKit figures left to chance celebrate with KayKit's clips, the best three jumping and
        // cheering on the steps, the rows behind cheering, waving to the fans and taunting.
        const has = clipTest(this.players);
        const stepClips = ['clip:Jump_Full_Long', 'clip:Cheering', 'clip:Cheering'];
        const rowClips = ['clip:Cheering', 'clip:Waving', 'clip:Jump_Full_Short', 'clip:Skeletons_Taunt_Longer', 'clip:Waving', 'clip:Push_Ups'];
        this.slots.forEach((slot, i) => {
            const p = players[i];
            if (!choseCelebration(style, p.index)) {
                const want = i < 3 ? stepClips[i] : rowClips[(i - 3) % rowClips.length];
                const pick = has(i, want) ? want : kaykitCelebration(has, i, random);
                if (pick) {
                    slot.dance = pick;
                }
            }
            slot.dance = usableDance(this.players, i, slot.dance, random);
        });
        this.dance(0);
        this.confetti(g, style);
        if (app.players) {
            app.players.group.visible = false;
        }
        app.ball.mesh.visible = false;
        app.ball.blob.visible = false;
        app.ball.trail.visible = false;
    }

    // A draw, the best four of each team on a low stage, a half for each, their best player in
    // front applauding the other side, the rest behind cheering and waving to the fans.
    buildDraw(header, lists) {
        const app = this.app;
        const g = new THREE.Group();
        app.view.scene.add(g);
        this.group = g;
        const picks = lists.map((l) => l.slice(0, 4));
        const everyone = [...picks[0], ...picks[1]];
        if (!everyone.length) {
            app.cameras.setMode('tactical');
            return;
        }
        const size = everyone.reduce((m, st) => {
            const f = figureSize(app.players, st.k);
            return { radius: Math.max(m.radius, f.radius), height: Math.max(m.height, f.height) };
        }, { radius: 0.35, height: 1.85 });
        const R = size.radius;
        const spacing = 2 * R + 0.55;
        const stageH = 0.3;
        const rows = Math.max(1, Math.max(picks[0].length, picks[1].length) - 1);
        const half = R + 0.45 + rows * spacing * 0.9 + spacing * 0.45 + R + 0.4;
        const depth = 2 * spacing + 1.2;
        const styles = app.styles || [null, null];
        picks.forEach((list, team) => {
            const tint = styles[team] ? threeColorOf(styles[team].kit.primary).lerp(new THREE.Color(0xe8eaed), 0.82) : new THREE.Color(0xe8eaed);
            const box = new THREE.Mesh(new THREE.BoxGeometry(half, stageH, depth), new THREE.MeshStandardMaterial({ color: tint, roughness: 0.6 }));
            box.position.set((team === 0 ? -1 : 1) * (half / 2 + 0.05), stageH / 2, depth / 2 - spacing - 0.6);
            box.castShadow = true;
            box.receiveShadow = true;
            g.add(box);
        });
        const players = everyone.map((s) => header.players[s.k]);
        const pose = { players: players.length, nodes: new Float32Array(players.length * PLAYER_FLOATS) };
        const random = seeded(players.length * 7 + 5);
        this.slots = [];
        picks.forEach((list, team) => {
            const side = team === 0 ? -1 : 1;
            const style = styles[team];
            list.forEach((st, i) => {
                const p = header.players[st.k];
                // The best player stands in front near the middle and turns a little towards the
                // other team, the others stand in a row behind him.
                const front = i === 0;
                const x = front ? side * (R + 0.45) : side * (R + 0.45 + (i - 1) * spacing * 0.9 + spacing * 0.45);
                const y = front ? 0 : spacing;
                const heading = front ? -side * 0.45 : 0;
                const dance = front || i % 2 === 0 ? (roleName('clap') || showDance(style, p.index, random)) : showDance(style, p.index, random);
                this.slots.push({ x, y, z: stageH, heading, dance, phase: random() * 2, team, index: p.index, front });
            });
        });
        // Wider than the stage, so its right half stays clear of the panel of cards.
        this.frameShot((2 * half + 1) * 1.45, size.height, stageH);
        this.pose = pose;
        this.clock = 0;
        const podiumHeader = { players: players.map((p) => ({ team: p.team, index: p.index, goalkeeper: p.goalkeeper, controlled: false })), teams: header.teams };
        this.players = new Players(g, podiumHeader, app.styles, app.assets, app.teamRigs || [null, null]);
        shareColors(this.players, app.players);
        this.frame = { playerFlags: new Uint8Array(players.length).fill(1), ownedPlayer: -1 };
        const has = clipTest(this.players);
        this.slots.forEach((slot, i) => {
            if (!choseCelebration(styles[slot.team], slot.index) || slot.front) {
                const want = slot.front ? 'clip:Waving' : ['clip:Cheering', 'clip:Waving', 'clip:Idle_B'][i % 3];
                if (has(i, want)) {
                    slot.dance = want;
                }
            }
            slot.dance = usableDance(this.players, i, slot.dance, random);
        });
        this.dance(0);
        this.confetti(g, null, [styles[0] ? styles[0].kit.primary : [255, 111, 0], styles[1] ? styles[1].kit.primary : [255, 160, 0], [255, 255, 255]]);
        if (app.players) {
            app.players.group.visible = false;
        }
        app.ball.mesh.visible = false;
        app.ball.blob.visible = false;
        app.ball.trail.visible = false;
    }

    // A shot that holds the whole podium and the rows behind it, whatever the size of the models.
    frameShot(width, height, top) {
        const app = this.app;
        const director = app.director;
        const sh = director.shot;
        const aspect = Math.max(0.5, app.view.aspect || 16 / 9);
        const fov = 40;
        const half = Math.tan(THREE.MathUtils.degToRad(fov / 2));
        const tall = top + height * 1.15;
        const dist = Math.max(10, (width / 2 + 1) / (half * aspect), (tall / 2 + 1) / half);
        // The card of the MVP sits on the right, so the podium moves left of it.
        const shift = dist * half * aspect * 0.28;
        sh.target.set(shift, tall * 0.5, 0);
        sh.pos.set(shift, tall * 0.5 + dist * 0.12, dist);
        sh.fov = fov;
        sh.rate = 2;
        sh.snap = true;
        director.useScripted();
        app.cameras.jump();
        this.spread = width;
    }

    // Poses everyone on the podium at time t and draws them.
    dance(t) {
        if (!this.players || !this.slots) {
            return;
        }
        const nodes = this.pose.nodes;
        this.slots.forEach((s, i) => {
            const o = i * PLAYER_FLOATS;
            posePlayer(s.dance, t + s.phase, nodes, o, s.x, s.y, s.heading || 0, 'show');
            for (let n = 0; n < 14; n++) {
                nodes[o + n * 7 + 2] += s.z;
            }
            if (s.dance && s.dance.startsWith('clip:') && typeof this.players.playClip === 'function') {
                this.players.playClip(i, s.dance, t + s.phase);
            }
        });
        this.players.update(this.pose, this.frame);
    }

    confetti(group, style, mix = null) {
        const count = 600;
        const geo = new THREE.PlaneGeometry(0.09, 0.05);
        const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
        const mesh = new THREE.InstancedMesh(geo, mat, count);
        const colors = mix || (style ? [style.kit.primary, style.kit.secondary, [255, 255, 255]] : [[255, 111, 0], [255, 160, 0], [232, 234, 237]]);
        this.bits = [];
        for (let i = 0; i < count; i++) {
            const c = colors[i % colors.length];
            mesh.setColorAt(i, new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace));
            this.bits.push({
                p: new THREE.Vector3((Math.random() - 0.5) * Math.max(12, (this.spread || 0) + 4), 4 + Math.random() * 10, (Math.random() - 0.5) * 8),
                v: new THREE.Vector3((Math.random() - 0.5) * 0.6, -0.8 - Math.random() * 0.8, (Math.random() - 0.5) * 0.6),
                r: new THREE.Euler(Math.random() * 6, Math.random() * 6, Math.random() * 6),
                w: new THREE.Vector3(Math.random() * 6, Math.random() * 6, Math.random() * 6),
            });
        }
        mesh.frustumCulled = false;
        group.add(mesh);
        this.confettiMesh = mesh;
    }

    update(dt) {
        if (!this.confettiMesh) {
            return;
        }
        this.clock += dt;
        this.dance(this.clock);
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const one = new THREE.Vector3(1, 1, 1);
        this.bits.forEach((b, i) => {
            b.p.addScaledVector(b.v, dt);
            b.p.x += Math.sin(performance.now() / 700 + i) * dt * 0.3;
            if (b.p.y < 0.02) {
                b.p.y = 10 + Math.random() * 4;
            }
            b.r.x += b.w.x * dt;
            b.r.y += b.w.y * dt;
            b.r.z += b.w.z * dt;
            q.setFromEuler(b.r);
            m.compose(b.p, q, one);
            this.confettiMesh.setMatrixAt(i, m);
        });
        this.confettiMesh.instanceMatrix.needsUpdate = true;
    }

    hide() {
        const app = this.app;
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
        if (this.group) {
            if (this.players) {
                this.players.dispose();
                this.players = null;
            }
            this.group.removeFromParent();
            this.group.traverse((obj) => {
                if (obj.geometry) {
                    obj.geometry.dispose();
                }
                if (obj.material) {
                    obj.material.dispose();
                }
            });
            this.group = null;
            this.confettiMesh = null;
            this.slots = null;
        }
        if (app.players) {
            app.players.group.visible = true;
        }
        app.ball.mesh.visible = true;
        app.ball.blob.visible = true;
        app.ball.trail.visible = app.ball.trailOn;
        if (this.previousCamera) {
            app.cameras.setMode(this.previousCamera);
            this.previousCamera = null;
        }
    }
}

// The MVP spotlight, the play of the match loops in the main view, slowed near its key moment,
// while the MVP dances in a cone of light on a dark stage beside it with a name plate.
export class Spotlight {
    constructor(app) {
        this.app = app;
        this.overlay = null;
    }

    get showing() {
        return !!this.overlay;
    }

    // The clip, the latest goal, else the latest assist, save or tackle, else the longest run.
    clip(s) {
        const last = (list) => (list.length ? list[list.length - 1] : null);
        let key = last(s.goalFrames);
        let what = 'Goal';
        if (key == null) {
            key = last(s.assistFrames);
            what = 'Assist';
        }
        if (key == null) {
            key = last(s.saveFrames);
            what = 'Save';
        }
        if (key == null) {
            key = last(s.tackleFrames);
            what = 'Tackle';
        }
        if (key != null) {
            // A goal's clip ends on the frame before it, the kick-off after it stays unseen.
            const end = what === 'Goal' || what === 'Assist' ? key - 1 : key + 12;
            return { begin: key - 60, end, key, what, camera: what === 'Save' ? 'goal' : 'lowside' };
        }
        if (s.run) {
            const end = Math.min(s.run.end + 10, s.run.start + 120);
            return { begin: s.run.start - 10, end, key: Math.min(s.run.end, end - 4), what: 'On the ball', camera: 'orbit' };
        }
        return null;
    }

    show(s) {
        const app = this.app;
        const tl = app.timeline;
        if (!tl || this.showing) {
            return;
        }
        const clip = this.clip(s);
        this.state = {
            t: tl.t, playing: tl.playing, speed: tl.speed, direction: tl.direction, following: tl.following,
            camera: app.cameras.mode, focus: app.cameras.focusIndex,
        };
        app.hud.freeze(tl.store.get(tl.last));
        if (clip) {
            clip.begin = Math.max(tl.first, clip.begin);
            clip.end = Math.min(tl.last, clip.end);
            this.loop = clip;
            app.cameras.focusIndex = s.k;
            app.cameras.setMode(clip.camera);
            tl.following = false;
            tl.seek(clip.begin);
            tl.direction = 1;
            tl.playing = true;
            app.lastIndex = tl.index;
            app.afterSeek();
        }
        this.mvp = s;
        const info = playerInfo(app, s.k);
        const style = app.styles ? app.styles[s.team] : null;
        this.danceName = showDance(style, s.index);
        const stage = h('div', { class: 'spot-stage' });
        const count = (value, one, many) => `${value} ${value === 1 ? one : many}`;
        const stats = [
            count(s.goals, 'goal', 'goals'), count(s.assists, 'assist', 'assists'), count(s.passes, 'pass', 'passes'),
            count(s.tackles, 'tackle', 'tackles'), s.goalkeeper || s.saves ? count(s.saves, 'save', 'saves') : null,
        ].filter(Boolean).join(', ');
        this.overlay = h('div', { id: 'spotlight' },
            h('div', { class: 'mvp-flash', 'aria-hidden': 'true' }),
            h('div', { class: 'spot-side' },
                h('div', { class: 'mvp-rays', 'aria-hidden': 'true' }),
                mvpMedal(style ? [style.kit.primary, style.kit.secondary] : [[194, 65, 12], [248, 249, 250]]), stage,
                h('div', { class: 'nameplate', style: style ? `border-top-color:${css(style.kit.primary)}` : '' },
                    h('div', { class: 'mvp-label' }, 'Most valuable player'),
                    h('div', { class: 'mvp-name' }, h('span', { class: 'mvp-number' }, String(info.number)), info.name),
                    h('div', { class: 'mvp-team' }, info.badge ? `${info.badge}  ` : '', info.teamName),
                    h('div', { class: 'small muted' }, stats),
                    h('button', { class: 'btn primary', onclick: () => app.postMatch.skip() }, 'Next'))),
            h('div', { class: 'spot-frame' }, h('div', { class: 'spot-caption' }, clip ? `Player of the match, ${clip.what.toLowerCase()}` : 'Player of the match')));
        document.getElementById('watch').appendChild(this.overlay);
        document.getElementById('app').classList.add('spot-show');
        this.buildStage(stage, s);
        app.sound.fanfare();
        this.overlay.querySelector('button').focus();
        this.overlay.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                e.stopPropagation();
                app.postMatch.skip();
            }
        });
        this.overlay.addEventListener('click', (e) => {
            if (!e.target.closest('button')) app.postMatch.skip();
        });
    }

    buildStage(stage, s) {
        const app = this.app;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        // The neutral curve keeps the hue of a bright kit where filmic curves wash it pale.
        renderer.toneMapping = THREE.NeutralToneMapping !== undefined ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.05;
        renderer.shadowMap.enabled = true;
        renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
        stage.appendChild(renderer.domElement);
        // A dark studio behind the figure, lit from above in the middle.
        stage.style.background = 'radial-gradient(ellipse 80% 70% at 50% 38%, #2b2e38 0%, #15161b 55%, #08090b 100%)';
        const scene = new THREE.Scene();
        // The figure's own size frames the shot, big models stand back further.
        const size = figureSize(app.players, s.k);
        const tall = size.height;
        const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 80);
        const dist = 2.9 * tall + 0.3 + size.radius;
        camera.position.set(0, tall * 0.62, dist);
        camera.lookAt(0, tall * 0.5, 0);
        scene.add(new THREE.HemisphereLight(0xc4d2e8, 0x24201c, 0.55));
        // The key, a warm spot from above and in front, with the cone of light it throws, about
        // as bright on the figure as daylight on the pitch.
        const spot = new THREE.SpotLight(0xffe7c8, 220, 0, 0.4, 0.55, 2);
        spot.position.set(0.8, 7.5, 3.2);
        spot.target.position.set(0, tall * 0.45, 0);
        spot.castShadow = true;
        spot.shadow.mapSize.set(1024, 1024);
        spot.shadow.bias = -0.0005;
        scene.add(spot, spot.target);
        // A soft fill from the front left so faces under hats and hoods read, and a cool rim
        // from behind that lifts the figure off the dark.
        const fill = new THREE.DirectionalLight(0xfff6ec, 0.9);
        fill.position.set(-3, tall * 0.9, 6);
        scene.add(fill);
        const rim = new THREE.DirectionalLight(0xbcd6ff, 2.4);
        rim.position.set(1.8, tall * 1.7, -5);
        scene.add(rim);
        const rim2 = new THREE.DirectionalLight(0xffd8b0, 1.2);
        rim2.position.set(-2.2, tall * 1.2, -4);
        scene.add(rim2);
        const floor = new THREE.Mesh(new THREE.CircleGeometry(9, 64), new THREE.MeshStandardMaterial({ color: 0x101115, roughness: 0.92 }));
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        scene.add(floor);
        // A warm pool on the floor and a visible cone, both additive so they glow.
        const pool = new THREE.Mesh(new THREE.CircleGeometry(1.6 + size.radius, 48), new THREE.MeshBasicMaterial({
            color: 0xffc77a, transparent: true, opacity: 0.16, depthWrite: false, blending: THREE.AdditiveBlending,
        }));
        pool.rotation.x = -Math.PI / 2;
        pool.position.y = 0.01;
        scene.add(pool);
        const coneGeo = new THREE.ConeGeometry(2.0 + size.radius, 8, 48, 1, true);
        coneGeo.translate(0, -4, 0);
        const cone = new THREE.Mesh(coneGeo, beamMaterial(0xffd29a, 0.3));
        cone.position.set(0, 8, 0);
        scene.add(cone);
        // The floor fades into the dark of the studio rather than ending at an edge.
        scene.fog = new THREE.Fog(0x0a0b0e, dist + 1, dist + 9);
        const p = app.source.header.players[s.k];
        const header = { players: [{ team: p.team, index: p.index, goalkeeper: p.goalkeeper, controlled: false }], teams: app.source.header.teams };
        const players = new Players(scene, header, app.styles, app.assets, app.teamRigs || [null, null]);
        shareColors(players, app.players);
        // A KayKit player of the match left to chance cheers with KayKit's own clip.
        const style = app.styles ? app.styles[p.team] : null;
        if (!choseCelebration(style, p.index)) {
            const has = clipTest(players);
            this.danceName = has(0, 'clip:Cheering') ? 'clip:Cheering' : (kaykitCelebration(has, 0, Math.random) || this.danceName);
        }
        this.danceName = usableDance(players, 0, this.danceName);
        this.stageState = {
            renderer, scene, camera, players, stage, dist, clock: performance.now(),
            pose: { players: 1, nodes: new Float32Array(PLAYER_FLOATS) },
            frame: { playerFlags: new Uint8Array([1]), ownedPlayer: -1 },
        };
    }

    update() {
        const app = this.app;
        if (!this.showing) {
            return;
        }
        const tl = app.timeline;
        const c = this.loop;
        if (c && tl) {
            tl.following = false;
            tl.direction = 1;
            tl.speed = tl.t > c.key - 8 && tl.t < c.key + 4 ? 0.35 : 1;
            if (tl.t >= c.end || !tl.playing) {
                tl.seek(c.begin);
                tl.playing = true;
                app.lastIndex = tl.index;
                app.afterSeek();
            }
        }
        const st = this.stageState;
        if (!st) {
            return;
        }
        const w = st.stage.clientWidth;
        const hgt = st.stage.clientHeight;
        const ratio = st.renderer.getPixelRatio();
        if (w && hgt && (st.renderer.domElement.width !== Math.floor(w * ratio) || st.renderer.domElement.height !== Math.floor(hgt * ratio))) {
            st.renderer.setSize(w, hgt, false);
            st.camera.aspect = w / hgt;
            st.camera.updateProjectionMatrix();
        }
        // The loop of the play of the match shows the moves, not the goal graphic of the play.
        const goal = app.hud.el && app.hud.el.goal;
        if (goal && goal.classList.contains('show')) {
            app.hud.hideGoal();
        }
        const t = (performance.now() - st.clock) / 1000;
        const push = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : Math.pow(Math.max(0, 1 - t / 1.2), 3);
        st.camera.position.z = st.dist * (1 + push * 0.65);
        posePlayer(this.danceName, t, st.pose.nodes, 0, 0, 0, Math.sin(t * 0.4) * 0.2, 'show');
        if (this.danceName && this.danceName.startsWith('clip:') && typeof st.players.playClip === 'function') {
            st.players.playClip(0, this.danceName, t);
        }
        st.players.update(st.pose, st.frame);
        st.renderer.render(st.scene, st.camera);
    }

    hide(restore = true) {
        const app = this.app;
        if (!this.overlay) {
            return;
        }
        this.overlay.remove();
        this.overlay = null;
        document.getElementById('app').classList.remove('spot-show');
        if (this.stageState) {
            const st = this.stageState;
            st.players.dispose();
            st.scene.traverse((obj) => {
                if (obj.geometry) {
                    obj.geometry.dispose();
                }
                if (obj.material) {
                    obj.material.dispose();
                }
                if (obj.isLight && obj.shadow && obj.shadow.map) {
                    obj.shadow.map.dispose();
                }
            });
            st.renderer.dispose();
            st.renderer.forceContextLoss();
            this.stageState = null;
        }
        const s = this.state;
        const tl = app.timeline;
        if (s && tl && restore) {
            tl.speed = s.speed;
            tl.direction = s.direction;
            app.cameras.focusIndex = s.focus;
            app.cameras.setMode(s.camera);
            if (s.following) {
                tl.goLive();
            } else {
                tl.seek(s.t);
                tl.playing = false;
            }
            app.lastIndex = tl.index;
            app.afterSeek();
            app.updatePlayUi();
        }
        if (tl && !restore) {
            tl.speed = s ? s.speed : 1;
        }
        this.loop = null;
    }
}
