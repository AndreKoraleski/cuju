// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The referee's decisions as on television. When a foul, a card or an offside stops play, the
// view rewinds a few seconds and replays the moment in slow motion from a clear angle, with rings
// under the offender and the victim, an arrow of the challenge, a frozen flash on the contact
// and a banner of the decision, or the offside line at the pass. A card brings the referee to the
// offender with the card raised and a lower third of the card, then the restart follows.

import * as THREE from 'three';
import { NODES_FLOAT, PLAYER_FLOATS, isCut } from './format.js';
import { teamNames } from './hud.js';
import { HALF_H, HALF_W } from './pitch.js';
import {
    WIPE, blendPose, clamp, headingOf, headingTo, namedPose, rgb, showPointer, smooth,
} from './transition.js';

const STOP = 0.8;
const FREEZE = 1.9;
const CARD = 3.4;
// Frames of the replay before the contact, and those of them shown slowly.
const LEAD = 32;
const SLOW_FRAMES = 9;
const SLOW = 0.35;

const COLORS = { foul: '#ff8a1f', yellow: '#ffd400', red: '#e0242b', offside: '#35a7ff' };
const TITLES = { foul: 'Foul', yellow: 'Yellow card', red: 'Red card', offside: 'Offside' };

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

function flatMaterial(color, opacity) {
    return new THREE.MeshBasicMaterial({
        color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide,
    });
}

// The marks drawn on the pitch, rings, an arrow and the offside lines, all flat on the grass.
class Telestration {
    constructor(scene) {
        this.group = new THREE.Group();
        this.group.visible = false;
        this.group.renderOrder = 3;
        scene.add(this.group);
        const ring = new THREE.RingGeometry(0.72, 0.92, 48);
        ring.rotateX(-Math.PI / 2);
        const disc = new THREE.CircleGeometry(0.72, 40);
        disc.rotateX(-Math.PI / 2);
        this.rings = [0, 1].map(() => {
            const g = new THREE.Group();
            const r = new THREE.Mesh(ring, flatMaterial(0xffffff, 0.95));
            const d = new THREE.Mesh(disc, flatMaterial(0xffffff, 0.22));
            r.renderOrder = 3;
            d.renderOrder = 3;
            g.add(r, d);
            this.group.add(g);
            return g;
        });
        const plane = new THREE.PlaneGeometry(1, 1);
        plane.rotateX(-Math.PI / 2);
        plane.translate(0.5, 0, 0);
        this.shaft = new THREE.Mesh(plane, flatMaterial(0xffffff, 0.9));
        const head = new THREE.BufferGeometry();
        head.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, -0.55, 0.9, 0, 0, 0, 0, 0.55], 3));
        this.head = new THREE.Mesh(head, flatMaterial(0xffffff, 0.95));
        this.arrow = new THREE.Group();
        this.arrow.add(this.shaft, this.head);
        this.shaft.renderOrder = 3;
        this.head.renderOrder = 3;
        this.group.add(this.arrow);
        const lineGeo = new THREE.PlaneGeometry(1, 1);
        lineGeo.rotateX(-Math.PI / 2);
        this.lines = [0, 1].map(() => {
            const m = new THREE.Mesh(lineGeo, flatMaterial(0xffffff, 0.92));
            m.renderOrder = 3;
            this.group.add(m);
            return m;
        });
        this.shade = new THREE.Mesh(lineGeo, flatMaterial(0x35a7ff, 0.16));
        this.shade.renderOrder = 2;
        this.group.add(this.shade);
    }

    clear() {
        this.group.visible = false;
        for (const r of this.rings) {
            r.visible = false;
        }
        this.arrow.visible = false;
        this.lines[0].visible = false;
        this.lines[1].visible = false;
        this.shade.visible = false;
    }

    ring(i, x, y, color, pulse) {
        const g = this.rings[i];
        g.visible = true;
        this.group.visible = true;
        g.position.set(x, 0.035 + i * 0.002, -y);
        g.scale.setScalar(1 + 0.08 * Math.sin(pulse * 6));
        g.children[0].material.color.set(color);
        g.children[1].material.color.set(color);
    }

    // An arrow on the grass from a to b, drawn to fraction f of its length.
    arrowTo(ax, ay, bx, by, f, color) {
        const len = Math.hypot(bx - ax, by - ay) * f;
        if (len < 0.3) {
            this.arrow.visible = false;
            return;
        }
        this.group.visible = true;
        this.arrow.visible = true;
        const angle = Math.atan2(by - ay, bx - ax);
        this.arrow.position.set(ax, 0.04, -ay);
        this.arrow.rotation.set(0, angle, 0);
        const shaft = Math.max(0.1, len - 0.85);
        this.shaft.scale.set(shaft, 1, 0.32);
        this.head.position.set(shaft - 0.05, 0, 0);
        this.shaft.material.color.set(color);
        this.head.material.color.set(color);
    }

    // The offside lines across the pitch, the defence at lineX and the attacker at attackX.
    offside(lineX, attackX, attack, f) {
        this.group.visible = true;
        const len = HALF_H * 2 + 2;
        const [defence, attacker] = this.lines;
        defence.visible = true;
        defence.position.set(lineX, 0.03, 0);
        defence.scale.set(0.16, 1, len * f);
        defence.material.color.set(COLORS.offside);
        attacker.visible = true;
        attacker.position.set(attackX, 0.031, 0);
        attacker.scale.set(0.12, 1, len * f);
        attacker.material.color.set(COLORS.red);
        // The ground beyond the defence, towards the goal the attacker runs at.
        const goal = attack * HALF_W;
        this.shade.visible = true;
        this.shade.position.set((lineX + goal) / 2, 0.025, 0);
        this.shade.scale.set(Math.abs(goal - lineX), 1, len * f);
    }

    dispose() {
        this.group.removeFromParent();
    }
}

export class Decisions {
    constructor(app, transitions) {
        this.app = app;
        this.tr = transitions;
        transitions.decisions = this;
        this.marks = new Telestration(app.view.scene);
        const watch = document.getElementById('watch');
        this.flash = el('div', 'decision-flash', watch);
        this.banner = el('div', 'decision-banner', watch);
        this.bannerLabel = el('div', 'db-label', this.banner);
        const body = el('div', 'db-body', this.banner);
        this.bannerNumber = el('b', 'db-number', body);
        const text = el('div', 'db-text', body);
        this.bannerName = el('div', 'db-name', text);
        this.bannerTeam = el('div', 'db-team', text);
        this.third = el('div', 'card-third', watch);
        const card = el('div', 'ct-card', this.third);
        this.cardYellow = el('i', 'ct-yellow', card);
        this.cardRed = el('i', 'ct-red', card);
        this.thirdBadge = el('b', 'ct-badge', this.third);
        const ct = el('div', 'ct-text', this.third);
        this.thirdKind = el('div', 'ct-kind', ct);
        this.thirdName = el('div', 'ct-name', ct);
        this.thirdTeam = el('div', 'ct-team', ct);
        this.thirdMinute = el('div', 'ct-minute', this.third);
        this.hint = el('button', 'decision-skip', watch, 'Skip, Enter or click');
        this.hint.type = 'button';
        this.hint.addEventListener('click', () => this.tr.skip());
        // A click on the view skips a decision's replay.
        document.getElementById('stage').addEventListener('click', () => {
            if (this.tr.seq && this.tr.seq.decision) {
                this.tr.skip();
            }
        });
        this.tmp = new Float32Array(PLAYER_FLOATS);
        this.v = new THREE.Vector3();
    }

    // Name, number and team of player k, from the roster of the team's style when it has one.
    info(k) {
        const app = this.app;
        const header = app.source.header;
        const p = header.players[k] || { team: 0, index: 0 };
        const style = app.styles ? app.styles[p.team] : null;
        const roster = style && style.players ? style.players[p.index] : null;
        const team = header.teams[p.team] || {};
        return {
            team: p.team,
            number: roster && roster.number != null ? roster.number : (p.goalkeeper ? 1 : p.index + 1),
            name: roster && roster.name ? roster.name : (p.goalkeeper ? 'Goalkeeper' : `Player ${p.index + 1}`),
            teamName: teamNames(header)[p.team],
            badge: (style && style.badge) || (team.name || '?').slice(0, 1).toUpperCase(),
            color: style ? rgb(style.kit.primary) : (p.team ? '#2a56b8' : '#b3262e'),
            trim: style ? rgb(style.kit.secondary) : '#ffffff',
        };
    }

    // The scenes of a decision. Play stops with the whistle, the replay runs to the contact or
    // the pass and freezes on it, a card is shown, then the restart.
    scenes(seq) {
        const tr = this.tr;
        const app = this.app;
        const d = seq.ev.decision;
        const tl = app.timeline;
        const store = tl.store;
        const sps = tl.sps;
        seq.decision = d;
        const T = seq.T;
        const C = clamp(d.contact, store.first, T);
        // The replay starts a few seconds back, never across a cut or the restart before.
        let start = C;
        const earlier = tr.events.filter((e) => e.index < T && e !== seq.ev).pop();
        const floor = Math.max(store.first, earlier ? earlier.index + 1 : store.first);
        while (start > floor && C - start < LEAD) {
            const a = store.get(start - 1);
            const b = store.get(start);
            if (!a || !b || isCut(a, b)) {
                break;
            }
            start--;
        }
        const slowFrom = Math.max(start, C - SLOW_FRAMES);
        const fast = (slowFrom - start) / sps;
        const slow = (C - slowFrom) / (sps * SLOW);
        const replay = fast + slow;
        seq.replay = { start, C, slowFrom, fast };
        if (d.type === 'offside') {
            this.prepareOffside(seq);
        } else {
            this.prepareFoul(seq);
        }
        const kind = d.type === 'red' && d.secondYellow ? 'second' : d.type;
        const scenes = [];
        scenes.push({
            name: 'stop', dur: STOP + WIPE / 2,
            at: () => T - 1,
            dress: (pose, l) => tr.dressStage(pose, seq, l, 0.25),
            enter: () => {
                tr.whistle(d.type === 'red' || d.type === 'yellow' ? [0.25, 0.5] : [0.3]);
                this.hint.classList.add('show');
                showPointer(true);
            },
        });
        scenes.push({
            name: 'replay', dur: WIPE / 2 + replay, wipe: true, wipeLabel: 'Replay',
            at: (s, l) => {
                const u = Math.max(0, l - WIPE / 2);
                if (u < fast) {
                    return start + u * sps;
                }
                return Math.min(C, slowFrom + (u - fast) * sps * SLOW);
            },
            dress: (pose, l, s, sample) => {
                this.keepVisible(seq, sample);
                this.dressReplay(pose, seq, -1);
            },
            enter: () => {
                app.hud.setReplay(true);
                seq.replayBadge = true;
                tr.takeCamera(this.replayShot(seq, null), true);
            },
        });
        scenes.push({
            name: 'freeze', dur: FREEZE + WIPE / 2,
            at: () => C,
            dress: (pose, l, s, sample) => {
                this.keepVisible(seq, sample);
                this.dressReplay(pose, seq, l);
            },
            enter: () => this.showBanner(d, kind),
            // The flash follows the sequence's clock, so a pause or the speed holds it too.
            update: (s, l) => {
                this.flash.style.opacity = (0.85 * (1 - smooth(l / 0.5))).toFixed(3);
            },
            exit: () => {
                this.flash.style.opacity = '0';
                this.hideBanner();
            },
        });
        if (d.type === 'yellow' || d.type === 'red') {
            const color = d.type === 'red' ? 'red' : 'yellow';
            scenes.push({
                name: 'card', dur: WIPE / 2 + CARD + WIPE / 2, wipe: true, wipeLabel: TITLES[kind === 'second' ? 'red' : d.type],
                at: () => T,
                dress: (pose, l, s, sample) => {
                    this.keepVisible(seq, sample);
                    this.dressCard(pose, seq, l, color);
                },
                enter: () => {
                    this.marks.clear();
                    app.hud.setReplay(false);
                    seq.replayBadge = false;
                    this.prepareCard(seq);
                    tr.takeCamera(seq.card.shot, true);
                },
                update: (s, l) => {
                    if (l > WIPE / 2 + 0.9 && !seq.thirdShown) {
                        seq.thirdShown = true;
                        this.showThird(d, kind, store.get(T));
                    }
                },
                exit: () => this.hideThird(),
            });
        }
        const setup = tr.setupScene(seq);
        const enter = setup.enter;
        setup.enter = () => {
            this.marks.clear();
            this.hint.classList.remove('show');
            showPointer(false);
            app.hud.setReplay(false);
            seq.replayBadge = false;
            enter();
        };
        scenes.push(setup);
        return scenes;
    }

    end(seq) {
        this.flash.style.opacity = '0';
        this.marks.clear();
        this.hideBanner();
        this.hideThird();
        if (this.hint.classList.contains('show')) {
            showPointer(false);
        }
        this.hint.classList.remove('show');
        if (seq && seq.replayBadge) {
            this.app.hud.setReplay(false);
        }
    }

    // A player sent off leaves the frames at the call, yet the replay and the card show him.
    // The sample's frame is swapped for a copy that keeps him on the pitch, which the page
    // reads after the hooks, so the stored frames stay as they are.
    keepVisible(seq, sample) {
        const d = seq.decision;
        const f = sample && sample.a;
        if (!f) {
            return;
        }
        const missing = [d.offender, d.victim].filter((k) => k >= 0 && !(f.playerFlags[k] & 1));
        if (!missing.length) {
            return;
        }
        if (!seq.shown || seq.shown.from !== f) {
            const flags = f.playerFlags.slice();
            for (const k of missing) {
                flags[k] |= 1;
            }
            seq.shown = { from: f, frame: { ...f, playerFlags: flags } };
        }
        sample.a = seq.shown.frame;
    }

    // Foul replays.

    prepareFoul(seq) {
        const d = seq.decision;
        const store = this.app.timeline.store;
        const C = seq.replay.C;
        const pos = (j, k) => {
            const f = store.get(j);
            if (!f || k < 0) {
                return null;
            }
            const o = NODES_FLOAT + k * PLAYER_FLOATS;
            return [f.data[o], f.data[o + 1]];
        };
        const from = pos(Math.max(seq.replay.start, C - 8), d.offender);
        const to = pos(C, d.victim) || pos(C, d.offender);
        seq.arrow = from && to ? { from, to } : null;
    }

    // Offside replays, the line of the second last defender at the pass.
    prepareOffside(seq) {
        const app = this.app;
        const d = seq.decision;
        const header = app.source.header;
        const f = app.timeline.store.get(seq.replay.C);
        const team = header.players[d.offender].team;
        const attack = team === 0 ? 1 : -1;
        const depths = [];
        for (let k = 0; k < header.players.length; k++) {
            if (header.players[k].team !== team && (f.playerFlags[k] & 1)) {
                depths.push(f.data[NODES_FLOAT + k * PLAYER_FLOATS] * attack);
            }
        }
        depths.sort((a, b) => b - a);
        const second = Math.min(HALF_W, Math.max(0, depths.length > 1 ? depths[1] : 0, f.data[8] * attack));
        const o = NODES_FLOAT + d.offender * PLAYER_FLOATS;
        seq.offside = { lineX: second * attack, attackX: f.data[o], attackY: f.data[o + 1], attack };
    }

    // A clear angle of the challenge, from the main stand side and low, or along the offside
    // line so it stands upright on the screen.
    replayShot(seq, mid) {
        if (seq.offside) {
            const s = seq.offside;
            // On the line itself, high on the near side, with the attacker in the middle.
            const near = Math.min(HALF_H + 14, -s.attackY + 26);
            return {
                pos: new THREE.Vector3(s.lineX, 11, near),
                target: new THREE.Vector3(s.lineX, 0.4, -s.attackY - 2),
                fov: 40, rate: 2,
            };
        }
        const m = mid || this.midpoint(seq, null);
        return {
            pos: new THREE.Vector3(m.x - 4, 5.6, m.z + 16),
            target: new THREE.Vector3(m.x, 0.8, m.z),
            fov: 30, rate: 2.5,
        };
    }

    // The middle of the offender and the victim in three's frame, from the pose or the contact.
    midpoint(seq, pose) {
        const d = seq.decision;
        const ks = [d.offender, d.victim].filter((k) => k >= 0);
        const out = new THREE.Vector3();
        if (!ks.length) {
            return out;
        }
        for (const k of ks) {
            if (pose) {
                out.x += pose.nodes[k * PLAYER_FLOATS];
                out.z -= pose.nodes[k * PLAYER_FLOATS + 1];
            } else {
                const f = this.app.timeline.store.get(seq.replay.C);
                out.x += f.data[NODES_FLOAT + k * PLAYER_FLOATS];
                out.z -= f.data[NODES_FLOAT + k * PLAYER_FLOATS + 1];
            }
        }
        return out.divideScalar(ks.length);
    }

    // The marks during the replay, the arrow grows over the last second before the contact. A
    // frozen time of zero or more is the time since the freeze on the contact.
    dressReplay(pose, seq, frozenFor) {
        const frozen = frozenFor >= 0;
        const d = seq.decision;
        const app = this.app;
        const tl = app.timeline;
        const color = COLORS[d.type] || COLORS.foul;
        const nodes = pose.nodes;
        const pulse = performance.now() / 1000;
        if (d.offender >= 0) {
            this.marks.ring(0, nodes[d.offender * PLAYER_FLOATS], nodes[d.offender * PLAYER_FLOATS + 1], color, pulse);
        }
        if (d.victim >= 0) {
            this.marks.ring(1, nodes[d.victim * PLAYER_FLOATS], nodes[d.victim * PLAYER_FLOATS + 1], '#ffffff', pulse + 0.5);
        } else {
            this.marks.rings[1].visible = false;
        }
        if (seq.arrow) {
            const f = frozen ? 1 : clamp((tl.t - (seq.replay.C - 8)) / 8, 0, 1);
            const a = seq.arrow;
            this.marks.arrowTo(a.from[0], a.from[1], a.to[0], a.to[1], f, color);
        }
        if (seq.offside) {
            const s = seq.offside;
            if (frozen) {
                this.marks.offside(s.lineX, nodes[d.offender * PLAYER_FLOATS], s.attack, Math.max(0.02, smooth(frozenFor / 0.45)));
            }
        } else if (!frozen) {
            // The camera keeps the two players in the middle of the shot.
            const shot = app.cameras.scripted;
            const m = this.midpoint(seq, pose);
            if (shot && app.cameras.mode === 'scripted') {
                const want = this.replayShot(seq, m);
                shot.pos.copy(want.pos);
                shot.target.copy(want.target);
            }
        }
    }

    // Cards.

    prepareCard(seq) {
        const app = this.app;
        const d = seq.decision;
        const store = app.timeline.store;
        let f = store.get(seq.T);
        const gone = !(f.playerFlags[d.offender] & 1);
        if (gone) {
            // Sent off at the call, he is shown where the foul was.
            f = store.get(seq.replay.C) || f;
        }
        const o = NODES_FLOAT + d.offender * PLAYER_FLOATS;
        const ox = f.data[o];
        const oy = f.data[o + 1];
        // The referee runs in from the side of play away from the nearest touch line.
        const side = oy > 0 ? -1 : 1;
        const from = [clamp(ox + 3, -HALF_W - 1, HALF_W + 1), clamp(oy + side * 8, -HALF_H - 1, HALF_H + 1)];
        const dx = from[0] - ox;
        const dy = from[1] - oy;
        const dl = Math.hypot(dx, dy) || 1;
        const to = [ox + (dx / dl) * 1.7, oy + (dy / dl) * 1.7];
        // The camera looks across the pair, from the side, at chest height.
        const mx = (ox + to[0]) / 2;
        const my = (oy + to[1]) / 2;
        let px = -dy / dl;
        let py = dx / dl;
        if (py > 0) {
            px = -px;
            py = -py;
        }
        const shot = {
            pos: new THREE.Vector3(mx + px * 6.2, 2.0, -(my + py * 6.2)),
            target: new THREE.Vector3(mx, 1.45, -my),
            fov: 34, rate: 3,
        };
        seq.card = { from, to, ox, oy, shot, gone, heading: headingOf(f.data, NODES_FLOAT + d.offender * PLAYER_FLOATS) };
    }

    // The referee runs to the offender and raises the card, the offender holds his head.
    dressCard(pose, seq, l, color) {
        const c = seq.card;
        if (!c) {
            return;
        }
        const tr = this.tr;
        const d = seq.decision;
        const u = l - WIPE / 2 + 0.25;
        const run = 1.0;
        tr.refAlong(c.from, c.to, Math.max(0, u), run, c.ox, c.oy, 'card_raise', run - 0.05);
        tr.showReferee(true, u > run + 0.1 ? color : null);
        // The offender turns to the referee and puts his hands on his head.
        const o = d.offender * PLAYER_FLOATS;
        const heading = headingTo(c.to[0] - c.ox, c.to[1] - c.oy);
        const turn = c.gone ? 1 : smooth((u - 0.3) / 0.5);
        if (c.gone) {
            // His frames are off the pitch, so the whole pose is drawn here.
            namedPose('idle', u, heading, c.ox, c.oy, pose.nodes, o);
        }
        if (turn > 0) {
            namedPose('hands_on_head', u - 0.3, heading, c.ox, c.oy, this.tmp, 0);
            for (let i = 2; i < 7; i++) {
                this.tmp[i] = pose.nodes[o + i];
            }
            blendPose(pose.nodes, o, this.tmp, 0, turn, pose.nodes, o);
        }
    }

    // Graphics.

    showBanner(d, kind) {
        const who = this.info(d.offender);
        this.banner.style.setProperty('--decision', COLORS[kind === 'second' ? 'red' : d.type] || COLORS.foul);
        this.banner.style.setProperty('--team', who.color);
        this.bannerLabel.textContent = kind === 'second' ? 'Second yellow' : TITLES[d.type];
        this.bannerNumber.textContent = who.number;
        this.bannerName.textContent = who.name;
        this.bannerTeam.textContent = d.type === 'offside' ? `${who.teamName}, free kick to the defence` : who.teamName;
        this.banner.classList.add('show');
    }

    hideBanner() {
        this.banner.classList.remove('show');
    }

    showThird(d, kind, frame) {
        const who = this.info(d.offender);
        const minute = frame ? Math.floor(frame.timeMs / 60000) + 1 : 0;
        this.third.classList.remove('yellow', 'red', 'second', 'show');
        this.third.classList.add(kind === 'second' ? 'second' : d.type);
        this.third.style.setProperty('--team', who.color);
        this.thirdBadge.textContent = who.badge;
        this.thirdBadge.style.background = who.color;
        this.thirdBadge.style.color = who.trim;
        this.thirdKind.textContent = kind === 'second' ? 'Second yellow, sent off' : (d.type === 'red' ? 'Red card, sent off' : 'Yellow card');
        this.thirdName.textContent = `${who.number}  ${who.name}`;
        this.thirdTeam.textContent = who.teamName;
        this.thirdMinute.textContent = minute ? `${minute}′` : '';
        void this.third.offsetWidth;
        this.third.classList.add('show');
    }

    hideThird() {
        this.third.classList.remove('show');
    }
}
