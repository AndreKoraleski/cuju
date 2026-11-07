// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Team styles, kits and the textures drawn from them.

import * as THREE from 'three';

export const PATTERNS = ['plain', 'stripes', 'hoops', 'halves', 'sash', 'checkers'];
export const SKINS = ['mixed', 'light', 'medium', 'dark'];
export const HAIRS = ['mixed', 'bald', 'short', 'long'];
// Per player choices of the roster, grf keeps GRF's own look for that index.
export const PLAYER_SKINS = ['grf', 'light', 'tan', 'medium', 'dark'];
export const PLAYER_HAIRS = ['grf', 'bald', 'short01', 'short02', 'medium01', 'long01', 'long02'];
export const HAIR_COLORS_NAMED = ['grf', 'black', 'brown', 'darkblonde', 'blonde', 'red'];
export const FORMATS = { '5v5': 5, '11v11': 11 };
// A celebration the page knows, or clip and the name of an animation inside a model, as the
// server checks them.
const CELEBRATION_NAME = /^[a-z0-9_]{1,40}(:[A-Za-z0-9_. -]{1,60})?$/;
// A model of the roster, team follows the team's model. Shipped ids are checked by the server.
const MODEL_NAME = /^[A-Za-z0-9_.-]{1,64}$/;
const HEX_COLOR = /^#[0-9a-f]{6}$/;
// The most color pairs the server keeps for one model.
export const MAX_RECOLORS = 32;

export function isCelebrationName(name) {
    return typeof name === 'string' && CELEBRATION_NAME.test(name);
}

// A roster entry with every field present.
export function rosterEntry(p, i) {
    const e = p && typeof p === 'object' ? p : {};
    const number = Number.isInteger(e.number) && e.number >= 0 && e.number <= 99 ? e.number : i + 1;
    return {
        name: typeof e.name === 'string' ? e.name.slice(0, 24) : '',
        number,
        skin: PLAYER_SKINS.includes(e.skin) ? e.skin : 'grf',
        hair: PLAYER_HAIRS.includes(e.hair) ? e.hair : 'grf',
        hair_color: HAIR_COLORS_NAMED.includes(e.hair_color) ? e.hair_color : 'grf',
        model: typeof e.model === 'string' && MODEL_NAME.test(e.model) ? e.model : 'team',
        celebration: isCelebrationName(e.celebration) ? e.celebration : 'team',
    };
}

// The color pairs of every model, lower case #rrggbb on both sides, at most 32 per model. GRF's
// own model takes the kit instead, so it holds no pairs.
export function recolorPairs(value) {
    const out = {};
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return out;
    }
    for (const [model, pairs] of Object.entries(value)) {
        if (model === 'grf' || model === 'team' || !MODEL_NAME.test(model) || !pairs || typeof pairs !== 'object') {
            continue;
        }
        const clean = {};
        let n = 0;
        for (const [from, to] of Object.entries(pairs)) {
            const a = String(from).toLowerCase();
            const b = String(to).toLowerCase();
            if (HEX_COLOR.test(a) && HEX_COLOR.test(b) && n < MAX_RECOLORS) {
                clean[a] = b;
                n++;
            }
        }
        out[model] = clean;
    }
    return out;
}

function celebrationList(value) {
    return Array.isArray(value) ? value.filter(isCelebrationName).slice(0, 32) : [];
}

// One to four colors of the fans, or null to wear the kit.
function crowdColors(value) {
    const list = value && Array.isArray(value.colors) ? value.colors : [];
    const colors = list.filter((c) => Array.isArray(c) && c.length >= 3 && c.slice(0, 3).every(Number.isFinite))
        .slice(0, 4).map((c) => color3(c, [0, 0, 0]));
    return colors.length ? colors : null;
}

// The model a style asks for, grf, custom or a shipped model id. Older styles held an object.
export function modelChoice(style) {
    const m = style && style.model;
    if (typeof m === 'string' && m) {
        return m;
    }
    if (m && typeof m === 'object') {
        return m.source === 'shipped' ? (m.name || 'grf') : (m.source || 'grf');
    }
    return 'grf';
}
export const NUMBER_COLORS = ['white', 'black', 'secondary'];

const SKIN_TONES = {
    light: ['#f3d2b8', '#ecc19f', '#e6b58f'],
    medium: ['#d39e74', '#c68c5f', '#b47a50'],
    dark: ['#8d5a36', '#6f4329', '#5a341f'],
};
const HAIR_COLORS = ['#1c140f', '#3b2616', '#6b4423', '#a07443', '#d8c48a', '#0d0d0d'];
const KEEPER_CHOICES = [[235, 220, 40], [40, 200, 90], [240, 120, 20], [230, 60, 200],
    [40, 220, 220], [30, 30, 30], [150, 240, 60]];

// The server takes colours only as three whole numbers from 0 to 255, so every colour is rounded
// here, the fallback included.
function color3(value, fallback) {
    const source = Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((v) => Number.isFinite(v))
        ? value : fallback;
    return source.slice(0, 3).map((v) => Math.max(0, Math.min(255, Math.round(v))));
}

// A darker shade of a colour, for the trim and shorts of a keeper kit made up here.
function shade(c, f) {
    return color3(c.map((v) => v * f), c);
}

// The server's defaults, as _teams.py holds them in _KIT, _GOALKEEPER and _BUILTIN_KIT.
export const DEFAULT_KIT = {
    pattern: 'plain', primary: [214, 48, 49], secondary: [255, 255, 255], shorts: [255, 255, 255], socks: [214, 48, 49],
};
export const DEFAULT_GOALKEEPER = {
    primary: [46, 204, 113], secondary: [20, 20, 20], shorts: [20, 20, 20], socks: [46, 204, 113],
};
export const BUILTIN_KIT = {
    pattern: 'plain', primary: [9, 132, 227], secondary: [255, 255, 255], shorts: [9, 132, 227], socks: [255, 255, 255],
};

function distance(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function linear(v) {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function labF(t) {
    return t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116;
}

// CIE Lab under D65 of an sRGB color given as three numbers from 0 to 255.
export function rgbToLab(c, out = [0, 0, 0]) {
    const r = linear(c[0]);
    const g = linear(c[1]);
    const b = linear(c[2]);
    const x = labF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const y = labF(0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
    const z = labF((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883);
    out[0] = 116 * y - 16;
    out[1] = 500 * (x - y);
    out[2] = 200 * (y - z);
    return out;
}

function labInv(t) {
    return t ** 3 > 216 / 24389 ? t ** 3 : (116 * t - 16) / (24389 / 27);
}

function gamma(v) {
    const c = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, c * 255));
}

// The sRGB color of a Lab color, clipped to the gamut, as three numbers from 0 to 255.
export function labToRgb(lab, out = [0, 0, 0]) {
    const fy = (lab[0] + 16) / 116;
    const x = 0.95047 * labInv(fy + lab[1] / 500);
    const y = labInv(fy);
    const z = 1.08883 * labInv(fy - lab[2] / 200);
    out[0] = gamma(3.2404542 * x - 1.5371385 * y - 0.4985314 * z);
    out[1] = gamma(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z);
    out[2] = gamma(0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
    return out;
}

// CIEDE2000 between two Lab colors, about 2 is a just noticeable difference.
export function ciede2000(p, q) {
    const [L1, a1, b1] = p;
    const [L2, a2, b2] = q;
    const C1 = Math.hypot(a1, b1);
    const C2 = Math.hypot(a2, b2);
    const Cm7 = ((C1 + C2) / 2) ** 7;
    const G = 0.5 * (1 - Math.sqrt(Cm7 / (Cm7 + 25 ** 7)));
    const a1p = a1 * (1 + G);
    const a2p = a2 * (1 + G);
    const C1p = Math.hypot(a1p, b1);
    const C2p = Math.hypot(a2p, b2);
    const hue = (b, a) => {
        if (a === 0 && b === 0) {
            return 0;
        }
        const h = Math.atan2(b, a) * 180 / Math.PI;
        return h < 0 ? h + 360 : h;
    };
    const h1 = hue(b1, a1p);
    const h2 = hue(b2, a2p);
    const dL = L2 - L1;
    const dC = C2p - C1p;
    let dh = 0;
    if (C1p * C2p !== 0) {
        dh = h2 - h1;
        if (dh > 180) {
            dh -= 360;
        } else if (dh < -180) {
            dh += 360;
        }
    }
    const dH = 2 * Math.sqrt(C1p * C2p) * Math.sin((dh * Math.PI) / 360);
    const Lm = (L1 + L2) / 2;
    const Cm = (C1p + C2p) / 2;
    let hm = h1 + h2;
    if (C1p * C2p !== 0) {
        if (Math.abs(h1 - h2) > 180) {
            hm += h1 + h2 < 360 ? 360 : -360;
        }
        hm /= 2;
    }
    const rad = Math.PI / 180;
    const T = 1 - 0.17 * Math.cos((hm - 30) * rad) + 0.24 * Math.cos(2 * hm * rad)
        + 0.32 * Math.cos((3 * hm + 6) * rad) - 0.2 * Math.cos((4 * hm - 63) * rad);
    const dTheta = 30 * Math.exp(-(((hm - 275) / 25) ** 2));
    const Cm7p = Cm ** 7;
    const Rc = 2 * Math.sqrt(Cm7p / (Cm7p + 25 ** 7));
    const Sl = 1 + (0.015 * (Lm - 50) ** 2) / Math.sqrt(20 + (Lm - 50) ** 2);
    const Sc = 1 + 0.045 * Cm;
    const Sh = 1 + 0.015 * Cm * T;
    const Rt = -Math.sin(2 * dTheta * rad) * Rc;
    const l = dL / Sl;
    const c = dC / Sc;
    const hh = dH / Sh;
    return Math.sqrt(l * l + c * c + hh * hh + Rt * c * hh);
}

// The perceptual difference of two sRGB colors, CIEDE2000.
export function deltaE(a, b) {
    return ciede2000(rgbToLab(a), rgbToLab(b));
}

// The share of the shirt that the secondary color covers, by pattern.
const SECONDARY_SHARE = { plain: 0, stripes: 0.5, hoops: 0.5, halves: 0.5, sash: 0.15, checkers: 0.5 };

function shirtColors(kit) {
    const share = SECONDARY_SHARE[kit.pattern] || 0;
    return share ? [[kit.primary, 1 - share], [kit.secondary, share]] : [[kit.primary, 1]];
}

// How different two shirts look from a distance, the larger of the two directed mean distances
// from each color of one shirt to the nearest color of the other, weighted by area. Stripes of
// green and yellow against stripes of yellow and green measure zero.
function shirtDifference(a, b) {
    const directed = (x, y) => x.reduce((sum, [c, w]) => sum + w * Math.min(...y.map(([d]) => deltaE(c, d))), 0);
    const sa = shirtColors(a);
    const sb = shirtColors(b);
    return Math.max(directed(sa, sb), directed(sb, sa));
}

// The difference of two whole kits, mostly the shirt, which fills most of a player from above.
export function kitDifference(a, b) {
    return 0.8 * shirtDifference(a, b) + 0.2 * deltaE(a.shorts, b.shorts);
}

// Kits closer than this look alike on the pitch.
const KIT_CLASH = 22;
const WHITE = [245, 245, 245];
const BLACK = [24, 24, 28];

// The away kits to try, in order, the home kit's colors swapped, then a white or a black shirt
// with a sash of the team's primary, then plain white and plain black.
function awayCandidates(kit) {
    const trim = kit.primary;
    return [
        { ...kit, primary: kit.secondary, secondary: kit.primary, shorts: kit.socks, socks: kit.shorts },
        { pattern: 'sash', primary: WHITE, secondary: trim, shorts: WHITE, socks: trim },
        { pattern: 'sash', primary: BLACK, secondary: trim, shorts: BLACK, socks: trim },
        { pattern: 'plain', primary: WHITE, secondary: trim, shorts: trim, socks: WHITE },
        { pattern: 'plain', primary: BLACK, secondary: trim, shorts: trim, socks: BLACK },
    ].map((k) => ({ ...k, primary: k.primary.slice(), secondary: k.secondary.slice(), shorts: k.shorts.slice(), socks: k.socks.slice() }));
}

// The kit the right team wears against the left team's kit, its own when they differ enough.
export function awayKit(home, kit) {
    if (kitDifference(home, kit) >= KIT_CLASH) {
        return null;
    }
    let best = null;
    let bestD = -1;
    for (const c of awayCandidates(kit)) {
        const d = kitDifference(home, c);
        if (d >= KIT_CLASH) {
            return c;
        }
        if (d > bestD) {
            bestD = d;
            best = c;
        }
    }
    return best;
}

// The colour among the keeper choices farthest from every colour given.
export function contrasting(avoid, skip = 0) {
    const ranked = KEEPER_CHOICES
        .map((c) => ({ c, d: Math.min(...avoid.map((a) => distance(a, c))) }))
        .sort((x, y) => y.d - x.d);
    return ranked[Math.min(skip, ranked.length - 1)].c.slice();
}

// Fills every field of a style, from a header or registry team. Older headers only carry
// color and secondary, those become a plain kit.
export function normalizeStyle(team, side = 0) {
    const legacyPrimary = color3(team && team.color, side ? [40, 80, 200] : DEFAULT_KIT.primary);
    const legacySecondary = color3(team && team.secondary, side ? [30, 30, 40] : DEFAULT_KIT.secondary);
    const s = (team && team.style) || {};
    const kit = s.kit || {};
    const primary = color3(kit.primary, legacyPrimary);
    const secondary = color3(kit.secondary, legacySecondary);
    const out = {
        kit: {
            pattern: PATTERNS.includes(kit.pattern) ? kit.pattern : 'plain',
            primary,
            secondary,
            shorts: color3(kit.shorts, secondary),
            socks: color3(kit.socks, primary),
        },
        skin: SKINS.includes(s.skin) ? s.skin : 'mixed',
        hair: HAIRS.includes(s.hair) ? s.hair : 'mixed',
        badge: typeof s.badge === 'string' ? Array.from(s.badge).slice(0, 3).join('') : '',
        number: NUMBER_COLORS.includes(s.number) ? s.number : 'white',
        model: modelChoice(s),
        players: Array.isArray(s.players) ? s.players.map(rosterEntry) : [],
        colors: recolorPairs(s.colors),
        celebrations: celebrationList(s.celebrations),
        crowd: { colors: crowdColors(s.crowd) || [primary, secondary] },
    };
    const gk = s.goalkeeper || {};
    const keeper = contrasting([primary, secondary, out.kit.shorts], side);
    out.goalkeeper = {
        primary: color3(gk.primary, keeper),
        secondary: color3(gk.secondary, shade(keeper, 0.6)),
        shorts: color3(gk.shorts, shade(keeper, 0.45)),
        socks: color3(gk.socks, keeper),
    };
    return out;
}

// The style as the server takes it, only the entries that _teams.py style accepts. The viewer's own
// style carries more, such as the skin, hair and artwork it draws with, which the server refuses.
export function serverStyle(style) {
    const parts = (kit, defaults) => Object.fromEntries(['primary', 'secondary', 'shorts', 'socks']
        .map((k) => [k, color3(kit && kit[k], defaults[k])]));
    const kit = style.kit || {};
    const out = {
        model: modelChoice(style),
        kit: { pattern: PATTERNS.includes(kit.pattern) ? kit.pattern : 'plain', ...parts(kit, DEFAULT_KIT) },
        goalkeeper: parts(style.goalkeeper, DEFAULT_GOALKEEPER),
        badge: typeof style.badge === 'string' ? Array.from(style.badge).slice(0, 3).join('') : '',
        number: NUMBER_COLORS.includes(style.number) ? style.number : 'white',
        players: (Array.isArray(style.players) ? style.players : []).map((p, i) => rosterEntry(p, i)),
        colors: recolorPairs(style.colors),
        celebrations: celebrationList(style.celebrations),
    };
    const fans = crowdColors(style.crowd);
    if (fans) {
        out.crowd = { colors: fans };
    }
    return out;
}

// Keeps the two keepers apart when both teams left them at their defaults.
export function separateKeepers(styles, teams) {
    // The right team wears its away kit when its kit looks like the left team's, as on a real
    // matchday, judged by perceptual difference over shirt, pattern and shorts.
    const away = awayKit(styles[0].kit, styles[1].kit);
    if (away) {
        styles[1].kit = away;
        styles[1].away = true;
    }
    const a = styles[0].goalkeeper.primary;
    const b = styles[1].goalkeeper.primary;
    const explicit = teams[1] && teams[1].style && teams[1].style.goalkeeper;
    if (distance(a, b) < 60 && !explicit) {
        const avoid = [styles[0].kit.primary, styles[1].kit.primary, styles[1].kit.secondary, a];
        const c = contrasting(avoid);
        styles[1].goalkeeper = {
            primary: c, secondary: shade(c, 0.6), shorts: shade(c, 0.45), socks: c,
        };
    }
}

export function css(c) {
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function hex(c) {
    return '#' + c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
}

export function fromHex(h) {
    const v = parseInt(h.replace('#', ''), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function threeColor(c) {
    return new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
}

function hash(n) {
    let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    return (h ^ (h >>> 16)) >>> 0;
}

// Per player colours for the parts that are not the shirt.
export function playerLook(style, player, k) {
    const h = hash(player.team * 97 + player.index * 13 + k);
    let tones;
    if (style.skin === 'mixed') {
        tones = [...SKIN_TONES.light, ...SKIN_TONES.medium, ...SKIN_TONES.dark];
    } else {
        tones = SKIN_TONES[style.skin];
    }
    let hair = style.hair;
    if (hair === 'mixed') {
        hair = ['short', 'short', 'short', 'long', 'bald'][h % 5];
    }
    const skin = new THREE.Color(tones[h % tones.length]);
    const kit = player.goalkeeper ? style.goalkeeper : style.kit;
    return {
        skin,
        hairStyle: hair,
        hair: hair === 'bald' ? skin.clone() : new THREE.Color(HAIR_COLORS[(h >> 4) % HAIR_COLORS.length]),
        shorts: threeColor(kit.shorts),
        socks: threeColor(kit.socks),
        shirt: threeColor(kit.primary),
        boots: new THREE.Color(0x121212),
        number: style.players && style.players[player.index] ? style.players[player.index].number
            : (player.goalkeeper ? 1 : (player.index + 1)),
        name: style.players && style.players[player.index] ? style.players[player.index].name : '',
    };
}

// The shirt texture for a capsule torso. Capsule u runs around starting at the player's
// right side, so the chest is at u 0.25 and the back at u 0.75, v runs from bottom to top.
export function shirtTexture(style, goalkeeper) {
    const w = 512;
    const h = 256;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const kit = goalkeeper ? style.goalkeeper : style.kit;
    const pattern = goalkeeper ? 'plain' : style.kit.pattern;
    const a = css(kit.primary);
    const b = css(kit.secondary);
    ctx.fillStyle = a;
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = b;
    if (pattern === 'stripes') {
        const n = 16;
        for (let k = 0; k < n; k += 2) {
            ctx.fillRect((k / n) * w, 0, w / n, h);
        }
    } else if (pattern === 'hoops') {
        const n = 8;
        for (let k = 1; k < n; k += 2) {
            ctx.fillRect(0, (k / n) * h, w, h / n);
        }
    } else if (pattern === 'halves') {
        ctx.fillRect(0, 0, w * 0.25, h);
        ctx.fillRect(w * 0.75, 0, w * 0.25, h);
    } else if (pattern === 'sash') {
        // A band from the right shoulder across the chest to the left hip, and round the back.
        ctx.beginPath();
        ctx.moveTo(w * 0.1, 0);
        ctx.lineTo(w * 0.22, 0);
        ctx.lineTo(w * 0.48, h);
        ctx.lineTo(w * 0.36, h);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(w * 0.6, 0);
        ctx.lineTo(w * 0.72, 0);
        ctx.lineTo(w * 0.98, h);
        ctx.lineTo(w * 0.86, h);
        ctx.closePath();
        ctx.fill();
    } else if (pattern === 'checkers') {
        const nx = 16;
        const ny = 8;
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                if ((i + j) % 2) {
                    ctx.fillRect((i / nx) * w, (j / ny) * h, w / nx, h / ny);
                }
            }
        }
    }
    // Collar and cuffs in the secondary colour, the top rows of the capsule.
    ctx.fillStyle = goalkeeper ? css(kit.secondary) : b;
    ctx.fillRect(0, 0, w, h * 0.04);
    if (style.badge) {
        // The badge on the left chest, which is u a little past 0.25 towards 0.5.
        const bx = w * 0.3;
        const by = h * 0.3;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.beginPath();
        ctx.arc(bx, by, h * 0.075, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = css(kit.primary);
        ctx.font = `700 ${Math.round(h * 0.075)}px system-ui, "Segoe UI Emoji", "Apple Color Emoji", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(style.badge, bx, by + 1, h * 0.14);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
}

export function numberColor(style, goalkeeper) {
    const kit = goalkeeper ? style.goalkeeper : style.kit;
    if (style.number === 'black') {
        return threeColor([20, 20, 20]);
    }
    if (style.number === 'secondary') {
        return threeColor(kit.secondary);
    }
    return threeColor([250, 250, 250]);
}

// Kits for GRF's player.glb, painted in its kit UV space from kit_mask.png. The front of the
// shirt is the left half of the texture and the back the right half, each seen from outside,
// v grows downwards with the shirt above row 0.41, shorts to 0.57 and socks below.
const SHIRT_BOTTOM = 0.41;

function patternMask(pattern, u, v) {
    const back = u >= 0.5;
    const cu = back ? (u - 0.5) * 2 : u * 2;
    switch (pattern) {
    case 'stripes':
        return Math.floor(cu * 14) & 1;
    case 'hoops':
        return Math.floor(v / 0.05) & 1;
    case 'halves':
        // The player's right half, on the left of the front panel and the right of the back.
        return back ? (cu > 0.5 ? 1 : 0) : (cu < 0.5 ? 1 : 0);
    case 'sash': {
        const t = Math.min(1, v / SHIRT_BOTTOM);
        const centre = back ? 0.78 - 0.5 * t : 0.22 + 0.5 * t;
        return Math.abs(cu - centre) < 0.085 ? 1 : 0;
    }
    case 'checkers':
        return (Math.floor(cu * 14) + Math.floor(v / 0.05)) & 1;
    default:
        return 0;
    }
}

// The shirt mask as an alpha channel, used to clip artwork to the shirt.
const shirtAlpha = new WeakMap();
function shirtAlphaCanvas(mask) {
    if (!shirtAlpha.has(mask)) {
        const canvas = document.createElement('canvas');
        canvas.width = mask.width;
        canvas.height = mask.height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(mask.width, mask.height);
        for (let i = 0; i < img.data.length; i += 4) {
            img.data[i + 3] = mask.data[i];
        }
        ctx.putImageData(img, 0, 0);
        shirtAlpha.set(mask, canvas);
    }
    return shirtAlpha.get(mask);
}

// The flat front and back artwork over the shirt region, clipped to the shirt.
function drawShirtArt(ctx, mask, art) {
    if (!art || (!art.front && !art.back)) {
        return;
    }
    const w = mask.width;
    const h = mask.height;
    const layer = document.createElement('canvas');
    layer.width = w;
    layer.height = h;
    const lc = layer.getContext('2d');
    if (art.front) {
        lc.drawImage(art.front, 0, 0, w / 2, h * SHIRT_BOTTOM);
    }
    if (art.back) {
        lc.drawImage(art.back, w / 2, 0, w / 2, h * SHIRT_BOTTOM);
    }
    lc.globalCompositeOperation = 'destination-in';
    lc.drawImage(shirtAlphaCanvas(mask), 0, 0);
    ctx.drawImage(layer, 0, 0);
}

// Paints the shared part of a kit, without the number, as a canvas.
export function paintKitBase(mask, style, goalkeeper) {
    const { width: w, height: h, data: m } = mask;
    const kit = goalkeeper ? style.goalkeeper : style.kit;
    const pattern = goalkeeper ? 'plain' : style.kit.pattern;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const out = ctx.createImageData(w, h);
    const d = out.data;
    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            const r = m[i] / 255;
            const g = m[i + 1] / 255;
            const b = m[i + 2] / 255;
            const shirt = r > 0 && patternMask(pattern, (x + 0.5) / w, v) ? kit.secondary : kit.primary;
            d[i] = r * shirt[0] + g * kit.shorts[0] + b * kit.socks[0];
            d[i + 1] = r * shirt[1] + g * kit.shorts[1] + b * kit.socks[1];
            d[i + 2] = r * shirt[2] + g * kit.shorts[2] + b * kit.socks[2];
            d[i + 3] = 255;
        }
    }
    ctx.putImageData(out, 0, 0);
    drawShirtArt(ctx, mask, style.art);
    if (style.badge) {
        // On the left chest, the right side of the front panel.
        const bx = w * 0.33;
        const by = h * 0.12;
        const r = w * 0.035;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.beginPath();
        ctx.arc(bx, by, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = css(kit.primary);
        ctx.font = `700 ${Math.round(r * 1.05)}px system-ui, "Segoe UI Emoji", "Apple Color Emoji", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(style.badge, bx, by + 1, r * 1.8);
    }
    return canvas;
}

// One player's kit texture, the base with the shirt number on the back.
export function kitTexture(base, style, goalkeeper, number) {
    const canvas = document.createElement('canvas');
    canvas.width = base.width;
    canvas.height = base.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(base, 0, 0);
    if (number != null) {
        const c = numberColor(style, goalkeeper);
        ctx.fillStyle = `#${c.getHexString(THREE.SRGBColorSpace)}`;
        ctx.font = `800 ${Math.round(base.height * 0.11)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(number), base.width * 0.75, base.height * 0.19, base.width * 0.2);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.flipY = false;
    tex.anisotropy = 4;
    return tex;
}
