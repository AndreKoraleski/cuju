// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The graphics and helpers of the montage, loaded into the page by render_promo.py. The graphics
// take the page's own theme, type and broadcast graphics, the lower third, the banner of a goal,
// the stinger and the logo wipe.

window.promo = (() => {
    const css = `
#promo-layer { position: fixed; inset: 0; z-index: 2147483000; pointer-events: none; overflow: hidden; }
#promo-layer #stinger { z-index: 30; }
.promo-third { position: absolute; left: 32px; bottom: 40px; display: flex; align-items: stretch;
    min-width: 280px; border: 1px solid var(--border); background: var(--overlay);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3); color: var(--text); font-family: var(--font);
    animation: lower-third-in 0.4s cubic-bezier(0.7, 0, 0.3, 1); }
.promo-third .lt-number { min-width: 60px; background: var(--accent); color: var(--on-accent);
    font-family: var(--broadcast-font, sans-serif); font-weight: 800; font-size: 20px; }
.promo-third .lt-name { font-size: 30px; }
.promo-third .lt-sub { font-size: 16px; letter-spacing: 0.08em; text-transform: uppercase; }
.promo-banner { position: absolute; left: 40px; bottom: 48px; max-width: calc(100% - 80px);
    padding: 14px 26px 16px 20px; border: 1px solid var(--border); border-left: 5px solid var(--accent);
    background: var(--overlay); box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); color: var(--text);
    font-family: var(--font); animation: promo-reveal 0.5s cubic-bezier(0.7, 0, 0.3, 1) both; }
.promo-banner .pb-head { display: flex; align-items: center; gap: 14px; }
.promo-banner .pb-chip { flex: none; width: 10px; height: 38px; background: var(--accent); }
.promo-banner .pb-title { font-size: 44px; font-variant: small-caps; letter-spacing: 0.06em; line-height: 1.05;
    animation: promo-pop 0.7s ease-out both; }
.promo-banner .pb-title.code { font-family: Consolas, "Cascadia Mono", "Courier New", monospace;
    font-variant: normal; letter-spacing: 0; font-size: 42px; }
.promo-banner .pb-sub { margin: 8px 0 0 24px; color: var(--muted); font-family: var(--broadcast-font, sans-serif);
    font-size: 17px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    animation: promo-fade 0.5s ease-out 0.25s both; }
.promo-logo { position: absolute; inset: 0; display: grid; place-items: center;
    background: linear-gradient(180deg, var(--stage-top), var(--stage-bottom)); color: var(--text);
    font-family: var(--font); }
.promo-logo .pl-inner { display: flex; flex-direction: column; align-items: center; gap: 18px;
    animation: promo-rise 0.8s cubic-bezier(0.2, 0.9, 0.2, 1) both; }
.promo-logo img { width: 300px; height: auto; filter: var(--logo-filter); }
.promo-logo .pl-rule { width: 0; height: 4px; background: var(--accent); animation: promo-rule 0.7s ease-out 0.25s forwards; }
.promo-logo .pl-line { font-size: 40px; letter-spacing: 0.02em; animation: promo-fade 0.6s ease-out 0.35s both; }
.promo-logo .pl-sub { color: var(--muted); font-family: var(--broadcast-font, sans-serif); font-size: 20px;
    font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; animation: promo-fade 0.6s ease-out 0.55s both; }
.promo-logo .pl-code { padding: 8px 18px; border: 1px solid var(--border); background: var(--surface);
    font-family: Consolas, "Cascadia Mono", "Courier New", monospace; font-size: 30px;
    animation: promo-fade 0.6s ease-out 0.45s both; }
.promo-fade-out { animation: promo-out 0.6s ease-in forwards; }
.promo-third .lt-number:empty { min-width: 12px; padding: 0; }
#scoreboard, #goal-score, #goal-number, #goal-minute, #goal-assist, .mvp-number, .mvp-card .stat, .db-number, .ct-minute { display: none !important; }
.promo-banner.center { left: 50%; bottom: 64px; width: max-content; max-width: none; white-space: nowrap; transform: translateX(-50%); animation: none; }
.promo-banner.center .pb-title { font-size: 50px; }
.promo-ripple { position: absolute; width: 24px; height: 24px; margin: -12px 0 0 -12px; border: 3px solid var(--accent);
    border-radius: 50%; animation: promo-ripple 0.7s ease-out forwards; }
.promo-scrim { position: absolute; inset: 0; background: var(--scrim); animation: promo-fade 0.5s ease-out both; }
.promo-file { position: absolute; left: 50%; top: 50%; min-width: 560px; transform: translate(-50%, -50%);
    padding: 22px 28px 24px; border: 1px solid var(--border); border-left: 5px solid var(--accent);
    background: var(--surface); box-shadow: 0 18px 60px rgba(0, 0, 0, 0.5); color: var(--text); font-family: var(--font);
    animation: promo-pop-card 0.5s cubic-bezier(0.2, 0.9, 0.2, 1) both; }
.promo-file .pf-kicker { color: var(--muted); font-family: var(--broadcast-font, sans-serif); font-size: 15px; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase; }
.promo-file .pf-name { margin: 6px 0 14px; font-size: 38px; font-variant: small-caps; letter-spacing: 0.04em; }
.promo-file .pf-line { padding: 6px 0; border-top: 1px solid var(--border); font-family: Consolas, "Cascadia Mono", "Courier New", monospace;
    font-size: 20px; }
.promo-file .pf-line i { color: var(--muted); font-family: var(--font); font-style: normal; font-size: 18px; }
@keyframes promo-ripple { from { opacity: 1; transform: scale(1); } to { opacity: 0; transform: scale(4); } }
@keyframes promo-pop-card { from { opacity: 0; transform: translate(-50%, -46%) scale(0.96); } to { opacity: 1; transform: translate(-50%, -50%); } }
@keyframes promo-reveal { from { clip-path: inset(0 100% 0 0); transform: translateX(-16px); } to { clip-path: inset(0 0 0 0); transform: none; } }
@keyframes promo-pop { from { letter-spacing: 0.4em; opacity: 0; } to { opacity: 1; } }
@keyframes promo-fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes promo-rise { from { opacity: 0; transform: translateY(18px) scale(0.96); } to { opacity: 1; transform: none; } }
@keyframes promo-rule { to { width: 340px; } }
@keyframes promo-out { to { opacity: 0; } }
`;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.append(style);
    const layer = document.createElement('div');
    layer.id = 'promo-layer';
    document.body.append(layer);
    const stinger = document.getElementById('stinger');
    layer.append(stinger);
    const make = (tag, cls, parent, text) => {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text != null) node.textContent = text;
        if (parent) parent.append(node);
        return node;
    };
    return {
        layer,
        // Moves the graphics into an open modal dialog, which sits above the rest of the page.
        host(node) { (node || document.body).append(layer); },
        clear() { for (const node of [...layer.children]) if (node !== stinger) node.remove(); },
        third(name, sub, number) {
            const t = make('div', 'promo-third', layer);
            make('span', 'lt-number', t, number || '');
            const body = make('span', 'lt-body', t);
            make('span', 'lt-name', body, name);
            make('span', 'lt-sub', body, sub);
            return t;
        },
        banner(title, sub, code) {
            const b = make('div', 'promo-banner', layer);
            const head = make('div', 'pb-head', b);
            make('i', 'pb-chip', head);
            make('span', code ? 'pb-title code' : 'pb-title', head, title);
            if (sub) make('div', 'pb-sub', b, sub);
            return b;
        },
        // A line of the page's banner, centered at the foot of the picture.
        headline(title) {
            const b = promo.banner(title, '');
            b.classList.add('center');
            return b;
        },
        // Moves the view of the page in on an element, as a camera would.
        zoomTo(selector, scale, seconds) {
            const app = document.getElementById('app');
            const r = document.querySelector(selector).getBoundingClientRect();
            app.style.transformOrigin = `${r.left + r.width / 2}px ${r.top + r.height / 2}px`;
            app.style.transition = `transform ${seconds}s cubic-bezier(0.45, 0, 0.2, 1)`;
            void app.offsetWidth;
            app.style.transform = `scale(${scale})`;
        },
        // A press on an element, the button held down and a ring spreading from it.
        press(selector, down) {
            const target = typeof selector === 'string' ? document.querySelector(selector) : selector;
            target.style.transition = 'transform 0.12s, filter 0.12s';
            target.style.transform = down ? 'scale(0.94)' : '';
            target.style.filter = down ? 'brightness(0.85)' : '';
            if (down) {
                const r = target.getBoundingClientRect();
                const ring = make('div', 'promo-ripple', layer);
                ring.style.left = `${r.left + r.width / 2}px`;
                ring.style.top = `${r.top + r.height / 2}px`;
            }
        },
        // A card of a saved file and what it holds, over a scrim.
        file(kicker, name, lines) {
            make('div', 'promo-scrim', layer);
            const card = make('div', 'promo-file', layer);
            make('div', 'pf-kicker', card, kicker);
            make('div', 'pf-name', card, name);
            for (const [path, note] of lines) {
                const line = make('div', 'pf-line', card, path + '  ');
                make('i', '', line, note);
            }
            return card;
        },
        // Hides every line of text in root that holds a digit, since the montage shows no numbers.
        hideNumbers(root) {
            for (const node of root.querySelectorAll('*')) {
                if (!node.children.length && /\d/.test(node.textContent)) node.style.visibility = 'hidden';
            }
        },
        // Gives the roster of each side names, an array of names per side or null.
        async names(sides) {
            await promo.styles(sides.map((list) => list && { style: { players: list.map((name) => ({ name })) } }));
        },
        logo(line, sub, code) {
            const c = make('div', 'promo-logo', layer);
            const inner = make('div', 'pl-inner', c);
            const img = make('img', '', inner);
            img.src = 'assets/cuju-logo-384.png';
            img.alt = '';
            make('div', 'pl-rule', inner);
            if (line) make('div', 'pl-line', inner, line);
            if (code) make('div', 'pl-code', inner, code);
            if (sub) make('div', 'pl-sub', inner, sub);
            return c;
        },
        fadeOut() { for (const node of layer.children) if (node !== stinger) node.classList.add('promo-fade-out'); },
        stingIn() { stinger.hidden = false; stinger.classList.remove('out'); void stinger.offsetWidth; stinger.classList.add('in'); },
        stingOut() { stinger.hidden = false; stinger.classList.remove('in'); void stinger.offsetWidth; stinger.classList.add('out'); },
        stingCovered() {
            stinger.hidden = false;
            stinger.classList.remove('in', 'out');
            stinger.classList.add('in');
            for (const a of stinger.getAnimations({ subtree: true })) a.finish();
        },
        // The page's own logo wipe, at progress p from 0 to 1, covering from about 0.3 to 0.7.
        wipe(p, label) {
            const w = cuju.transition.wipe;
            if (!promo.wipeHome) {
                promo.wipeHome = w.root.parentNode;
                layer.insertBefore(w.root, stinger);
                w.root.style.zIndex = 20;
            }
            if (cuju.styles) w.setTeams(cuju.styles);
            w.show(p, label || '');
        },
        async match(title) {
            const list = await (await fetch('/api/matches')).json();
            const m = list.find((x) => x.title === title);
            if (!m) throw new Error('no recording titled ' + title);
            return m.id;
        },
        async field(preset, changes) {
            const { presetField, cleanField } = await import('./js/field.js');
            cuju.applyField(cleanField({ ...presetField(preset || 'Summer afternoon'), ...(changes || {}) }));
        },
        async open(title, seconds) {
            await cuju.openMatch(await promo.match(title));
            for (let i = 0; i < 200 && !(cuju.players && cuju.timeline); i++) await new Promise((r) => setTimeout(r, 50));
            // The team sheets close at once, without their wipe, and give the camera back.
            cuju.director.closeLineups(false);
            if (seconds != null) {
                cuju.timeline.seek(cuju.timeline.first + seconds * cuju.timeline.sps);
                cuju.afterSeek();
            }
            // Play starts with the capture, so a scene opens where its storyboard puts it.
            cuju.timeline.pause();
            promo.held = true;
        },
        go() {
            if (promo.held) {
                promo.held = false;
                cuju.timeline.play(1);
            }
        },
        events() {
            return cuju.transition.events.filter((e) => e.goal || e.decision).map((e) => ({
                index: e.index, seconds: (e.index - cuju.timeline.first) / cuju.timeline.sps,
                goal: e.goal, team: e.goalTeam, decision: e.decision ? e.decision.type : null,
            }));
        },
        async styles(changes) {
            const h = cuju.source.header;
            changes.forEach((c, side) => {
                if (c) h.teams[side] = { ...h.teams[side], ...c, style: { ...(h.teams[side].style || {}), ...(c.style || {}) } };
            });
            await cuju.rebuildPlayers();
        },
        async team(name) {
            const { api } = await import('./js/api.js');
            let t = (await api.teams()).find((x) => x.name === name);
            if (!t) {
                t = await api.uploadTeam(name, 'agent.py', new Blob(['def agent(observation):\n    return [0]\n']), '11v11');
                const kit = { pattern: 'plain', primary: [255, 111, 0], secondary: [32, 33, 36], shorts: [32, 33, 36], socks: [255, 111, 0] };
                t = await api.updateTeam(t.id, { style: { kit, crowd: { colors: [kit.primary, kit.secondary] } } });
            }
            return t;
        },
        async editor(name, section) {
            const t = await promo.team(name);
            cuju.showPage('teams');
            await cuju.library.refresh();
            const entry = cuju.library.teams.find((x) => x.id === t.id);
            await cuju.editor.open(entry);
            if (section) cuju.editor.showSection(section);
            // The line under the team's name counts its players, and the montage shows no numbers.
            if (cuju.editor.titleMeta) cuju.editor.titleMeta.style.visibility = 'hidden';
            document.activeElement && document.activeElement.blur();
            return entry;
        },
    };
})();
