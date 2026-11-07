// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Team artwork from the server, the flat shirt front and back, and the four sponsor boards.

import { api } from './api.js';

export const SHIRT_SLOTS = ['shirt-front', 'shirt-back'];
export const SPONSOR_SLOTS = ['sponsor-1', 'sponsor-2', 'sponsor-3', 'sponsor-4'];
export const SPONSOR_SIZE = { width: 1024, height: 128 };
// The shirt panels cover half the kit texture across and 0.41 of it down, in kit UV space.
export const SHIRT_SIZE = { width: 512, height: 420 };

const cache = new Map();
let generation = 0;

// Forgets loaded artwork, after an edit.
export function invalidateArt() {
    generation++;
    cache.clear();
}

async function bitmap(id, slot) {
    try {
        return await createImageBitmap(await api.image(id, slot));
    } catch (err) {
        console.warn('team image not loaded', id, slot, err);
        return null;
    }
}

// Returns the front, back and sponsors of a registry team, missing pieces null or left out.
export function loadTeamArt(team) {
    if (!team || !Array.isArray(team.images) || team.images.length === 0) {
        return Promise.resolve({ front: null, back: null, sponsors: [] });
    }
    const key = `${generation}:${team.id}:${team.images.slice().sort().join(',')}`;
    if (!cache.has(key)) {
        const has = (slot) => team.images.includes(slot);
        cache.set(key, (async () => {
            const [front, back, ...sponsors] = await Promise.all([
                has('shirt-front') ? bitmap(team.id, 'shirt-front') : null,
                has('shirt-back') ? bitmap(team.id, 'shirt-back') : null,
                ...SPONSOR_SLOTS.map((s) => (has(s) ? bitmap(team.id, s) : null)),
            ]);
            return { front, back, sponsors: sponsors.filter(Boolean) };
        })());
    }
    return cache.get(key);
}
