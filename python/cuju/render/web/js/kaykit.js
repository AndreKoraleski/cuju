// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The animation clips of Kay Lousberg's KayKit Character Animations, made for Rig_Medium, the
// rig that every shipped KayKit character shares. The files load once, the first time a model on
// that rig is used, and their clips join every such rig, so a player plays them by name.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const PACKS = ['general', 'movementbasic', 'movementadvanced', 'simulation', 'special', 'combatmelee'];
const URL_OF = (pack) => `assets/animations/kaykit-rig-medium-${pack}.glb`;

// Clips with nothing to show on a football pitch, or that only reset the rig.
const SKIPPED = /^(T-Pose|EXPERIMENTAL_.*)$/;

// Bones every model on Rig_Medium carries, how a rig of the family is told apart.
const FAMILY_BONES = ['root', 'hips', 'spine', 'chest', 'head', 'upperarml', 'upperarmr', 'upperlegl', 'upperlegr'];

let library = null;

// Every clip of the packs that loaded, a pack that fails is left out quietly.
export function kaykitClips() {
    if (!library) {
        library = Promise.all(PACKS.map(async (pack) => {
            try {
                const gltf = await new GLTFLoader().loadAsync(URL_OF(pack));
                return (gltf.animations || []).filter((c) => c && c.name && !SKIPPED.test(c.name));
            } catch {
                return [];
            }
        })).then((lists) => {
            const seen = new Set();
            const out = [];
            for (const clip of lists.flat()) {
                if (!seen.has(clip.name)) {
                    seen.add(clip.name);
                    out.push(clip);
                }
            }
            return out;
        });
    }
    return library;
}

// The names of the objects of a scene, as GLTFLoader leaves them.
function namesOf(scene) {
    const names = new Set();
    scene.traverse((obj) => {
        if (obj.name) {
            names.add(obj.name);
        }
    });
    return names;
}

// Whether a scene is a model on Rig_Medium.
export function isKayKitScene(scene) {
    if (!scene) {
        return false;
    }
    const names = namesOf(scene);
    return names.has('Rig_Medium') || FAMILY_BONES.every((n) => names.has(n));
}

// Adds the KayKit clips to a rig on Rig_Medium, each cut down to the tracks of bones the model
// has, so a model without hand slots binds every track it keeps. A clip the model's own file
// carries under the same name stays. Resolves to the rig, with or without the clips.
export async function attachKayKitClips(rig) {
    if (!rig || !rig.scene || !Array.isArray(rig.clips) || rig.kaykit || !isKayKitScene(rig.scene)) {
        return rig;
    }
    rig.kaykit = true;
    try {
        const clips = await kaykitClips();
        const names = namesOf(rig.scene);
        const own = new Set(rig.clips.map((c) => c.name));
        for (const clip of clips) {
            if (own.has(clip.name)) {
                continue;
            }
            const tracks = clip.tracks.filter((track) => {
                const node = THREE.PropertyBinding.parseTrackName(track.name).nodeName;
                return names.has(node);
            });
            if (tracks.length) {
                rig.clips.push(new THREE.AnimationClip(clip.name, clip.duration, tracks));
            }
        }
    } catch {
        // The procedural moves stand in for any clip that did not load.
    }
    return rig;
}

// Clips for each role on a KayKit model, the first one the model has wins. A name with a greater
// than sign plays its first clip once and then loops the second.
export const KAYKIT_ROLES = {
    celebrate: ['clip:Cheering', 'clip:Jump_Full_Long', 'clip:Skeletons_Taunt_Longer', 'clip:Jump_Full_Short', 'clip:Waving', 'clip:Push_Ups'],
    jump: ['clip:Jump_Full_Short', 'clip:Jump_Full_Long', 'clip:Cheering'],
    cheer: ['clip:Cheering', 'clip:Jump_Full_Short'],
    taunt: ['clip:Skeletons_Taunt_Longer', 'clip:Skeletons_Taunt', 'clip:Cheering'],
    showoff: ['clip:Push_Ups', 'clip:Skeletons_Taunt'],
    wave: ['clip:Waving', 'clip:Cheering'],
    huddle: ['clip:Cheering', 'clip:Idle_B'],
    idle: ['clip:Idle_A', 'clip:Idle_B'],
    calm: ['clip:Idle_B', 'clip:Idle_A'],
    dejected: ['clip:Sit_Floor_Down>Sit_Floor_Idle', 'clip:Crouching', 'clip:Idle_B'],
    crouch: ['clip:Crouching', 'clip:Idle_B'],
    floored: ['clip:Lie_Down>Lie_Idle', 'clip:Sit_Floor_Down>Sit_Floor_Idle'],
    handshake: ['clip:Interact', 'clip:Idle_A'],
    punch: ['clip:Melee_Unarmed_Attack_Punch_A', 'clip:Cheering'],
    kick: ['clip:Melee_Unarmed_Attack_Kick', 'clip:Cheering'],
};

// The celebrations a KayKit scorer picks from when his team leaves the choice open, with the
// air punch and the kick of the combat pack for a little more swagger.
export const KAYKIT_CELEBRATIONS = [
    'clip:Cheering', 'clip:Jump_Full_Long', 'clip:Skeletons_Taunt_Longer', 'clip:Push_Ups', 'clip:Waving',
    'clip:Jump_Full_Short', 'clip:Melee_Unarmed_Attack_Punch_A', 'clip:Skeletons_Taunt',
];

// The KayKit clips the team editor offers as celebrations, with their names in the editor.
export const KAYKIT_MENU = [
    ['Cheering', 'Cheering'],
    ['Waving', 'Waving to the fans'],
    ['Jump_Full_Short', 'Jump'],
    ['Jump_Full_Long', 'Big jump'],
    ['Skeletons_Taunt', 'Taunt'],
    ['Skeletons_Taunt_Longer', 'Long taunt'],
    ['Push_Ups', 'Push ups'],
    ['Sit_Ups', 'Sit ups'],
    ['Melee_Unarmed_Attack_Punch_A', 'Air punch'],
    ['Melee_Unarmed_Attack_Kick', 'Kick at the air'],
    ['Dodge_Forward', 'Forward roll'],
    ['Sit_Floor_Down>Sit_Floor_Idle', 'Sit on the grass'],
    ['Lie_Down>Lie_Idle', 'Lie on the grass'],
    ['Idle_B', 'Stand cool'],
];

const LABELS = new Map(KAYKIT_MENU);

// The editor's name of a KayKit clip, or null for another name.
export function kaykitLabel(name) {
    return LABELS.get(String(name || '').replace(/^clip:/, '')) || null;
}

// The clips the editor lists for a rig, by name without the clip prefix. A KayKit rig lists the
// menu above, then the clips of its own file, any other rig every clip it has.
export function menuClips(rig) {
    if (!rig || !Array.isArray(rig.clips)) {
        return [];
    }
    if (!rig.kaykit) {
        return rig.clips.map((c) => c.name);
    }
    const has = (name) => name.split('>').every((part) => rig.clips.some((c) => c.name === part));
    const own = ((rig.options && rig.options.animations) || []).map((c) => c && c.name).filter(Boolean);
    return [...KAYKIT_MENU.map(([name]) => name).filter(has), ...own.filter((n) => !LABELS.has(n))];
}
