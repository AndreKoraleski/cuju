// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Player figures. Skinned glTF models are retargeted onto the 14 nodes saved in each frame, the
// default assets/player.glb or a team's own model. Without a model, a figure of capsules and
// spheres rides the same nodes.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import { NODE_COUNT, NODE_NAMES, NODE_PARENTS, PLAYER_FLOATS } from './format.js';
import {
    ciede2000, hex, kitTexture, numberColor, paintKitBase, playerLook, shirtTexture, threeColor,
} from './kits.js';
import { Recolor, kitPairs, shownColors } from './recolor.js';
import { fallbackCelebration, has as hasAnimation, pickCelebration } from './animate.js';
import { attachKayKitClips } from './kaykit.js';

// The largest team model read, the server takes at most 30 MB so this only stops a runaway file.
const MAX_MODEL_BYTES = 64 * 1024 * 1024;

// Local rest positions from GRF's player.object. GRF's animations replace every local rotation,
// so the rest pose, the bind pose of player.glb, has identity rotations everywhere.
const REST_LOCAL = [
    [0, 0, 0], [0, 0, 0.96], [0, 0, 0.15], [0, -0.03, 0.5], [0.16, -0.01, 0.48], [-0.01, 0, -0.33],
    [-0.16, -0.01, 0.48], [0.01, 0, -0.33], [0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
    [-0.087, 0, -0.01], [0, 0, -0.42], [0, -0.04, -0.44],
];

const ONE = new THREE.Vector3(1, 1, 1);
const ZERO_SCALE = new THREE.Matrix4().makeScale(0, 0, 0);
const tmpPos = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const tmpMat = new THREE.Matrix4();
const blendMat = new THREE.Matrix4();
const blendMat2 = new THREE.Matrix4();
const blendPa = new THREE.Vector3();
const blendPb = new THREE.Vector3();
const blendQa = new THREE.Quaternion();
const blendQb = new THREE.Quaternion();
const blendSa = new THREE.Vector3();
const blendSb = new THREE.Vector3();

// Rest world transforms in the engine frame, as matrices.
function restWorldEngine() {
    const world = [];
    for (let i = 0; i < NODE_COUNT; i++) {
        const local = new THREE.Matrix4().makeTranslation(...REST_LOCAL[i]);
        world.push(NODE_PARENTS[i] < 0 ? local : world[NODE_PARENTS[i]].clone().multiply(local));
    }
    return world;
}

// GRF's reference height and the tallest a model is drawn, 1.25 times that.
const GRF_HEIGHT = 1.92;
const MAX_MODEL_HEIGHT = 1.25 * GRF_HEIGHT;

const ENGINE_TO_THREE = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
const THREE_TO_ENGINE = ENGINE_TO_THREE.clone().invert();
const REST_ENGINE = restWorldEngine();
// Rest rotations in three's frame, the G of the retargeting.
const REST_THREE_QUAT = REST_ENGINE.map((w) => {
    const m = ENGINE_TO_THREE.clone().multiply(w).multiply(THREE_TO_ENGINE);
    const q = new THREE.Quaternion();
    m.decompose(new THREE.Vector3(), q, new THREE.Vector3());
    return q;
});
const REST_BODY_HEIGHT = new THREE.Vector3().setFromMatrixPosition(REST_ENGINE[1]).z;

// A pose buffer of one player in the rest pose, at the origin facing engine minus y.
export function restPoseNodes(heading = 0, out = new Float32Array(PLAYER_FLOATS)) {
    const yaw = new THREE.Matrix4().makeRotationZ(heading);
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    for (let i = 0; i < NODE_COUNT; i++) {
        yaw.clone().multiply(REST_ENGINE[i]).decompose(p, q, s);
        out.set([p.x, p.y, p.z, q.x, q.y, q.z, q.w], i * 7);
    }
    return out;
}

// A procedural walk cycle in the engine frame for the team editor's preview. Every joint turns
// about its own x axis and the stride grows with speed up to 6 meters per second.
export function walkPoseNodes(phase, speed, heading = 0, out = new Float32Array(PLAYER_FLOATS)) {
    const stride = Math.min(1, speed / 6);
    const s = Math.sin(phase);
    const c = Math.cos(phase);
    const bob = 0.03 * stride * Math.abs(c);
    const lean = -0.12 * stride;
    const angles = [0, lean, lean * 0.5, -lean, 0.5 * stride * s, -0.5 - 0.4 * stride,
        -0.5 * stride * s, -0.5 - 0.4 * stride, -0.55 * stride * s,
        0.15 + 0.9 * stride * Math.max(0, -c), -0.2 * stride * s, 0.55 * stride * s,
        0.15 + 0.9 * stride * Math.max(0, c), 0.2 * stride * s];
    const world = [];
    const xAxis = new THREE.Vector3(1, 0, 0);
    for (let i = 0; i < NODE_COUNT; i++) {
        const q = new THREE.Quaternion().setFromAxisAngle(xAxis, angles[i]);
        const pos = new THREE.Vector3(...REST_LOCAL[i]);
        if (i === 1) {
            pos.z -= bob;
        }
        const local = new THREE.Matrix4().compose(pos, q, ONE);
        if (i === 0) {
            local.makeRotationZ(heading);
        }
        world.push(NODE_PARENTS[i] < 0 ? local : world[NODE_PARENTS[i]].clone().multiply(local));
    }
    const v = new THREE.Vector3();
    const r = new THREE.Quaternion();
    const sc = new THREE.Vector3();
    for (let i = 0; i < NODE_COUNT; i++) {
        world[i].decompose(v, r, sc);
        out.set([v.x, v.y, v.z, r.x, r.y, r.z, r.w], i * 7);
    }
    return out;
}

// Three.js world matrix of one node from the pose buffer.
function nodeMatrix(nodes, offset, out) {
    tmpPos.set(nodes[offset], nodes[offset + 2], -nodes[offset + 1]);
    nodeQuaternion(nodes, offset, tmpQuat);
    return out.compose(tmpPos, tmpQuat, ONE);
}

function nodeQuaternion(nodes, offset, out) {
    out.set(nodes[offset + 3], nodes[offset + 5], -nodes[offset + 4], nodes[offset + 6]);
    if (out.lengthSq() < 1e-8) {
        out.identity();
    }
    return out;
}

// One part of the simple figure, with its node, geometry, colour slot and local transform in
// the node's three.js axes, x to the player's left, y up the body or limb and z forward.
function figureParts() {
    const capsule = (r, len) => new THREE.CapsuleGeometry(r, len, 4, 12);
    const sphere = (r) => new THREE.SphereGeometry(r, 16, 12);
    const part = (node, geometry, pos, slot, scale = [1, 1, 1], extra = {}) => ({
        node, geometry, slot, ...extra,
        local: new THREE.Matrix4().compose(new THREE.Vector3(...pos), new THREE.Quaternion(), new THREE.Vector3(...scale)),
    });
    const numberPlane = new THREE.PlaneGeometry(0.17, 0.17);
    numberPlane.rotateY(Math.PI);
    const parts = [
        part(1, sphere(0.15), [0, 0.02, 0], 'shorts', [1.15, 0.75, 0.8]),
        part(2, capsule(0.13, 0.26), [0, 0.25, 0.01], 'shirt', [1.25, 1, 0.78], { textured: true }),
        part(2, numberPlane, [0, 0.31, -0.098], 'number', [1, 1, 1], { numbered: true }),
        part(3, new THREE.CylinderGeometry(0.05, 0.055, 0.12, 10), [0, -0.01, 0.01], 'skin'),
        part(3, sphere(0.105), [0, 0.12, 0.015], 'skin', [0.95, 1.08, 1.05]),
        part(3, sphere(0.11), [0, 0.15, -0.005], 'hair', [1, 0.8, 1.05], { hair: ['short', 'long'] }),
        part(3, sphere(0.1), [0, 0.07, -0.05], 'hair', [1.02, 1.4, 0.7], { hair: ['long'] }),
    ];
    for (const [shoulder, elbow] of [[4, 5], [6, 7]]) {
        parts.push(part(shoulder, capsule(0.056, 0.2), [0, -0.14, 0], 'shirt'));
        parts.push(part(elbow, capsule(0.042, 0.2), [0, -0.13, 0], 'skin'));
        parts.push(part(elbow, sphere(0.05), [0, -0.3, 0.005], 'skin', [0.8, 1.2, 1]));
    }
    for (const [thigh, knee, ankle] of [[8, 9, 10], [11, 12, 13]]) {
        parts.push(part(thigh, new THREE.CylinderGeometry(0.095, 0.085, 0.2, 12), [0, -0.08, 0], 'shorts'));
        parts.push(part(thigh, capsule(0.072, 0.26), [0, -0.23, 0], 'skin'));
        parts.push(part(knee, capsule(0.058, 0.3), [0, -0.22, 0.02], 'socks'));
        parts.push(part(ankle, new THREE.BoxGeometry(0.095, 0.075, 0.26), [0, -0.04, 0.06], 'boots'));
    }
    return parts;
}

// Shirt numbers 0 to 99 in a 10 by 10 grid, the cell of a number is the number itself. White on
// transparent, tinted by the material colour, and shared by every figure, so never disposed.
const ATLAS_SIDE = 10;
let numberAtlas = null;
function numberAtlasTexture() {
    if (numberAtlas) {
        return numberAtlas;
    }
    const cell = 64;
    const canvas = document.createElement('canvas');
    canvas.width = cell * ATLAS_SIDE;
    canvas.height = cell * ATLAS_SIDE;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.font = `800 ${cell * 0.78}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let n = 0; n < ATLAS_SIDE * ATLAS_SIDE; n++) {
        const i = n % ATLAS_SIDE;
        const j = Math.floor(n / ATLAS_SIDE);
        ctx.fillText(String(n), i * cell + cell / 2, j * cell + cell / 2 + 3, cell * 0.92);
    }
    numberAtlas = new THREE.CanvasTexture(canvas);
    numberAtlas.colorSpace = THREE.SRGBColorSpace;
    return numberAtlas;
}

// Draws the numbers from the atlas, the instance attribute picks the cell. The canvas is flipped
// on upload, so the first row of cells sits at the top of v.
function numberMaterial(color) {
    const mat = new THREE.MeshStandardMaterial({
        map: numberAtlasTexture(), color, transparent: true, alphaTest: 0.4, roughness: 0.6,
        polygonOffset: true, polygonOffsetFactor: -2, depthWrite: false,
    });
    mat.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nattribute float aNumber;')
            .replace('#include <uv_vertex>', `#include <uv_vertex>
                vMapUv = vec2((uv.x + mod(aNumber, ${ATLAS_SIDE}.0)) / ${ATLAS_SIDE}.0,
                    (uv.y + (${ATLAS_SIDE - 1}.0 - floor(aNumber / ${ATLAS_SIDE}.0))) / ${ATLAS_SIDE}.0);`);
    };
    mat.customProgramCacheKey = () => 'cuju-number';
    return mat;
}

// How much of a player is hidden, for a camera inside or just behind its head.
const HIDE_NONE = 0;
const HIDE_HEAD = 1;
const HIDE_ALL = 2;
const HIDE_MODES = { none: HIDE_NONE, head: HIDE_HEAD, all: HIDE_ALL };

// The eyes of GRF's player from its neck node, in the engine frame, forward is minus y.
const GRF_EYE = new THREE.Vector3(0, -0.1, 0.15);
const tmpEye = new THREE.Vector3();

// The middle of the head of GRF's player from its neck node and its radius, for a player whose
// head was not measured.
const GRF_HEAD = { local: new THREE.Vector3(0, -0.02, 0.11), radius: 0.16 };
// The room left between a head and its glass helmet, in meters.
const HELMET_MARGIN = 0.04;

// The eyes of a player from the neck node of a pose, scaled about its ground point, in three's
// frame, or another point given in the neck node's frame.
function neckEye(nodes, base, scale, out, offset = GRF_EYE) {
    const o = base + 3 * 7;
    tmpQuat.set(nodes[o + 3], nodes[o + 4], nodes[o + 5], nodes[o + 6]);
    if (tmpQuat.lengthSq() < 1e-8) {
        tmpQuat.identity();
    }
    tmpEye.copy(offset).applyQuaternion(tmpQuat);
    const gx = nodes[base];
    const gy = nodes[base + 1];
    const x = gx + (nodes[o] + tmpEye.x - gx) * scale;
    const y = gy + (nodes[o + 1] + tmpEye.y - gy) * scale;
    const z = (nodes[o + 2] + tmpEye.z) * scale;
    return out.set(x, z, -y);
}

// The simple figures of some players, grouped by kit so each group shares one shirt texture.
class Figures {
    constructor(parent, header, styles, members) {
        this.group = new THREE.Group();
        parent.add(this.group);
        this.parts = figureParts();
        this.kits = [];
        const byKit = new Map();
        for (const k of members) {
            const p = header.players[k];
            const key = `${p.team}:${p.goalkeeper ? 1 : 0}`;
            if (!byKit.has(key)) {
                byKit.set(key, []);
            }
            byKit.get(key).push(k);
        }
        const plain = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0 });
        this.disposables = [plain];
        for (const [key, list] of byKit) {
            const [team, gk] = key.split(':').map(Number);
            const style = styles[team] || styles[0];
            const shirt = new THREE.MeshStandardMaterial({ map: shirtTexture(style, !!gk), roughness: 0.7 });
            const number = numberMaterial(numberColor(style, !!gk));
            this.disposables.push(shirt, shirt.map, number);
            const looks = list.map((k) => playerLook(style, header.players[k], k));
            const meshes = this.parts.map((part) => {
                let geometry = part.geometry;
                let material = plain;
                if (part.textured) {
                    material = shirt;
                } else if (part.numbered) {
                    material = number;
                    geometry = part.geometry.clone();
                    const numbers = new Float32Array(list.length);
                    looks.forEach((l, i) => {
                        numbers[i] = Math.min(99, Math.max(0, Math.round(l.number) || 0));
                    });
                    geometry.setAttribute('aNumber', new THREE.InstancedBufferAttribute(numbers, 1));
                    this.disposables.push(geometry);
                }
                const mesh = new THREE.InstancedMesh(geometry, material, list.length);
                mesh.castShadow = !part.numbered;
                mesh.receiveShadow = true;
                mesh.frustumCulled = false;
                mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
                if (!part.textured && !part.numbered) {
                    looks.forEach((l, i) => mesh.setColorAt(i, l[part.slot]));
                    mesh.instanceColor.needsUpdate = true;
                }
                this.group.add(mesh);
                return mesh;
            });
            this.kits.push({ list, looks, meshes });
        }
        for (const p of this.parts) {
            this.disposables.push(p.geometry);
        }
        this.node = new THREE.Matrix4();
    }

    update(nodes, visible, hidden) {
        for (const kit of this.kits) {
            kit.list.forEach((k, i) => {
                const base = k * PLAYER_FLOATS;
                const look = kit.looks[i];
                const mode = hidden ? hidden[k] : 0;
                for (let j = 0; j < this.parts.length; j++) {
                    const part = this.parts[j];
                    if (!visible[k] || mode === HIDE_ALL || (mode === HIDE_HEAD && part.node === 3)
                        || (part.hair && !part.hair.includes(look.hairStyle))) {
                        kit.meshes[j].setMatrixAt(i, ZERO_SCALE);
                        continue;
                    }
                    nodeMatrix(nodes, base + part.node * 7, this.node);
                    tmpMat.multiplyMatrices(this.node, part.local);
                    kit.meshes[j].setMatrixAt(i, tmpMat);
                }
            });
            for (const mesh of kit.meshes) {
                mesh.instanceMatrix.needsUpdate = true;
            }
        }
    }

    eye(nodes, k, out) {
        return neckEye(nodes, k * PLAYER_FLOATS, 1, out);
    }

    helmet(nodes, k, out) {
        neckEye(nodes, k * PLAYER_FLOATS, 1, out, GRF_HEAD.local);
        return GRF_HEAD.radius + HELMET_MARGIN;
    }

    headBone() {
        return null;
    }

    size() {
        return { radius: 0.32, height: 1.85 };
    }

    dispose() {
        this.group.removeFromParent();
        for (const d of this.disposables) {
            d.dispose();
        }
        for (const kit of this.kits) {
            for (const m of kit.meshes) {
                m.dispose();
            }
        }
    }
}

const PATTERN_INDEX = { plain: 0, stripes: 1, hoops: 2, halves: 3, sash: 4, checkers: 5 };

// Tints a model material, the whole of it one kit slot, with the team kit. Shirt patterns are
// drawn from the position in the bind pose, so they need no particular UV layout.
function applyKit(material, slot, kit, unit) {
    const uniforms = {
        kitPrimary: { value: threeColor(kit.primary) },
        kitSecondary: { value: threeColor(kit.secondary) },
        kitShorts: { value: threeColor(kit.shorts) },
        kitSocks: { value: threeColor(kit.socks) },
        kitPattern: { value: PATTERN_INDEX[kit.pattern] || 0 },
        kitSlot: { value: { shirt: 1, shorts: 2, socks: 3 }[slot] || 0 },
        kitUnit: { value: unit },
    };
    material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nvarying vec3 vKitPos;\nuniform float kitUnit;')
            .replace('#include <begin_vertex>', '#include <begin_vertex>\nvKitPos = position * kitUnit;');
        shader.fragmentShader = shader.fragmentShader
            .replace('#include <common>', `#include <common>
                varying vec3 vKitPos;
                uniform vec3 kitPrimary;
                uniform vec3 kitSecondary;
                uniform vec3 kitShorts;
                uniform vec3 kitSocks;
                uniform float kitPattern;
                uniform float kitSlot;
                vec3 kitShirtColor(vec3 p) {
                    float m = 0.0;
                    if (kitPattern == 1.0) m = step(0.5, fract(p.x * 9.0));
                    else if (kitPattern == 2.0) m = step(0.5, fract(p.y * 7.0));
                    else if (kitPattern == 3.0) m = step(p.x, 0.0);
                    else if (kitPattern == 4.0) m = step(abs(p.x + p.y - 1.3), 0.07);
                    else if (kitPattern == 5.0) m = mod(floor(p.x * 9.0) + floor(p.y * 9.0), 2.0);
                    return mix(kitPrimary, kitSecondary, m);
                }`)
            .replace('#include <map_fragment>', `#include <map_fragment>
                {
                    float lum = clamp(dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114)) * 1.25, 0.0, 1.0);
                    if (kitSlot == 1.0) diffuseColor.rgb = kitShirtColor(vKitPos) * lum;
                    else if (kitSlot == 2.0) diffuseColor.rgb = kitShorts * lum;
                    else if (kitSlot == 3.0) diffuseColor.rgb = kitSocks * lum;
                }`);
    };
    material.customProgramCacheKey = () => 'cuju-kit-slot';
}

function kitSlotOf(name, named, onlyKit) {
    const lower = (name || '').toLowerCase();
    if (named) {
        for (const slot of ['shirt', 'shorts', 'socks', 'skin', 'hair', 'boots']) {
            if ([].concat(named[slot] || []).includes(name)) {
                return slot;
            }
        }
    }
    if (/shirt|jersey/.test(lower)) return 'shirt';
    if (/short/.test(lower)) return 'shorts';
    if (/sock/.test(lower)) return 'socks';
    if (onlyKit) return null;
    if (/skin/.test(lower)) return 'skin';
    if (/hair/.test(lower)) return 'hair';
    if (/boot|shoe/.test(lower)) return 'boots';
    return null;
}

// Common rig names, used to fill a joint mapping. Mixamo first, then a few other conventions.
const RIG_NAMES = {
    player: ['root', 'Root', 'Armature'],
    body: ['Hips', 'hips', 'pelvis', 'Pelvis'],
    middle: ['Spine2', 'Spine1', 'spine_02', 'Chest', 'chest', 'Spine', 'spine'],
    neck: ['Neck', 'neck', 'neck_01', 'Head', 'head'],
    left_shoulder: ['LeftArm', 'upperarm_l', 'UpperArm.L', 'upper_arm.L', 'LeftUpperArm'],
    left_elbow: ['LeftForeArm', 'lowerarm_l', 'LowerArm.L', 'forearm.L', 'LeftLowerArm'],
    right_shoulder: ['RightArm', 'upperarm_r', 'UpperArm.R', 'upper_arm.R', 'RightUpperArm'],
    right_elbow: ['RightForeArm', 'lowerarm_r', 'LowerArm.R', 'forearm.R', 'RightLowerArm'],
    left_thigh: ['LeftUpLeg', 'thigh_l', 'UpperLeg.L', 'thigh.L', 'LeftUpperLeg'],
    left_knee: ['LeftLeg', 'calf_l', 'LowerLeg.L', 'shin.L', 'LeftLowerLeg'],
    left_ankle: ['LeftFoot', 'foot_l', 'Foot.L', 'foot.L'],
    right_thigh: ['RightUpLeg', 'thigh_r', 'UpperLeg.R', 'thigh.R', 'RightUpperLeg'],
    right_knee: ['RightLeg', 'calf_r', 'LowerLeg.R', 'shin.R', 'RightLowerLeg'],
    right_ankle: ['RightFoot', 'foot_r', 'Foot.R', 'foot.R'],
};

function stripPrefix(name) {
    return name.replace(/^mixamorig\d*[:_]?/i, '').replace(/^.*[:|]/, '');
}

// Lists the skinned bones of a model, or throws when it has no skin.
export function skinnedBones(scene) {
    const bones = [];
    const seen = new Set();
    scene.traverse((obj) => {
        if (obj.isSkinnedMesh && obj.skeleton) {
            for (const b of obj.skeleton.bones) {
                if (!seen.has(b)) {
                    seen.add(b);
                    bones.push(b);
                }
            }
        }
    });
    if (bones.length === 0) {
        throw new Error('This model has no skin, so it cannot be animated. Export it with its armature and skin weights.');
    }
    return bones;
}

// The skinned bone names as written in the file, GLTFLoader renames nodes for its own use.
export function fileBoneNames(gltf) {
    const json = gltf.parser && gltf.parser.json;
    return skinnedBones(gltf.scene).map((b) => {
        const assoc = gltf.parser && gltf.parser.associations.get(b);
        const index = assoc ? assoc.nodes : undefined;
        return json && index !== undefined && json.nodes[index] && json.nodes[index].name ? json.nodes[index].name : b.name;
    });
}

// Fills a joint mapping by exact names first, then common rig names.
export function guessJoints(boneNames) {
    const joints = {};
    const exact = new Set(boneNames);
    const stripped = new Map(boneNames.map((n) => [stripPrefix(n).toLowerCase(), n]));
    for (const node of NODE_NAMES) {
        if (exact.has(node)) {
            joints[node] = node;
            continue;
        }
        for (const candidate of RIG_NAMES[node]) {
            const hit = stripped.get(candidate.toLowerCase());
            if (hit) {
                joints[node] = hit;
                break;
            }
        }
    }
    return joints;
}

// A skinned model retargeted onto the 14 nodes. A mapped bone's world rotation is the saved
// rotation times the inverse of GRF's rest rotation times the bone's own bind rotation, so the
// model's bind pose need not match GRF's. The pelvis follows the saved body position with its
// height scaled to the model, every other bone keeps its own length from its parent, and
// unmapped bones keep their bind transform relative to their parent. The rig only reads its
// scene, which stays with whoever loaded it.
export class Rig {
    constructor(scene, joints, options = {}) {
        this.scene = scene;
        this.options = options;
        scene.updateMatrixWorld(true);
        const bones = skinnedBones(scene);
        const inverse = new Map();
        scene.traverse((obj) => {
            if (obj.isSkinnedMesh && obj.skeleton) {
                obj.skeleton.bones.forEach((b, i) => inverse.set(b, obj.skeleton.boneInverses[i]));
            }
        });
        const bind = new Map(bones.map((b) => [b, inverse.has(b) ? inverse.get(b).clone().invert() : b.matrixWorld.clone()]));
        const nodeOf = new Map();
        for (const [node, boneName] of Object.entries(joints || {})) {
            const i = NODE_NAMES.indexOf(node);
            // GLTFLoader drops dots and similar from node names, mappings use the file's names.
            const clean = THREE.PropertyBinding.sanitizeNodeName(String(boneName));
            const bone = bones.find((b) => b.name === boneName) || bones.find((b) => b.name === clean);
            if (i >= 0 && bone) {
                nodeOf.set(bone, i);
            }
        }
        const bodyBone = bones.find((b) => nodeOf.get(b) === 1);
        if (!bodyBone) {
            throw new Error('The joint mapping has no bone for body, the pelvis.');
        }
        // Order bones parents first, and find the skeleton roots.
        const depth = (b) => {
            let d = 0;
            for (let p = b.parent; p; p = p.parent) {
                d++;
            }
            return d;
        };
        const ordered = bones.slice().sort((a, b) => depth(a) - depth(b));
        const index = new Map(ordered.map((b, i) => [b, i]));
        const parentBone = (b) => {
            for (let p = b.parent; p; p = p.parent) {
                if (index.has(p)) {
                    return p;
                }
            }
            return null;
        };
        // A missing player joint means the skeleton root plays the player node.
        if (![...nodeOf.values()].includes(0)) {
            const root = ordered[0];
            if (!nodeOf.has(root)) {
                nodeOf.set(root, 0);
            }
        }
        // A model kept in the engine frame, z up, is turned into three's frame first, glTF's
        // usual y up models need no turn.
        const frame = options.engineFrame ? ENGINE_TO_THREE.clone() : new THREE.Matrix4();
        const box = new THREE.Box3().setFromObject(scene).applyMatrix4(frame);
        const bodyBind = new THREE.Vector3().setFromMatrixPosition(bind.get(bodyBone)).applyMatrix4(frame);
        const floor = Number.isFinite(box.min.y) ? box.min.y : 0;
        let pelvis = bodyBind.y - floor;
        // Units, a model whose pelvis stands above ten units is taken to be in centimeters.
        let unit = Number(options.unit) || (pelvis > 10 ? 0.01 : 1);
        // A model much taller than GRF's player is scaled down to MAX_MODEL_HEIGHT, so big
        // characters keep their look without towering over the others.
        const tall = Number.isFinite(box.max.y) ? (box.max.y - box.min.y) * unit : 0;
        this.fit = tall > MAX_MODEL_HEIGHT ? MAX_MODEL_HEIGHT / tall : 1;
        this.standing = tall > 0.3 ? tall * this.fit : GRF_HEIGHT;
        unit *= this.fit;
        pelvis *= unit;
        if (!(pelvis > 0.2)) {
            pelvis = REST_BODY_HEIGHT;
        }
        this.heightRatio = pelvis / REST_BODY_HEIGHT;
        this.unit = unit;
        const S = new THREE.Matrix4().makeScale(unit, unit, unit).premultiply(frame);
        this.bones = ordered.map((b) => {
            const world = S.clone().multiply(bind.get(b));
            const parent = parentBone(b);
            const parentWorld = parent ? S.clone().multiply(bind.get(parent)) : null;
            const local = parentWorld ? parentWorld.clone().invert().multiply(world) : world.clone();
            const p = new THREE.Vector3();
            const q = new THREE.Quaternion();
            const s = new THREE.Vector3();
            world.decompose(p, q, s);
            const node = nodeOf.has(b) ? nodeOf.get(b) : -1;
            return {
                name: b.name,
                node,
                parent: parent ? index.get(parent) : -1,
                local,
                localOffset: new THREE.Vector3().setFromMatrixPosition(local),
                scale: s,
                // The inverse of G times C, applied after the saved rotation.
                correction: node >= 0 ? REST_THREE_QUAT[node].clone().invert().multiply(q) : null,
                position: p,
                rotation: q,
            };
        });
        this.alignLimbs();
        // Animation clips the file carries, which a player can play in place of the nodes.
        this.clips = Array.isArray(options.animations) ? options.animations.filter((c) => c && c.name) : [];
        this.placement = S;
        this.findHead(Number.isFinite(box.max.y) ? box.max.y * unit : NaN);
    }

    // The head bone, the bone mapped to the neck or a descendant named head, the bones that a
    // camera inside the head hides, and the eyes in the head bone's own frame.
    findHead(top) {
        const neck = this.bones.findIndex((b) => b.node === 3);
        const below = (j, root) => {
            for (let p = this.bones[j].parent; p >= 0; p = this.bones[p].parent) {
                if (p === root) {
                    return true;
                }
            }
            return false;
        };
        let head = neck;
        if (neck >= 0 && !/head/i.test(this.bones[neck].name)) {
            const named = this.bones.findIndex((b, j) => /head/i.test(b.name) && !/end|top|nub/i.test(b.name) && below(j, neck));
            if (named >= 0) {
                head = named;
            }
        }
        this.headIndex = head;
        this.headMask = this.bones.map((b, j) => head >= 0 && (j === head || below(j, head)));
        this.eyeLocal = null;
        if (head >= 0) {
            const hb = this.bones[head];
            const rise = Number.isFinite(top) && top > hb.position.y + 0.05 ? top - hb.position.y : 0.3 * this.heightRatio;
            const eye = new THREE.Vector3(hb.position.x, hb.position.y + 0.45 * rise, hb.position.z + 0.3 * rise);
            this.eyeLocal = eye.applyMatrix4(new THREE.Matrix4().compose(hb.position, hb.rotation, hb.scale).invert());
        }
    }

    // The standing height and the horizontal radius of the model as drawn in GRF's rest pose,
    // from its skinned vertices, measured once.
    size() {
        if (this.measured) {
            return this.measured;
        }
        let radius = 0;
        let top = 0;
        try {
            const inst = this.instantiate();
            this.pose(inst, restPoseNodes(0), 0);
            const v = new THREE.Vector4();
            const w = new THREE.Vector4();
            const m = new THREE.Matrix4();
            inst.root.traverse((obj) => {
                if (!obj.isSkinnedMesh) {
                    return;
                }
                const g = obj.geometry;
                const pos = g.attributes.position;
                const si = g.attributes.skinIndex;
                const sw = g.attributes.skinWeight;
                for (let i = 0; i < pos.count; i++) {
                    v.set(pos.getX(i), pos.getY(i), pos.getZ(i), 1).applyMatrix4(obj.bindMatrix);
                    w.set(0, 0, 0, 0);
                    for (let c = 0; c < 4; c++) {
                        const weight = sw.getComponent(i, c);
                        if (weight) {
                            const b = si.getComponent(i, c);
                            m.multiplyMatrices(obj.skeleton.bones[b].matrixWorld, obj.skeleton.boneInverses[b]);
                            w.addScaledVector(v.clone().applyMatrix4(m), weight);
                        }
                    }
                    radius = Math.max(radius, Math.hypot(w.x, w.z));
                    top = Math.max(top, w.y);
                }
            });
            this.release(inst);
        } catch (err) {
            console.warn('model size not measured', err);
        }
        this.measured = {
            radius: radius > 0.05 ? radius : 0.35,
            height: top > 0.3 ? top : this.standing,
        };
        return this.measured;
    }

    // The sphere round the head, its middle in the head bone's own frame and its radius in
    // meters, from the skinned vertices that follow the head bone or a bone below it and the
    // rigid meshes hung below them, which takes in the hair and anything worn on the head.
    // Measured once in GRF's rest pose.
    headSphere() {
        if (this.headMeasured !== undefined) {
            return this.headMeasured;
        }
        this.headMeasured = null;
        if (this.headIndex < 0) {
            return null;
        }
        try {
            const inst = this.instantiate();
            this.pose(inst, restPoseNodes(0), 0);
            const onHead = new Set(inst.bones.filter((b, j) => b && this.headMask[j]));
            const points = [];
            const box = new THREE.Box3();
            const m = new THREE.Matrix4();
            inst.root.traverse((obj) => {
                if (obj.isMesh && !obj.isSkinnedMesh) {
                    // A rigid mesh worn on the head, such as a hat, hangs below a head bone.
                    const rel = new THREE.Matrix4();
                    let node = obj;
                    while (node && !onHead.has(node)) {
                        node.updateMatrix();
                        rel.premultiply(node.matrix);
                        node = node.parent;
                    }
                    if (node) {
                        m.multiplyMatrices(node.matrixWorld, rel);
                        const pos = obj.geometry.attributes.position;
                        for (let i = 0; i < pos.count; i++) {
                            const p = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(m);
                            points.push(p);
                            box.expandByPoint(p);
                        }
                    }
                    return;
                }
                if (!obj.isSkinnedMesh) {
                    return;
                }
                const g = obj.geometry;
                const pos = g.attributes.position;
                const si = g.attributes.skinIndex;
                const sw = g.attributes.skinWeight;
                for (let i = 0; i < pos.count; i++) {
                    let best = 0;
                    for (let c = 1; c < 4; c++) {
                        if (sw.getComponent(i, c) > sw.getComponent(i, best)) {
                            best = c;
                        }
                    }
                    const bone = obj.skeleton.bones[si.getComponent(i, best)];
                    if (!onHead.has(bone)) {
                        continue;
                    }
                    const b = si.getComponent(i, best);
                    m.multiplyMatrices(bone.matrixWorld, obj.skeleton.boneInverses[b]);
                    const p = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(obj.bindMatrix).applyMatrix4(m);
                    points.push(p);
                    box.expandByPoint(p);
                }
            });
            if (points.length) {
                const center = box.getCenter(new THREE.Vector3());
                let radius = 0;
                for (const p of points) {
                    radius = Math.max(radius, p.distanceTo(center));
                }
                const local = center.clone().applyMatrix4(inst.world[this.headIndex].clone().invert());
                this.headMeasured = { local, radius };
            }
            this.release(inst);
        } catch (err) {
            console.warn('head not measured', err);
        }
        return this.headMeasured;
    }

    // The names of the clips as celebrations name them.
    clipNames() {
        return this.clips.map((c) => `clip:${c.name}`);
    }

    clip(name) {
        const bare = String(name || '').replace(/^clip:/, '');
        return this.clips.find((c) => c.name === bare) || null;
    }

    // The eyes of a posed instance in three's frame.
    eye(instance, out) {
        if (this.headIndex < 0 || !this.eyeLocal) {
            return null;
        }
        return out.copy(this.eyeLocal).applyMatrix4(instance.world[this.headIndex]);
    }

    // Limbs of most rigs rest in a T or A pose while GRF's rest pose hangs the arms down and
    // stands the legs straight. Each mapped limb bone gets the turn that points its bind
    // direction, towards its child bone, along GRF's rest direction of the same limb, so the
    // saved rotations act on a model standing as GRF's player does. The trunk, neck and
    // feet keep their bind turn, which keeps the facing.
    alignLimbs() {
        const restPos = REST_ENGINE.map((w) => new THREE.Vector3().setFromMatrixPosition(w).applyMatrix4(ENGINE_TO_THREE));
        const down = new THREE.Vector3(0, -1, 0);
        const childNode = { 4: 5, 5: null, 6: 7, 7: null, 8: 9, 9: 10, 11: 12, 12: 13 };
        const byNode = new Map(this.bones.map((b, i) => [b.node, i]));
        this.bones.forEach((info, i) => {
            if (!(info.node in childNode)) {
                return;
            }
            const c = childNode[info.node];
            const want = c == null ? down.clone() : restPos[c].clone().sub(restPos[info.node]).normalize();
            let child = c != null && byNode.has(c) ? this.bones[byNode.get(c)] : null;
            if (!child) {
                child = this.bones.find((b) => b.parent === i) || null;
            }
            if (!child) {
                return;
            }
            const have = child.position.clone().sub(info.position);
            if (have.lengthSq() < 1e-10) {
                return;
            }
            const turn = new THREE.Quaternion().setFromUnitVectors(have.normalize(), want);
            info.correction = REST_THREE_QUAT[info.node].clone().invert().multiply(turn).multiply(info.rotation);
        });
    }

    instantiate() {
        const root = SkeletonUtils.clone(this.scene);
        const byName = new Map();
        root.traverse((obj) => {
            if (obj.isBone && !byName.has(obj.name)) {
                byName.set(obj.name, obj);
            }
            if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
                obj.frustumCulled = false;
            }
        });
        const bones = this.bones.map((info) => {
            const bone = byName.get(info.name);
            if (bone) {
                bone.matrixAutoUpdate = false;
                bone.matrixWorldAutoUpdate = false;
            }
            return bone || null;
        });
        // Geometries and textures stay shared with the rig's scene, only the materials dress
        // makes and the cloned skeletons belong to the instance.
        return { root, bones, world: this.bones.map(() => new THREE.Matrix4()), materials: [] };
    }

    release(instance) {
        for (const m of instance.materials) {
            m.dispose();
        }
        disposeSkeletons(instance.root);
    }

    // Poses an instance from one player's nodes. With hideHead the head bone and the bones below
    // it shrink to a point, which hides the head, its hair and anything worn on it.
    pose(instance, nodes, base, hideHead = false) {
        const world = instance.world;
        for (let j = 0; j < this.bones.length; j++) {
            const info = this.bones[j];
            const out = world[j];
            const parent = info.parent >= 0 ? world[info.parent] : null;
            if (info.node < 0) {
                if (parent) {
                    out.multiplyMatrices(parent, info.local);
                } else {
                    out.copy(info.local);
                }
            } else {
                const o = base + info.node * 7;
                nodeQuaternion(nodes, o, tmpQuat).multiply(info.correction);
                if (info.node === 1) {
                    tmpPos.set(nodes[o], nodes[o + 2] * this.heightRatio, -nodes[o + 1]);
                } else if (info.node === 0 || !parent) {
                    tmpPos.set(nodes[o], nodes[o + 2], -nodes[o + 1]);
                } else {
                    tmpPos.copy(info.localOffset).applyMatrix4(parent);
                }
                out.compose(tmpPos, tmpQuat, info.scale);
            }
            const bone = instance.bones[j];
            if (bone) {
                if (hideHead && this.headMask[j]) {
                    const h = world[this.headIndex].elements;
                    bone.matrixWorld.copy(ZERO_SCALE).setPosition(h[12], h[13], h[14]);
                } else {
                    bone.matrixWorld.copy(out);
                }
            }
        }
    }

    // Poses an instance from a clip of the model at time t seconds, standing on the ground point
    // and heading of node player of the pose, blended over the pose of the nodes by weight. A
    // weight of one draws the clip alone. Below one, each bone's turn and offset from its parent
    // blend between the two poses, so limbs keep their lengths while the clip fades in or out.
    // The bones stay driven by hand, so a later pose from the nodes alone replaces it whole.
    poseClip(instance, clip, t, weight, nodes, base, hideHead = false) {
        if (!instance.mixer) {
            instance.mixer = new THREE.AnimationMixer(instance.root);
        }
        if (instance.clip !== clip) {
            instance.mixer.stopAllAction();
            instance.action = instance.mixer.clipAction(clip);
            instance.action.play();
            instance.clip = clip;
        }
        const d = clip.duration || 0;
        instance.action.time = d > 0 ? Math.max(0, Math.min(d - 1e-4, t)) : 0;
        instance.mixer.update(0);
        nodeQuaternion(nodes, base, tmpQuat);
        tmpPos.set(0, 0, 1).applyQuaternion(tmpQuat);
        const heading = Math.atan2(tmpPos.x, tmpPos.z);
        const root = instance.root;
        root.matrixAutoUpdate = false;
        // The player node's height lifts the clip with it, onto a podium step or a stage.
        root.matrix.makeRotationY(heading).setPosition(nodes[base], nodes[base + 2] || 0, -nodes[base + 1]).multiply(this.placement);
        for (const b of instance.bones) {
            if (b) {
                b.matrixAutoUpdate = true;
                b.matrixWorldAutoUpdate = true;
            }
        }
        root.updateMatrixWorld(true);
        for (const b of instance.bones) {
            if (b) {
                b.matrixAutoUpdate = false;
                b.matrixWorldAutoUpdate = false;
            }
        }
        // Skinning reads the bones alone, so the root goes back to where the nodes keep it.
        root.matrix.identity();
        root.matrixAutoUpdate = true;
        const world = instance.world;
        if (weight >= 1) {
            for (let j = 0; j < this.bones.length; j++) {
                const bone = instance.bones[j];
                if (bone) {
                    world[j].copy(bone.matrixWorld);
                } else {
                    const p = this.bones[j].parent;
                    world[j].multiplyMatrices(p >= 0 ? world[p] : root.matrixWorld, this.bones[j].local);
                }
            }
        } else {
            const clipWorld = instance.clipWorld || (instance.clipWorld = this.bones.map(() => new THREE.Matrix4()));
            for (let j = 0; j < this.bones.length; j++) {
                const bone = instance.bones[j];
                const p = this.bones[j].parent;
                if (bone) {
                    clipWorld[j].copy(bone.matrixWorld);
                } else {
                    clipWorld[j].multiplyMatrices(p >= 0 ? clipWorld[p] : root.matrixWorld, this.bones[j].local);
                }
            }
            this.pose(instance, nodes, base, false);
            const nodeWorld = instance.nodeWorld || (instance.nodeWorld = this.bones.map(() => new THREE.Matrix4()));
            for (let j = 0; j < this.bones.length; j++) {
                nodeWorld[j].copy(world[j]);
            }
            for (let j = 0; j < this.bones.length; j++) {
                const p = this.bones[j].parent;
                if (p >= 0) {
                    blendMat.copy(nodeWorld[p]).invert().multiply(nodeWorld[j]);
                    blendMat2.copy(clipWorld[p]).invert().multiply(clipWorld[j]);
                } else {
                    blendMat.copy(nodeWorld[j]);
                    blendMat2.copy(clipWorld[j]);
                }
                blendMat.decompose(blendPa, blendQa, blendSa);
                blendMat2.decompose(blendPb, blendQb, blendSb);
                blendPa.lerp(blendPb, weight);
                blendQa.slerp(blendQb, weight);
                blendSa.lerp(blendSb, weight);
                blendMat.compose(blendPa, blendQa, blendSa);
                if (p >= 0) {
                    world[j].multiplyMatrices(world[p], blendMat);
                } else {
                    world[j].copy(blendMat);
                }
            }
        }
        for (let j = 0; j < this.bones.length; j++) {
            const bone = instance.bones[j];
            if (!bone) {
                continue;
            }
            if (hideHead && this.headIndex >= 0 && this.headMask[j]) {
                const h = world[this.headIndex].elements;
                bone.matrixWorld.copy(ZERO_SCALE).setPosition(h[12], h[13], h[14]);
            } else {
                bone.matrixWorld.copy(world[j]);
            }
        }
    }

    // Drops the clip an instance last played, the nodes alone pose it from then on.
    stopClip(instance) {
        if (!instance.clip) {
            return;
        }
        instance.mixer.stopAllAction();
        instance.clip = null;
        instance.action = null;
    }

    // Recolours an instance's materials with a kit, per the manifest or by material names.
    dress(instance, style, goalkeeper, look) {
        const kit = goalkeeper ? { ...style.goalkeeper, pattern: 'plain' } : style.kit;
        const named = this.options.materials || null;
        const onlyKit = !!this.options.onlyKit;
        const copies = new Map();
        instance.root.traverse((obj) => {
            if (!obj.isMesh) {
                return;
            }
            const swap = (mat) => {
                if (!copies.has(mat)) {
                    let copy = mat;
                    const slot = kitSlotOf(mat.name, named, onlyKit);
                    if (slot === 'shirt' || slot === 'shorts' || slot === 'socks') {
                        copy = mat.clone();
                        applyKit(copy, slot, kit, this.unit);
                    } else if (slot && look && look[slot]) {
                        copy = mat.clone();
                        copy.color = look[slot].clone();
                    }
                    if (copy !== mat) {
                        instance.materials.push(copy);
                    }
                    copies.set(mat, copy);
                }
                return copies.get(mat);
            };
            obj.material = Array.isArray(obj.material) ? obj.material.map(swap) : swap(obj.material);
        });
    }
}

// The renderer gives every skeleton it draws a bone texture, which only the skeleton frees.
function disposeSkeletons(root) {
    root.traverse((obj) => {
        if (obj.isSkinnedMesh && obj.skeleton) {
            obj.skeleton.dispose();
        }
    });
}

// Players drawn with rigs, one instance each, each player on the rig of its own model. recolorOf
// gives the shared recolor of a team, model and role, the keepers apart from the outfield, or null
// to keep the model's colors.
class Rigged {
    constructor(parent, header, styles, members, rigOf, recolorOf) {
        this.group = new THREE.Group();
        parent.add(this.group);
        this.entries = [];
        this.byK = new Map();
        try {
            for (const k of members) {
                const p = header.players[k];
                const style = styles[p.team] || styles[0];
                const { rig, model } = rigOf(k);
                const instance = rig.instantiate();
                const entry = { k, rig, model, team: p.team, gk: !!p.goalkeeper, instance, request: null, drawnClip: false };
                this.entries.push(entry);
                this.byK.set(k, entry);
                const recolor = recolorOf(p.team, model, rig, !!p.goalkeeper);
                if (recolor) {
                    recolor.apply(instance);
                    entry.recolor = recolor;
                }
                rig.dress(instance, style, p.goalkeeper, playerLook(style, p, k));
                this.group.add(instance.root);
            }
        } catch (err) {
            this.dispose();
            throw err;
        }
    }

    update(nodes, visible, hidden) {
        for (const e of this.entries) {
            const mode = hidden ? hidden[e.k] : 0;
            const on = !!visible[e.k] && mode !== HIDE_ALL;
            e.instance.root.visible = on;
            if (!on) {
                continue;
            }
            const base = e.k * PLAYER_FLOATS;
            e.hideHead = mode === HIDE_HEAD;
            const r = e.request;
            if (r && r.weight > 0) {
                e.rig.poseClip(e.instance, r.clip, r.t, r.weight, nodes, base, e.hideHead);
                e.drawnClip = true;
            } else {
                e.rig.pose(e.instance, nodes, base, e.hideHead);
                e.drawnClip = false;
            }
        }
        // A clip lasts one drawing, a player whose clip is not asked for again stands on his
        // nodes the next time.
        for (const e of this.entries) {
            e.request = null;
        }
    }

    // Recolored materials may be added after the instances were built, by a change of pairs.
    attach(team, model, gk, recolor) {
        for (const e of this.entries) {
            if (e.team === team && e.model === model && e.gk === gk && e.recolor !== recolor) {
                recolor.apply(e.instance);
                e.recolor = recolor;
            }
        }
    }

    // Asks for a clip on player k for the next drawing, at time t and weight, or none.
    request(k, clip, t = 0, weight = 1) {
        const e = this.byK.get(k);
        if (!e) {
            return false;
        }
        e.request = clip && weight > 0 ? { clip, t, weight } : null;
        return true;
    }

    // Drops the clip of player k at once, the nodes pose him from the next drawing.
    stop(k) {
        const e = this.byK.get(k);
        if (e) {
            e.request = null;
            e.rig.stopClip(e.instance);
        }
    }

    eye(nodes, k, out) {
        const e = this.byK.get(k);
        const head = e.rig.headIndex >= 0 ? e.instance.bones[e.rig.headIndex] : null;
        if (e.drawnClip && head && e.rig.eyeLocal) {
            return out.copy(e.rig.eyeLocal).applyMatrix4(head.matrixWorld);
        }
        e.rig.pose(e.instance, nodes, k * PLAYER_FLOATS, !!e.hideHead);
        return e.rig.eye(e.instance, out) || neckEye(nodes, k * PLAYER_FLOATS, e.rig.heightRatio, out);
    }

    headBone(k) {
        const e = this.byK.get(k);
        return e && e.rig.headIndex >= 0 ? e.instance.bones[e.rig.headIndex] : null;
    }

    // The middle of player k's helmet as drawn, and its radius, 0 for none.
    helmet(nodes, k, out) {
        const e = this.byK.get(k);
        const head = e ? e.rig.headSphere() : null;
        if (!head) {
            return 0;
        }
        const bone = e.instance.bones[e.rig.headIndex];
        out.copy(head.local).applyMatrix4(bone ? bone.matrixWorld : e.instance.world[e.rig.headIndex]);
        return head.radius + HELMET_MARGIN;
    }

    size(k) {
        return this.byK.get(k).rig.size();
    }

    dispose() {
        this.group.removeFromParent();
        for (const e of this.entries) {
            if (e.recolor) {
                e.recolor.forget(e.instance);
            }
            if (e.instance.mixer) {
                e.instance.mixer.stopAllAction();
                e.instance.mixer.uncacheRoot(e.instance.root);
            }
            e.rig.release(e.instance);
        }
        this.entries = [];
        this.byK.clear();
    }
}

// The radius and height of GRF's player at the reference height, from its bind pose, which is
// the straight rest pose, in the engine frame.
const grfSizes = new WeakMap();
function grfSize(grf) {
    if (!grfSizes.has(grf)) {
        let radius = 0;
        let top = 0;
        grf.scene.traverse((obj) => {
            if (obj.isMesh && obj.geometry.attributes.position) {
                const pos = obj.geometry.attributes.position;
                for (let i = 0; i < pos.count; i++) {
                    radius = Math.max(radius, Math.hypot(pos.getX(i), pos.getY(i)));
                    top = Math.max(top, pos.getZ(i));
                }
            }
        });
        grfSizes.set(grf, { radius: radius || 0.32, height: top || grf.referenceHeight });
    }
    return grfSizes.get(grf);
}

// The head of GRF's player at the reference height, from the vertices of its bind pose that
// follow the neck bone, its middle in the frame of the neck node of the rest pose.
const grfHeads = new WeakMap();
function grfHead(grf) {
    if (!grfHeads.has(grf)) {
        let head = null;
        try {
            const points = [];
            const box = new THREE.Box3();
            grf.scene.traverse((obj) => {
                if (!obj.isSkinnedMesh) {
                    return;
                }
                const neck = obj.skeleton.bones.findIndex((b) => b.name === 'neck');
                const pos = obj.geometry.attributes.position;
                const si = obj.geometry.attributes.skinIndex;
                const sw = obj.geometry.attributes.skinWeight;
                for (let i = 0; neck >= 0 && i < pos.count; i++) {
                    let best = 0;
                    for (let c = 1; c < 4; c++) {
                        if (sw.getComponent(i, c) > sw.getComponent(i, best)) {
                            best = c;
                        }
                    }
                    if (si.getComponent(i, best) === neck) {
                        const p = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i));
                        points.push(p);
                        box.expandByPoint(p);
                    }
                }
            });
            if (points.length) {
                const center = box.getCenter(new THREE.Vector3());
                let radius = 0;
                for (const p of points) {
                    radius = Math.max(radius, p.distanceTo(center));
                }
                const rest = restPoseNodes(0);
                const o = 3 * 7;
                const turn = new THREE.Quaternion(rest[o + 3], rest[o + 4], rest[o + 5], rest[o + 6]).invert();
                const local = center.sub(new THREE.Vector3(rest[o], rest[o + 1], rest[o + 2])).applyQuaternion(turn);
                head = { local, radius };
            }
        } catch (err) {
            console.warn('GRF head not measured', err);
        }
        grfHeads.set(grf, head || GRF_HEAD);
    }
    return grfHeads.get(grf);
}

// Hides the head of GRF's player by its skin weights. The head hangs on the neck bone, and every
// fragment weighted mostly to it is discarded while hide.value is 1, so no triangle shared with
// the collar stretches. The shadow keeps the head.
function hideHeadByWeight(material, neck, hide) {
    material.onBeforeCompile = (shader) => {
        shader.uniforms.cujuHeadBone = { value: neck };
        shader.uniforms.cujuHideHead = hide;
        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', `#include <common>
                uniform float cujuHeadBone;
                varying float vCujuHead;`)
            .replace('#include <skinbase_vertex>', `#include <skinbase_vertex>
                vCujuHead = dot(skinWeight, vec4(equal(skinIndex, vec4(cujuHeadBone))));`);
        shader.fragmentShader = shader.fragmentShader
            .replace('#include <common>', `#include <common>
                uniform float cujuHideHead;
                varying float vCujuHead;`)
            .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
                if (cujuHideHead > 0.5 && vCujuHead > 0.5) discard;`);
    };
    material.customProgramCacheKey = () => 'cuju-grf-head';
}

// GRF's own player, player.glb, kept in the engine frame. Its bind pose has identity rotations
// and its inverse binds are pure translations, so a bone's world matrix is the saved node
// transform, turned into three's frame, and that reproduces GRF's skinning exactly. GRF scales
// the body by height over 1.92 meters about the player's ground point and moves the hair to the
// scaled neck without scaling it.
class GrfPlayers {
    constructor(parent, header, styles, members, grf, options = {}) {
        this.group = new THREE.Group();
        parent.add(this.group);
        this.grf = grf;
        // A referee or another figure brings a kit canvas of its own and no number.
        this.kitCanvas = options.kitCanvas || null;
        this.byK = new Map();
        // Kit textures and cloned materials belong here, the geometries and the textures in grf
        // are shared by every instance and stay with the assets.
        this.disposables = [];
        this.entries = [];
        try {
            this.build(header, styles, members);
        } catch (err) {
            this.dispose();
            throw err;
        }
        this.node = new THREE.Matrix4();
        this.scaled = new THREE.Matrix4();
        this.neck = new THREE.Vector3();
    }

    build(header, styles, members) {
        const grf = this.grf;
        const bases = new Map();
        const kitBase = (team, gk) => {
            const key = `${team}:${gk}`;
            if (!bases.has(key)) {
                bases.set(key, paintKitBase(grf.mask, styles[team] || styles[0], !!gk));
            }
            return bases.get(key);
        };
        for (const k of members) {
            const p = header.players[k];
            const style = styles[p.team] || styles[0];
            const look = grfLook(grf, style, p);
            const root = SkeletonUtils.clone(grf.scene);
            const entry = {
                k, root, bones: new Array(NODE_COUNT).fill(null), hair: null, scale: look.height / grf.referenceHeight,
                hideHead: { value: 0 },
            };
            this.entries.push(entry);
            this.byK.set(k, entry);
            const bones = entry.bones;
            const kit = this.kitCanvas
                ? kitTexture(this.kitCanvas, style, false, null)
                : kitTexture(kitBase(p.team, p.goalkeeper ? 1 : 0), style, !!p.goalkeeper, playerLook(style, p, k).number);
            this.disposables.push(kit);
            const copies = new Map();
            root.traverse((obj) => {
                const i = NODE_NAMES.indexOf(obj.name);
                if (obj.isBone && i >= 0) {
                    obj.matrixAutoUpdate = false;
                    obj.matrixWorldAutoUpdate = false;
                    bones[i] = obj;
                }
                if (obj.isMesh) {
                    obj.castShadow = true;
                    obj.receiveShadow = true;
                    obj.frustumCulled = false;
                    const neck = obj.isSkinnedMesh ? obj.skeleton.bones.findIndex((b) => b.name === 'neck') : -1;
                    const swap = (mat) => {
                        if (!copies.has(mat)) {
                            const copy = this.dress(mat, look, kit);
                            if (neck >= 0) {
                                hideHeadByWeight(copy, neck, entry.hideHead);
                            }
                            copies.set(mat, copy);
                        }
                        return copies.get(mat);
                    };
                    obj.material = Array.isArray(obj.material) ? obj.material.map(swap) : swap(obj.material);
                }
            });
            if (look.hair) {
                const hair = SkeletonUtils.clone(look.hair);
                hair.matrixAutoUpdate = false;
                hair.traverse((obj) => {
                    if (obj.isMesh && !Array.isArray(obj.material)) {
                        obj.castShadow = true;
                        obj.material = obj.material.clone();
                        obj.material.map = look.hairColor;
                        obj.material.color.set(0xffffff);
                        obj.material.needsUpdate = true;
                        this.disposables.push(obj.material);
                    }
                });
                entry.hair = hair;
                this.group.add(hair);
            }
            this.group.add(root);
        }
    }

    dress(mat, look, kit) {
        const copy = mat.clone();
        this.disposables.push(copy);
        const name = (mat.name || '').toLowerCase();
        if (name === 'kit') {
            copy.color.set(0xffffff);
            copy.map = kit;
            if (this.grf.kitNormal) {
                copy.normalMap = this.grf.kitNormal;
            }
            copy.roughness = 0.8;
            copy.metalness = 0;
        } else if (name === 'skin' && look.skin) {
            copy.color.set(0xffffff);
            copy.map = look.skin;
            copy.roughness = 0.65;
        } else if (name === 'shoe' && this.grf.shoe) {
            copy.color.set(0xffffff);
            copy.map = this.grf.shoe;
        } else if (name === 'sole' && this.grf.sole) {
            copy.color.set(0xffffff);
            copy.map = this.grf.sole;
        }
        copy.needsUpdate = true;
        return copy;
    }

    update(nodes, visible, hidden) {
        for (const e of this.entries) {
            const mode = hidden ? hidden[e.k] : 0;
            const on = !!visible[e.k] && mode !== HIDE_ALL;
            e.root.visible = on;
            e.hideHead.value = mode === HIDE_HEAD ? 1 : 0;
            if (e.hair) {
                e.hair.visible = on && mode !== HIDE_HEAD;
            }
            if (!on) {
                continue;
            }
            const base = e.k * PLAYER_FLOATS;
            // Scaling about the ground point g moves g to the origin, scales and moves it back, in
            // three's frame g has y 0.
            const gx = nodes[base];
            const gz = -nodes[base + 1];
            const s = e.scale;
            this.scaled.set(
                s, 0, 0, gx * (1 - s),
                0, s, 0, 0,
                0, 0, s, gz * (1 - s),
                0, 0, 0, 1,
            );
            for (let i = 0; i < NODE_COUNT; i++) {
                const bone = e.bones[i];
                if (!bone) {
                    continue;
                }
                nodeMatrix(nodes, base + i * 7, this.node);
                bone.matrixWorld.multiplyMatrices(this.scaled, this.node).multiply(ENGINE_TO_THREE);
            }
            if (e.hair) {
                const o = base + 3 * 7;
                nodeMatrix(nodes, o, this.node);
                this.neck.set(nodes[o], nodes[o + 2], -nodes[o + 1]).applyMatrix4(this.scaled);
                this.node.setPosition(this.neck);
                e.hair.matrix.multiplyMatrices(this.node, ENGINE_TO_THREE);
            }
        }
    }

    eye(nodes, k, out) {
        const e = this.byK.get(k);
        return neckEye(nodes, k * PLAYER_FLOATS, e ? e.scale : 1, out);
    }

    headBone(k) {
        const e = this.byK.get(k);
        return e ? e.bones[3] : null;
    }

    helmet(nodes, k, out) {
        const e = this.byK.get(k);
        const s = e ? e.scale : 1;
        const head = grfHead(this.grf);
        neckEye(nodes, k * PLAYER_FLOATS, s, out, head.local);
        return head.radius * s + HELMET_MARGIN;
    }

    size(k) {
        const e = this.byK.get(k);
        const s = e ? e.scale : 1;
        const base = grfSize(this.grf);
        return { radius: base.radius * s, height: base.height * s };
    }

    dispose() {
        this.group.removeFromParent();
        for (const d of this.disposables) {
            d.dispose();
        }
        for (const e of this.entries) {
            disposeSkeletons(e.root);
        }
        this.disposables = [];
        this.entries = [];
        this.byK.clear();
    }
}

const SKIN_FILES = { light: ['skin_01.png', 'skin_02.png'], medium: ['skin_03.png'], dark: ['skin_04.png'] };
const ROSTER_SKIN = { light: 'skin_01.png', tan: 'skin_02.png', medium: 'skin_03.png', dark: 'skin_04.png' };

// Skin, hair and height of a GRF player, the roster entry when it chooses, else GRF's own look
// for the index, the height always GRF's.
function grfLook(grf, style, p) {
    const entry = grf.appearance[p.index % Math.max(1, grf.appearance.length)] || {};
    const roster = (style.players && style.players[p.index]) || {};
    const h = (p.index * 7 + p.team * 3) >>> 0;
    let skin = grf.textures[entry.skin] || grf.textures['skin_01.png'] || null;
    if (roster.skin && roster.skin !== 'grf') {
        skin = grf.textures[ROSTER_SKIN[roster.skin]] || skin;
    } else if (style.skin && style.skin !== 'mixed' && SKIN_FILES[style.skin]) {
        const files = SKIN_FILES[style.skin];
        skin = grf.textures[files[h % files.length]] || skin;
    }
    let hair = grf.hair[entry.hair] || null;
    if (roster.hair && roster.hair !== 'grf') {
        hair = roster.hair === 'bald' ? null : (grf.hair[`hair_${roster.hair}.glb`] || hair);
    } else if (style.hair === 'bald') {
        hair = null;
    } else if (style.hair === 'short' || style.hair === 'long') {
        const fits = Object.keys(grf.hair).filter((f) => f.includes(style.hair));
        hair = fits.length ? grf.hair[fits[h % fits.length]] : hair;
    }
    let hairColor = grf.textures[entry.hair_color] || grf.textures['hair_black.png'] || null;
    if (roster.hair_color && roster.hair_color !== 'grf') {
        hairColor = grf.textures[`hair_${roster.hair_color}.png`] || hairColor;
    }
    return { skin, hair, hairColor, height: entry.height_m || grf.referenceHeight };
}

async function loadTexture(loader, url, srgb = true) {
    const tex = await loader.loadAsync(url);
    tex.flipY = false;
    tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.anisotropy = 4;
    return tex;
}

async function loadImageData(url) {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`${url} answered ${res.status}`);
    }
    const bitmap = await createImageBitmap(await res.blob());
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// Loads the assets the converter wrote, see assets/manifest.json. Any failure leaves the simple
// figures and the procedural ball, goals and stands in place.
export async function loadAssets(base = 'assets/') {
    const assets = {
        manifest: null, grf: null, ball: null, goals: null, crowd: null, adboards: [], pitch: null, error: null,
    };
    try {
        const res = await fetch(`${base}manifest.json`, { cache: 'no-cache' });
        if (!res.ok) {
            throw new Error(`manifest.json answered ${res.status}`);
        }
        assets.manifest = await res.json();
    } catch (err) {
        assets.error = `No assets, using simple figures. ${err.message || err}`;
        console.info(assets.error);
        return assets;
    }
    const manifest = assets.manifest;
    const files = manifest.files || [];
    const byRole = (role) => files.filter((f) => f.role === role);
    const loader = new GLTFLoader();
    const textures = new THREE.TextureLoader();
    const tryLoad = async (what, fn) => {
        try {
            return await fn();
        } catch (err) {
            console.warn(`${what} not loaded`, err);
            return null;
        }
    };
    const [ball, goals, crowd, pitch] = await Promise.all([
        tryLoad('ball', async () => (await loader.loadAsync(`${base}ball.glb`)).scene),
        tryLoad('goals', async () => (await loader.loadAsync(`${base}goal.glb`)).scene),
        tryLoad('crowd', () => textures.loadAsync(`${base}crowd.webp`)),
        tryLoad('pitch', () => textures.loadAsync(`${base}pitch.webp`)),
    ]);
    assets.ball = ball;
    assets.goals = goals;
    assets.crowd = crowd ? crowd.image : null;
    if (pitch) {
        pitch.colorSpace = THREE.SRGBColorSpace;
        pitch.anisotropy = 8;
        assets.pitch = pitch;
    }
    for (const f of byRole('adboard')) {
        const img = await tryLoad(f.file, () => textures.loadAsync(base + f.file));
        if (img) {
            assets.adboards.push(img.image);
        }
    }
    assets.grf = await tryLoad('player', async () => {
        const gltf = await loader.loadAsync(`${base}player.glb`);
        const want = files.map((f) => f.file).filter((n) => /^(skin_|hair_).*\.png$/.test(n) || n === 'shoe_sole.png');
        const loaded = await Promise.all(want.map((n) => loadTexture(textures, base + n)));
        const tex = Object.fromEntries(want.map((n, i) => [n, loaded[i]]));
        const hair = {};
        await Promise.all(byRole('hair').map(async (f) => {
            const h = await tryLoad(f.file, async () => (await loader.loadAsync(base + f.file)).scene);
            if (h) {
                hair[f.file] = h;
            }
        }));
        // Every kit, the goalkeepers' too, is painted from the mask in the team's colours.
        const [mask, kitNormal, shoe] = await Promise.all([
            loadImageData(`${base}kit_mask.png`),
            tryLoad('kit normal', () => loadTexture(textures, `${base}kit_normal.webp`, false)),
            tryLoad('shoe', () => loadTexture(textures, `${base}shoe.webp`)),
        ]);
        return {
            scene: gltf.scene,
            mask,
            kitNormal,
            shoe,
            sole: tex['shoe_sole.png'] || null,
            textures: tex,
            hair,
            appearance: (manifest.appearance && manifest.appearance.players) || [],
            referenceHeight: (manifest.player_scale && manifest.player_scale.reference_height_m) || 1.92,
        };
    });
    if (!assets.grf) {
        assets.error = 'The player model did not load, using simple figures.';
    }
    return assets;
}

// A model a team uploaded is someone else's file. Everything it needs must sit inside the glb,
// so the loader is only let at data and blob URLs, the ones GLTFLoader makes for embedded
// buffers and images, and any other URL turns into an empty data URL that fails to load.
function sealedManager() {
    const manager = new THREE.LoadingManager();
    manager.setURLModifier((url) => (/^(data|blob):/i.test(url) ? url : 'data:,'));
    return manager;
}

// Throws a readable error for a glb that names outside files, before the loader sees it.
function checkSelfContained(buffer) {
    const view = new DataView(buffer);
    if (buffer.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67) {
        throw new Error('This model is not a binary glTF file, export it as .glb.');
    }
    const length = view.getUint32(12, true);
    if (view.getUint32(16, true) !== 0x4e4f534a || 20 + length > buffer.byteLength) {
        throw new Error('This .glb file is damaged, its JSON chunk is missing.');
    }
    const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, length)));
    const outside = [...(json.buffers || []), ...(json.images || [])]
        .filter((entry) => typeof entry.uri === 'string' && !/^data:/i.test(entry.uri));
    if (outside.length) {
        throw new Error('This model refers to files outside the .glb, export it with every buffer and texture embedded.');
    }
}

// Parses the bytes of a team's own model, the upload of the team editor included.
export async function parseModel(buffer) {
    if (buffer.byteLength > MAX_MODEL_BYTES) {
        throw new Error(`This model is ${Math.ceil(buffer.byteLength / 1048576)} MB, at most ${MAX_MODEL_BYTES / 1048576} MB is read.`);
    }
    checkSelfContained(buffer);
    return new GLTFLoader(sealedManager()).parseAsync(buffer, '');
}

async function fetchModel(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
        throw new Error(`model download failed, ${res.status}`);
    }
    if (Number(res.headers.get('Content-Length')) > MAX_MODEL_BYTES) {
        throw new Error(`This model is larger than ${MAX_MODEL_BYTES / 1048576} MB.`);
    }
    return res.arrayBuffer();
}

// The registry gives a model's size and joints but nothing that changes with every upload, so the
// rig is keyed on a digest of the bytes themselves.
async function digest(buffer) {
    if (globalThis.crypto && crypto.subtle) {
        const sum = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
        return Array.from(sum, (b) => b.toString(16).padStart(2, '0')).join('');
    }
    // Pages served over plain http away from localhost have no crypto.subtle, FNV stands in.
    const bytes = new Uint8Array(buffer);
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
        h = Math.imul(h ^ bytes[i], 16777619);
    }
    return `${bytes.length}.${(h >>> 0).toString(16)}`;
}

// The newest rig of each team. A rig replaced by a new upload is not disposed, players built
// before it may still draw its scene.
const teamRigCache = new Map();

// Builds a rig for a team's own model from the registry, null when it cannot. A failure is not
// kept, so the next rebuild tries again.
export async function loadTeamRig(team) {
    if (!team || !team.model) {
        return null;
    }
    try {
        const buffer = await fetchModel(`/api/teams/${encodeURIComponent(team.id)}/model`);
        const key = `${await digest(buffer)}.${JSON.stringify(team.model.joints)}`;
        let cached = teamRigCache.get(team.id);
        if (!cached || cached.key !== key) {
            const entry = { key, rig: null };
            // A model on KayKit's rig also plays the clips of KayKit's animation packs.
            entry.rig = parseModel(buffer).then((gltf) => attachKayKitClips(rigFromGltf(gltf, team.model.joints, { onlyKit: true })));
            entry.rig.catch(() => {
                if (teamRigCache.get(team.id) === entry) {
                    teamRigCache.delete(team.id);
                }
            });
            teamRigCache.set(team.id, entry);
            cached = entry;
        }
        return await cached.rig;
    } catch (err) {
        console.warn('team model not used', team.id, err);
        return null;
    }
}

// A rig from a loaded glTF, with the animation clips the file carries.
export function rigFromGltf(gltf, joints, options = {}) {
    return new Rig(gltf.scene, joints, { ...options, animations: gltf.animations || [] });
}

// The clips of a rig as celebrations name them, clip and the clip's name.
export function clipNames(rig) {
    return rig && rig.clipNames ? rig.clipNames() : [];
}

// The model a roster entry asks for, team follows the team's model.
export function playerModel(style, index) {
    const roster = style && style.players ? style.players[index] : null;
    if (roster && roster.model && roster.model !== 'team') {
        return roster.model;
    }
    return (style && style.model) || 'grf';
}

// Every model a team's roster uses, the team's own model included, for loading rigs.
export function teamModels(style, size = null) {
    const out = new Set([(style && style.model) || 'grf']);
    const roster = (style && style.players) || [];
    roster.slice(0, size == null ? roster.length : size).forEach((p, i) => out.add(playerModel(style, i)));
    return [...out];
}

// The rigs of each team as a map from model to rig. A team may give one rig, the rig of its own
// model, as older callers do, or a Map or plain object of rigs by model.
function rigTables(rigs, styles) {
    return [0, 1].map((team) => {
        const r = rigs ? rigs[team] : null;
        const table = new Map();
        if (r instanceof Rig) {
            const style = styles[team] || styles[0];
            table.set(style && style.model && style.model !== 'grf' ? style.model : 'custom', r);
        } else if (r instanceof Map) {
            for (const [m, rig] of r) {
                if (rig instanceof Rig) {
                    table.set(m, rig);
                }
            }
        } else if (r && typeof r === 'object') {
            for (const [m, rig] of Object.entries(r)) {
                if (rig instanceof Rig) {
                    table.set(m, rig);
                }
            }
        }
        return table;
    });
}

// Two models look alike on the pitch when their first main colors, as each team shows them, are
// closer than ALIKE in CIEDE2000 and their second ones closer than ALIKE_SECOND.
const ALIKE = 10;
const ALIKE_SECOND = 18;

function alike(a, b) {
    if (!a.length || !b.length) {
        return false;
    }
    const d0 = ciede2000(a[0].lab, b[0].lab);
    const d1 = a[1] && b[1] ? ciede2000(a[1].lab, b[1].lab) : d0;
    return d0 < ALIKE && d1 < ALIKE_SECOND;
}

// After one team is recolored, the main colors must differ by this much, else the other team
// takes its kit's primary as well.
const APART = 25;

function clearlyApart(a, b) {
    return !a.length || !b.length || ciede2000(a[0].lab, b[0].lab) >= APART;
}

// The pairs of every team and model in use, the team's own, and where two teams would look alike
// the main color of the model turned into the kit's primary, the right team's first, then the
// left team's when the two are still not clearly apart. The main color is the heaviest that is
// not skin.
function resolvePairs(styles, used, clash) {
    const pairs = [0, 1].map((team) => {
        const style = styles[team] || styles[0];
        const table = new Map();
        for (const m of used[team].keys()) {
            table.set(m, { ...((style.colors && style.colors[m]) || {}) });
        }
        return table;
    });
    if (!clash) {
        return pairs;
    }
    // The models as drawn, each dressed in its kit with the team's own pairs over it.
    const shown = (team, m) => shownColors(used[team].get(m), { ...kitPairs(used[team].get(m), (styles[team] || styles[0]).kit), ...pairs[team].get(m) });
    const turn = (team, m) => {
        const main = shownColors(used[team].get(m), {})[0];
        if (main) {
            pairs[team].get(m)[main.hex] = hex((styles[team] || styles[0]).kit.primary);
        }
    };
    const done = [new Set(), new Set()];
    for (const m1 of used[1].keys()) {
        for (const m0 of used[0].keys()) {
            if (!alike(shown(0, m0), shown(1, m1))) {
                continue;
            }
            if (!done[1].has(m1)) {
                turn(1, m1);
                done[1].add(m1);
            }
            if (!clearlyApart(shown(0, m0), shown(1, m1)) && !done[0].has(m0)) {
                turn(0, m0);
                done[0].add(m0);
            }
        }
    }
    return pairs;
}

// Clips that settle into a pose, which hold their last frame rather than start over.
const ONCE_CLIP = /(_Down|_StandUp|^PickUp$|^Throw$|^Interact$|^Hit_|Death|_Pose$|Spawn|Awaken|Resurrect|^Use_Item$|^Dodge_|^Jump_Start$|^Jump_Land$)/;

function loopTime(clip, t) {
    const d = clip.duration || 0;
    return d > 0 ? t % d : 0;
}

// Glass helmets, a bubble round the head of every player, one instanced sphere drawn after the
// players. Only the side facing the camera is drawn, so a camera inside a bubble sees through it,
// and the glass never writes depth. It is clear in the middle, with a faint tint, a brighter rim
// where it turns away and the highlight of the sun.
const HELMET_VERTEX = `
varying vec3 vNormal;
varying vec3 vView;
void main() {
    mat4 m = modelMatrix * instanceMatrix;
    vec4 world = m * vec4(position, 1.0);
    vNormal = normalize(mat3(m) * normal);
    vView = cameraPosition - world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
}
`;
const HELMET_FRAGMENT = `
uniform vec3 uTint;
uniform vec3 uSunDir;
varying vec3 vNormal;
varying vec3 vView;
void main() {
    vec3 n = normalize(vNormal);
    vec3 v = normalize(vView);
    float facing = clamp(dot(n, v), 0.0, 1.0);
    float rim = pow(1.0 - facing, 3.0);
    float spec = pow(max(dot(n, normalize(uSunDir + v)), 0.0), 120.0);
    float alpha = clamp(0.04 + 0.5 * rim + 0.9 * spec, 0.0, 0.85);
    gl_FragColor = vec4(uTint * (0.5 + rim) + vec3(spec), alpha);
}
`;

class Helmets {
    constructor(parent, count) {
        this.geometry = new THREE.SphereGeometry(1, 24, 16);
        this.material = new THREE.ShaderMaterial({
            vertexShader: HELMET_VERTEX,
            fragmentShader: HELMET_FRAGMENT,
            uniforms: { uTint: { value: new THREE.Color(0xbfe6ff) }, uSunDir: { value: new THREE.Vector3(0, 1, 0) } },
            transparent: true,
            depthWrite: false,
        });
        this.mesh = new THREE.InstancedMesh(this.geometry, this.material, Math.max(1, count));
        this.mesh.frustumCulled = false;
        this.mesh.renderOrder = 2;
        this.mesh.visible = false;
        this.center = new THREE.Vector3();
        this.scale = new THREE.Vector3();
        this.matrix = new THREE.Matrix4();
        this.turn = new THREE.Quaternion();
        parent.add(this.mesh);
    }

    // Puts a bubble on every player on the pitch whose head is drawn, from the parts that draw
    // them.
    update(nodes, where, visible, hidden, sun) {
        for (let k = 0; k < where.length; k++) {
            const part = where[k];
            let radius = 0;
            if (part && part.helmet && visible[k] && !hidden[k]) {
                radius = part.helmet(nodes, k, this.center);
            }
            if (radius > 0) {
                this.matrix.compose(this.center, this.turn, this.scale.setScalar(radius));
            } else {
                this.matrix.copy(ZERO_SCALE);
            }
            this.mesh.setMatrixAt(k, this.matrix);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        if (sun) {
            this.material.uniforms.uSunDir.value.copy(sun);
        }
        this.mesh.visible = true;
    }

    dispose() {
        this.geometry.dispose();
        this.material.dispose();
        this.mesh.removeFromParent();
    }
}

export class Players {
    // rigs holds, per team, a Map or object of rigs by model, or the one rig of the team's own
    // model, assets the GRF model. Each player draws the model its roster entry names, or the
    // team's, and falls back to GRF's player when that model has no rig. Kits always come from
    // the team colors, and other models wear them too, each main color of the model taking the
    // shirt, trim, shorts or socks of the kit or the keeper's kit, under the team's own color
    // pairs. With options.clash false the models of two teams keep their pairs even when they
    // look alike. The rigs and the assets belong to the caller and outlive the players.
    constructor(scene, header, styles, assets, rigs = [null, null], options = {}) {
        this.header = header;
        this.styles = styles;
        this.count = header.players.length;
        this.visible = new Uint8Array(this.count).fill(1);
        this.hidden = new Uint8Array(this.count);
        this.group = new THREE.Group();
        scene.add(this.group);
        const tables = rigTables(rigs, styles);
        this.choice = header.players.map((p) => {
            const style = styles[p.team] || styles[0];
            const wanted = playerModel(style, p.index);
            const rig = wanted !== 'grf' && tables[p.team] ? tables[p.team].get(wanted) || null : null;
            return { model: rig ? wanted : 'grf', wanted, rig };
        });
        const used = [new Map(), new Map()];
        header.players.forEach((p, k) => {
            const c = this.choice[k];
            if (c.rig && used[p.team]) {
                used[p.team].set(c.model, c.rig);
            }
        });
        this.used = used;
        try {
            this.pairs = resolvePairs(styles, used, options.clash !== false);
        } catch (err) {
            console.warn('model colors not compared', err);
            this.pairs = resolvePairs(styles, used, false);
        }
        this.recolors = new Map();
        const custom = [];
        const grf = [];
        const simple = [];
        header.players.forEach((p, k) => {
            if (this.choice[k].rig) {
                custom.push(k);
            } else if (assets && assets.grf) {
                grf.push(k);
            } else {
                simple.push(k);
            }
        });
        this.parts = [];
        this.where = new Array(this.count).fill(null);
        this.rigged = null;
        if (custom.length) {
            try {
                this.rigged = new Rigged(this.group, header, styles, custom,
                    (k) => this.choice[k], (team, model, rig, gk) => this.recolorFor(team, model, rig, gk));
                this.parts.push(this.rigged);
                custom.forEach((k) => {
                    this.where[k] = this.rigged;
                });
            } catch (err) {
                console.warn('custom models failed, using GRF players', err);
                this.rigged = null;
                custom.forEach((k) => {
                    this.choice[k] = { ...this.choice[k], model: 'grf', rig: null };
                });
                (assets && assets.grf ? grf : simple).push(...custom);
            }
        }
        if (grf.length) {
            try {
                const part = new GrfPlayers(this.group, header, styles, grf, assets.grf);
                this.parts.push(part);
                grf.forEach((k) => {
                    this.where[k] = part;
                });
            } catch (err) {
                console.warn('GRF players failed, using simple figures', err);
                simple.push(...grf);
            }
        }
        if (simple.length) {
            const part = new Figures(this.group, header, styles, simple);
            this.parts.push(part);
            simple.forEach((k) => {
                this.where[k] = part;
            });
        }
        this.markers = new Markers(this.group, header, styles);
        this.helmets = null;
        this.arenaGroup = null;
    }

    // The pairs a model is drawn with, its colors dressed in the kit of the role, the keeper's or
    // the outfield's, as the match shows it, the away kit included, and the team's own pairs over
    // them. A pair of a color to itself keeps that color.
    drawnPairs(team, model, rig, gk) {
        const style = this.styles[team] || this.styles[0];
        const kit = gk ? { ...style.goalkeeper, pattern: 'plain' } : style.kit;
        const own = this.pairs[team] ? this.pairs[team].get(model) : null;
        return { ...kitPairs(rig, kit), ...(own || {}) };
    }

    // The shared recolor of a team, model and role, made when its pairs change anything.
    recolorFor(team, model, rig, gk = false) {
        const key = `${team}:${model}:${gk ? 'gk' : 'field'}`;
        if (this.recolors.has(key)) {
            return this.recolors.get(key);
        }
        const pairs = this.drawnPairs(team, model, rig, gk);
        if (!Object.keys(pairs).length) {
            return null;
        }
        try {
            const r = new Recolor(rig, pairs);
            this.recolors.set(key, r);
            return r;
        } catch (err) {
            console.warn('model not recolored', model, err);
            this.recolors.set(key, null);
            return null;
        }
    }

    // Redraws the colors of a team on one model in place, for a live preview, after its pairs or
    // its kit changed. Pairs given replace the team's own. Returns false when no player of that
    // team draws that model.
    setColors(team, model, pairs) {
        const rig = this.used[team] ? this.used[team].get(model) : null;
        if (!rig) {
            return false;
        }
        if (pairs !== undefined) {
            this.pairs[team].set(model, { ...(pairs || {}) });
        }
        for (const gk of [false, true]) {
            const key = `${team}:${model}:${gk ? 'gk' : 'field'}`;
            const r = this.recolors.get(key);
            if (r) {
                r.set(this.drawnPairs(team, model, rig, gk));
                continue;
            }
            this.recolors.delete(key);
            const made = this.recolorFor(team, model, rig, gk);
            if (made && this.rigged) {
                this.rigged.attach(team, model, gk, made);
            }
        }
        return true;
    }

    // Redraws every model of a team after its kit changed, its own pairs kept.
    setKit(team) {
        for (const model of this.used[team] ? this.used[team].keys() : []) {
            this.setColors(team, model);
        }
    }

    // The pairs a team and model are drawn with, the clash recolor included.
    colorsOf(team, model) {
        return { ...((this.pairs[team] && this.pairs[team].get(model)) || {}) };
    }

    // The model player k draws, grf when its own had no rig.
    modelOf(k) {
        return this.choice[k] ? this.choice[k].model : 'grf';
    }

    rigOf(k) {
        return this.choice[k] ? this.choice[k].rig : null;
    }

    update(pose, frame) {
        for (let k = 0; k < this.count; k++) {
            // Bit 0 is on the pitch, a player sent off disappears.
            this.visible[k] = frame.playerFlags[k] & 1;
        }
        for (const part of this.parts) {
            part.update(pose.nodes, this.visible, this.hidden);
        }
        this.markers.update(pose, frame, this.visible, this.hidden);
        this.updateHelmets(pose);
    }

    // Glass helmets while the arena of the scene asks for them, as the space station does. A
    // player whose head a camera hides wears none.
    updateHelmets(pose) {
        const scene = this.group.parent;
        if (!this.arenaGroup || this.arenaGroup.parent !== scene) {
            this.arenaGroup = scene ? scene.children.find((o) => o.name === 'arena') || null : null;
        }
        const arena = this.arenaGroup ? this.arenaGroup.userData : null;
        if (!arena || !arena.helmets) {
            if (this.helmets) {
                this.helmets.mesh.visible = false;
            }
            return;
        }
        if (!this.helmets) {
            this.helmets = new Helmets(this.group, this.count);
        }
        this.helmets.update(pose.nodes, this.where, this.visible, this.hidden, arena.sun);
    }

    // Hides player k for a camera at its eyes, none, head for the head and hair, or all.
    setHidden(k, mode = 'none') {
        if (k >= 0 && k < this.count) {
            this.hidden[k] = HIDE_MODES[mode] || 0;
        }
    }

    // The eyes of player k in three's frame, from the model's own head bone, or from GRF's neck.
    eye(pose, k, out = new THREE.Vector3()) {
        const part = this.where[k];
        if (!part) {
            return this.position(pose, k, out).setY(1.7);
        }
        return part.eye(pose.nodes, k, out);
    }

    // The bone that carries player k's head, whose matrixWorld follows the pose, or null.
    headBone(k) {
        const part = this.where[k];
        return part ? part.headBone(k) : null;
    }

    // The horizontal radius in meters of player k's model as drawn, from its ground point to the
    // farthest point of the body standing at rest, arms down.
    radius(k) {
        const part = this.where[k];
        return part ? part.size(k).radius : 0.32;
    }

    // The standing height in meters of player k's model as drawn.
    height(k) {
        const part = this.where[k];
        return part ? part.size(k).height : 1.85;
    }

    // The animation clips of player k's model, named as celebrations name them.
    clipNames(k) {
        return clipNames(this.rigOf(k));
    }

    // The clip that name plays at time t seconds on player k's model, with its own time, or
    // null when the model lacks it. A name of two clips joined by a greater than sign plays the
    // first once and then loops the second. A clip that settles into a pose plays once and holds
    // its last frame, the others loop.
    resolveClip(k, name, t) {
        const rig = this.rigOf(k);
        if (!rig || typeof rig.clip !== 'function' || !name) {
            return null;
        }
        const parts = String(name).replace(/^clip:/, '').split('>');
        const first = rig.clip(parts[0]);
        if (!first) {
            return null;
        }
        const time = Math.max(0, Number(t) || 0);
        if (parts.length > 1) {
            const then = rig.clip(parts[1]);
            if (then && time >= first.duration) {
                return { clip: then, t: loopTime(then, time - first.duration) };
            }
            return { clip: first, t: Math.min(time, first.duration) };
        }
        return { clip: first, t: ONCE_CLIP.test(first.name) ? Math.min(time, first.duration) : loopTime(first, time) };
    }

    // Whether player k's model has the clip that name plays.
    hasClip(k, name) {
        return !!this.resolveClip(k, name, 0);
    }

    // The length in seconds of one play of a clip on player k's model, both parts of a joined
    // name, or 0 when the model lacks it.
    clipDuration(k, name) {
        const rig = this.rigOf(k);
        if (!rig || typeof rig.clip !== 'function' || !name) {
            return 0;
        }
        return String(name).replace(/^clip:/, '').split('>').reduce((sum, part) => {
            const c = rig.clip(part);
            return sum + (c ? c.duration : 0);
        }, 0);
    }

    // Plays a clip of player k's model at time t seconds for the next drawing, standing where
    // node player of the pose stands, blended over the pose of the nodes by weight from 0 to 1,
    // which defaults to 1, the clip alone. A clip lasts one drawing, so a caller asks again every
    // frame, and a player not asked for, or asked at weight 0, is drawn from his nodes exactly.
    // False when the model has no such clip, then the caller poses a procedural animation.
    playClip(k, name, t, weight) {
        const found = this.rigged ? this.resolveClip(k, name, t) : null;
        if (!found) {
            return false;
        }
        let w = weight == null ? 1 : Number(weight);
        w = w > 0 ? Math.min(1, w) : 0;
        return this.rigged.request(k, found.clip, found.t, w);
    }

    // Draws player k from his nodes from the next drawing on.
    stopClip(k) {
        if (this.rigged && this.rigOf(k)) {
            this.rigged.stop(k);
        }
    }

    // The celebration player k performs after a goal, a clip of its model where it names one the
    // model has, else a name of the animation library.
    celebration(k, seed = 0) {
        const p = this.header.players[k];
        const style = this.styles[p.team] || this.styles[0];
        const name = pickCelebration(style, p.index, seed);
        if (name.startsWith('clip:')) {
            return this.hasClip(k, name) ? name : fallbackCelebration(null, seed + k);
        }
        return hasAnimation(name) ? name : fallbackCelebration(name, seed + k);
    }

    // Ground position of a player in three's frame.
    position(pose, k, out) {
        const o = k * PLAYER_FLOATS;
        return out.set(pose.nodes[o], 0, -pose.nodes[o + 1]);
    }

    dispose() {
        for (const part of this.parts) {
            part.dispose();
        }
        this.parts = [];
        for (const r of this.recolors.values()) {
            if (r) {
                r.dispose();
            }
        }
        this.recolors.clear();
        this.markers.dispose();
        if (this.helmets) {
            this.helmets.dispose();
            this.helmets = null;
        }
        this.group.removeFromParent();
    }
}

// Rings on the ground. Every player stands on a thin ring of its team's color, faint enough to
// sit quietly at broadcast distance, the controlled players on a wider ring and the owner of the
// ball on a pulsing one.
class Markers {
    constructor(parent, header, styles) {
        this.group = new THREE.Group();
        parent.add(this.group);
        const ring = new THREE.RingGeometry(0.55, 0.72, 40);
        ring.rotateX(-Math.PI / 2);
        // The engine hands a side's control to the designated player as play moves, and a frame
        // marks the player controlled on its step, so every player of a side with an agent gets
        // a ring that shows while the player is controlled.
        this.header = header;
        const agentTeams = new Set(header.players.filter((p) => p.controlled).map((p) => p.team));
        this.controlled = header.players.map((p) => {
            if (!agentTeams.has(p.team)) {
                return null;
            }
            const style = styles[p.team] || styles[0];
            const mat = new THREE.MeshBasicMaterial({
                color: threeColor(style.kit.primary), transparent: true, opacity: 0.85, depthWrite: false,
            });
            mat.color.lerp(new THREE.Color(0xffffff), 0.35);
            const mesh = new THREE.Mesh(ring, mat);
            mesh.renderOrder = 2;
            this.group.add(mesh);
            return mesh;
        });
        const teamGeo = new THREE.RingGeometry(0.34, 0.42, 36);
        teamGeo.rotateX(-Math.PI / 2);
        this.teamRing = new THREE.InstancedMesh(teamGeo, new THREE.MeshBasicMaterial({
            transparent: true, opacity: 0.42, depthWrite: false,
        }), header.players.length);
        this.teamRing.frustumCulled = false;
        this.teamRing.renderOrder = 1;
        this.teamRing.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        const white = new THREE.Color(0xffffff);
        header.players.forEach((p, k) => {
            const style = styles[p.team] || styles[0];
            this.teamRing.setColorAt(k, threeColor(style.kit.primary).lerp(white, 0.2));
        });
        this.teamRing.instanceColor.needsUpdate = true;
        this.group.add(this.teamRing);
        this.ringMatrix = new THREE.Matrix4();
        const ownerGeo = new THREE.RingGeometry(0.8, 0.95, 48);
        ownerGeo.rotateX(-Math.PI / 2);
        this.owner = new THREE.Mesh(ownerGeo, new THREE.MeshBasicMaterial({
            color: 0xfff6c0, transparent: true, opacity: 0.9, depthWrite: false,
        }));
        this.owner.renderOrder = 2;
        this.group.add(this.owner);
        this.start = performance.now();
        this.ring = ring;
    }

    // Shows or hides the team rings, and sets how strongly they show, 0.42 by default.
    setTeamRings(visible, opacity = null) {
        this.teamRing.visible = !!visible;
        if (opacity != null) {
            this.teamRing.material.opacity = opacity;
        }
    }

    dispose() {
        this.group.removeFromParent();
        this.ring.dispose();
        for (const m of this.controlled) {
            if (m) {
                m.material.dispose();
            }
        }
        this.teamRing.geometry.dispose();
        this.teamRing.material.dispose();
        this.teamRing.dispose();
        this.owner.geometry.dispose();
        this.owner.material.dispose();
    }

    update(pose, frame, visible, hidden = null) {
        // Frames of an older cuju carry no mark of control, the header's agents keep their rings.
        const flags = frame && frame.playerFlags;
        let marked = false;
        if (flags) {
            for (let k = 0; k < flags.length && !marked; k++) {
                marked = !!(flags[k] & 8);
            }
        }
        for (let k = 0; k < this.controlled.length; k++) {
            const o = k * PLAYER_FLOATS;
            const on = !!visible[k] && !(hidden && hidden[k] === HIDE_ALL);
            if (on) {
                this.ringMatrix.makeTranslation(pose.nodes[o], 0.022, -pose.nodes[o + 1]);
                this.teamRing.setMatrixAt(k, this.ringMatrix);
            } else {
                this.teamRing.setMatrixAt(k, ZERO_SCALE);
            }
            const m = this.controlled[k];
            if (!m) {
                continue;
            }
            m.visible = !!visible[k] && (marked ? !!(flags[k] & 8) : !!this.header.players[k].controlled);
            m.position.set(pose.nodes[o], 0.03, -pose.nodes[o + 1]);
        }
        this.teamRing.instanceMatrix.needsUpdate = true;
        const owner = frame.ownedPlayer;
        if (owner >= 0 && owner < this.controlled.length && visible[owner]) {
            const o = owner * PLAYER_FLOATS;
            this.owner.visible = true;
            this.owner.position.set(pose.nodes[o], 0.035, -pose.nodes[o + 1]);
            const s = 1 + Math.sin((performance.now() - this.start) / 200) * 0.06;
            this.owner.scale.set(s, 1, s);
        } else {
            this.owner.visible = false;
        }
    }
}

// The referee's kit, painted from GRF's kit template, black with a white collar, white cuffs and
// white tops to the socks.
const REFEREE_BLACK = [22, 22, 24];
const REFEREE_WHITE = [240, 240, 240];

function paintRefereeKit(mask) {
    const { width: w, height: h, data: m } = mask;
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
            const u = (x + 0.5) / w;
            const cu = u >= 0.5 ? (u - 0.5) * 2 : u * 2;
            const r = m[i] / 255;
            const b = m[i + 2] / 255;
            let c = REFEREE_BLACK;
            if (r > 0.5 && v < 0.02 && Math.abs(cu - 0.5) < 0.16) {
                c = REFEREE_WHITE;
            } else if (r > 0.5 && v < 0.16 && (cu < 0.035 || cu > 0.965)) {
                c = REFEREE_WHITE;
            } else if (b > 0.5 && v < 0.6) {
                c = REFEREE_WHITE;
            }
            d[i] = c[0];
            d[i + 1] = c[1];
            d[i + 2] = c[2];
            d[i + 3] = 255;
        }
    }
    ctx.putImageData(out, 0, 0);
    return canvas;
}

const CARD_COLORS = { yellow: 0xf4d20b, red: 0xd71920 };

// The referee, GRF's player in black, with a card in the right hand on request. update takes one
// player's nodes in the engine frame, as a Float32Array of PLAYER_FLOATS, and whether it shows.
export function createReferee(scene, assets) {
    const group = new THREE.Group();
    group.name = 'referee';
    scene.add(group);
    const header = { players: [{ team: 0, index: 6, goalkeeper: false, controlled: false }], teams: [{}, {}] };
    const kit = { pattern: 'plain', primary: REFEREE_BLACK, secondary: REFEREE_WHITE, shorts: REFEREE_BLACK, socks: REFEREE_BLACK };
    const style = {
        kit, goalkeeper: kit, skin: 'mixed', hair: 'short', badge: '', number: 'white', players: [], art: null,
    };
    let body = null;
    if (assets && assets.grf) {
        try {
            body = new GrfPlayers(group, header, [style, style], [0], assets.grf, {
                kitCanvas: paintRefereeKit(assets.grf.mask),
            });
        } catch (err) {
            console.warn('referee model failed, using a simple figure', err);
            body = null;
        }
    }
    if (!body) {
        body = new Figures(group, header, [style, style], [0]);
    }
    const scale = body.entries && body.entries[0] ? body.entries[0].scale : 1;
    const cardGeo = new THREE.BoxGeometry(0.075, 0.105, 0.005);
    const cardMats = Object.fromEntries(Object.entries(CARD_COLORS).map(([k, c]) => [k, new THREE.MeshStandardMaterial({
        color: c, emissive: c, emissiveIntensity: 0.25, roughness: 0.5,
    })]));
    const card = new THREE.Mesh(cardGeo, cardMats.yellow);
    card.castShadow = true;
    card.visible = false;
    card.matrixAutoUpdate = false;
    group.add(card);
    const visible = new Uint8Array(1);
    const q = new THREE.Quaternion();
    const p = new THREE.Vector3();
    const along = new THREE.Vector3();
    const face = new THREE.Vector3();
    const side = new THREE.Vector3();
    const basis = new THREE.Matrix4();
    let kind = null;
    return {
        group,
        update(nodes, show = true) {
            visible[0] = show ? 1 : 0;
            body.update(nodes, visible, null);
            card.visible = !!kind && !!show;
            if (!card.visible) {
                return;
            }
            // At the fingertips of the right hand, the long side along the forearm and the face
            // turned to where the referee faces, whatever the twist of the arm.
            const o = 7 * 7;
            q.set(nodes[o + 3], nodes[o + 4], nodes[o + 5], nodes[o + 6]);
            if (q.lengthSq() < 1e-8) {
                q.identity();
            }
            along.set(0, 0, -1).applyQuaternion(q);
            p.copy(along).multiplyScalar(0.36);
            const gx = nodes[0];
            const gy = nodes[1];
            const x = gx + (nodes[o] + p.x - gx) * scale;
            const y = gy + (nodes[o + 1] + p.y - gy) * scale;
            const z = (nodes[o + 2] + p.z) * scale;
            q.set(nodes[3], nodes[4], nodes[5], nodes[6]);
            if (q.lengthSq() < 1e-8) {
                q.identity();
            }
            face.set(0, -1, 0).applyQuaternion(q).addScaledVector(along, -along.dot(face));
            if (face.lengthSq() < 1e-6) {
                face.set(0, 0, 1);
            }
            face.normalize();
            side.crossVectors(along, face);
            basis.makeBasis(side, along, face).premultiply(ENGINE_TO_THREE);
            card.matrix.copy(basis).setPosition(x, z, -y);
        },
        // Shows a yellow or a red card in the right hand, null puts it away.
        setCard(next) {
            kind = next === 'yellow' || next === 'red' ? next : null;
            if (kind) {
                card.material = cardMats[kind];
            }
            card.visible = !!kind;
        },
        eye(nodes, out = new THREE.Vector3()) {
            return neckEye(nodes, 0, scale, out);
        },
        dispose() {
            body.dispose();
            cardGeo.dispose();
            for (const m of Object.values(cardMats)) {
                m.dispose();
            }
            group.removeFromParent();
        },
    };
}
