// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The ball, its rolling spin, a soft contact shadow and an optional trail.

import * as THREE from 'three';
import { BALL_RADIUS } from './pitch.js';

const TRAIL_POINTS = 48;

function makeBallTexture() {
    const w = 512;
    const h = 256;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#f7f7f4';
    ctx.fillRect(0, 0, w, h);
    // Panels placed on the sphere by longitude and latitude, drawn on the equirectangular map.
    const panels = [[0, 90], [0, -90]];
    for (let k = 0; k < 5; k++) {
        panels.push([k * 72, 26.6], [k * 72 + 36, -26.6]);
    }
    const img = ctx.getImageData(0, 0, w, h);
    const centres = panels.map(([lon, lat]) => {
        const a = THREE.MathUtils.degToRad(lon);
        const b = THREE.MathUtils.degToRad(lat);
        return new THREE.Vector3(Math.cos(b) * Math.cos(a), Math.sin(b), Math.cos(b) * Math.sin(a));
    });
    const v = new THREE.Vector3();
    for (let y = 0; y < h; y++) {
        const lat = (0.5 - (y + 0.5) / h) * Math.PI;
        for (let x = 0; x < w; x++) {
            const lon = ((x + 0.5) / w) * Math.PI * 2;
            v.set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon));
            let best = 0;
            for (const c of centres) {
                best = Math.max(best, v.dot(c));
            }
            const i = (y * w + x) * 4;
            if (best > 0.93) {
                img.data[i] = 30; img.data[i + 1] = 34; img.data[i + 2] = 48;
            } else if (best > 0.91) {
                img.data[i] = 120; img.data[i + 1] = 120; img.data[i + 2] = 125;
            }
        }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
}

function makeBlobTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, 'rgba(0, 0, 0, 0.55)');
    g.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

export class Ball {
    constructor(scene) {
        // The outer group carries position and spin, the inner model sits in it.
        this.mesh = new THREE.Group();
        this.model = new THREE.Mesh(
            new THREE.SphereGeometry(BALL_RADIUS, 32, 20),
            new THREE.MeshStandardMaterial({ map: makeBallTexture(), roughness: 0.42, metalness: 0 }),
        );
        this.model.castShadow = true;
        this.ownsModel = true;
        this.mesh.add(this.model);
        scene.add(this.mesh);
        this.blob = new THREE.Mesh(
            new THREE.PlaneGeometry(1, 1),
            new THREE.MeshBasicMaterial({ map: makeBlobTexture(), transparent: true, depthWrite: false }),
        );
        this.blob.rotation.x = -Math.PI / 2;
        this.blob.renderOrder = 1;
        scene.add(this.blob);
        this.spin = new THREE.Quaternion();
        this.last = null;
        this.velocity = new THREE.Vector3();
        const positions = new Float32Array(TRAIL_POINTS * 3);
        const colors = new Float32Array(TRAIL_POINTS * 4);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 4));
        this.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({
            vertexColors: true, transparent: true, depthWrite: false,
        }));
        this.trail.frustumCulled = false;
        this.trail.visible = false;
        scene.add(this.trail);
        this.trailCount = 0;
        this.trailOn = false;
        this.trailClock = 0;
        this.axis = new THREE.Vector3();
        this.delta = new THREE.Quaternion();
    }

    // Swaps in GRF's ball, a model in the engine frame, turned into three's frame here. Only the
    // procedural sphere is disposed, a clone of a loaded model shares its geometry, material and
    // texture with the assets.
    useModel(scene) {
        const model = scene.clone();
        model.rotation.x = -Math.PI / 2;
        model.traverse((obj) => {
            if (obj.isMesh) {
                obj.castShadow = true;
            }
        });
        this.mesh.remove(this.model);
        if (this.ownsModel) {
            this.model.geometry.dispose();
            this.model.material.map.dispose();
            this.model.material.dispose();
            this.ownsModel = false;
        }
        this.model = model;
        this.mesh.add(model);
    }

    setTrail(on) {
        this.trailOn = on;
        this.trail.visible = on;
        this.trailCount = 0;
    }

    // Forgets the motion history, after a seek or a cut the spin must not jump.
    reset() {
        this.last = null;
        this.trailCount = 0;
    }

    update(pose, dt) {
        const b = pose.ball;
        const pos = this.mesh.position.set(b[0], b[2], -b[1]);
        if (pose.ballKnownRotation) {
            this.mesh.quaternion.set(b[3], b[5], -b[4], b[6]);
        } else if (this.last && dt > 0) {
            this.velocity.subVectors(pos, this.last).divideScalar(dt);
            const ground = pos.y < BALL_RADIUS * 2.5;
            const dist = Math.hypot(pos.x - this.last.x, pos.z - this.last.z);
            if (ground && dist > 1e-5 && dist < 3) {
                // Rolling without slipping turns about up cross velocity by distance over radius.
                this.axis.set(pos.z - this.last.z, 0, -(pos.x - this.last.x)).normalize();
                this.delta.setFromAxisAngle(this.axis, dist / BALL_RADIUS);
                this.spin.premultiply(this.delta);
            } else if (!ground && dist < 3) {
                // In flight the ball keeps a gentle topspin along its path.
                this.axis.set(this.velocity.z, 0, -this.velocity.x);
                if (this.axis.lengthSq() > 1e-6) {
                    this.delta.setFromAxisAngle(this.axis.normalize(), dt * 6);
                    this.spin.premultiply(this.delta);
                }
            }
            this.mesh.quaternion.copy(this.spin);
        }
        if (!this.last) {
            this.last = new THREE.Vector3();
        }
        this.last.copy(pos);
        const h = Math.max(0, pos.y - BALL_RADIUS);
        const size = 0.42 + h * 0.18;
        this.blob.scale.set(size, size, 1);
        this.blob.position.set(pos.x, 0.012, pos.z);
        this.blob.material.opacity = Math.max(0.15, 1 - h * 0.12);
        if (this.trailOn) {
            this.updateTrail(pos, dt);
        }
    }

    updateTrail(pos, dt) {
        this.trailClock += dt;
        const p = this.trail.geometry.attributes.position;
        const c = this.trail.geometry.attributes.color;
        if (this.trailClock > 1 / 30 || this.trailCount === 0) {
            this.trailClock = 0;
            p.array.copyWithin(3, 0, (TRAIL_POINTS - 1) * 3);
            this.trailCount = Math.min(TRAIL_POINTS, this.trailCount + 1);
        }
        p.array[0] = pos.x;
        p.array[1] = pos.y;
        p.array[2] = pos.z;
        for (let i = 0; i < TRAIL_POINTS; i++) {
            const a = i < this.trailCount ? (1 - i / this.trailCount) * 0.55 : 0;
            c.array[i * 4] = 1;
            c.array[i * 4 + 1] = 0.95;
            c.array[i * 4 + 2] = 0.75;
            c.array[i * 4 + 3] = a;
        }
        this.trail.geometry.setDrawRange(0, this.trailCount);
        p.needsUpdate = true;
        c.needsUpdate = true;
    }
}
