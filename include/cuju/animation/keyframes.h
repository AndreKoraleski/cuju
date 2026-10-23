// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The pose of an animation node at a frame, interpolated from its keys or read from the table.
// Exact: the same function fills the table at load, so a read equals a computation bit for bit.
#pragma once

#include "../quaternion.h"
#include "../vector3.h"
#include "animation_data.h"

namespace cuju {
namespace animation {

CUJU_FUNCTION_NO_INLINE inline void ComputeInterpolatedValues(const Animation &animationData,
                                                              const Span<KeyFrame> &keys, int frame,
                                                              Quaternion &orientation,
                                                              Vector3 &position) {
    struct WeighedKey {
        const KeyFrame *keyFrame = nullptr;
        float influence = 0.0f;
        int frame = 0;
    } weighedKeys[2];
    int weighedKeysSize = 0;

    position = Vector3(0);
    orientation = Quaternion::Identity();

    if (frame > 0 && frame < animationData.frameCount) {
        for (size_t key = 0; key < keys.size(); key++) {
            if (keys[key].frame < frame) weighedKeysSize = 0;
            weighedKeys[weighedKeysSize].keyFrame = &keys[key];
            weighedKeys[weighedKeysSize++].frame = keys[key].frame;
            if (keys[key].frame >= frame) break;
        }
        // Exact: the ratios and the bias divide in double and round to float once.
        float bias = 0;
        if (weighedKeysSize == 1) {
            weighedKeys[0].influence = 1;
        } else if (weighedKeysSize == 2) {
            int distance = weighedKeys[1].frame - weighedKeys[0].frame;
            float distance1 = frame - weighedKeys[0].frame;
            float distance2 = weighedKeys[1].frame - frame;
            float ratio1 = (float)(1 - distance1 / (distance * 1.0));
            float ratio2 = (float)(1 - distance2 / (distance * 1.0));
            weighedKeys[0].influence = ratio1;
            weighedKeys[1].influence = ratio2;
            int relativePosition = frame - weighedKeys[0].frame;
            bias = (float)((relativePosition * 1.0) / (distance * 1.0));
        }
        if (weighedKeysSize > 1) {
            orientation = weighedKeys[0].keyFrame->orientation.GetSphericallyInterpolated(
                bias, weighedKeys[1].keyFrame->orientation);
            position = weighedKeys[0].keyFrame->position * weighedKeys[0].influence +
                       weighedKeys[1].keyFrame->position * weighedKeys[1].influence;
        } else {
            orientation = weighedKeys[0].keyFrame->orientation;
            position = weighedKeys[0].keyFrame->position * weighedKeys[0].influence;
        }
    } else if (frame >= animationData.frameCount) {
        for (size_t key = keys.size(); key-- > 0;) {
            weighedKeys[weighedKeysSize].keyFrame = &keys[key];
            weighedKeys[weighedKeysSize++].frame = keys[key].frame;
            if (weighedKeysSize == 2) break;
        }
        if (weighedKeysSize == 0) return;
        if (weighedKeysSize == 1) {
            orientation = weighedKeys[0].keyFrame->orientation;
            position = weighedKeys[0].keyFrame->position;
            return;
        }
        float keySpacing = weighedKeys[0].frame - weighedKeys[1].frame;
        float framesPastLastKey = frame - weighedKeys[0].frame;
        float bias = 1 + ((1 / keySpacing) * framesPastLastKey);
        orientation = weighedKeys[1].keyFrame->orientation.GetSphericallyInterpolated(
            bias, weighedKeys[0].keyFrame->orientation);
        position = weighedKeys[1].keyFrame->position * (1 - bias) +
                   weighedKeys[0].keyFrame->position * bias;
    } else {
        orientation = keys[0].orientation;
        position = keys[0].position;
    }
}

CUJU_FUNCTION inline void GetInterpolatedValues(const Animation &animationData,
                                                const NodeAnimation &node, int frame,
                                                Quaternion &orientation, Vector3 &position) {
    if (node.interpolation && frame < node.interpolationFrames) {
        const InterpolationEntry &entry = node.interpolation[frame < 0 ? 0 : frame];
        orientation = entry.orientation;
        position = entry.position;
        return;
    }
    ComputeInterpolatedValues(animationData, node.keys, frame, orientation, position);
}

CUJU_FUNCTION inline void BlendFrames(Quaternion orientationAtFrame, const Vector3 &positionAtFrame,
                                      const Quaternion &orientationAtNextFrame,
                                      const Vector3 &positionAtNextFrame, float bias,
                                      Quaternion &orientation, Vector3 &position) {
    orientationAtFrame.MakeSameNeighborhood(orientationAtNextFrame);
    orientation =
        orientationAtFrame.GetLinearlyInterpolated(bias, orientationAtNextFrame).GetNormalized();
    position = positionAtFrame * (1.0f - bias) + positionAtNextFrame * bias;
}

CUJU_FUNCTION inline void GetInterpolatedValuesZ(const Animation &animationData,
                                                 const NodeAnimation &node, int frame,
                                                 Quaternion &orientation, Vector3 &position,
                                                 float &z) {
    if (node.interpolation && frame < node.interpolationFrames) {
        const InterpolationEntry &entry = node.interpolation[frame < 0 ? 0 : frame];
        orientation = entry.orientation;
        position = entry.position;
        z = entry.angleZ;
        return;
    }
    ComputeInterpolatedValues(animationData, node.keys, frame, orientation, position);
    float x, y;
    orientation.GetAngles(x, y, z);
}

// Covers frames 0 to `frameCount + 1`, the frames that the simulation reads. The halfway poses use
// `BlendFrames`, as an application does.
inline void BuildInterpolationTable(AnimationStore &store) {
    size_t total = 0;
    for (const Animation &animationData : store.animations)
        total += (size_t)animationData.nodes.size() * (animationData.frameCount + 2);
    store.interpolation.assign(total, InterpolationEntry{});
    store.middle.assign(total, MiddleEntry{});
    size_t offset = 0;
    for (const Animation &animationData : store.animations) {
        const int frames = animationData.frameCount + 2;
        for (const NodeAnimation &constantNode : animationData.nodes) {
            // The store owns the nodes, so the write is safe.
            NodeAnimation &node = const_cast<NodeAnimation &>(constantNode);
            for (int frame = 0; frame < frames; frame++) {
                InterpolationEntry &entry = store.interpolation[offset + frame];
                ComputeInterpolatedValues(animationData, node.keys, frame, entry.orientation,
                                          entry.position);
                float x, y;
                entry.orientation.GetAngles(x, y, entry.angleZ);
            }
            for (int frame = 0; frame + 1 < frames; frame++) {
                const InterpolationEntry &at = store.interpolation[offset + frame];
                const InterpolationEntry &next = store.interpolation[offset + frame + 1];
                MiddleEntry &middle = store.middle[offset + frame];
                BlendFrames(at.orientation, at.position, next.orientation, next.position, 0.5f,
                            middle.orientation, middle.position);
                middle.normalizedOrientation = middle.orientation.GetNormalized();
                const float squaredLength = middle.orientation.GetSquaredMagnitude();
                middle.normalizedKeepsZeros = squaredLength >= 0.000001f && squaredLength < 2.0f;
            }
            node.interpolation = store.interpolation.data() + offset;
            node.middle = store.middle.data() + offset;
            node.interpolationFrames = frames;
            offset += frames;
        }
    }
}

inline std::unique_ptr<AnimationStore> ParseAnimationTables(const std::string &bytes) {
    std::unique_ptr<AnimationStore> store = ParseAnimations(bytes);
    BuildInterpolationTable(*store);
    return store;
}

CUJU_FUNCTION inline void GetKeyFrameZ(const Animation &animationData, int bodyPart, int frame,
                                       Quaternion &orientation, Vector3 &position, float &z) {
    for (const NodeAnimation &node : animationData.nodes)
        if (node.bodyPart == bodyPart) {
            GetInterpolatedValuesZ(animationData, node, frame, orientation, position, z);
            return;
        }
    float x, y;
    orientation.GetAngles(x, y, z);
}

CUJU_FUNCTION inline bool GetKeyFrame(const Animation &animationData, int bodyPart, int frame,
                                      Quaternion &orientation, Vector3 &position) {
    for (const NodeAnimation &node : animationData.nodes) {
        if (node.bodyPart == bodyPart) {
            GetInterpolatedValues(animationData, node, frame, orientation, position);
            for (const KeyFrame &key : node.keys)
                if (key.frame == frame) return true;
            return false;
        }
    }
    return false;
}

}  // namespace animation
}  // namespace cuju
