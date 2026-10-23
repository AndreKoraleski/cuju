// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Applies one frame of an animation to the nodes of a humanoid, with GRF's simple smoothing.
#pragma once

#include <cstdint>
#include <cstring>

#include "../profiling.h"
#include "../warp.h"

#include "../libm/inverse_trigonometric_float.h"
#include "../libm/powf.h"
#include "../math_utilities.h"
#include "../state/environment_state.h"
#include "keyframes.h"
#include "skeleton.h"

namespace cuju {
namespace animation {

CUJU_FUNCTION inline float Curve(float source, float bias) {
    return (libm::sinf((source - 0.5f) * PI) * 0.5f + 0.5f) * bias + source * (1.0f - bias);
}

enum {
    BODY_PART_LEFT_KNEE = 4,
    BODY_PART_RIGHT_KNEE = 5,
    BODY_PART_LEFT_ANKLE = 6,
    BODY_PART_RIGHT_ANKLE = 7,
    BODY_PART_LEFT_ELBOW = 10,
    BODY_PART_RIGHT_ELBOW = 11,
    BODY_PART_BODY = 12,
    BODY_PART_PLAYER = 13
};

struct AnimationApplyPlan {
    int16_t entryIndex[state::BODY_PARTS];
    bool entryNew[state::BODY_PARTS];
    float beginBias;
};

CUJU_FUNCTION inline float SmoothBeginBias(int frame) {
    return libm::powf(Curve(1.0f - NormalizedClamp((float)frame, 0, 8), 1.0f), 0.5f);
}

// `SmoothBeginBias(0..8)`. The frame clamps, so 9 entries cover every input.
CUJU_TABLE_DEFINITION(float, SMOOTH_BEGIN_BIAS,
                      {0x1p+0f, 0x1.f6297cp-1f, 0x1.d906bep-1f, 0x1.a9b664p-1f, 0x1.6a09e6p-1f,
                       0x1.1c73b4p-1f, 0x1.87de2ap-2f, 0x1.8f8b88p-3f, 0x0p+0f});

CUJU_FUNCTION inline float SmoothBeginBiasFromTable(int frame) {
    return CUJU_TABLE(SMOOTH_BEGIN_BIAS)[frame < 0 ? 0 : frame > 8 ? 8 : frame];
}

CUJU_FUNCTION inline int JointSpeedClass(int bodyPart) {
    if (bodyPart == BODY_PART_LEFT_ELBOW || bodyPart == BODY_PART_RIGHT_ELBOW ||
        bodyPart == BODY_PART_LEFT_KNEE || bodyPart == BODY_PART_RIGHT_KNEE)
        return 1;
    if (bodyPart == BODY_PART_LEFT_ANKLE || bodyPart == BODY_PART_RIGHT_ANKLE) return 2;
    return 0;
}

CUJU_FUNCTION inline float JointSpeedLimit(int speedClass, float beginBias) {
    float maximumAnglePerSecond = 7.5f * PI;
    if (speedClass == 1)
        maximumAnglePerSecond *= 1.2f;
    else if (speedClass == 2)
        maximumAnglePerSecond *= 1.6f;
    return (0.3f + 0.7f * (1.0f - beginBias)) * maximumAnglePerSecond;
}

CUJU_FUNCTION inline float JointSpeed(float dot, int timeDifferenceMilliseconds) {
    return 2.0f * libm::acosf(clamp(dot, -1.0f, 1.0f)) /
           ((float)timeDifferenceMilliseconds * 0.001f);
}

#if !defined(__CUDA_ARCH__)
// The smallest dot within the speed limit over a 10 ms tick, per speed class and frame 0..8.
// Exact: `acosf` never rises on [0, 1], so at or above it the limit holds without `acosf`.
struct JointSpeedThresholds {
    float dot[3][9];
    JointSpeedThresholds() {
        for (int speedClass = 0; speedClass < 3; speedClass++)
            for (int frame = 0; frame < 9; frame++) {
                const float limit = JointSpeedLimit(speedClass, SmoothBeginBiasFromTable(frame));
                uint32_t low = 0, high = 0x3f800000u;
                while (low < high) {
                    const uint32_t middle = low + (high - low) / 2;
                    float value;
                    memcpy(&value, &middle, sizeof(value));
                    if (JointSpeed(value, 10) > limit)
                        low = middle + 1;
                    else
                        high = middle;
                }
                memcpy(&dot[speedClass][frame], &low, sizeof(float));
            }
    }
};

inline const JointSpeedThresholds &GetJointSpeedThresholds() {
    static const JointSpeedThresholds thresholds;
    return thresholds;
}

// `middle.orientation.GetSphericallyInterpolated(bias, to).GetNormalized()` from the normalized
// copy, for a zero `bias` and `normalizedKeepsZeros`. False for a NaN dot, left to the full path.
// Exact: at |dot| >= 1 the slerp returns its own orientation. Below, the half-angle sine is at
// least 3.4e-4, so its zero-bias path keeps each nonzero element and gives each zero the sign of
// a sum of two zeros, which `Normalize`, a positive factor, and `normalizedKeepsZeros` preserve.
inline bool SteadySlerp(const MiddleEntry &middle, float bias, const Quaternion &to,
                        Quaternion &result) {
    const Quaternion &from = middle.orientation;
    Quaternion target = to;
    // Exact: the dot of `GetSphericallyInterpolated`, in float, in the order w, x, y, z.
    float cosHalfTheta =
        from.elements[3] * target.elements[3] + from.elements[0] * target.elements[0] +
        from.elements[1] * target.elements[1] + from.elements[2] * target.elements[2];
    if (cosHalfTheta < 0) {
        target = -target;
        cosHalfTheta = -cosHalfTheta;
    }
    if (fabsf(cosHalfTheta) >= 1.0f) {
        result = middle.normalizedOrientation;
        return true;
    }
    if (!(fabsf(cosHalfTheta) < 1.0f)) return false;
    for (int element = 0; element < 4; element++) {
        const float own = middle.normalizedOrientation.elements[element];
        result.elements[element] =
            own != 0.0f ? own
            : (std::signbit(own) && std::signbit(target.elements[element]) != std::signbit(bias))
                ? -0.0f
                : 0.0f;
    }
    return true;
}
#endif

// Each node takes the last history entry of its body part or appends one, in node order.
// `ParseAnimations` rejects repeated body parts, so one pass before the appends suffices.
CUJU_FUNCTION inline void AnimationApplyPrepare(const Animation &animationData,
                                                state::HumanoidState &humanoid, bool smooth,
                                                int frame, AnimationApplyPlan &plan) {
    plan.beginBias = smooth ? SmoothBeginBiasFromTable(frame) : 0.0f;
    int16_t *entryIndex = plan.entryIndex;
    bool *entryNew = plan.entryNew;
    const int nodeCount = animationData.nodes.size();
    if (nodeCount > state::BODY_PARTS)
        CUJU_FATAL("an animation moves more body parts than a humanoid has");
    if (smooth) {
        auto &movementHistory = humanoid.movementHistory;
        int16_t lastEntry[state::BODY_PARTS];
        for (int bodyPart = 0; bodyPart < state::BODY_PARTS; bodyPart++) lastEntry[bodyPart] = -1;
        for (int historyIndex = 0; historyIndex < movementHistory.count; historyIndex++) {
            const int32_t bodyPart = movementHistory[historyIndex].nodeName;
            if (bodyPart >= 0 && bodyPart < state::BODY_PARTS)
                lastEntry[bodyPart] = (int16_t)historyIndex;
        }
        for (int node = 0; node < nodeCount; node++) {
            const int bodyPart = animationData.nodes[node].bodyPart;
            int found = lastEntry[bodyPart];
            entryNew[node] = found < 0;
            if (found < 0) {
                state::MovementHistoryEntry newEntry{};
                newEntry.nodeName = bodyPart;
                // `AnimationApplyNode` fills in the pose.
                newEntry.timeDifferenceMilliseconds = 10;
                movementHistory.push_back(newEntry);
                found = movementHistory.count - 1;
            }
            entryIndex[node] = (int16_t)found;
        }
    }
}

// Runs on one lane, inside a region.
CUJU_FUNCTION_NO_INLINE inline void AnimationApplyNode(
    const Animation &animationData, const Skeleton &skeleton, state::HumanoidState &humanoid,
    int nodeIndex, int frame, int timeOffsetMilliseconds, bool smooth, float smoothFactor,
    const Vector3 &basePosition, float baseRotation, state::BiasedOffsets &offsets,
    int timeDifferenceMilliseconds, bool noPosition, const AnimationApplyPlan &plan) {
    const int16_t *entryIndex = plan.entryIndex;
    const bool *entryNew = plan.entryNew;
    const NodeAnimation &nodeAnimation = animationData.nodes[nodeIndex];
    const int bodyPart = nodeAnimation.bodyPart;
    state::NodeState &mapNode = humanoid.nodes[skeleton.nodeMap[bodyPart]];

    Quaternion orientation;
    Vector3 position;
#if !defined(__CUDA_ARCH__)
    // The middle entry while `orientation` still equals it, else null. The table holds the middle
    // of every frame whose two poses it holds.
    const MiddleEntry *middlePose = nullptr;
    if (nodeAnimation.middle && timeOffsetMilliseconds == -1 && frame >= 0 &&
        frame + 1 < nodeAnimation.interpolationFrames) {
        middlePose = &nodeAnimation.middle[frame];
        orientation = middlePose->orientation;
        position = middlePose->position;
    } else
#endif
    {
        Quaternion orientationAtFrame, orientationAtNextFrame;
        Vector3 positionAtFrame, positionAtNextFrame;
        float bias = 0.5f;
        if (timeOffsetMilliseconds != -1) bias = clamp(timeOffsetMilliseconds / 10.0f, 0.0f, 1.0f);
        GetInterpolatedValues(animationData, nodeAnimation, frame, orientationAtFrame,
                              positionAtFrame);
        GetInterpolatedValues(animationData, nodeAnimation, frame + 1, orientationAtNextFrame,
                              positionAtNextFrame);
        BlendFrames(orientationAtFrame, positionAtFrame, orientationAtNextFrame,
                    positionAtNextFrame, bias, orientation, position);
    }

    if (bodyPart == BODY_PART_PLAYER) {
        if (noPosition) {
            position.coordinates[0] = 0;
            position.coordinates[1] = 0;
        } else {
            position.Rotate2D(baseRotation);
        }
    }
    if (bodyPart == BODY_PART_BODY) {
        Quaternion rotationZ;
        rotationZ.SetAngleAxis(baseRotation, Vector3(0, 0, 1));
        orientation = rotationZ * orientation;
        orientation.Normalize();
#if !defined(__CUDA_ARCH__)
        middlePose = nullptr;
#endif
    }

    {
        state::BiasedOffset &offset = offsets.elements[bodyPart];
        if (offset.bias != 0.0) {
            offset.orientation.MakeSameNeighborhood(orientation);
            orientation = orientation.GetLinearlyInterpolated(offset.bias, offset.orientation)
                              .GetNormalized();
#if !defined(__CUDA_ARCH__)
            middlePose = nullptr;
#endif
        }
    }

    if (smooth) {
        state::MovementHistoryEntry *entry = &humanoid.movementHistory[entryIndex[nodeIndex]];
        if (entryNew[nodeIndex]) {
            entry->position = position;
            entry->orientation = orientation;
        }

        const float beginBias = plan.beginBias;
        float currentBias = 0.0f + beginBias * smoothFactor * 0.5f;

        if (bodyPart != BODY_PART_PLAYER) {
            const Quaternion previousOrientation = entry->orientation;
            Quaternion currentOrientation = mapNode.rotation;
            currentOrientation.MakeSameNeighborhood(previousOrientation);
            if (timeDifferenceMilliseconds > 0) {
                bool steady = false;
#if !defined(__CUDA_ARCH__)
                if (middlePose && middlePose->normalizedKeepsZeros && currentBias == 0.0f)
                    steady = SteadySlerp(*middlePose, currentBias, currentOrientation, orientation);
#endif
                if (!steady)
                    orientation =
                        orientation.GetSphericallyInterpolated(currentBias, currentOrientation)
                            .GetNormalized();
                float dot = orientation.MakeSameNeighborhood(currentOrientation);
                const int speedClass = JointSpeedClass(bodyPart);
                const float maximumAnglePerSecond = JointSpeedLimit(speedClass, beginBias);
#if !defined(__CUDA_ARCH__)
                const bool withinLimit =
                    timeDifferenceMilliseconds == 10 &&
                    dot >= GetJointSpeedThresholds().dot[speedClass][frame < 0   ? 0
                                                                     : frame > 8 ? 8
                                                                                 : frame];
#else
                const bool withinLimit = false;
#endif
                if (!withinLimit) {
                    float anglePerSecond = JointSpeed(dot, timeDifferenceMilliseconds);
                    if (anglePerSecond > maximumAnglePerSecond) {
                        float allowedFraction = maximumAnglePerSecond / anglePerSecond;
                        Quaternion desiredRotation =
                            currentOrientation.GetRotationTo(orientation).GetNormalized();
                        orientation = (desiredRotation.GetRotationMultipliedBy(allowedFraction) *
                                       currentOrientation)
                                          .GetNormalized();
                    }
                }
            }
            entry->orientation = currentOrientation;
            entry->timeDifferenceMilliseconds = timeDifferenceMilliseconds;
        } else {
            const Vector3 currentPosition = mapNode.position;
            if (timeDifferenceMilliseconds > 0 && beginBias > 0.01f) {
                position.coordinates[2] = position.coordinates[2] * (1.0f - currentBias) +
                                          currentPosition.coordinates[2] * currentBias;
                float maximumMetersPerSecond = 2.8f;
                if (animationData.outgoingSpecialState != STRING_EMPTY)
                    maximumMetersPerSecond = 6.0f;
                float allowedDistance =
                    maximumMetersPerSecond * ((float)timeDifferenceMilliseconds * 0.001f);
                float newZ = position.coordinates[2] + basePosition.coordinates[2];
                float desiredDistance = fabsf(newZ - currentPosition.coordinates[2]);
                float heightBias = 1.0f;
                if (desiredDistance > allowedDistance)
                    heightBias = allowedDistance / desiredDistance;
                newZ = (newZ * heightBias + currentPosition.coordinates[2] * (1.0f - heightBias)) *
                           beginBias +
                       newZ * (1.0f - beginBias);
                position = position.Get2D() + Vector3(0, 0, newZ);
            }
            entry->position = currentPosition;
            entry->timeDifferenceMilliseconds = timeDifferenceMilliseconds;
        }
    }

    if (bodyPart != BODY_PART_PLAYER) {
        Quaternion currentOrientation = mapNode.rotation;
        orientation.MakeSameNeighborhood(currentOrientation);
        mapNode.rotation = orientation;
    } else {
        mapNode.position = position + basePosition;
    }
}

// Every player and the three officials.
constexpr int MAXIMUM_PUT_HUMANOIDS = 2 * state::MAXIMUM_TEAM_PLAYERS + 3;
// The humanoids are independent, so their nodes split over the lanes together and the result
// equals applying each humanoid on its own.
CUJU_FUNCTION_NO_INLINE inline void FetchPutBuffersBatch(const AnimationCollection &collection,
                                                         const Skeleton *const *skeletons,
                                                         state::HumanoidState *const *humanoids,
                                                         int humanoidCount) {
    CUJU_PROFILE_SCOPE(ANIMATION_APPLY);
    if (humanoidCount > MAXIMUM_PUT_HUMANOIDS)
        CUJU_FATAL("a match holds more humanoids than a frame update takes");
    AnimationApplyPlan plans[MAXIMUM_PUT_HUMANOIDS];
    int firstNode[MAXIMUM_PUT_HUMANOIDS + 1];
    firstNode[0] = 0;
    for (int humanoidIndex = 0; humanoidIndex < humanoidCount; humanoidIndex++) {
        const state::AnimationApplyBuffer &buffer = humanoids[humanoidIndex]->animationApplyBuffer;
        const Animation &animationData = collection.animations[buffer.animation];
        AnimationApplyPrepare(animationData, *humanoids[humanoidIndex], buffer.smooth,
                              buffer.frameNumber, plans[humanoidIndex]);
        firstNode[humanoidIndex + 1] = firstNode[humanoidIndex] + (int)animationData.nodes.size();
    }
    WarpFor(firstNode[humanoidCount], [&](int item) {
        int humanoidIndex = 0;
        while (firstNode[humanoidIndex + 1] <= item) humanoidIndex++;
        state::HumanoidState &humanoid = *humanoids[humanoidIndex];
        state::AnimationApplyBuffer &buffer = humanoid.animationApplyBuffer;
        AnimationApplyNode(collection.animations[buffer.animation], *skeletons[humanoidIndex],
                           humanoid, item - firstNode[humanoidIndex], buffer.frameNumber, -1,
                           buffer.smooth, buffer.smoothFactor, buffer.position,
                           buffer.orientation.value(), buffer.offsets, 10, buffer.noPosition,
                           plans[humanoidIndex]);
    });
}

}  // namespace animation
}  // namespace cuju
