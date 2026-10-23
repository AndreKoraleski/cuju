// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Animation selection of players and officials, a crude filter followed by stable sorts.
// Exact: every expression keeps GRF's operation order, precision and conversions.
#pragma once

#include "../warp.h"

#include "../profiling.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "../animation/animation_derived.h"
#include "../libm/powf.h"
#include "../libm/sincosf.h"
#include "humanoid.h"
#include "player_possession.h"

namespace cuju {
namespace simulation {

using animation::AnimationDerived;

enum SelectResult { SELECT_FALSE = 0, SELECT_TRUE = 1, UNSUPPORTED = 2 };

struct SelectContext {
    World &world;
    const SimulationContext &context;
    const AnimationDerived *derived;
    int humanoidIndex;
    CUJU_FUNCTION HumanoidState &GetHumanoid() {
        return world.serialized.match.players[humanoidIndex].base.humanoid;
    }
    CUJU_FUNCTION const AnimationDerived &GetDerived(int index) const { return derived[index]; }
};

CUJU_FUNCTION inline int GetVelocityID(int velocity, bool treatDribbleAsWalk = false) {
    int id = 0;
    switch (velocity) {
        case e_Velocity_Idle:
            id = 0;
            break;
        case e_Velocity_Dribble:
            id = 1;
            break;
        case e_Velocity_Walk:
            id = 2;
            break;
        case e_Velocity_Sprint:
            id = 3;
            break;
        default:
            id = 0;
            break;
    }
    if (treatDribbleAsWalk && id > 1) id--;
    return id;
}

CUJU_FUNCTION inline float Curve(float source, float bias = 1.0f) {
    return (libm::sinf((source - 0.5f) * PI) * 0.5f + 0.5f) * bias + source * (1.0f - bias);
}

// A zero vector gives NaN components.
CUJU_FUNCTION inline void NormalizeTo(Vector3 &vector, float length) {
    float factor = length / sqrtf(vector.GetDotProduct(vector));
    vector.coordinates[0] *= factor;
    vector.coordinates[1] *= factor;
    vector.coordinates[2] *= factor;
}
CUJU_FUNCTION inline Vector3 GetNormalizedTo(const Vector3 &vector, float length) {
    Vector3 result = vector;
    NormalizeTo(result, length);
    return result;
}
CUJU_FUNCTION inline void NormalizeMaximum(Vector3 &vector, float length) {
    if (vector.GetLength() > length) {
        vector.Normalize();
        vector *= length;
    }
}
CUJU_FUNCTION inline Vector3 GetNormalizedMaximum(const Vector3 &vector, float length) {
    if (vector.GetLength() > length) return vector.GetNormalized(0) * length;
    return vector;
}
// Exact: float differences whose squares add up in double.
CUJU_FUNCTION inline float GetDistance(const Vector3 &first, const Vector3 &second) {
    float differenceX = first.coordinates[0] - second.coordinates[0],
          differenceY = first.coordinates[1] - second.coordinates[1],
          differenceZ = first.coordinates[2] - second.coordinates[2];
    if (differenceX == 0.0f && differenceY == 0.0f && differenceZ == 0.0f) return 0.0f;
    double doubleX = differenceX, doubleY = differenceY, doubleZ = differenceZ;
    float length = (float)sqrt(doubleX * doubleX + doubleY * doubleY + doubleZ * doubleZ);
    if (length < ABOVE_ONE_MILLIONTH) length = 0;
    return length;
}

// Exact: each angle is a double product with `PI`, rounded to float.
CUJU_FUNCTION inline void PreferredDirectionAngles(float out[11]) {
    const double piFractions[11] = {0,    0.111, -0.111, 0.25,  -0.25, 0.5,
                                    -0.5, 0.75,  -0.75,  0.999, -0.999};
    for (int direction = 0; direction < 11; direction++)
        out[direction] = (float)(piFractions[direction] * (double)PI);
}
// `Vector3(0, -1, 0).GetRotated2D(angle)` for each `PreferredDirectionAngles` entry, but the
// vector itself for the angle 0.
CUJU_TABLE_DEFINITION(Vector3, PREFERRED_DIRECTION_VECTORS,
                      {
                          {0x0p+0f, -0x1p+0f, 0x0p+0f},
                          {0x1.5de48ap-2f, -0x1.e12f08p-1f, 0x0p+0f},
                          {-0x1.5de48ap-2f, -0x1.e12f08p-1f, 0x0p+0f},
                          {0x1.6a09e6p-1f, -0x1.6a09e6p-1f, 0x0p+0f},
                          {-0x1.6a09e6p-1f, -0x1.6a09e6p-1f, 0x0p+0f},
                          {0x1p+0f, 0x1.777a5cp-25f, 0x0p+0f},
                          {-0x1p+0f, 0x1.777a5cp-25f, 0x0p+0f},
                          {0x1.6a09e6p-1f, 0x1.6a09e6p-1f, 0x0p+0f},
                          {-0x1.6a09e6p-1f, 0x1.6a09e6p-1f, 0x0p+0f},
                          {0x1.9bc4e4p-9f, 0x1.ffff5ap-1f, 0x0p+0f},
                          {-0x1.9bc4e4p-9f, 0x1.ffff5ap-1f, 0x0p+0f},
                      });

CUJU_FUNCTION_NO_INLINE inline Vector3 ForceIntoPreferredDirectionVector(const Vector3 &source) {
    float bestDot = -1.0f;
    Vector3 bestValue;
    for (int direction = 0; direction < 11; direction++) {
        const Vector3 candidate = CUJU_TABLE(PREFERRED_DIRECTION_VECTORS)[direction];
        float dotProduct = candidate.GetDotProduct(source);
        if (dotProduct > bestDot) {
            bestDot = dotProduct;
            bestValue = candidate;
        }
    }
    return bestValue;
}
CUJU_FUNCTION_NO_INLINE inline float ForceIntoPreferredDirectionAngle(float angle) {
    float angles[11];
    PreferredDirectionAngles(angles);
    float bestAngleDifference = 10000.0;
    float bestValue = 0;
    for (int direction = 0; direction < 11; direction++) {
        float difference = fabsf(angles[direction] - angle);
        if (difference < bestAngleDifference) {
            bestAngleDifference = difference;
            bestValue = angles[direction];
        }
    }
    return bestValue;
}
CUJU_FUNCTION_NO_INLINE inline float ForceIntoAllowedBodyDirectionAngle(float angle) {
    const double piFractions[5] = {0, 0.25, -0.25, 0.75, -0.75};
    float bestAngleDifference = 10000.0;
    float bestValue = 0;
    for (int direction = 0; direction < 5; direction++) {
        float candidate = (float)(piFractions[direction] * (double)PI);
        float difference = fabsf(candidate - angle);
        if (difference < bestAngleDifference) {
            bestAngleDifference = difference;
            bestValue = candidate;
        }
    }
    return bestValue;
}

CUJU_FUNCTION inline float CalculateBiasForFastCornering(const Vector3 &currentMovement,
                                                         const Vector3 &desiredMovement,
                                                         float velocityExponent, float bias) {
    float angle = desiredMovement.GetNormalized(currentMovement).GetAngle2D(currentMovement);
    float currentMovementBias = libm::sinf(fabsf(angle) - 0.5f * PI) * 0.5f + 0.5f;
    float velocityBias =
        libm::powf(clamp(currentMovement.GetLength() / (SPRINT_VELOCITY - 0.5f), 0.0f, 1.0f),
                   velocityExponent);
    return velocityBias * currentMovementBias * bias;
}

CUJU_FUNCTION inline float StretchSprintTo(const float &inputVelocity,
                                           float inputSpaceMaximumVelocity,
                                           float targetMaximumVelocity) {
    if (inputVelocity < WALK_SPRINT_SWITCH) return inputVelocity;
    float sprintPart = inputVelocity - WALK_SPRINT_SWITCH;
    float oldLength = inputSpaceMaximumVelocity - WALK_SPRINT_SWITCH;
    float newLength = targetMaximumVelocity - WALK_SPRINT_SWITCH;
    float toNewFactor = newLength / oldLength;
    float stretchedSprintPart = sprintPart * toNewFactor;
    return WALK_SPRINT_SWITCH + stretchedSprintPart;
}

CUJU_FUNCTION inline Vector3 GetFrontOfFootOffsetRelative(float velocity, float bodyAngleRelative,
                                                          float height) {
    float fullDistanceFactor = 1.0f;
    float distance =
        0.34f + velocity * DEFAULT_TOUCH_OFFSET_MILLISECONDS * 0.001f * fullDistanceFactor;
    Vector3 forwardOffset = Vector3(0, -distance * 0.8f, 0);
    float bodyAngle = bodyAngleRelative;
    if (velocity < IDLE_DRIBBLE_SWITCH) bodyAngle = 0;
    Vector3 angled = Vector3(0, -distance * 0.2f, 0).GetRotated2D(bodyAngle);
    return (forwardOffset + angled) * (1.0f - clamp((height - 0.11f) / 4.0f, 0.0f, 0.5f));
}

// Speed: 16-bit positions take a third of the stack of key-animation pairs, and every sort of
// one key type shares this instance.
template <class Key>
CUJU_FUNCTION_NO_INLINE inline void SortByLaneKeys(AnimationList &dataSet, const Key *mine,
                                                   int lane, int step) {
    static_assert(MAXIMUM_ANIMATIONS <= 32767, "int16 positions");
    const int count = dataSet.count;
#if !defined(__CUDA_ARCH__)
    for (int element = 0; element < count; element++)
        if (mine[element] != mine[element]) CUJU_FATAL("an animation sort key is NaN");
#endif
#if CUJU_WARP_LANES
    // Speed: a stable rank sort in registers, 4 keys per lane, fits every list of the simulation.
    if (step == 32 && count <= 128) {
        Key laneKeys[4];
        int laneValues[4], rank[4] = {0, 0, 0, 0};
        for (int chunk = 0; chunk < 4; chunk++) {
            const int element = chunk * 32 + lane;
            laneValues[chunk] = element < count ? dataSet[element] : 0;
            laneKeys[chunk] = element < count ? mine[chunk] : Key();
        }
        const int chunkCount = (count + 31) / 32;
        for (int otherChunk = 0; otherChunk < chunkCount; otherChunk++)
            for (int sourceLane = 0; sourceLane < 32; sourceLane++) {
                const int other = otherChunk * 32 + sourceLane;
                if (other >= count) break;
                const Key otherKey = __shfl_sync(0xffffffffu, laneKeys[otherChunk], sourceLane);
                for (int chunk = 0; chunk < 4; chunk++) {
                    const int element = chunk * 32 + lane;
                    if (element < count && (otherKey < laneKeys[chunk] ||
                                            (!(laneKeys[chunk] < otherKey) && other < element)))
                        rank[chunk]++;
                }
            }
        for (int chunk = 0; chunk < chunkCount; chunk++)
            for (int sourceLane = 0; sourceLane < 32; sourceLane++) {
                const int element = chunk * 32 + sourceLane;
                if (element >= count) break;
                const int sortedPosition = __shfl_sync(0xffffffffu, rank[chunk], sourceLane);
                dataSet[sortedPosition] = __shfl_sync(0xffffffffu, laneValues[chunk], sourceLane);
            }
        __syncwarp();
        return;
    }
#endif
    Key keys[MAXIMUM_ANIMATIONS];
    int16_t positions[2][MAXIMUM_ANIMATIONS];
#if CUJU_WARP_LANES
    if (step == 32) {
#pragma unroll 1
        for (int element = 0; element < count; element++) {
            const int chunk = element >> 5;
            keys[element] = __shfl_sync(
                0xffffffffu, chunk * 32 + lane < count ? mine[chunk] : Key(), element & 31);
        }
    } else
#endif
    {
        for (int element = 0; element < count; element++) keys[element] = mine[element];
    }
    for (int element = 0; element < count; element++) positions[0][element] = (int16_t)element;
    int source = 0;
    for (int width = 1; width < count; width *= 2, source ^= 1) {
        const int16_t *input = positions[source];
        int16_t *output = positions[source ^ 1];
        for (int low = 0; low < count; low += 2 * width) {
            int middle = low + width < count ? low + width : count,
                high = low + 2 * width < count ? low + 2 * width : count;
            int left = low, right = middle, written = low;
            while (left < middle && right < high)
                output[written++] =
                    (keys[input[right]] < keys[input[left]]) ? input[right++] : input[left++];
            while (left < middle) output[written++] = input[left++];
            while (right < high) output[written++] = input[right++];
        }
    }
    int16_t *sorted = positions[source ^ 1];
    for (int element = 0; element < count; element++)
        sorted[element] = (int16_t)dataSet[positions[source][element]];
    for (int element = 0; element < count; element++) dataSet[element] = sorted[element];
}

// Exact: each sort compares a pure key of one animation with `<`, a strict weak order without
// NaN, so any stable sort of the precomputed keys gives GRF's order.
template <class Key, class KeyFunction>
CUJU_FUNCTION_NO_INLINE inline void StableSortByKey(AnimationList &dataSet, KeyFunction key) {
    CUJU_PROFILE_SCOPE(STABLE_SORT);
    const int count = dataSet.count;
    if (count < 2) return;
    Key mine[MAXIMUM_ANIMATIONS];
    int lane = 0, step = 1;
#if CUJU_WARP_LANES
    bool nested;
    WarpSplit(nested, lane, step);
    if (!nested) WarpRegionBegin();
#endif
    int keyCount = 0;
    for (int element = lane; element < count; element += step)
        mine[keyCount++] = key(dataSet[element]);
#if CUJU_WARP_LANES
    if (!nested) WarpRegionEnd();
#endif
    SortByLaneKeys<Key>(dataSet, mine, lane, step);
}

constexpr int MAXIMUM_CHAIN_KEYS = 8;

CUJU_FUNCTION inline bool ChainKeysPrecede(const double *left, const double *right, int keyCount) {
    for (int key = 0; key < keyCount; key++) {
        if (left[key] < right[key]) return true;
        if (right[key] < left[key]) return false;
    }
    return false;
}

// Exact: the first after a chain of stable sorts has the smallest keys in order, the last sort's
// key first, and the earliest position on a full tie. Doubles hold float and int keys exactly.
template <class KeyFunction>
CUJU_FUNCTION_NO_INLINE inline int FirstOfSortChain(const AnimationList &dataSet, int keyCount,
                                                    KeyFunction keys) {
    const int count = dataSet.count;
    if (count < 2) return dataSet.at(0);
    int lane = 0, step = 1;
#if CUJU_WARP_LANES
    bool nested;
    WarpSplit(nested, lane, step);
    if (!nested) WarpRegionBegin();
#endif
    int best = -1;
    double bestKeys[MAXIMUM_CHAIN_KEYS];
    for (int element = lane; element < count; element += step) {
        double elementKeys[MAXIMUM_CHAIN_KEYS];
        keys((int)dataSet[element], elementKeys);
#if !defined(__CUDA_ARCH__)
        for (int key = 0; key < keyCount; key++)
            if (elementKeys[key] != elementKeys[key]) CUJU_FATAL("an animation sort key is NaN");
#endif
        if (best < 0 || ChainKeysPrecede(elementKeys, bestKeys, keyCount)) {
            best = element;
            for (int key = 0; key < keyCount; key++) bestKeys[key] = elementKeys[key];
        }
    }
#if CUJU_WARP_LANES
    if (!nested) {
        WarpRegionEnd();
        for (int offset = 16; offset > 0; offset >>= 1) {
            const int otherBest = __shfl_xor_sync(0xffffffffu, best, offset);
            double otherKeys[MAXIMUM_CHAIN_KEYS];
            for (int key = 0; key < keyCount; key++)
                otherKeys[key] = __shfl_xor_sync(0xffffffffu, bestKeys[key], offset);
            if (otherBest >= 0 &&
                (best < 0 || ChainKeysPrecede(otherKeys, bestKeys, keyCount) ||
                 (!ChainKeysPrecede(bestKeys, otherKeys, keyCount) && otherBest < best))) {
                best = otherBest;
                for (int key = 0; key < keyCount; key++) bestKeys[key] = otherKeys[key];
            }
        }
        best = __shfl_sync(0xffffffffu, best, 0);
    }
#endif
    return dataSet[best];
}

// Directions are relative to the humanoid's angle, and velocities are categories.
struct CrudeSelectionQuery {
    bool byFunctionType = false;
    int functionType = 0;
    bool heedForcedFoot = false;
    // The default 1 is `e_Foot_Right`.
    int strongFoot = 1;
    bool bySide = false;
    Vector3 lookAtVectorRelative;
    bool allowLastDitchAnimations = false;
    bool byIncomingVelocity = false, incomingVelocityStrict = false,
         incomingVelocityNoDribbleToIdle = false, incomingVelocityNoDribbleToSprint = false,
         incomingVelocityForceLinearity = false;
    int incomingVelocity = 0;
    bool byOutgoingVelocity = false;
    int outgoingVelocity = 0;
    bool byPickupBall = false, pickupBall = true;
    bool byIncomingBodyDirection = false;
    Vector3 incomingBodyDirection;
    bool incomingBodyDirectionStrict = false, incomingBodyDirectionForceLinearity = false;
    bool byIncomingBallDirection = false;
    Vector3 incomingBallDirection;
    bool byOutgoingBallDirection = false;
    Vector3 outgoingBallDirection;
    bool byTripType = false;
    int tripType = 0;
    int32_t incomingSpecialState = animation::STRING_EMPTY,
            incomingRetainState = animation::STRING_EMPTY;
    float specialVariable1 = 0, specialVariable2 = 0;
};

CUJU_FUNCTION_NO_INLINE inline bool CheckFunctionType(int animationType, int functionType) {
    using namespace animation;
    switch (functionType) {
        case e_FunctionType_Movement:
            return animationType == e_AnimationString_Movement;
        case e_FunctionType_BallControl:
            return animationType == e_AnimationString_BallControl;
        case e_FunctionType_Trap:
            return animationType == e_AnimationString_Trap;
        case e_FunctionType_ShortPass:
            return animationType == e_AnimationString_ShortPass;
        case e_FunctionType_LongPass:
            return animationType == e_AnimationString_LongPass;
        case e_FunctionType_HighPass:
            return animationType == e_AnimationString_HighPass;
        case e_FunctionType_Shot:
            return animationType == e_AnimationString_Shot;
        case e_FunctionType_Deflect:
            return animationType == e_AnimationString_Deflect;
        case e_FunctionType_Catch:
            return animationType == e_AnimationString_Catch;
        case e_FunctionType_Interfere:
            return animationType == e_AnimationString_Interfere;
        case e_FunctionType_Trip:
            return animationType == e_AnimationString_Trip;
        case e_FunctionType_Sliding:
            return animationType == e_AnimationString_Sliding;
        case e_FunctionType_Special:
            return animationType == e_AnimationString_Special;
        default:
            return false;
    }
}

struct alignas(128) CrudeRecord {
    struct AnimationFields {
        int32_t animationType, currentFoot;
        Vector3 incomingBodyDirection;
        float incomingVelocity, outgoingVelocity, specialVariable1, specialVariable2;
        int32_t incomingRetainState, incomingSpecialState;
        bool lastDitch;
    } animationFields;
    struct DerivedFields {
        Vector3 crudeOutDirectionRotated, crudeOutBodyDirection, crudeInBallAdapted,
            crudeBallDirectionNormalized;
        float crudeTurnAngle, crudeInBodyFixAbsolute, incomingMaximumDeviation,
            outgoingMaximumDeviation;
        int32_t tripTypeRounded, forcedFootWhich;
        bool crudeInBallNonzero, outgoingRetainRaw, touchFootLeft;
    } derivedFields;
};
static_assert(sizeof(CrudeRecord) == 128, "one cache line");
CUJU_FUNCTION inline CrudeRecord MakeCrudeRecord(const animation::Animation &sourceAnimation,
                                                 const AnimationDerived &sourceDerived) {
    CrudeRecord record;
    record.animationFields.animationType = sourceAnimation.animationType;
    record.animationFields.currentFoot = sourceAnimation.currentFoot;
    record.animationFields.incomingBodyDirection = sourceAnimation.incomingBodyDirection;
    record.animationFields.incomingVelocity = sourceAnimation.incomingVelocity;
    record.animationFields.outgoingVelocity = sourceAnimation.outgoingVelocity;
    record.animationFields.specialVariable1 = sourceAnimation.specialVariable1;
    record.animationFields.specialVariable2 = sourceAnimation.specialVariable2;
    record.animationFields.incomingRetainState = sourceAnimation.incomingRetainState;
    record.animationFields.incomingSpecialState = sourceAnimation.incomingSpecialState;
    record.animationFields.lastDitch = sourceAnimation.lastDitch;
    record.derivedFields.crudeOutDirectionRotated = sourceDerived.crudeOutDirectionRotated;
    record.derivedFields.crudeOutBodyDirection = sourceDerived.crudeOutBodyDirection;
    record.derivedFields.crudeInBallAdapted = sourceDerived.crudeInBallAdapted;
    record.derivedFields.crudeBallDirectionNormalized = sourceDerived.crudeBallDirectionNormalized;
    record.derivedFields.crudeTurnAngle = sourceDerived.crudeTurnAngle;
    record.derivedFields.crudeInBodyFixAbsolute = sourceDerived.crudeInBodyFixAbsolute;
    record.derivedFields.incomingMaximumDeviation = sourceDerived.incomingMaximumDeviation;
    record.derivedFields.outgoingMaximumDeviation = sourceDerived.outgoingMaximumDeviation;
    record.derivedFields.tripTypeRounded = sourceDerived.tripTypeRounded;
    record.derivedFields.forcedFootWhich = sourceDerived.forcedFootWhich;
    record.derivedFields.crudeInBallNonzero = sourceDerived.crudeInBallNonzero;
    record.derivedFields.outgoingRetainRaw = sourceDerived.outgoingRetainRaw;
    record.derivedFields.touchFootLeft = sourceDerived.touchFootLeft;
    return record;
}

// Exact: per-animation terms that the host computes with the selection's own expressions, so
// each equals its inline computation bit for bit on either backend.
struct SelectionRecord {
    float incomingBodyAngles[6];
    float bodyTurnAngle;
    int32_t incomingBodyIndex;
    Vector3 preferredOutgoingDirection, preferredOutgoingMovement;
    float preferredOutgoingAngle;
    float lookAngle;
    // Index of frame 0 in `SimulationContext::smoothedMovement`, -1 below two positions.
    int32_t smoothedMovementOffset;
    // Exact: the angle of a finite vector to itself is +0, as `x * y - y * x` is +0 and the dot
    // product is not negative.
    bool ballDirectionFinite;
};

inline SelectionRecord MakeSelectionRecord(const animation::Animation &animationData,
                                           const AnimationDerived &derivedData,
                                           int32_t smoothedMovementOffset) {
    SelectionRecord record{};
    for (int direction = 0; direction < 6; direction++)
        record.incomingBodyAngles[direction] = animationData.incomingBodyDirection.GetAngle2D(
            CUJU_TABLE(ALLOWED_BODY_DIRECTIONS)[direction]);
    record.bodyTurnAngle =
        animationData.incomingBodyDirection.GetAngle2D(derivedData.crudeOutBodyDirection);
    record.incomingBodyIndex =
        ForceIntoAllowedBodyDirectionIndex(animationData.incomingBodyDirection);
    record.preferredOutgoingDirection =
        ForceIntoPreferredDirectionVector(animationData.outgoingDirection);
    record.preferredOutgoingMovement =
        record.preferredOutgoingDirection * RangeVelocity(animationData.outgoingVelocity);
    record.preferredOutgoingAngle =
        ForceIntoPreferredDirectionAngle(animationData.outgoingAngle.value());
    record.lookAngle = ForceIntoAllowedBodyDirectionAngle(animationData.outgoingBodyAngle.value()) +
                       record.preferredOutgoingAngle;
    record.smoothedMovementOffset =
        animationData.positions.size() >= 2 ? smoothedMovementOffset : -1;
    const Vector3 &ballDirection = derivedData.crudeBallDirectionNormalized;
    record.ballDirectionFinite = std::isfinite(ballDirection.coordinates[0]) &&
                                 std::isfinite(ballDirection.coordinates[1]) &&
                                 std::isfinite(ballDirection.coordinates[2]);
    return record;
}

// `fabsf(ALLOWED_BODY_DIRECTIONS[i].GetAngle2D(ALLOWED_BODY_DIRECTIONS[j])) / PI` at `6 * i + j`.
// Speed: the lanes read different entries, so the table lives in global memory on the device.
CUJU_TABLE_DEFINITION_GLOBAL(float, ALLOWED_BODY_DIRECTION_KEYS,
                             {0x0p+0f,   0x1p-2f, 0x1p-2f, 0x1.8p-1f, 0x1.8p-1f, 0x0p+0f,
                              0x1p-2f,   0x0p+0f, 0x1p-1f, 0x1p-1f,   0x1p+0f,   0x1p+0f,
                              0x1p-2f,   0x1p-1f, 0x0p+0f, 0x1p+0f,   0x1p-1f,   0x0p+0f,
                              0x1.8p-1f, 0x1p-1f, 0x1p+0f, 0x0p+0f,   0x1p-1f,   0x0p+0f,
                              0x1.8p-1f, 0x1p+0f, 0x1p-1f, 0x1p-1f,   0x0p+0f,   0x0p+0f,
                              0x0p+0f,   0x1p+0f, 0x0p+0f, 0x0p+0f,   0x0p+0f,   0x0p+0f});
// `fabsf(FixAngle(ALLOWED_BODY_DIRECTIONS[i].GetAngle2D()))` for each entry `i`.
CUJU_TABLE_DEFINITION(float, ALLOWED_BODY_DIRECTION_FIX_ABSOLUTE,
                      {0x0p+0f, 0x1.921fbp-1f, 0x1.921fcp-1f, 0x1.2d97c8p+1f, 0x1.2d97c8p+1f,
                       0x1.921fb6p+0f});
// `fabsf(Vector3(0, -1, 0).GetAngle2D(ALLOWED_BODY_DIRECTIONS[i]))` for each entry `i`.
CUJU_TABLE_DEFINITION(float, ALLOWED_BODY_DIRECTION_IDLE_ABSOLUTE,
                      {0x0p+0f, 0x1.921fb6p-1f, 0x1.921fb6p-1f, 0x1.2d97c8p+1f, 0x1.2d97c8p+1f,
                       0x0p+0f});

// Bit for bit. A state loaded from elsewhere may hold a body direction outside the table.
CUJU_FUNCTION inline int AllowedBodyDirectionEntry(const Vector3 &direction) {
    for (int entry = 0; entry < 6; entry++) {
        const Vector3 &allowed = CUJU_TABLE(ALLOWED_BODY_DIRECTIONS)[entry];
        if (libm::FloatBits(direction.coordinates[0]) == libm::FloatBits(allowed.coordinates[0]) &&
            libm::FloatBits(direction.coordinates[1]) == libm::FloatBits(allowed.coordinates[1]) &&
            libm::FloatBits(direction.coordinates[2]) == libm::FloatBits(allowed.coordinates[2]))
            return entry;
    }
    return -1;
}

constexpr int FUNCTION_TYPE_SLOTS = 16;
inline void BuildFunctionIndex(const animation::AnimationCollection &collection,
                               std::vector<int16_t> &animations, std::vector<int32_t> &offset) {
    animations.clear();
    offset.assign(FUNCTION_TYPE_SLOTS + 1, 0);
    for (int functionType = 0; functionType < FUNCTION_TYPE_SLOTS; functionType++) {
        offset[functionType] = (int32_t)animations.size();
        for (int animationIndex = 0; animationIndex < (int)collection.animations.size();
             animationIndex++)
            if (CheckFunctionType(collection.animations[animationIndex].animationType,
                                  functionType))
                animations.push_back((int16_t)animationIndex);
    }
    offset[FUNCTION_TYPE_SLOTS] = (int32_t)animations.size();
}

CUJU_FUNCTION_NO_INLINE inline void CrudeSelection(const SelectContext &selectContext,
                                                   const CrudeSelectionQuery &query,
                                                   AnimationList &dataSet) {
    CUJU_PROFILE_SCOPE(CRUDE_SELECTION);
    dataSet.clear();
    const animation::AnimationCollection &collection = *selectContext.context.animations;
    const float marginRadians = 0.06f * PI;
    Vector3 fencedDirection;
    float queryIncomingToFenceAngle = 0.0f;
    if (query.bySide) {
        fencedDirection = query.lookAtVectorRelative.GetRotated2D(PI);
        queryIncomingToFenceAngle = fencedDirection.GetAngle2D(query.incomingBodyDirection);
    }
    const bool byBodyDirection =
        query.byIncomingBodyDirection &&
        !(query.byIncomingVelocity && query.incomingVelocity == e_Velocity_Idle);
    const int queryBodyEntry =
        byBodyDirection ? AllowedBodyDirectionEntry(query.incomingBodyDirection) : -1;
    float queryInBodyFixAbsolute = 0.0f, queryIdleBodyAngleAbsolute = 0.0f;
    if (queryBodyEntry >= 0) {
        queryInBodyFixAbsolute = CUJU_TABLE(ALLOWED_BODY_DIRECTION_FIX_ABSOLUTE)[queryBodyEntry];
        queryIdleBodyAngleAbsolute =
            CUJU_TABLE(ALLOWED_BODY_DIRECTION_IDLE_ABSOLUTE)[queryBodyEntry];
    } else if (byBodyDirection) {
        queryInBodyFixAbsolute = fabsf(FixAngle(query.incomingBodyDirection.GetAngle2D()));
        queryIdleBodyAngleAbsolute =
            fabsf(Vector3(0, -1, 0).GetAngle2D(query.incomingBodyDirection));
    }
    const SelectionRecord *selection = selectContext.context.selection;
    // Exact: the index holds what `CheckFunctionType` accepts, in increasing order, so the result
    // equals a scan of every animation.
    const int16_t *list = nullptr;
    int candidateCount = (int)collection.animations.size();
    if (query.byFunctionType && selectContext.context.functionTypeAnimations &&
        query.functionType >= 0 && query.functionType < FUNCTION_TYPE_SLOTS) {
        list = selectContext.context.functionTypeAnimations +
               selectContext.context.functionTypeOffsets[query.functionType];
        candidateCount = selectContext.context.functionTypeOffsets[query.functionType + 1] -
                         selectContext.context.functionTypeOffsets[query.functionType];
    }
    const bool queryInBallNonzero = query.incomingBallDirection.GetLength() != 0.0f;
    Vector3 queryInBallAdapted = query.incomingBallDirection;
    queryInBallAdapted.coordinates[2] *= 0.4f;
    if (queryInBallNonzero) queryInBallAdapted.Normalize();
    const Vector3 queryOutBall2D = query.outgoingBallDirection.Get2D();
    // Exact: the null test of `GetNormalized`, which returns its fallback for a null vector.
    const bool queryOutBallNull = fabsf(queryOutBall2D.coordinates[0]) < 0.000001f &&
                                  fabsf(queryOutBall2D.coordinates[1]) < 0.000001f &&
                                  fabsf(queryOutBall2D.coordinates[2]) < 0.000001f;
    const Vector3 queryOutBallNormalized =
        queryOutBallNull ? Vector3(0) : queryOutBall2D.GetNormalized(Vector3(0));
    auto pass = [&](int animationIndex) -> bool {
        CrudeRecord temporary;
        const CrudeRecord &record =
            selectContext.context.crude
                ? selectContext.context.crude[animationIndex]
                : (temporary = MakeCrudeRecord(collection.animations[animationIndex],
                                               selectContext.GetDerived(animationIndex)),
                   temporary);
        const CrudeRecord::AnimationFields &animationFields = record.animationFields;
        const CrudeRecord::DerivedFields &derivedFields = record.derivedFields;
        int animationType = animationFields.animationType;
        if (query.byFunctionType && !list && !CheckFunctionType(animationType, query.functionType))
            return false;

        if (query.byIncomingVelocity) {
            int animationIncomingVelocity =
                FloatToVelocityCategory(animationFields.incomingVelocity);
            if (!query.incomingVelocityStrict) {
                if (query.incomingVelocityNoDribbleToIdle &&
                    animationIncomingVelocity == e_Velocity_Idle &&
                    query.incomingVelocity == e_Velocity_Dribble)
                    return false;
                if (animationIncomingVelocity == e_Velocity_Idle &&
                    query.incomingVelocity == e_Velocity_Walk)
                    return false;
                if (animationIncomingVelocity == e_Velocity_Idle &&
                    query.incomingVelocity == e_Velocity_Sprint)
                    return false;
                if (animationIncomingVelocity == e_Velocity_Dribble &&
                    query.incomingVelocity == e_Velocity_Idle)
                    return false;
                if (animationIncomingVelocity == e_Velocity_Walk &&
                    query.incomingVelocity == e_Velocity_Idle)
                    return false;
                if (animationIncomingVelocity == e_Velocity_Sprint &&
                    query.incomingVelocity == e_Velocity_Idle)
                    return false;
                if (query.incomingVelocityNoDribbleToSprint &&
                    animationIncomingVelocity == e_Velocity_Sprint &&
                    query.incomingVelocity == e_Velocity_Dribble)
                    return false;
                if (query.incomingVelocityForceLinearity) {
                    float animationIncoming = RangeVelocity(animationFields.incomingVelocity);
                    float animationOutgoing = RangeVelocity(animationFields.outgoingVelocity);
                    float queryIncoming = VelocityCategoryToFloat(query.incomingVelocity);
                    if (FloatToVelocityCategory(animationIncoming) == e_Velocity_Dribble)
                        animationIncoming = WALK_VELOCITY;
                    if (FloatToVelocityCategory(animationOutgoing) == e_Velocity_Dribble)
                        animationOutgoing = WALK_VELOCITY;
                    if (FloatToVelocityCategory(queryIncoming) == e_Velocity_Dribble)
                        queryIncoming = WALK_VELOCITY;
                    // GRF: `std::max` and `std::min`, with `queryIncoming` as the first argument.
                    float upperVelocity =
                        (queryIncoming < animationOutgoing) ? animationOutgoing : queryIncoming;
                    float lowerVelocity =
                        (animationOutgoing < queryIncoming) ? animationOutgoing : queryIncoming;
                    if (animationIncoming > upperVelocity) return false;
                    if (animationIncoming < lowerVelocity) return false;
                }
            } else {
                if (animationIncomingVelocity != query.incomingVelocity) return false;
            }
        }

        if (query.byOutgoingVelocity &&
            FloatToVelocityCategory(animationFields.outgoingVelocity) != query.outgoingVelocity)
            return false;

        if (query.bySide) {
            const Vector3 &animationIncomingDirection = animationFields.incomingBodyDirection;
            const Vector3 &animationOutgoingDirection = derivedFields.crudeOutDirectionRotated;
            const float animationTurnAngle = derivedFields.crudeTurnAngle;
            if (fabsf(animationTurnAngle) > 0.06f * PI) {
                int animationSide = (animationTurnAngle > 0) ? 0 : 1;
                float animationIncomingToFenceAngle =
                    fencedDirection.GetAngle2D(animationIncomingDirection);
                float fenceToOutgoingAngle = animationOutgoingDirection.GetAngle2D(fencedDirection);
                int animationIncomingToFenceSide = (animationIncomingToFenceAngle > 0) ? 0 : 1;
                int queryIncomingToFenceSide = (queryIncomingToFenceAngle > 0) ? 0 : 1;
                int fenceToAnimationOutgoingSide = (fenceToOutgoingAngle > 0) ? 0 : 1;
                if (animationIncomingToFenceSide == animationSide &&
                    fenceToAnimationOutgoingSide == animationSide &&
                    fabsf(animationIncomingToFenceAngle + fenceToOutgoingAngle) < PI)
                    return false;
                // GRF: adds the side number 0 or 1 to an angle in radians, kept as written.
                if (queryIncomingToFenceSide == animationSide &&
                    fenceToAnimationOutgoingSide == animationSide &&
                    fabsf((float)queryIncomingToFenceSide + fenceToOutgoingAngle) < PI)
                    return false;
            }
        }

        if (query.byPickupBall) {
            if ((!derivedFields.outgoingRetainRaw && query.pickupBall) ||
                (derivedFields.outgoingRetainRaw && !query.pickupBall))
                return false;
        }

        if (!query.allowLastDitchAnimations && animationFields.lastDitch) return false;

        if (byBodyDirection) {
            if (FloatToVelocityCategory(animationFields.incomingVelocity) != e_Velocity_Idle) {
                Vector3 animationIncomingBodyDirection = animationFields.incomingBodyDirection;
                if (derivedFields.crudeInBodyFixAbsolute > queryInBodyFixAbsolute + marginRadians)
                    return false;
                const Vector3 &animationOutgoingBodyDirection = derivedFields.crudeOutBodyDirection;
                const float queryOffsetAngle =
                    selection && queryBodyEntry >= 0
                        ? selection[animationIndex].incomingBodyAngles[queryBodyEntry]
                        : animationIncomingBodyDirection.GetAngle2D(query.incomingBodyDirection);
                if (query.incomingBodyDirectionStrict) {
                    if (fabsf(queryOffsetAngle) > marginRadians) return false;
                } else {
                    if (fabsf(queryOffsetAngle) > 0.5f * PI + marginRadians) return false;
                }
                if (query.incomingBodyDirectionForceLinearity) {
                    float bodyTurnAngle = selection ? selection[animationIndex].bodyTurnAngle
                                                    : animationIncomingBodyDirection.GetAngle2D(
                                                          animationOutgoingBodyDirection);
                    if ((bodyTurnAngle > marginRadians && queryOffsetAngle > marginRadians) ||
                        (bodyTurnAngle < -marginRadians && queryOffsetAngle < -marginRadians))
                        return false;
                    if (fabsf(bodyTurnAngle) + fabsf(queryOffsetAngle) > PI + marginRadians)
                        return false;
                }
            } else {
                if (query.incomingBodyDirectionStrict) {
                    if (queryIdleBodyAngleAbsolute > marginRadians) return false;
                } else {
                    if (queryIdleBodyAngleAbsolute > 0.25f * PI + marginRadians) return false;
                }
            }
        }

        if (query.byIncomingBallDirection) {
            if (derivedFields.crudeInBallNonzero && queryInBallNonzero) {
                float ballDirectionAngle =
                    fabsf(queryInBallAdapted.GetAngle2D(derivedFields.crudeInBallAdapted));
                float maximumDeviation = derivedFields.incomingMaximumDeviation;
                if (maximumDeviation == 0.0f) {
                    maximumDeviation = collection.maximumIncomingBallDirectionDeviation.value();
                    if (animationType == animation::e_AnimationString_Deflect)
                        maximumDeviation = 0.4f * PI;
                }
                if (ballDirectionAngle > maximumDeviation) return false;
            }
        }

        if (query.byOutgoingBallDirection) {
            const Vector3 &animationBallDirection = derivedFields.crudeBallDirectionNormalized;
            // Exact: a null query compares the direction with itself, +0 when it is finite.
            float ballDirectionAngle =
                queryOutBallNull && selection && selection[animationIndex].ballDirectionFinite
                    ? 0.0f
                    : fabsf((queryOutBallNull ? animationBallDirection : queryOutBallNormalized)
                                .GetAngle2D(animationBallDirection));
            float maximumDeviation = derivedFields.outgoingMaximumDeviation;
            if ((double)maximumDeviation == 0.0)
                maximumDeviation = collection.maximumOutgoingBallDirectionDeviation.value();
            if (ballDirectionAngle > maximumDeviation) return false;
        }

        if (query.incomingSpecialState != animationFields.incomingSpecialState) return false;
        if ((query.functionType == e_FunctionType_Deflect ||
             ((query.incomingRetainState != animation::STRING_EMPTY) !=
              (animationFields.incomingRetainState != animation::STRING_EMPTY))) &&
            query.incomingRetainState != animationFields.incomingRetainState)
            return false;
        if (query.specialVariable1 != animationFields.specialVariable1) return false;
        if (query.specialVariable2 != animationFields.specialVariable2) return false;

        if (query.byTripType && derivedFields.tripTypeRounded != query.tripType) return false;

        if (query.heedForcedFoot && derivedFields.forcedFootWhich != 0) {
            int animationFoot = derivedFields.touchFootLeft ? 0 : 1;
            if (animationFields.currentFoot == 0) animationFoot = animationFoot == 0 ? 1 : 0;
            if (derivedFields.forcedFootWhich == 1 && query.strongFoot != animationFoot)
                return false;
            if (derivedFields.forcedFootWhich == 2 && query.strongFoot == animationFoot)
                return false;
        }
        return true;
    };
    if (candidateCount > AnimationList::capacity()) InlineVectorOverflow();
    // Speed: the count stays in a register until the end.
    int count = 0;
    WarpFilterEmit(
        0, candidateCount, [&](int candidate) { return pass(list ? list[candidate] : candidate); },
        [&](int candidate) { dataSet[count++] = (int16_t)(list ? list[candidate] : candidate); });
    dataSet.count = count;
}

// Directions are relative to the humanoid's angle, and velocities are categories.
struct Predicates {
    int desiredFoot = 0;
    int incomingVelocity = 0;
    Vector3 relativeDesiredDirection;
    float corneringBias = 0.0f;
    int desiredVelocity = 0;
    Vector3 relativeIncomingBodyDirection;
    // World coordinates, unlike the directions.
    Vector3 lookAt;
    Vector3 relativeDesiredTripDirection;
    float idle = 0.0f;
};

// Exact: `std::max(left, right)`, so `left` wins a tie or a NaN.
CUJU_FUNCTION inline float StandardMaximum(float left, float right) {
    return (left < right) ? right : left;
}

CUJU_FUNCTION inline float MovementSimilarityOf(const Vector3 &outgoingDirection,
                                                const Vector3 &outgoingMovement,
                                                const Vector3 &relativeDesiredDirection,
                                                int desiredVelocity, float corneringBias) {
    Vector3 desiredMovement = relativeDesiredDirection * VelocityCategoryToFloat(desiredVelocity);
    desiredMovement = desiredMovement * (1.0f - corneringBias);
    float value = (desiredMovement - outgoingMovement).GetLength();
    value -= fabsf(relativeDesiredDirection.GetDotProduct(outgoingDirection)) * 4.0f;
    return value;
}
CUJU_FUNCTION inline float GetMovementSimilarity(const animation::Animation &animationData,
                                                 const Vector3 &relativeDesiredDirection,
                                                 int desiredVelocity, float corneringBias) {
    Vector3 outgoingDirection = ForceIntoPreferredDirectionVector(animationData.outgoingDirection);
    float outgoingVelocity = RangeVelocity(animationData.outgoingVelocity);
    Vector3 outgoingMovement = outgoingDirection * outgoingVelocity;
    return MovementSimilarityOf(outgoingDirection, outgoingMovement, relativeDesiredDirection,
                                desiredVelocity, corneringBias);
}

CUJU_FUNCTION_NO_INLINE inline float DirectionSimilarityRating(
    const animation::Animation &animationData, const Predicates &predicates,
    const Vector3 &relativeLookAt) {
    Vector3 relativeDesiredBodyDirection =
        (relativeLookAt - animationData.translation).GetNormalized(Vector3(0, -1, 0));
    float maximumAngleSmuggle = 0.1f * PI;
    float outgoingAngle =
        animationData.outgoingDirection
            .GetRotated2D(clamp(
                predicates.relativeDesiredDirection.GetAngle2D(animationData.outgoingDirection),
                -maximumAngleSmuggle, maximumAngleSmuggle))
            .GetAngle2D(Vector3(0, -1, 0));
    Vector3 predictedOutgoingBodyDirection =
        animationData.outgoingBodyDirection.GetRotated2D(outgoingAngle);
    float rating = fabsf(predictedOutgoingBodyDirection.GetAngle2D(relativeDesiredBodyDirection));
    rating += fabsf(animationData.outgoingBodyAngle.value()) * 0.05f;
    return rating;
}

CUJU_FUNCTION inline float KeyIdle(const animation::Animation &animationData,
                                   const Predicates &predicates) {
    return fabsf(animationData.idleLevel - predicates.idle);
}
CUJU_FUNCTION inline int KeyFoot(const animation::Animation &animationData,
                                 const Predicates &predicates) {
    int rating = 1;
    if (animationData.currentFoot == predicates.desiredFoot) rating = 0;
    if (FloatToVelocityCategory(animationData.incomingVelocity) == e_Velocity_Idle) rating = 0;
    return rating;
}
CUJU_FUNCTION inline float KeyIncomingBodyDirection(const animation::Animation &animationData,
                                                    const Predicates &predicates) {
    float rating = fabsf(ForceIntoAllowedBodyDirectionVector(animationData.incomingBodyDirection)
                             .GetAngle2D(ForceIntoAllowedBodyDirectionVector(
                                 predicates.relativeIncomingBodyDirection))) /
                   PI;
    if (FloatToVelocityCategory(animationData.incomingVelocity) == e_Velocity_Idle) rating = 0;
    return rating;
}
CUJU_FUNCTION inline float KeyIncomingVelocity(const animation::Animation &animationData,
                                               const Predicates &predicates) {
    int currentVelocityID = GetVelocityID(predicates.incomingVelocity);
    int incomingID = GetVelocityID(FloatToVelocityCategory(animationData.incomingVelocity));
    float rating = fabsf(clamp((float)(incomingID - currentVelocityID), -3, 3));
    int outgoingID = GetVelocityID(FloatToVelocityCategory(animationData.outgoingVelocity));
    if (incomingID > std::max(currentVelocityID, outgoingID)) rating += 0.5f;
    if (incomingID < std::min(currentVelocityID, outgoingID)) rating += 0.5f;
    return rating;
}

// Exact: the search compares `GetLength`, and the result is `GetDistance`.
CUJU_FUNCTION_NO_INLINE inline float ClosestOpponentDistance(const World &world, int player) {
    const int opponentTeam = 1 - world.playerTeamID[player];
    const Vector3 position = Humanoid(world, player).spatialState.position;
    float closestDistance = 10000;
    int closest = -1;
    for (int other = 0; other < world.serialized.match.playerCount; other++) {
        if (world.playerTeamID[other] != opponentTeam ||
            !world.serialized.match.players[other].base.isActive)
            continue;
        float distance = (Humanoid(world, other).spatialState.position - position).GetLength();
        if (distance < closestDistance) {
            closestDistance = distance;
            closest = other;
        }
    }
    if (closest < 0) CUJU_FATAL("a player has no active opponent");
    return GetDistance(Humanoid(world, closest).spatialState.position, position);
}

// `Curve(NormalizedClamp(10 * frame, 0, 160), 1)` for frames 0 to 16, which covers every frame.
// Speed: the lanes read different frames, so the table lives in global memory on the device.
CUJU_TABLE_DEFINITION_GLOBAL(float, PHYSICS_CURVED_TIME,
                             {0x0p+0f, 0x1.3ad04p-7f, 0x1.37ca2p-5f, 0x1.59267p-4f, 0x1.2bec34p-3f,
                              0x1.c71898p-3f, 0x1.3c10eap-2f, 0x1.9c1d2p-2f, 0x1p-1f, 0x1.31f17p-1f,
                              0x1.61f78cp-1f, 0x1.8e39dap-1f, 0x1.b504f4p-1f, 0x1.d4db32p-1f,
                              0x1.ec835ep-1f, 0x1.fb14cp-1f, 0x1p+0f});

CUJU_FUNCTION inline float PhysicsCurvedTime(int frame) {
    return CUJU_TABLE(PHYSICS_CURVED_TIME)[frame < 0 ? 0 : frame > 16 ? 16 : frame];
}

struct Selector {
    World &world;
    const SimulationContext &context;
    const AnimationDerived *derived;
    int humanoidIndex;
    Predicates predicates;
    struct TurnCache {
        bool valid = false;
        uint32_t angleBits = 0;
        libm::SineCosine turn{};
    };
    // Speed: cached within a selection and keyed by the bits of the angle.
    TurnCache forwardTurn, backTurn;

    CUJU_FUNCTION Selector(World &inputWorld, const SimulationContext &inputContext, int inputIndex)
        : world(inputWorld),
          context(inputContext),
          derived(inputContext.derived),
          humanoidIndex(inputIndex) {}

    CUJU_FUNCTION static libm::SineCosine CachedTurn(TurnCache &cache, float angle,
                                                     float argument) {
        const uint32_t bits = libm::FloatBits(angle);
        if (!cache.valid || cache.angleBits != bits) {
            cache.turn = libm::sincosf(argument);
            cache.angleBits = bits;
            cache.valid = true;
        }
        return cache.turn;
    }
    CUJU_FUNCTION libm::SineCosine Turn(float angle) {
        return CachedTurn(forwardTurn, angle, angle);
    }
    // Exact: computed apart from `Turn`, since `libm::cosf` is not shown to be even.
    CUJU_FUNCTION libm::SineCosine BackTurn(float angle) {
        return CachedTurn(backTurn, angle, -angle);
    }
    CUJU_FUNCTION Vector3 RelativeLookAt() {
        const state::SpatialState &spatialState = GetSpatialState();
        return RotatedBy(predicates.lookAt - spatialState.position,
                         BackTurn(spatialState.angle.value()));
    }

    CUJU_FUNCTION float MovementSimilarityKey(int index) {
        if (context.selection) {
            const SelectionRecord &record = context.selection[index];
            return MovementSimilarityOf(record.preferredOutgoingDirection,
                                        record.preferredOutgoingMovement,
                                        predicates.relativeDesiredDirection,
                                        predicates.desiredVelocity, predicates.corneringBias);
        }
        return GetMovementSimilarity(A(index), predicates.relativeDesiredDirection,
                                     predicates.desiredVelocity, predicates.corneringBias);
    }
    CUJU_FUNCTION void SortByIncomingBodyDirection(AnimationList &dataSet) {
        // A single animation computes no key.
        const int queryIndex = dataSet.count > 1 ? IncomingBodyDirectionQuery() : 0;
        Sort<float>(dataSet, [&](int animationIndex) {
            return IncomingBodyDirectionKey(animationIndex, queryIndex);
        });
    }
    CUJU_FUNCTION int IncomingBodyDirectionQuery() {
        return ForceIntoAllowedBodyDirectionIndex(predicates.relativeIncomingBodyDirection);
    }
    CUJU_FUNCTION float IncomingBodyDirectionKey(int animationIndex, int queryIndex) {
        if (!context.selection) return KeyIncomingBodyDirection(A(animationIndex), predicates);
        float rating = CUJU_TABLE(
            ALLOWED_BODY_DIRECTION_KEYS)[6 * context.selection[animationIndex].incomingBodyIndex +
                                         queryIndex];
        if (FloatToVelocityCategory(A(animationIndex).incomingVelocity) == e_Velocity_Idle)
            rating = 0;
        return rating;
    }
    CUJU_FUNCTION float PreferredOutgoingAngle(int index) {
        if (context.selection) return context.selection[index].preferredOutgoingAngle;
        return ForceIntoPreferredDirectionAngle(A(index).outgoingAngle.value());
    }
    CUJU_FUNCTION float LookAngle(int index) {
        if (context.selection) return context.selection[index].lookAngle;
        const animation::Animation &animationData = A(index);
        return ForceIntoAllowedBodyDirectionAngle(animationData.outgoingBodyAngle.value()) +
               ForceIntoPreferredDirectionAngle(animationData.outgoingAngle.value());
    }
    CUJU_FUNCTION Vector3 SmoothedMovement(int index, uint32_t frame) {
        const Span<Vector3> &positions = A(index).positions;
        if (context.selection && context.smoothedMovement) {
            const int32_t offset = context.selection[index].smoothedMovementOffset;
            if (offset >= 0 && frame < (uint32_t)positions.size())
                return context.smoothedMovement[offset + frame];
        }
        return CalculateMovementAtFrame(positions, frame, 1);
    }

    CUJU_FUNCTION bool Official() const {
        return humanoidIndex >= world.serialized.match.playerCount;
    }
    CUJU_FUNCTION state::PlayerState &GetPlayer() {
        return world.serialized.match.players[humanoidIndex];
    }
    CUJU_FUNCTION HumanoidState &GetHumanoid() {
        return Official() ? world.serialized.match
                                .officials[humanoidIndex - world.serialized.match.playerCount]
                                .humanoid
                          : GetPlayer().base.humanoid;
    }
    CUJU_FUNCTION state::SpatialState &GetSpatialState() { return GetHumanoid().spatialState; }
    CUJU_FUNCTION const animation::Animation &A(int index) const {
        return context.animations->animations[index];
    }
    CUJU_FUNCTION const AnimationDerived &GetDerived(int index) const { return derived[index]; }
    CUJU_FUNCTION SelectContext SC() {
        return SelectContext{world, context, derived, humanoidIndex};
    }
    CUJU_FUNCTION int MentalImage() {
        return MentalImageIndex(world, GetHumanoid().mentalImageTime);
    }
    CUJU_FUNCTION Vector3 MentalBallPredictionAt(int timeMilliseconds) {
        return MentalBallPrediction(world, MentalImage(), timeMilliseconds);
    }
    CUJU_FUNCTION float Statistic(int statistic) {
        return Official() ? context.staticData->data[humanoidIndex].statistics[statistic]
                          : GetStatistic(world, context, humanoidIndex, statistic);
    }
    CUJU_FUNCTION float BaseLastTouchBias(int decayMilliseconds) {
        uint64_t last = Official()
                            ? world.serialized.match
                                  .officials[humanoidIndex - world.serialized.match.playerCount]
                                  .lastTouchTimeMilliseconds
                            : GetPlayer().base.lastTouchTimeMilliseconds;
        if (decayMilliseconds > 0)
            return 1.0f - clamp((float)(world.serialized.match.actualTimeMilliseconds - last) /
                                    (float)decayMilliseconds,
                                0.0f, 1.0f);
        return 0.0f;
    }

    CUJU_FUNCTION void SetMovementSimilarityPredicate(const Vector3 &relativeDesiredDirection,
                                                      int desiredVelocity) {
        predicates.relativeDesiredDirection = relativeDesiredDirection;
        predicates.desiredVelocity = desiredVelocity;
        predicates.corneringBias = CalculateBiasForFastCornering(
            Vector3(0, -1.0f * GetSpatialState().floatVelocity, 0),
            relativeDesiredDirection * VelocityCategoryToFloat(desiredVelocity), 1.0f, 0.9f);
    }

    template <class Key, class KeyFunction>
    CUJU_FUNCTION void Sort(AnimationList &dataSet, KeyFunction key) {
        StableSortByKey<Key>(dataSet, key);
    }
    CUJU_FUNCTION void SortBaseAnimation(AnimationList &dataSet) {
        Sort<int>(dataSet,
                  [&](int animationIndex) { return A(animationIndex).baseAnimation ? 0 : 1; });
    }

    CUJU_FUNCTION bool AllowLastDitch(bool includingPossessionAmount = true) {
        if (includingPossessionAmount &&
            TeamOfPlayer(world, humanoidIndex).teamPossessionAmount < 1.0f)
            return true;
        return ((int)GetPlayer().timeNeededToGetToBallOptimisticMilliseconds * 1.7f + 800 <
                (float)(int)GetPlayer().timeNeededToGetToBallMilliseconds);
    }

    CUJU_FUNCTION bool HighOrBouncyBall() {
        const state::BallSerialized &ball = world.serialized.match.ball;
        float ballHeightNextTick = BallPredict(ball, 10).coordinates[2];
        float ballHeightAtTouch =
            BallPredict(ball, DEFAULT_TOUCH_OFFSET_MILLISECONDS).coordinates[2];
        float ballBounce = fabsf(ball.momentum.coordinates[2]);
        if (ballHeightNextTick > 0.3f || ballHeightAtTouch > 0.3f) return true;
        if (ballBounce > 1.0f) return true;
        return false;
    }

    CUJU_FUNCTION_NO_INLINE void KeepBestDirectionAnimations(AnimationList &dataSet,
                                                             const state::PlayerCommand &command,
                                                             bool strict, float allowedAngle = 0,
                                                             int allowedVelocitySteps = 0,
                                                             int forcedQuadrantID = -1) {
        CUJU_PROFILE_SCOPE(KEEP_BEST_DIRECTION);
        const animation::AnimationCollection &collection = *context.animations;
        int bestQuadrantID = forcedQuadrantID;
        if (bestQuadrantID == -1) {
            Sort<float>(dataSet,
                        [&](int animationIndex) { return MovementSimilarityKey(animationIndex); });
            if (strict && command.desiredFunctionType != e_FunctionType_Movement)
                SortBaseAnimation(dataSet);
            bestQuadrantID = A(dataSet[0]).quadrantID;
        }
        int kept = 1;
        WarpFilterEmit(
            1, dataSet.count,
            [&](int candidate) {
                const animation::Animation &animationData = A(dataSet[candidate]);
                if (strict) return animationData.quadrantID == bestQuadrantID;
                const animation::Quadrant &bestQuadrant = collection.quadrants.at(bestQuadrantID);
                const animation::Quadrant &quadrant =
                    collection.quadrants.at(animationData.quadrantID);
                bool predicate = true;
                if (!animationData.lastDitch) {
                    if (abs(GetVelocityID(quadrant.velocity, true) -
                            GetVelocityID(bestQuadrant.velocity, true)) > allowedVelocitySteps)
                        predicate = false;
                }
                if (fabsf(quadrant.angle.value() - bestQuadrant.angle.value()) > allowedAngle)
                    predicate = false;
                return predicate;
            },
            [&](int candidate) { dataSet[kept++] = dataSet[candidate]; });
        dataSet.count = kept;
    }

    CUJU_FUNCTION_NO_INLINE void KeepBestBodyDirectionAnimations(
        AnimationList &dataSet, const state::PlayerCommand &command, bool strict,
        float allowedAngle = 0) {
        CUJU_PROFILE_SCOPE(KEEP_BEST_BODY_DIRECTION);
        // A single animation computes no key.
        Vector3 relativeLookAt;
        if (dataSet.count > 1) relativeLookAt = RelativeLookAt();
        Sort<float>(dataSet, [&](int animationIndex) {
            return DirectionSimilarityRating(A(animationIndex), predicates, relativeLookAt);
        });
        if (strict && command.desiredFunctionType != e_FunctionType_Movement)
            SortBaseAnimation(dataSet);
        float bestLookAngle = LookAngle(dataSet[0]);
        int kept = 1;
        WarpFilterEmit(
            1, dataSet.count,
            [&](int candidate) {
                float animationLookAngle = LookAngle(dataSet[candidate]);
                float adaptedAllowedAngle = 0.06f * PI;
                if (!strict) adaptedAllowedAngle = allowedAngle;
                return fabsf(animationLookAngle - bestLookAngle) <= adaptedAllowedAngle;
            },
            [&](int candidate) { dataSet[kept++] = dataSet[candidate]; });
        dataSet.count = kept;
    }

    CUJU_FUNCTION_NO_INLINE int GetIdleMovementAnimationID() {
        CrudeSelectionQuery query;
        query.byFunctionType = true;
        query.functionType = e_FunctionType_Movement;
        query.byIncomingVelocity = true;
        query.incomingVelocity = e_Velocity_Idle;
        query.byOutgoingVelocity = true;
        query.outgoingVelocity = e_Velocity_Idle;
        AnimationList dataSet;
        CrudeSelection(SC(), query, dataSet);
        // GRF: sorts by idle level, incoming body direction, incoming velocity, direction and then
        // movement similarity. Each key reads only its own predicates, so all are set first.
        predicates.idle = 1;
        predicates.relativeIncomingBodyDirection = Vector3(0, -1, 0);
        predicates.incomingVelocity = e_Velocity_Idle;
        SetMovementSimilarityPredicate(Vector3(0, -1, 0), e_Velocity_Idle);
        predicates.lookAt = GetSpatialState().position +
                            RotatedBy(Vector3(0, -10, 0), Turn(GetSpatialState().angle.value()));
        // A single animation computes no key.
        Vector3 relativeLookAt;
        int queryIndex = 0;
        if (dataSet.count > 1) {
            relativeLookAt = RelativeLookAt();
            queryIndex = IncomingBodyDirectionQuery();
        }
        return FirstOfSortChain(dataSet, 5, [&](int animationIndex, double *keys) {
            keys[0] = MovementSimilarityKey(animationIndex);
            keys[1] = DirectionSimilarityRating(A(animationIndex), predicates, relativeLookAt);
            keys[2] = KeyIncomingVelocity(A(animationIndex), predicates);
            keys[3] = IncomingBodyDirectionKey(animationIndex, queryIndex);
            keys[4] = KeyIdle(A(animationIndex), predicates);
        });
    }

    CUJU_FUNCTION_NO_INLINE Vector3 CalculatePhysicsVector(
        int animationIndex, bool useDesiredMovement, const Vector3 &desiredMovement,
        bool useDesiredBodyDirection, const Vector3 &desiredBodyDirectionRelative,
        PositionList &positionsResult, float &rotationOffsetResult) {
        CUJU_PROFILE_SCOPE(PHYSICS_VECTOR);
        // GRF: `mod_CheatBodyDirection` is false, so this stays unused.
        (void)useDesiredBodyDirection;
        const animation::Animation &animationData = A(animationIndex);
        const AnimationDerived &derivedData = GetDerived(animationIndex);
        state::SpatialState &spatialState = GetSpatialState();
        const float angle = spatialState.angle.value();
        const libm::SineCosine turn = Turn(angle);
        positionsResult.clear();

        int animationTouchFrame = derivedData.touchFrame;
        bool touch = (animationTouchFrame > 0);
        float statisticAgility = Statistic(e_Statistic_PhysicalAgility);
        float statisticAcceleration = Statistic(e_Statistic_PhysicalAcceleration);
        float statisticDribble = Statistic(e_Statistic_TechnicalDribble);

        float incomingSwitchBias = 0.0f;
        float outgoingSwitchBias = 0.0f;
        const int animationType = animationData.animationType;
        using namespace animation;
        if (animationType == e_AnimationString_BallControl)
            outgoingSwitchBias = 0.0f;
        else if (animationType == e_AnimationString_Trap)
            outgoingSwitchBias = 0.0f;
        else if (animationType == e_AnimationString_Interfere)
            outgoingSwitchBias = 0.0f;
        else if (animationType == e_AnimationString_Deflect)
            outgoingSwitchBias = 1.0f;
        else if (animationType == e_AnimationString_Sliding)
            outgoingSwitchBias = 0.0f;
        else if (animationType == e_AnimationString_Special)
            outgoingSwitchBias = 1.0f;
        else if (animationType == e_AnimationString_Trip)
            outgoingSwitchBias = 0.5f;
        else if (touch)
            outgoingSwitchBias = 1.0f;
        if (animationData.incomingSpecialState != animation::STRING_EMPTY ||
            animationData.outgoingSpecialState != animation::STRING_EMPTY)
            outgoingSwitchBias = 1.0f;

        Vector3 animationIncomingMovement =
            RotatedBy(Vector3(0, -1, 0), turn) * RangeVelocity(animationData.incomingVelocity);
        Vector3 adaptedCurrentMovement = animationIncomingMovement * incomingSwitchBias +
                                         spatialState.movement * (1.0f - incomingSwitchBias);
        Vector3 predictedOutgoingMovement = RotatedBy(animationData.outgoingMovement, turn);
        Vector3 velocifiedDesiredMovement =
            useDesiredMovement ? desiredMovement : predictedOutgoingMovement;
        Vector3 adaptedDesiredMovement = predictedOutgoingMovement * outgoingSwitchBias +
                                         velocifiedDesiredMovement * (1.0f - outgoingSwitchBias);

        const float playerMaximumVelocity = GetMaximumVelocity(*context.staticData, humanoidIndex);
        float maximumVelocity = playerMaximumVelocity;
        if (touch) maximumVelocity *= 0.92f;

        const int timeStepMilliseconds = 10;
        float difficultyFactor = derivedData.animationDifficultyFactor;
        float difficultyPenaltyFactor = libm::PowfPositiveExponent(
            clamp((difficultyFactor - 0.0f) *
                      (1.0f - (statisticAgility * 0.2f + statisticAcceleration * 0.2f)) * 2.0f,
                  0.0f, 1.0f),
            0.7f);
        float powerFactor = 1.0f - clamp(libm::PowfPositiveExponent(BaseLastTouchBias(1000), 0.8f) *
                                             (0.8f - statisticDribble * 0.3f),
                                         0.0f, 0.4f);
        powerFactor *=
            1.0f - clamp(GetHumanoid().decayingPositionOffset.GetLength() *
                                 (10.0f - Statistic(e_Statistic_PhysicalBalance) * 5.0f) -
                             0.1f,
                         0.0f, 0.3f);

        Vector3 temporalMovement = adaptedCurrentMovement;
        Vector3 currentPosition;

        float physicsBias = 1.0f;
        float maximumAngleModificationUnder = 0.125f * PI;
        float maximumAngleModificationOver = 0.125f * PI;
        float maximumAngleModificationStraight = 0.125f * PI;
        if (touch) {
            float bonus =
                1.0f -
                libm::PowfPositiveExponent(
                    NormalizedClamp(
                        (adaptedCurrentMovement + predictedOutgoingMovement).GetLength() * 0.5f, 0,
                        SPRINT_VELOCITY),
                    0.8f) *
                    0.8f;
            bonus *= 0.6f + 0.4f * Statistic(e_Statistic_TechnicalBallControl);
            maximumAngleModificationUnder = 0.2f * PI * bonus;
            maximumAngleModificationOver = 0;
            maximumAngleModificationStraight = 0.1f * PI * bonus;
        }
        if (animationType == e_AnimationString_Sliding) {
            maximumAngleModificationUnder = 0.5f * PI;
            maximumAngleModificationOver = 0.5f * PI;
            maximumAngleModificationStraight = 0.5f * PI;
        }
        if (animationType == e_AnimationString_Movement) physicsBias *= 1.0f;
        if (animationType == e_AnimationString_BallControl) physicsBias *= 1.0f;
        if (animationType == e_AnimationString_Trap) physicsBias *= 1.0f;
        if (animationType == e_AnimationString_ShortPass) physicsBias *= 0.0f;
        if (animationType == e_AnimationString_HighPass) physicsBias *= 0.0f;
        if (animationType == e_AnimationString_Shot) physicsBias *= 0.0f;
        if (animationType == e_AnimationString_Interfere) physicsBias *= 0.5f;
        if (animationType == e_AnimationString_Deflect) physicsBias *= 0.0f;
        if (animationType == e_AnimationString_Sliding) physicsBias *= 1.0f;
        if (animationType == e_AnimationString_Trip) {
            if (derivedData.tripTypeIsOne)
                physicsBias *= 0.5f;
            else
                physicsBias *= 0.0f;
        }
        if (animationType == e_AnimationString_Special) physicsBias *= 0.0f;
        if (animationData.incomingSpecialState != animation::STRING_EMPTY) physicsBias *= 0.0f;

        // GRF: 0.5 plus `_default_AccelerationFactor`, which is 0.5.
        const float accelerationMultiplier = 0.5f + 0.5f;

        float cappedToDesiredAngle = 0;
        if (physicsBias > 0.0f) {
            Vector3 animationOutgoingVector = predictedOutgoingMovement.GetNormalized(0);
            if (FloatToVelocityCategory(predictedOutgoingMovement.GetLength()) == e_Velocity_Idle)
                animationOutgoingVector = RotatedBy(animationData.outgoingDirection, turn);
            Vector3 desiredVector = adaptedDesiredMovement.GetNormalized(0);
            if (FloatToVelocityCategory(adaptedDesiredMovement.GetLength()) == e_Velocity_Idle)
                desiredVector = RotatedBy(desiredBodyDirectionRelative, turn);
            float toDesiredAngle = desiredVector.GetAngle2D(animationOutgoingVector);
            if (fabsf(toDesiredAngle) <= 0.5f * PI || animationType == e_AnimationString_Sliding) {
                float animationChange =
                    animationOutgoingVector.GetAngle2D(spatialState.directionVector);
                if (fabsf(animationChange) > 0.06f * PI) {
                    int sign = SignSide(animationChange);
                    if (SignSide(toDesiredAngle) == sign) {
                        cappedToDesiredAngle = clamp(toDesiredAngle, -maximumAngleModificationOver,
                                                     maximumAngleModificationOver);
                    } else {
                        cappedToDesiredAngle = clamp(toDesiredAngle, -maximumAngleModificationUnder,
                                                     maximumAngleModificationUnder);
                    }
                } else {
                    cappedToDesiredAngle = clamp(toDesiredAngle, -maximumAngleModificationStraight,
                                                 maximumAngleModificationStraight);
                }
            }
        }

        const int effectiveFrameCount = animationData.frameCount - 1;
        const float adaptedDesiredVelocity = adaptedDesiredMovement.GetLength();
        const Vector3 adaptedDesiredDirection = adaptedDesiredMovement.GetNormalized(0);
        // Speed: the part of a frame free of the carried movement, which the lanes compute 32
        // ahead.
        struct PhysicsFrame {
            Vector3 animationMovement, adaptedAnimationMovement, normalizedMaximum,
                normalizedAdapted;
            float adaptedAnimationVelocity, rotationCosine, rotationSine, curvedTime;
        };
        auto precompute = [&](int frame) {
            const int timeMilliseconds = frame * timeStepMilliseconds;
            float frameBias = (timeMilliseconds + 10) / (float)((effectiveFrameCount + 1) * 10);

            float lagExponent = 1.0f;
            if (physicsBias > 0.0f && (animationType == e_AnimationString_BallControl ||
                                       animationType == e_AnimationString_Movement)) {
                // GRF: 0.5 is `_default_AgilityFactor`.
                lagExponent = 1.4f - 0.5f * 0.8f;
                lagExponent *= 1.2f - statisticAgility * 0.4f;
                if (touch) {
                    lagExponent += -0.1f + clamp(difficultyFactor * 0.4f, 0.0f, 0.5f);
                } else {
                    lagExponent += -0.2f + clamp(difficultyFactor * 0.2f, 0.0f, 0.2f);
                }
                lagExponent = clamp(lagExponent, 0.25f, 4.0f);
                if (touch && timeMilliseconds < animationTouchFrame * 10)
                    lagExponent = StandardMaximum(lagExponent, 0.7f);
                lagExponent = lagExponent * physicsBias + 1.0f * (1.0f - physicsBias);
            }
            // Exact: without physics bias the exponent is 1, and `powf(x, 1)` is `x` for non-NaN
            // `x`.
            float adaptedFrameBias =
                lagExponent == 1.0f ? frameBias : libm::powf(frameBias, lagExponent);
            Vector3 animationMovement = RotatedBy(
                SmoothedMovement(animationIndex,
                                 FloatToUnsigned32(effectiveFrameCount * adaptedFrameBias)),
                turn);

            float animationVelocity = animationMovement.GetLength();
            Vector3 adaptedAnimationMovement = animationMovement;
            float adaptedAnimationVelocity = animationVelocity;
            if (animationVelocity > WALK_SPRINT_SWITCH &&
                (animationType == e_AnimationString_Movement ||
                 animationType == e_AnimationString_BallControl ||
                 animationType == e_AnimationString_Trap)) {
                if (maximumVelocity > animationVelocity) {
                    adaptedAnimationVelocity = StretchSprintTo(
                        animationVelocity, ANIMATION_SPRINT_VELOCITY, maximumVelocity);
                    adaptedAnimationMovement =
                        adaptedAnimationMovement.GetNormalized(0) * adaptedAnimationVelocity;
                }
            }

            PhysicsFrame frameData;
            frameData.animationMovement = animationMovement;
            frameData.adaptedAnimationMovement = adaptedAnimationMovement;
            frameData.adaptedAnimationVelocity = adaptedAnimationVelocity;
            frameData.normalizedMaximum = GetNormalizedMaximum(adaptedAnimationMovement, 1.0f);
            frameData.normalizedAdapted = adaptedAnimationMovement.GetNormalized(0);
            const float rotation = cappedToDesiredAngle * frameBias;
            // Exact: below 2^-12 the sine is the argument and the cosine 1, by the test of `sinf`
            // and `cosf`. An animation without a turn rotates by zero every frame.
            if (libm::MagnitudeTopBits(rotation) < libm::MagnitudeTopBits(0x1p-12f)) {
                frameData.rotationCosine = 1.0f;
                frameData.rotationSine = rotation;
            } else {
                const libm::SineCosine frameTurn = libm::sincosf(rotation);
                frameData.rotationCosine = frameTurn.cosine;
                frameData.rotationSine = frameTurn.sine;
            }
            frameData.curvedTime = PhysicsCurvedTime(frame);
            return frameData;
        };
        if (animationData.frameCount > PositionList::capacity()) InlineVectorOverflow();
        WarpPipeline<PhysicsFrame>(
            animationData.frameCount, precompute, [&](int frame, const PhysicsFrame &frameData) {
                const int timeMilliseconds = frame * timeStepMilliseconds;
                float frameBias = (timeMilliseconds + 10) / (float)((effectiveFrameCount + 1) * 10);
                const Vector3 &animationMovement = frameData.animationMovement;
                Vector3 adaptedAnimationMovement = frameData.adaptedAnimationMovement;
                float adaptedAnimationVelocity = frameData.adaptedAnimationVelocity;
                float maximumSlower = 1.6f;
                if (touch) maximumSlower = 1.2f;
                float maximumFaster = 0.0f;
                const float temporalMovementLength = temporalMovement.GetLength();
                if (temporalMovementLength > adaptedAnimationVelocity)
                    maximumFaster =
                        MinimumFloat(0.0f + 1.0f * (1.0f - frameBias),
                                     StandardMaximum(maximumFaster, temporalMovementLength -
                                                                        adaptedAnimationVelocity));
                if (maximumFaster > 0)
                    maximumFaster *= StandardMaximum(
                        0.0f, frameData.normalizedMaximum.GetDotProduct(adaptedDesiredDirection));
                if (animationType == e_AnimationString_Sliding) maximumFaster = 100;
                const float desiredVelocity = adaptedDesiredVelocity;
                adaptedAnimationVelocity =
                    clamp(desiredVelocity, adaptedAnimationVelocity - maximumSlower,
                          adaptedAnimationVelocity + maximumFaster);
                adaptedAnimationMovement = frameData.normalizedAdapted * adaptedAnimationVelocity;

                Vector3 resultingPhysicsMovement = adaptedAnimationMovement;
                resultingPhysicsMovement = RotatedByCosineSine(
                    resultingPhysicsMovement, frameData.rotationCosine, frameData.rotationSine);
                resultingPhysicsMovement = resultingPhysicsMovement * physicsBias +
                                           animationMovement * (1.0f - physicsBias);
                Vector3 toDesired = (resultingPhysicsMovement - temporalMovement);

                // GRF: `mod_BrakeOnTouch` is false, so the humanoid never brakes on a touch.
                float penaltyBreakFactor = 0.0f;

                {
                    Vector3 predictedMovement = temporalMovement + toDesired;
                    float startVelocity = IDLE_DRIBBLE_SWITCH;
                    if (temporalMovementLength > startVelocity &&
                        predictedMovement.GetLength() > startVelocity) {
                        const Vector3 predictedDirection = predictedMovement.GetNormalized(),
                                      temporalDirection = temporalMovement.GetNormalized();
                        float maximumAngleFactor = 1.0f * (timeStepMilliseconds / 1000.0f);
                        maximumAngleFactor *= (0.7f + 0.3f * statisticAgility);
                        if (!touch) maximumAngleFactor *= 1.5f;
                        float maximumAngle = maximumAngleFactor * PI;
                        // Exact: GRF raises this to the power 1, and `powf(x, 1)` is `x` for
                        // non-NaN `x`.
                        float velocityFactor =
                            NormalizedClamp(temporalMovementLength, 0, SPRINT_VELOCITY);
                        maximumAngle = maximumAngle / (velocityFactor + 0.01f);
#if !defined(__CUDA_ARCH__)
                        // Exact: for dot > 0, |GetAngle2D| = atanf(|cross / dot|) <= |cross / dot|,
                        // so a ratio within the limit keeps the movement without the arc tangent.
                        const float cross =
                            predictedDirection.coordinates[0] * temporalDirection.coordinates[1] -
                            predictedDirection.coordinates[1] * temporalDirection.coordinates[0];
                        const float dot =
                            predictedDirection.coordinates[0] * temporalDirection.coordinates[0] +
                            predictedDirection.coordinates[1] * temporalDirection.coordinates[1];
                        const bool withinLimit = dot > 0.0f && fabsf(cross / dot) <= maximumAngle;
#else
              // Speed: on the GPU the test costs more than the arc tangent it saves.
              const bool withinLimit = false;
#endif
                        if (!withinLimit) {
                            float corneringAngle = predictedDirection.GetAngle2D(temporalDirection);
                            if (fabsf(corneringAngle) > maximumAngle) {
                                float overAngle = fabsf(corneringAngle) - maximumAngle;
                                toDesired +=
                                    -temporalMovement * clamp(overAngle / PI * 3.0f, 0.0f, 1.0f);
                            }
                        }
                    }
                }

                {
                    float maximumChange = 0.03f;
                    if (animationType == e_AnimationString_Trip) maximumChange *= 0.7f;
                    if (animationType == e_AnimationString_Sliding) maximumChange = 0.1f;
                    float velocityFactor = libm::PowfPositiveExponent(
                        NormalizedClamp(temporalMovementLength, 0, SPRINT_VELOCITY), 1.5f);
                    float firstStepFactor = velocityFactor;
                    if (animationType == e_AnimationString_Movement) firstStepFactor *= 0.4f;
                    maximumChange *=
                        (1.0f - firstStepFactor) + firstStepFactor * frameData.curvedTime;
                    maximumChange *= 1.2f - velocityFactor * 0.4f;
                    // GRF: 0.5 is `_default_AgilityFactor`.
                    maximumChange *= 0.75f + 0.5f * 0.5f;
                    maximumChange *= powerFactor;
                    float desiredLength = toDesired.GetLength();
                    float maximumAddition = maximumChange * timeStepMilliseconds;
                    NormalizeMaximum(toDesired, MinimumFloat(desiredLength, maximumAddition));
                }

                if (animationType != e_AnimationString_Sliding &&
                    animationType != e_AnimationString_Deflect) {
                    float velocityExponent = 1.8f;
                    float accelerationPower = 11.0f * accelerationMultiplier;
                    float falloffStartVelocity = IDLE_DRIBBLE_SWITCH;
                    const float nextLength = (temporalMovement + toDesired).GetLength();
                    if (nextLength > falloffStartVelocity) {
                        accelerationPower *= 1.0f - difficultyPenaltyFactor * 0.4f;
                        float velocityAirResistanceFactor =
                            clamp(libm::PowfPositiveExponent(
                                      clamp((temporalMovementLength - falloffStartVelocity) /
                                                (playerMaximumVelocity - falloffStartVelocity),
                                            0.0f, 1.0f),
                                      velocityExponent),
                                  0.0f, 1.0f);
                        Vector3 forwardVector;
                        if (nextLength > temporalMovementLength) {
                            Vector3 destination = temporalMovement + toDesired;
                            float velocity = temporalMovementLength;
                            float acceleration = nextLength - velocity;
                            forwardVector = destination.GetNormalized(0) * acceleration;
                        }
                        float accelerationAddition = forwardVector.GetLength();
                        float maximumAccelerationPerSecond = accelerationPower *
                                                             (1.0f - velocityAirResistanceFactor) *
                                                             (statisticAcceleration * 0.3f + 0.7f);
                        float maximumAccelerationAddition =
                            maximumAccelerationPerSecond * (timeStepMilliseconds / 1000.0f);
                        if (accelerationAddition > maximumAccelerationAddition) {
                            float remainingFactor =
                                maximumAccelerationAddition / accelerationAddition;
                            toDesired = toDesired - forwardVector * (1.0f - remainingFactor);
                        }
                    }
                }

                Vector3 temporaryTemporalMovement = temporalMovement + toDesired;
                if (timeMilliseconds >= (animationData.frameCount - 2) * 10) {
                    if (FloatToVelocityCategory(animationData.outgoingVelocity) ==
                            e_Velocity_Idle &&
                        FloatToVelocityCategory(temporaryTemporalMovement.GetLength()) !=
                            e_Velocity_Idle)
                        temporaryTemporalMovement = Vector3(0);
                    else if (FloatToVelocityCategory(animationData.outgoingVelocity) !=
                                 e_Velocity_Idle &&
                             FloatToVelocityCategory(temporaryTemporalMovement.GetLength()) ==
                                 e_Velocity_Idle)
                        temporaryTemporalMovement = GetNormalizedTo(
                            RotatedBy(animationData.outgoingMovement, turn), DRIBBLE_VELOCITY);
                }
                temporalMovement = temporaryTemporalMovement;
                if (timeMilliseconds >= (animationData.frameCount - 2) * 10)
                    penaltyBreakFactor = 0.0f;
                currentPosition += temporalMovement * (1.0f - penaltyBreakFactor) *
                                   (timeStepMilliseconds / 1000.0f);
                // The count is set once after the loop.
                positionsResult[frame] = currentPosition;
            });
        positionsResult.resize(animationData.frameCount > 0 ? animationData.frameCount : 0);

        Vector3 resultingMovement = temporalMovement;
        if (FloatToVelocityCategory(animationData.outgoingVelocity) != e_Velocity_Idle &&
            FloatToVelocityCategory(resultingMovement.GetLength()) != e_Velocity_Idle) {
            rotationOffsetResult = RotatedBy(resultingMovement, BackTurn(angle))
                                       .GetAngle2D(animationData.outgoingMovement);
        } else {
            rotationOffsetResult = cappedToDesiredAngle * physicsBias;
        }
        return resultingMovement;
    }

    CUJU_FUNCTION_NO_INLINE bool NeedTouch(int animationID, const state::PlayerCommand &command) {
        const animation::Animation &animationData = A(animationID);
        state::SpatialState &spatialState = GetSpatialState();
        // GRF: never true, since a bool as float is idle, but kept as written.
        if (FloatToVelocityCategory((float)(animationData.outgoingVelocity !=
                                            (float)e_Velocity_Idle)) != e_Velocity_Idle)
            return true;
        if (command.desiredVelocityFloat > IDLE_DRIBBLE_SWITCH) return true;
        if (fabsf(world.serialized.match.ball.momentum.GetLength()) > 2.0f) return true;
        Vector3 animationMovement =
            RotatedBy(animationData.outgoingMovement, Turn(spatialState.angle.value())) * 0.3f +
            spatialState.movement * 0.7f;
        float animationVelocity = animationMovement.GetLength();
        animationMovement.Normalize(spatialState.directionVector);
        animationMovement *= spatialState.movement.GetLength() * 0.8f + animationVelocity * 0.2f;
        Vector3 ballMovement =
            (MentalBallPredictionAt(250).Get2D() - MentalBallPredictionAt(240).Get2D()) * 100;
        if (fabsf(animationData.outgoingAngle.value()) > 0.125f * PI) return true;
        float distanceDeviation = (animationMovement - ballMovement).GetLength();
        if ((double)distanceDeviation >= 2.0) return true;
        float velocityDeviation = animationMovement.GetLength() - ballMovement.GetLength();
        if ((double)velocityDeviation < -1.4 || (double)velocityDeviation >= 0.7) return true;
        if (FloatToVelocityCategory(animationData.outgoingVelocity) != e_Velocity_Idle) {
            float angleDeviation =
                animationMovement.GetNormalized(spatialState.directionVector)
                    .GetDotProduct(ballMovement.GetNormalized(spatialState.directionVector));
            if ((double)angleDeviation < 0.975) return true;
        }
        return false;
    }

    CUJU_FUNCTION static Vector3 CalculateOutgoingMovement(const PositionList &positions) {
        if (positions.size() < 2) return Vector3(0);
        return (positions.at(positions.size() - 1) - positions.at(positions.size() - 2)) * 100.0f;
    }

    CUJU_FUNCTION_NO_INLINE void CalculatePredictedSituation(Vector3 &predictedPosition,
                                                             float &predictedAngle) {
        CUJU_PROFILE_SCOPE(PREDICTED_SITUATION);
        state::AnimationState &currentAnimation = GetHumanoid().currentAnimation;
        const animation::Animation &animationData = A(currentAnimation.animation);
        state::SpatialState &spatialState = GetSpatialState();
        if ((uint32_t)currentAnimation.positions.count > (uint32_t)currentAnimation.frameNumber) {
            predictedPosition =
                spatialState.position + currentAnimation.positions[animationData.frameCount - 1] +
                currentAnimation.actionSmuggle + currentAnimation.actionSmuggleSustain +
                currentAnimation.movementSmuggle;
        } else {
            predictedPosition =
                spatialState.position +
                animationData.translation.Get2D().GetRotated2D(spatialState.angle.value()) +
                currentAnimation.actionSmuggle + currentAnimation.actionSmuggleSustain +
                currentAnimation.movementSmuggle;
        }
        predictedAngle = spatialState.angle.value() + animationData.outgoingAngle.value() +
                         currentAnimation.rotationSmuggle.end.value();
        predictedAngle = ModulateIntoRange(-PI, PI, predictedAngle);
    }

    CUJU_FUNCTION int GetDesiredTimeToBallMilliseconds() {
        return FloatToInt32(clamp((float)GetPlayer().desiredTimeToBallMilliseconds,
                                  (float)(int)GetPlayer().timeNeededToGetToBallMilliseconds,
                                  1000000.0f));
    }

    CUJU_FUNCTION_NO_INLINE Vector3 CalculateMovementSmuggle() {
        state::AnimationState &currentAnimation = GetHumanoid().currentAnimation;
        const animation::Animation &animationData = A(currentAnimation.animation);
        const state::MatchState &match = world.serialized.match;
        if (TeamOfPlayer(world, humanoidIndex).designatedTeamPossessionPlayer != humanoidIndex ||
            match.designatedPossessionPlayer != humanoidIndex ||
            currentAnimation.touchFrame != -1 ||
            (currentAnimation.functionType == e_FunctionType_Trip &&
             !GetDerived(currentAnimation.animation).tripTypeIsOne) ||
            animationData.incomingSpecialState != animation::STRING_EMPTY ||
            animationData.outgoingSpecialState != animation::STRING_EMPTY || !match.inPlay ||
            match.inSetPiece || match.ballRetainer != -1)
            return Vector3(0);

        Vector3 toDesired;
        uint32_t timeToBallMilliseconds =
            (uint32_t)(int)GetPlayer().timeNeededToGetToBallMilliseconds;
        if (GetDesiredTimeToBallMilliseconds() > (int)timeToBallMilliseconds)
            timeToBallMilliseconds = (uint32_t)GetDesiredTimeToBallMilliseconds();
        uint32_t animationTimeMilliseconds = (uint32_t)animationData.frameCount * 10;
        uint32_t futureTimeMilliseconds = std::max(
            animationTimeMilliseconds + DEFAULT_TOUCH_OFFSET_MILLISECONDS, timeToBallMilliseconds);

        PositionList positions;
        for (const Vector3 &position : currentAnimation.positions) positions.push_back(position);
        Vector3 predictedOutgoingMovement = CalculateOutgoingMovement(positions);
        Vector3 predictedPosition;
        float predictedAngle;
        CalculatePredictedSituation(predictedPosition, predictedAngle);
        Vector3 ballPosition = MentalBallPredictionAt((int)futureTimeMilliseconds);
        float ballHeight = ballPosition.coordinates[2];
        Vector3 frontOfFootOffset =
            GetFrontOfFootOffsetRelative(predictedOutgoingMovement.GetLength(),
                                         animationData.outgoingBodyAngle.value(), ballHeight)
                .GetRotated2D(predictedAngle);
        Vector3 desiredBallPosition = predictedPosition + frontOfFootOffset;

        if (!GetPlayer().hasPossession) {
            Vector3 ballStart = MentalBallPredictionAt(0).Get2D();
            Vector3 ballEnd = MentalBallPredictionAt((int)futureTimeMilliseconds).Get2D();
            if ((ballStart - ballEnd).GetLength() < 0.5f) return Vector3(0);
            float fraction = 0.0f;
            if (!(ballStart == ballEnd)) {
                float lineDistance = (ballEnd - ballStart).GetLength();
                if (!(lineDistance < 0.000001f)) {
                    fraction = ((desiredBallPosition.coordinates[0] - ballStart.coordinates[0]) *
                                    (ballEnd.coordinates[0] - ballStart.coordinates[0]) +
                                (desiredBallPosition.coordinates[1] - ballStart.coordinates[1]) *
                                    (ballEnd.coordinates[1] - ballStart.coordinates[1])) /
                               (lineDistance * lineDistance);
                }
            }
            Vector3 closestBallPosition = ballStart + (ballEnd - ballStart) * fraction;
            toDesired = closestBallPosition - desiredBallPosition;
        } else {
            toDesired = ballPosition.Get2D() - desiredBallPosition;
        }

        uint32_t maximumEffectTimeThresholdMilliseconds = 250 + DEFAULT_TOUCH_OFFSET_MILLISECONDS;
        if (FloatToVelocityCategory(animationData.outgoingVelocity) == e_Velocity_Idle)
            maximumEffectTimeThresholdMilliseconds = 2000;
        float maximumEffectVelocity = DRIBBLE_WALK_SWITCH;
        float maximumSmugglePerSecond = 1.6f;
        if (futureTimeMilliseconds -
                (animationTimeMilliseconds + DEFAULT_TOUCH_OFFSET_MILLISECONDS) >
            maximumEffectTimeThresholdMilliseconds)
            return Vector3(0);

        Vector3 toDesiredMovement = toDesired / (animationTimeMilliseconds * 0.001f);
        Vector3 resultingMovement = predictedOutgoingMovement + toDesiredMovement;
        float predictedVelocity = predictedOutgoingMovement.GetLength();
        float resultingVelocity = resultingMovement.GetLength();
        if (resultingVelocity > predictedVelocity && resultingVelocity > maximumEffectVelocity)
            return Vector3(0);

        NormalizeMaximum(toDesired,
                         maximumSmugglePerSecond * ((animationData.frameCount - 1) * 0.01f));
        float removeDistance = 0.06f;
        toDesired = toDesired.GetNormalized(0) *
                    StandardMaximum(0.0f, toDesired.GetLength() - removeDistance);
        return toDesired;
    }

    CUJU_FUNCTION float LastTouchBiasAt(int decayMilliseconds, uint64_t timeMilliseconds) {
        uint64_t adaptedTimeMilliseconds = timeMilliseconds;
        if (timeMilliseconds == 0)
            adaptedTimeMilliseconds = world.serialized.match.actualTimeMilliseconds;
        if (decayMilliseconds > 0)
            return 1.0f - clamp((float)(adaptedTimeMilliseconds -
                                        GetPlayer().base.lastTouchTimeMilliseconds) /
                                    (float)decayMilliseconds,
                                0.0f, 1.0f);
        return 0.0f;
    }

    CUJU_FUNCTION float GetClosestOpponentDistance() {
        return ClosestOpponentDistance(world, humanoidIndex);
    }

    CUJU_FUNCTION_NO_INLINE static float GetBodyBallDistanceAdvantage(
        const animation::Animation &animationData, const Vector3 &touchMovement,
        const Vector3 &incomingMovement, const Vector3 &outgoingMovement, float outgoingAngle,
        const Vector3 &bodyPosition, const Vector3 &frontOfFootOffset,
        const Vector3 &animationBallPosition2D, const Vector3 &actualBallPosition2D,
        float radiusFactor, float radiusCheatDistance) {
        const float cheatFactor = 0.5f, cheatDistanceBonus = 0.02f;
        float incomingVelocity = incomingMovement.GetLength();
        float outgoingVelocity = outgoingMovement.GetLength();
        float averageInOutVelocity = (incomingMovement + outgoingMovement).GetLength() * 0.5f;
        float velocityChange = outgoingVelocity - incomingVelocity;
        float velocityChangePerSecond = velocityChange / (animationData.frameCount * 0.01f);
        (void)touchMovement;

        float bodyAnimationBallBonus =
            1.0f -
            Curve(NormalizedClamp(((bodyPosition + frontOfFootOffset.GetNormalized(0) * 0.1f) -
                                   animationBallPosition2D)
                                      .GetLength(),
                                  0.0f, 0.7f),
                  0.7f);
        float bodyActualBallBonus =
            1.0f -
            Curve(NormalizedClamp(((bodyPosition + frontOfFootOffset.GetNormalized(0) * 0.1f) -
                                   actualBallPosition2D)
                                      .GetLength(),
                                  0.0f, 0.7f),
                  0.4f);
        float velocityBonus =
            1.0f - NormalizedClamp(averageInOutVelocity, IDLE_VELOCITY, SPRINT_VELOCITY);
        float velocityChangeBonus =
            1.0f - NormalizedClamp(velocityChangePerSecond / 20.0f, -1.0f, 1.0f);

        float radius = radiusFactor;
        radius *= 1.0f + 1.0f * bodyAnimationBallBonus + 0.6f * bodyActualBallBonus +
                  0.8f * bodyAnimationBallBonus * bodyActualBallBonus + 0.0f * velocityChangeBonus +
                  1.0f * velocityBonus;
        float effectiveRadiusCheatDistance = radiusCheatDistance;

        Vector3 outgoingDirection;
        if (FloatToVelocityCategory(outgoingMovement.GetLength()) == e_Velocity_Idle) {
            outgoingDirection = Vector3(0, -1, 0).GetRotated2D(outgoingAngle);
        } else {
            outgoingDirection = outgoingMovement.GetNormalized();
        }
        Vector3 behindVectorUnscaled =
            -(incomingMovement * 0.1f + touchMovement * 0.2f + outgoingMovement * 0.7f);
        Vector3 behindVector =
            behindVectorUnscaled.GetNormalized(0) *
            libm::PowfPositiveExponent(
                NormalizedClamp(behindVectorUnscaled.GetLength(), 0, SPRINT_VELOCITY), 0.5f);
        float dot = Vector3(0, -1, 0).GetDotProduct(outgoingDirection);
        dot = 0.5f + clamp(dot * 0.5f + 0.5f, 0.0f, 1.0f) * 0.5f;
        behindVector *= dot;

        Vector3 animationToActualBall = actualBallPosition2D - animationBallPosition2D;
        {
            Vector3 straightAngleVectorUnscaled = incomingMovement;
            Vector3 straightAngleVector =
                straightAngleVectorUnscaled.GetNormalized(outgoingDirection);
            float toStraightAngle = Vector3(0, -1, 0).GetAngle2D(straightAngleVector);
            animationToActualBall.Rotate2D(toStraightAngle);
            float lateralRadiusFactor =
                0.6f - 0.3f * libm::PowfPositiveExponent(
                                  NormalizedClamp(straightAngleVectorUnscaled.GetLength(),
                                                  IDLE_VELOCITY, SPRINT_VELOCITY),
                                  0.7f);
            animationToActualBall.coordinates[0] /= lateralRadiusFactor;
            radius *= libm::powf(1.0f / lateralRadiusFactor, 0.5f);
            animationToActualBall.Rotate2D(-toStraightAngle);
        }
        Vector3 adaptedActualBallPosition2D = animationBallPosition2D + animationToActualBall;
        float radiusCheatBehindBias = 0.6f;
        Vector3 behindCenter =
            animationBallPosition2D +
            behindVector * (radius + (effectiveRadiusCheatDistance * radiusCheatBehindBias)) *
                cheatFactor;
        float result = 1.0f;
        float allowedRadius =
            (radius + effectiveRadiusCheatDistance) * cheatFactor + cheatDistanceBonus;
        if (GetDistance(adaptedActualBallPosition2D, behindCenter) > allowedRadius) result = 0.0f;
        return result;
    }

    struct CheatResult {
        int animationID = -1;
        int touchFrame = -1;
        Vector3 touchPosition, actionSmuggle;
        float rotationSmuggle = 0;
    };

#if !defined(__CUDA_ARCH__)
    // Speed: false only if no touch can pass the height test of `GetBestCheatableAnimationID`. The
    // bound takes the closest ball height from 6 ms before to 3 ms after the touch, the smallest
    // height factor, no doubling and the base subtraction, with a margin far above rounding.
    bool AnyTouchWithinHeight(const animation::Animation &animationData, int functionType,
                              float playerHeight) {
        const state::MatchState &match = world.serialized.match;
        const float positionZ = GetSpatialState().position.coordinates[2];
        for (int touch = 0; touch < (int)animationData.football.size(); touch++) {
            const int frame = animationData.football[touch].frame;
            const Vector3 absoluteBallPosition = BallPredict(match.ball, frame * 10);
            if (fabsf(absoluteBallPosition.coordinates[0]) >
                    PITCH_HALF_LENGTH + LINE_HALF_WIDTH + 0.11f ||
                fabsf(absoluteBallPosition.coordinates[1]) >
                    PITCH_HALF_WIDTH + LINE_HALF_WIDTH + 0.11f)
                continue;
            const float ballZ = MentalBallPredictionAt(frame * 10).coordinates[2];
            const float movementZ =
                (MentalBallPredictionAt(frame * 10 + 10).coordinates[2] - ballZ) * 100.0f;
            const float relativeZ = ballZ - positionZ;
            // The margin covers rounding only below these magnitudes. Beyond them the full test
            // runs.
            if (!(fabsf(relativeZ) < 100.0f && fabsf(movementZ) < 1000.0f)) return true;
            const float before = relativeZ - movementZ * 0.006f,
                        after = relativeZ + movementZ * 0.003f;
            const float low = (before < after ? before : after) - 0.0001f;
            const float high = (before < after ? after : before) + 0.0001f;
            const float animationZ = animationData.football[touch].position.coordinates[2] *
                                     (playerHeight / DEFAULT_PLAYER_HEIGHT);
            const float distance = animationZ < low    ? low - animationZ
                                   : animationZ > high ? animationZ - high
                                                       : 0.0f;
            float bound =
                distance *
                (1.0f - clamp((float)(((double)animationZ - 0.11) * (double)0.3f), 0.0f, 0.2f)) *
                0.7f;
            if (functionType == e_FunctionType_Deflect) bound *= 0.8f;
            if (animationData.baseAnimation) bound -= 0.15f;
            if (!(bound >= 0.2205f)) return true;
        }
        return false;
    }
#endif

    CUJU_FUNCTION_NO_INLINE int GetBestCheatableAnimationID(
        const AnimationList &sortedDataSet, bool useDesiredMovement,
        const Vector3 &desiredDirection, float desiredVelocityFloat, bool useDesiredBodyDirection,
        const Vector3 &desiredBodyDirectionRelative, PositionList &positionsResult,
        int &animationTouchFrameResult, Vector3 &touchPositionResult, Vector3 &actionSmuggleResult,
        float &rotationSmuggleResult, bool preferPassAndShot = false) {
        CUJU_PROFILE_SCOPE(BEST_CHEATABLE);
        const state::MatchState &match = world.serialized.match;
        state::SpatialState &spatialState = GetSpatialState();
        const float angle = spatialState.angle.value();
        const bool retainerIsMe = match.ballRetainer == humanoidIndex;
        if (match.ballRetainer != -1 && !retainerIsMe) return -1;

        const libm::SineCosine back = BackTurn(angle);
        Vector3 incomingMovement = RotatedBy(spatialState.movement, back);
        int bestAnimationID = -1;
        Vector3 bestActionSmuggle2D;
        Vector3 desiredMovement = desiredDirection * desiredVelocityFloat;
        float playerHeight = context.staticData->data[humanoidIndex].height;
        const int functionType = StringToFunctionType(A(sortedDataSet.at(0)).animationType);

        float temporaryRotationSmuggle = 0;
        float predictedAngle = 0;
        Vector3 adaptedOutgoingMovement;

        bool found = false;
        for (size_t entry = 0; entry < sortedDataSet.size() && !found; entry++) {
            const int animationIndex = sortedDataSet[entry];
            const animation::Animation &animationData = A(animationIndex);
            bool isBase = animationData.baseAnimation;
#if !defined(__CUDA_ARCH__)
            // Exact: an animation whose every touch misses in height is never chosen, but the last
            // one tried still runs for its path.
            if (entry + 1 < sortedDataSet.size() && !retainerIsMe &&
                !AnyTouchWithinHeight(animationData, functionType, playerHeight))
                continue;
#endif
            CalculatePhysicsVector(animationIndex, useDesiredMovement, desiredMovement,
                                   useDesiredBodyDirection, desiredBodyDirectionRelative,
                                   positionsResult, temporaryRotationSmuggle);
            predictedAngle = animationData.outgoingAngle.value() + temporaryRotationSmuggle;
            predictedAngle = ModulateIntoRange(-PI, PI, predictedAngle);

            Vector3 outgoingMovement = RotatedBy(CalculateOutgoingMovement(positionsResult), back);
            adaptedOutgoingMovement = outgoingMovement;
            const int totalTouches = (int)animationData.football.size();
            AnimationList touchIDs;
            for (int touch = totalTouches / 2; touch > -1; touch--) touchIDs.push_back(touch);
            for (int touch = totalTouches / 2 + 1; touch < totalTouches; touch++)
                touchIDs.push_back(touch);
            int defaultTouchFrame = GetDerived(animationIndex).touchFrame;

            for (int touchNumber = 0; touchNumber < totalTouches && !found; touchNumber++) {
                if (touchIDs[touchNumber] >= totalTouches)
                    CUJU_FATAL("a touch of an animation lies past its ball keys");
                Vector3 animationBallPosition =
                    animationData.football[touchIDs[touchNumber]].position;
                int animationTouchFrame = animationData.football[touchIDs[touchNumber]].frame;

                if (!retainerIsMe) {
                    Vector3 absoluteBallPosition =
                        BallPredict(match.ball, animationTouchFrame * 10);
                    if (fabsf(absoluteBallPosition.coordinates[0]) >
                            PITCH_HALF_LENGTH + LINE_HALF_WIDTH + 0.11f ||
                        fabsf(absoluteBallPosition.coordinates[1]) >
                            PITCH_HALF_WIDTH + LINE_HALF_WIDTH + 0.11f)
                        continue;
                }

                Vector3 touchMovement = RotatedBy(
                    CalculateMovementAtFrame(positionsResult, animationTouchFrame, 0), back);

                Vector3 ballPosition = MentalBallPredictionAt(animationTouchFrame * 10);
                Vector3 ballMovement = (MentalBallPredictionAt(animationTouchFrame * 10 + 10) -
                                        MentalBallPredictionAt(animationTouchFrame * 10)) *
                                       100.0f;
                ballPosition = RotatedBy(ballPosition - spatialState.position, back);
                ballMovement = RotatedBy(ballMovement, back);

                const Vector3 touchPosition =
                    RotatedBy(positionsResult.at(animationTouchFrame), back);
                Vector3 bodyPosition = touchPosition;
                bodyPosition.coordinates[2] = 0;

                // Exact: the yaw `z` comes from the interpolation table, which holds the same
                // `GetAngles`.
                Quaternion animationBodyRotation;
                Vector3 animationBodyPosition;
                float z;
                animation::GetKeyFrameZ(animationData, e_BodyPart_Player, animationTouchFrame,
                                        animationBodyRotation, animationBodyPosition, z);

                float animationBallHeight = animationBallPosition.coordinates[2];
                animationBallPosition =
                    (animationBallPosition - animationBodyPosition) + touchPosition;
                animationBallPosition.coordinates[2] =
                    animationBallHeight * (playerHeight / DEFAULT_PLAYER_HEIGHT);

                {
                    Vector3 ballBefore = ballPosition - ballMovement * 0.006f;
                    Vector3 ballAfter = ballPosition + ballMovement * 0.003f;
                    float fraction =
                        clamp(LineClosestToPoint(ballBefore, ballAfter, animationBallPosition),
                              0.0f, 1.0f);
                    ballPosition = ballBefore + (ballAfter - ballBefore) * fraction;
                }

                Vector3 actionSmuggle3D = ballPosition - animationBallPosition;
                Vector3 actionSmuggle2D = actionSmuggle3D.Get2D();

                float ballDistanceZ = fabsf(actionSmuggle3D.coordinates[2]);
                ballDistanceZ *=
                    1.0f - clamp((float)(((double)animationBallPosition.coordinates[2] - 0.11) *
                                         (double)0.3f),
                                 0.0f, 0.2f);
                ballDistanceZ *=
                    1.0f -
                    clamp((float)(((double)ballPosition.coordinates[2] - 0.11) * (double)0.4f),
                          0.0f, 0.3f);
                if (ballPosition.coordinates[2] > 1.8f &&
                    ballPosition.coordinates[2] > animationBallPosition.coordinates[2] + 0.12f)
                    ballDistanceZ *= 2.0f;
                if (ballPosition.coordinates[2] > 2.6f &&
                    ballPosition.coordinates[2] > animationBallPosition.coordinates[2] + 0.08f)
                    ballDistanceZ *= 20.0f;
                if (functionType == e_FunctionType_Deflect) ballDistanceZ *= 0.8f;
                if (retainerIsMe) ballDistanceZ = 0.0f;
                if (ballPosition.coordinates[2] < 0.5f && isBase)
                    ballDistanceZ = StandardMaximum(ballDistanceZ - 0.15f, 0.0f);

                if (ballDistanceZ < 0.22f) {
                    float touchFrameAwkwardness = NormalizedClamp(
                        (float)abs(defaultTouchFrame - animationTouchFrame), 0.0f, 4.0f);
                    touchFrameAwkwardness = touchFrameAwkwardness * touchFrameAwkwardness * 0.5f;
                    float touchFrameFactor = animationTouchFrame / 24.0f;
                    touchFrameFactor = libm::powf(touchFrameFactor, 0.7f);
                    if (touchFrameFactor > 1.0f)
                        touchFrameFactor = 1.0f + ((touchFrameFactor - 1.0f) * 0.5f);

                    float radiusCheatOffset = 0.0f;
                    float radiusFactor = 0.3f * (1.0f - touchFrameAwkwardness);
                    if (functionType == e_FunctionType_Deflect) {
                        radiusFactor *= 1.8f;
                        radiusCheatOffset += 0.4f;
                    }
                    if (functionType == e_FunctionType_Sliding) {
                        radiusFactor *= 0.2f;
                        radiusCheatOffset = 0.0f;
                    }
                    if (functionType == e_FunctionType_Interfere) {
                        radiusFactor *= 1.4f;
                        radiusCheatOffset += 0.2f;
                    }
                    if (functionType == e_FunctionType_ShortPass ||
                        functionType == e_FunctionType_LongPass ||
                        functionType == e_FunctionType_HighPass ||
                        functionType == e_FunctionType_Shot) {
                        radiusFactor *= 1.3f;
                        radiusCheatOffset += 0.15f;
                    }
                    if ((functionType == e_FunctionType_Trap ||
                         functionType == e_FunctionType_BallControl) &&
                        preferPassAndShot)
                        radiusFactor *= 0.3f;
                    radiusCheatOffset *= (1.0f - touchFrameAwkwardness);

                    float touchVelocity = touchMovement.GetLength();
                    Vector3 frontOfFootOffset =
                        GetFrontOfFootOffsetRelative(touchVelocity, z, ballPosition.coordinates[2]);
                    if (FloatToVelocityCategory(touchVelocity) != e_Velocity_Idle)
                        frontOfFootOffset.Rotate2D(
                            FixAngle(touchMovement.GetNormalized(Vector3(0, -1, 0)).GetAngle2D()));
                    float lastTouchBias = Curve(
                        LastTouchBiasAt(600, match.actualTimeMilliseconds +
                                                 (uint64_t)(int64_t)(animationTouchFrame * 10)),
                        1.0f);
                    if (lastTouchBias > 0.0f) {
                        float factor =
                            1.0f - lastTouchBias * 0.97f *
                                       (1.0f - Statistic(e_Statistic_TechnicalBallControl) * 0.1f);
                        radiusFactor *= factor;
                        radiusCheatOffset *= factor;
                    }
                    float touchFramedRadiusFactor = radiusFactor * touchFrameFactor;
                    float advantage = GetBodyBallDistanceAdvantage(
                        animationData, touchMovement, incomingMovement, adaptedOutgoingMovement,
                        predictedAngle, bodyPosition, frontOfFootOffset,
                        animationBallPosition.Get2D(), ballPosition.Get2D(),
                        touchFramedRadiusFactor, radiusCheatOffset);
                    if (advantage >= 1.0f || retainerIsMe) {
                        found = true;
                        bestAnimationID = animationIndex;
                        bestActionSmuggle2D = actionSmuggle2D;
                        animationTouchFrameResult = animationTouchFrame;
                    }
                }
            }
        }

        if (found) {
            touchPositionResult = MentalBallPredictionAt(animationTouchFrameResult * 10);
            Vector3 fullActionSmuggle = RotatedBy(bestActionSmuggle2D, Turn(angle));
            actionSmuggleResult = fullActionSmuggle;
            float smuggleDistance = actionSmuggleResult.GetLength();
            float adaptedCheatDiscardDistanceMultiplier = 0.4f;
            if (functionType != e_FunctionType_BallControl)
                adaptedCheatDiscardDistanceMultiplier *= 0.8f;
            if (touchPositionResult.coordinates[2] > 0.5f)
                adaptedCheatDiscardDistanceMultiplier *= 0.7f;
            if (touchPositionResult.coordinates[2] > 1.0f)
                adaptedCheatDiscardDistanceMultiplier *= 0.7f;
            float cheatDiscardDistanceBonus = 0.0f;
            if (functionType == e_FunctionType_Interfere) cheatDiscardDistanceBonus += 0.1f;
            smuggleDistance =
                clamp(smuggleDistance - (0.02f + cheatDiscardDistanceBonus), 0.0f, (float)100.0);
            smuggleDistance *= 1.0f - adaptedCheatDiscardDistanceMultiplier;
            smuggleDistance = MinimumFloat(smuggleDistance, actionSmuggleResult.GetLength() - 0.2f);
            if (retainerIsMe) smuggleDistance = 0.0f;
            actionSmuggleResult = actionSmuggleResult.GetNormalized(0) * smuggleDistance;
            {
                float toStraightAngle = angle + predictedAngle;
                actionSmuggleResult.Rotate2D(-toStraightAngle);
                float shortenForwardDistance = 0.02f;
                float allowForwardDistance =
                    0.25f *
                    (1.0f - libm::powf(NormalizedClamp(adaptedOutgoingMovement.GetLength(),
                                                       IDLE_VELOCITY, SPRINT_VELOCITY - 2.0f),
                                       0.6f)) *
                    (animationTouchFrameResult * 0.1f);
                if (functionType != e_FunctionType_BallControl &&
                    functionType != e_FunctionType_Trap)
                    allowForwardDistance += 0.1f;
                if (actionSmuggleResult.coordinates[1] < 0.0f)
                    actionSmuggleResult.coordinates[1] =
                        clamp(actionSmuggleResult.coordinates[1] + shortenForwardDistance,
                              -allowForwardDistance, 0.0f);
                actionSmuggleResult.Rotate2D(toStraightAngle);
            }
            actionSmuggleResult *=
                0.7f + 0.3f * NormalizedClamp(GetClosestOpponentDistance(), 0.6f, 1.2f);
        }
        rotationSmuggleResult = temporaryRotationSmuggle;
        return bestAnimationID;
    }

    CUJU_FUNCTION static float LineClosestToPoint(const Vector3 &lineStart, const Vector3 &lineEnd,
                                                  const Vector3 &point) {
        if (lineStart == lineEnd) return 0.0f;
        float lineDistance = (lineEnd - lineStart).GetLength();
        if (lineDistance < 0.000001f) return 0.0f;
        return ((point.coordinates[0] - lineStart.coordinates[0]) *
                    (lineEnd.coordinates[0] - lineStart.coordinates[0]) +
                (point.coordinates[1] - lineStart.coordinates[1]) *
                    (lineEnd.coordinates[1] - lineStart.coordinates[1])) /
               (lineDistance * lineDistance);
    }

    CUJU_FUNCTION_NO_INLINE static int StringToFunctionType(int animationType) {
        using namespace animation;
        switch (animationType) {
            case e_AnimationString_Movement:
                return e_FunctionType_Movement;
            case e_AnimationString_BallControl:
                return e_FunctionType_BallControl;
            case e_AnimationString_Trap:
                return e_FunctionType_Trap;
            case e_AnimationString_ShortPass:
                return e_FunctionType_ShortPass;
            case e_AnimationString_LongPass:
                return e_FunctionType_LongPass;
            case e_AnimationString_HighPass:
                return e_FunctionType_HighPass;
            case e_AnimationString_Shot:
                return e_FunctionType_Shot;
            case e_AnimationString_Deflect:
                return e_FunctionType_Deflect;
            case e_AnimationString_Catch:
                return e_FunctionType_Catch;
            case e_AnimationString_Interfere:
                return e_FunctionType_Interfere;
            case e_AnimationString_Trip:
                return e_FunctionType_Trip;
            case e_AnimationString_Sliding:
                return e_FunctionType_Sliding;
            case e_AnimationString_Special:
                return e_FunctionType_Special;
        }
        return e_FunctionType_None;
    }

    CUJU_FUNCTION SelectResult SelectAnimation(const state::PlayerCommand &command,
                                               int localInterruptAnimation, bool preferPassAndShot);
    CUJU_FUNCTION SelectResult BaseSelectAnimation(const state::PlayerCommand &command,
                                                   int localInterruptAnimation);
};

CUJU_FUNCTION_NO_INLINE inline SelectResult Selector::BaseSelectAnimation(
    const state::PlayerCommand &command, int localInterruptAnimation) {
    CUJU_PROFILE_SCOPE(BASE_SELECT_ANIMATION);
    HumanoidState &humanoid = GetHumanoid();
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    state::SpatialState &spatialState = GetSpatialState();
    const int functionType = command.desiredFunctionType;

    if (localInterruptAnimation != e_InterruptAnimation_ReQueue ||
        currentAnimation.frameNumber > 12) {
        const animation::Animation &current = A(currentAnimation.animation);
        spatialState.foot = current.outgoingFoot;
        if (current.outgoingSpecialState != animation::STRING_EMPTY) {
            spatialState.floatVelocity = 0;
            spatialState.velocityCategory = e_Velocity_Idle;
            spatialState.movement = Vector3(0);
        }
    }
    const animation::Animation &current = A(currentAnimation.animation);
    const float angle = spatialState.angle.value();

    CrudeSelectionQuery query;
    query.byFunctionType = true;
    query.functionType = functionType;
    query.byIncomingVelocity = true;
    query.incomingVelocity = spatialState.velocityCategory;
    query.incomingVelocityStrict = true;
    query.byIncomingBodyDirection = true;
    query.incomingBodyDirectionStrict = true;
    query.incomingBodyDirection = spatialState.relativeBodyDirectionVector;
    query.incomingVelocityForceLinearity = true;
    query.incomingBodyDirectionForceLinearity = true;
    if (functionType == e_FunctionType_Trip) {
        query.byTripType = true;
        query.tripType = command.tripType;
    }
    query.incomingSpecialState = current.outgoingSpecialState;
    if (world.serialized.match.ballRetainer == humanoidIndex)
        query.incomingRetainState = GetDerived(currentAnimation.animation).outgoingRetainRawValue;
    if (command.useSpecialVariable1) query.specialVariable1 = (float)command.specialVariable1;
    if (command.useSpecialVariable2) query.specialVariable2 = (float)command.specialVariable2;
    if (current.outgoingSpecialState != animation::STRING_EMPTY)
        query.incomingVelocity = e_Velocity_Idle;

    AnimationList dataSet;
    CrudeSelection(SC(), query, dataSet);
    if (dataSet.empty()) {
        if (functionType == e_FunctionType_Movement)
            dataSet.push_back(GetIdleMovementAnimationID());
        else
            return SELECT_FALSE;
    }

    float adaptedDesiredVelocityFloat = command.desiredVelocityFloat;
    if (command.useDesiredMovement) {
        Vector3 relativeDesiredDirection = RotatedBy(command.desiredDirection, BackTurn(angle));
        SetMovementSimilarityPredicate(relativeDesiredDirection,
                                       FloatToVelocityCategory(adaptedDesiredVelocityFloat));
        predicates.lookAt = command.desiredLookAt;
        if (functionType == e_FunctionType_Movement) {
            KeepBestDirectionAnimations(dataSet, command, true);
            if (command.useDesiredLookAt) KeepBestBodyDirectionAnimations(dataSet, command, true);
        } else {
            Sort<float>(dataSet,
                        [&](int animationIndex) { return MovementSimilarityKey(animationIndex); });
        }
    }

    // GRF: sorts by idle level, foot, incoming body direction, incoming velocity, trip direction
    // and then base animations. Each key reads only its own predicates, so all are set first.
    predicates.idle = 1;
    predicates.desiredFoot = spatialState.foot;
    predicates.relativeIncomingBodyDirection = spatialState.relativeBodyDirectionVector;
    predicates.incomingVelocity = spatialState.velocityCategory;
    if (command.useDesiredTripDirection)
        predicates.relativeDesiredTripDirection =
            RotatedBy(command.desiredTripDirection, BackTurn(angle));

    int selectedAnimationID = -1;
    PositionList temporaryPositions;
    float temporaryRotationSmuggle = 0;
    if (dataSet.empty()) return SELECT_FALSE;
    const bool byTripDirection = command.useDesiredTripDirection,
               byBase = functionType != e_FunctionType_Movement;
    // A single animation computes no key.
    const int queryIndex = dataSet.count > 1 ? IncomingBodyDirectionQuery() : 0;
    const int firstAnimation = FirstOfSortChain(
        dataSet, 4 + byTripDirection + byBase, [&](int animationIndex, double *keys) {
            int key = 0;
            if (byBase) keys[key++] = A(animationIndex).baseAnimation ? 0 : 1;
            if (byTripDirection)
                keys[key++] =
                    -GetDerived(animationIndex)
                         .bumpDirection.GetDotProduct(predicates.relativeDesiredTripDirection);
            keys[key++] = KeyIncomingVelocity(A(animationIndex), predicates);
            keys[key++] = IncomingBodyDirectionKey(animationIndex, queryIndex);
            keys[key++] = KeyFoot(A(animationIndex), predicates);
            keys[key++] = KeyIdle(A(animationIndex), predicates);
        });
    if (functionType == e_FunctionType_Movement || functionType == e_FunctionType_Trip ||
        functionType == e_FunctionType_Special) {
        selectedAnimationID = firstAnimation;
        const animation::Animation &nextAnimation = A(selectedAnimationID);
        Vector3 desiredMovement = command.desiredDirection * command.desiredVelocityFloat;
        Vector3 desiredBodyDirectionRelative = Vector3(0, -1, 0);
        if (command.useDesiredLookAt)
            desiredBodyDirectionRelative =
                (RotatedBy((command.desiredLookAt - spatialState.position).Get2D(),
                           BackTurn(angle)) -
                 nextAnimation.translation)
                    .GetNormalized(Vector3(0, -1, 0));
        CalculatePhysicsVector(selectedAnimationID, command.useDesiredMovement, desiredMovement,
                               command.useDesiredLookAt, desiredBodyDirectionRelative,
                               temporaryPositions, temporaryRotationSmuggle);
    }

    if (localInterruptAnimation == e_InterruptAnimation_ReQueue && selectedAnimationID != -1 &&
        currentAnimation.positions.count > 1 && temporaryPositions.size() > 1) {
        const animation::Animation &selected = A(selectedAnimationID);
        bool currentIdle = FloatToVelocityCategory(current.outgoingVelocity) == e_Velocity_Idle;
        if (currentAnimation.functionType == functionType &&
            ((!currentIdle && current.quadrantID == selected.quadrantID) ||
             (currentIdle && fabsf(PreferredOutgoingAngle(currentAnimation.animation) -
                                   PreferredOutgoingAngle(selectedAnimationID)) < 0.20f * PI)))
            selectedAnimationID = -1;
    }
    if (selectedAnimationID == -1) return SELECT_FALSE;

    humanoid.previousAnimationFrameNumber = currentAnimation.frameNumber;
    humanoid.previousAnimationFunctionType = currentAnimation.functionType;
    const animation::Animation &next = A(selectedAnimationID);
    currentAnimation.animation = selectedAnimationID;
    currentAnimation.id = selectedAnimationID;
    currentAnimation.functionType = functionType;
    currentAnimation.frameNumber = 0;
    currentAnimation.touchFrame = -1;
    currentAnimation.originatingInterrupt = localInterruptAnimation;
    currentAnimation.touchPosition = Vector3();
    const float maximumRotation = 0.25f * PI;
    SetRadian(currentAnimation.rotationSmuggle.begin,
              clamp(ModulateIntoRange(-PI, PI,
                                      spatialState.relativeBodyAngleNonQuantized.value() -
                                          next.incomingBodyAngle.value()) *
                        1.0f,
                    -maximumRotation, maximumRotation));
    SetRadian(currentAnimation.rotationSmuggle.end, temporaryRotationSmuggle);
    SetRadian(currentAnimation.rotationSmuggleOffset, 0);
    currentAnimation.actionSmuggle = Vector3();
    currentAnimation.actionSmuggleOffset = Vector3(0);
    currentAnimation.actionSmuggleSustain = Vector3(0);
    currentAnimation.actionSmuggleSustainOffset = Vector3(0);
    currentAnimation.movementSmuggle = Vector3(0);
    currentAnimation.movementSmuggleOffset = Vector3(0);
    currentAnimation.incomingMovement = spatialState.movement;
    currentAnimation.outgoingMovement = CalculateOutgoingMovement(temporaryPositions);
    currentAnimation.positions.count = 0;
    for (const Vector3 &position : temporaryPositions)
        currentAnimation.positions.push_back(position);
    currentAnimation.positionOffset = Vector3(0.0);
    currentAnimation.originatingCommand = command;
    return SELECT_TRUE;
}

CUJU_FUNCTION_NO_INLINE inline SelectResult Selector::SelectAnimation(
    const state::PlayerCommand &command, int localInterruptAnimation, bool preferPassAndShot) {
    CUJU_PROFILE_SCOPE(SELECT_ANIMATION);
    (void)preferPassAndShot;
    state::PlayerState &player = GetPlayer();
    HumanoidState &humanoid = GetHumanoid();
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    state::SpatialState &spatialState = GetSpatialState();
    const state::MatchState &match = world.serialized.match;
    const int functionType = command.desiredFunctionType;
    const bool retainerIsMe = match.ballRetainer == humanoidIndex;

    if (functionType != e_FunctionType_Movement && functionType != e_FunctionType_Trip &&
        functionType != e_FunctionType_Special && functionType != e_FunctionType_Sliding) {
        if ((MentalBallPredictionAt(200).Get2D() - spatialState.position).GetLength() > 10.0f)
            return SELECT_FALSE;
        if ((MentalBallPredictionAt(DEFAULT_TOUCH_OFFSET_MILLISECONDS).Get2D() -
             spatialState.position)
                    .GetLength() > 2.0f &&
            (MentalBallPredictionAt(DEFAULT_TOUCH_OFFSET_MILLISECONDS).Get2D() -
             (spatialState.position +
              spatialState.movement * (float)DEFAULT_TOUCH_OFFSET_MILLISECONDS * (float)0.001))
                    .GetLength() >
                (MentalBallPredictionAt(0).Get2D() - spatialState.position).GetLength())
            return SELECT_FALSE;
    }
    if (functionType != e_FunctionType_Movement && functionType != e_FunctionType_Trip &&
        functionType != e_FunctionType_Special && functionType != e_FunctionType_Sliding &&
        functionType != e_FunctionType_Deflect && !retainerIsMe) {
        if ((MentalBallPredictionAt(1000) - BallPredict(match.ball, 1000)).GetLength() > 2.0f)
            return SELECT_FALSE;
    }

    if (localInterruptAnimation == e_InterruptAnimation_ReQueue) {
        if (match.designatedPossessionPlayer < 0) return UNSUPPORTED;
        float focusDistance =
            (Humanoid(world, match.designatedPossessionPlayer).spatialState.position -
             spatialState.position)
                .GetLength();
        const int currentFunctionType = currentAnimation.functionType;
        const int effectiveFrames = A(currentAnimation.animation).frameCount - 1;
        if (currentFunctionType != e_FunctionType_Movement &&
            functionType == e_FunctionType_Movement)
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_Movement &&
            functionType == e_FunctionType_Movement &&
            (player.hasPossession || focusDistance > 12.0f))
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_Movement &&
            functionType == e_FunctionType_Movement &&
            currentAnimation.frameNumber + 6 > effectiveFrames)
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_Movement &&
            functionType == e_FunctionType_Movement && humanoid.requeueDelayFrames > 0)
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_BallControl &&
            functionType == e_FunctionType_BallControl &&
            (currentAnimation.frameNumber > 8 || humanoid.requeueDelayFrames > 0))
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_BallControl &&
            functionType == e_FunctionType_Trap)
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_Trap && functionType == e_FunctionType_Trap &&
            (currentAnimation.frameNumber + 6 > currentAnimation.touchFrame ||
             humanoid.requeueDelayFrames > 0))
            return SELECT_FALSE;
        if (currentFunctionType == e_FunctionType_Trap &&
            functionType == e_FunctionType_BallControl &&
            (currentAnimation.frameNumber + 6 > currentAnimation.touchFrame ||
             humanoid.requeueDelayFrames > 0))
            return SELECT_FALSE;

        const state::PlayerCommand &originatingCommand = currentAnimation.originatingCommand;
        if (originatingCommand.desiredFunctionType == functionType &&
            ((originatingCommand.desiredDirection * originatingCommand.desiredVelocityFloat) -
             (command.desiredDirection * command.desiredVelocityFloat))
                    .GetLength() < 1.5f)
            return SELECT_FALSE;

        if ((currentFunctionType == e_FunctionType_Movement &&
             functionType == e_FunctionType_Movement) ||
            (currentFunctionType == e_FunctionType_BallControl &&
             functionType == e_FunctionType_BallControl) ||
            (currentFunctionType == e_FunctionType_Trap &&
             functionType == e_FunctionType_BallControl) ||
            (currentFunctionType == e_FunctionType_Trap && functionType == e_FunctionType_Trap)) {
            Vector3 plannedMomentumChange =
                currentAnimation.outgoingMovement - currentAnimation.incomingMovement;
            Vector3 desiredMomentumChange =
                (command.desiredDirection * command.desiredVelocityFloat) - spatialState.movement;
            if ((desiredMomentumChange.GetDotProduct(plannedMomentumChange) > 0.0f &&
                 GetDistance(desiredMomentumChange, plannedMomentumChange) < 4.0f) ||
                desiredMomentumChange.GetDotProduct(plannedMomentumChange) > 0.8f ||
                GetDistance(desiredMomentumChange, plannedMomentumChange) < 2.0f)
                return SELECT_FALSE;
        }

        if (currentFunctionType == e_FunctionType_Movement &&
            functionType == e_FunctionType_BallControl &&
            (match.actualTimeMilliseconds - player.base.lastTouchTimeMilliseconds < 600 &&
             player.base.lastTouchType == e_TouchType_Intentional_Kicked) &&
            player.hasPossession) {
            float desiredMovementChange =
                (spatialState.movement - (command.desiredDirection * command.desiredVelocityFloat))
                    .GetLength();
            if (desiredMovementChange < 1.0f) return SELECT_FALSE;
        }
    }

    if (localInterruptAnimation != e_InterruptAnimation_ReQueue ||
        currentAnimation.frameNumber > 12) {
        const animation::Animation &current = A(currentAnimation.animation);
        spatialState.foot = current.outgoingFoot;
        if (current.outgoingSpecialState != animation::STRING_EMPTY) {
            spatialState.floatVelocity = 0;
            spatialState.velocityCategory = e_Velocity_Idle;
            spatialState.movement = Vector3(0);
        }
    }

    const animation::Animation &current = A(currentAnimation.animation);
    const float angle = spatialState.angle.value();

    CrudeSelectionQuery query;
    query.byFunctionType = true;
    query.functionType = functionType;
    if (query.functionType == e_FunctionType_LongPass)
        query.functionType = e_FunctionType_ShortPass;
    if (command.touchInformation.desiredPower != 0.0f) {
        query.byOutgoingBallDirection = true;
        query.outgoingBallDirection =
            RotatedBy(command.touchInformation.desiredDirection, BackTurn(angle));
    }
    query.byIncomingVelocity = true;
    query.incomingVelocity = spatialState.velocityCategory;
    if (query.functionType != e_FunctionType_Movement &&
        query.incomingVelocity == e_Velocity_Dribble)
        query.incomingVelocity = e_Velocity_Walk;
    query.incomingVelocityStrict = false;
    if (query.functionType != e_FunctionType_Movement &&
        query.functionType != e_FunctionType_BallControl) {
        query.incomingVelocityForceLinearity = false;
        if (query.functionType != e_FunctionType_Deflect) {
            query.incomingVelocityNoDribbleToSprint = true;
            if (query.functionType != e_FunctionType_ShortPass &&
                query.functionType != e_FunctionType_LongPass &&
                query.functionType != e_FunctionType_HighPass &&
                query.functionType != e_FunctionType_Shot) {
                query.incomingVelocityForceLinearity = true;
                query.incomingVelocityNoDribbleToIdle = true;
            } else {
                query.incomingVelocityForceLinearity = false;
                query.incomingVelocityNoDribbleToIdle = false;
            }
        } else {
            query.incomingVelocityNoDribbleToSprint = false;
            query.incomingVelocityNoDribbleToIdle = false;
        }
    } else {
        query.incomingVelocityStrict = true;
    }
    query.byIncomingBodyDirection = true;
    query.incomingBodyDirection = spatialState.relativeBodyDirectionVector;
    if (query.functionType != e_FunctionType_Movement) {
        query.incomingBodyDirectionStrict = false;
        if (query.functionType != e_FunctionType_Deflect) {
            query.incomingBodyDirectionForceLinearity =
                query.functionType != e_FunctionType_BallControl;
        } else {
            query.incomingBodyDirectionForceLinearity = false;
        }
    } else {
        query.incomingBodyDirectionStrict = true;
    }
    query.bySide = false;
    if (command.useDesiredLookAt && (current.outgoingSpecialState == animation::STRING_EMPTY) &&
        !retainerIsMe) {
        Vector3 playerLookAtVector = (command.desiredLookAt - spatialState.position)
                                         .GetNormalized(spatialState.directionVector);
        query.lookAtVectorRelative = RotatedBy(playerLookAtVector, BackTurn(angle));
        query.bySide = true;
    }
    if (command.onlyDeflectAnimationsThatPickupBall) {
        query.byPickupBall = true;
        query.pickupBall = true;
    }
    if (functionType == e_FunctionType_Trap || functionType == e_FunctionType_Interfere ||
        functionType == e_FunctionType_Deflect) {
        query.byIncomingBallDirection = true;
        query.incomingBallDirection =
            RotatedBy(MentalBallPredictionAt(180) - MentalBallPredictionAt(120), BackTurn(angle))
                .GetNormalized(Vector3(0));
    }
    query.allowLastDitchAnimations = AllowLastDitch();
    if (functionType == e_FunctionType_Trip) {
        query.byTripType = true;
        query.tripType = command.tripType;
    }
    query.incomingSpecialState = current.outgoingSpecialState;
    if (retainerIsMe)
        query.incomingRetainState = GetDerived(currentAnimation.animation).outgoingRetainRawValue;
    if (command.useSpecialVariable1) query.specialVariable1 = (float)command.specialVariable1;
    if (command.useSpecialVariable2) query.specialVariable2 = (float)command.specialVariable2;
    if (current.outgoingSpecialState != animation::STRING_EMPTY)
        query.incomingVelocity = e_Velocity_Idle;

    AnimationList dataSet;
    CrudeSelection(SC(), query, dataSet);
    if (dataSet.empty()) {
        if (functionType == e_FunctionType_Movement)
            dataSet.push_back(GetIdleMovementAnimationID());
        else
            return SELECT_FALSE;
    }

    if (command.useDesiredMovement) {
        Vector3 relativeDesiredDirection = RotatedBy(command.desiredDirection, BackTurn(angle));
        SetMovementSimilarityPredicate(relativeDesiredDirection,
                                       FloatToVelocityCategory(command.desiredVelocityFloat));
        predicates.lookAt = command.desiredLookAt;
        if (functionType == e_FunctionType_Movement) {
            KeepBestDirectionAnimations(dataSet, command, true);
            if (command.useDesiredLookAt) KeepBestBodyDirectionAnimations(dataSet, command, true);
        } else if (functionType == e_FunctionType_BallControl) {
            bool strict = !AllowLastDitch();
            float allowedBaseAngle = 0.0f * PI;
            if (command.useDesiredLookAt)
                KeepBestBodyDirectionAnimations(dataSet, command, strict, allowedBaseAngle);
            KeepBestDirectionAnimations(dataSet, command, strict, allowedBaseAngle, 0);
        } else if (functionType == e_FunctionType_Trap) {
            bool highOrBouncy = HighOrBouncyBall();
            bool strict = !(AllowLastDitch(false) || highOrBouncy);
            float allowedBaseAngle = 0.3f * PI;
            KeepBestDirectionAnimations(dataSet, command, strict, allowedBaseAngle, 2, -1);
            if (command.useDesiredLookAt)
                KeepBestBodyDirectionAnimations(dataSet, command, strict, allowedBaseAngle);
            if (!HighOrBouncyBall() && query.allowLastDitchAnimations == false) {
                Vector3 desiredMovement = command.desiredDirection * command.desiredVelocityFloat;
                const animation::Animation &best = A(dataSet[0]);
                Vector3 bestOutgoingMovement = RotatedBy(best.outgoingMovement, Turn(angle));
                float currentDesiredDot =
                    command.desiredDirection.GetDotProduct(spatialState.directionVector);
                bool allowAnimation = true;
                float angleDifference = fabsf(RotatedBy(best.outgoingDirection, Turn(angle))
                                                  .GetAngle2D(command.desiredDirection));
                if (angleDifference > 0.375f * PI) allowAnimation = false;
                Vector3 desiredBestDifference = bestOutgoingMovement - desiredMovement;
                if ((desiredBestDifference.GetLength() > WALK_VELOCITY + 0.5f &&
                     currentDesiredDot > 0.0f) ||
                    (desiredBestDifference.GetLength() > SPRINT_VELOCITY + 0.5f &&
                     currentDesiredDot <= 0.0f))
                    allowAnimation = false;
                if (!allowAnimation) return SELECT_FALSE;
            }
        } else if (functionType == e_FunctionType_Interfere) {
            bool strict = command.strictMovement == e_StrictMovement_True;
            float allowedAngle = 0.3f * PI;
            KeepBestDirectionAnimations(dataSet, command, strict, allowedAngle, 1);
            if (command.useDesiredLookAt)
                KeepBestBodyDirectionAnimations(dataSet, command, strict, allowedAngle);
        }
    }

    // Speed: a movement, trip or special needs only the first of the chain below, which
    // `FirstOfSortChain` finds without sorting. Each key reads only its own predicates.
    const bool firstOnly = functionType == e_FunctionType_Movement ||
                           functionType == e_FunctionType_Trip ||
                           functionType == e_FunctionType_Special;
    if (!firstOnly)
        Sort<double>(dataSet, [&](int animationIndex) {
            return GetDerived(animationIndex).priorityAbsolute;
        });
    int desiredIdleLevel = 0;
    if (!match.inPlay) desiredIdleLevel = 2;
    if (match.inSetPiece)
        desiredIdleLevel = 1;
    else if ((BallPredict(match.ball, 200) - spatialState.position).GetLength() > 16.0f)
        desiredIdleLevel = 1;
    predicates.idle = (float)desiredIdleLevel;
    if (!firstOnly)
        Sort<float>(dataSet,
                    [&](int animationIndex) { return KeyIdle(A(animationIndex), predicates); });
    predicates.desiredFoot = spatialState.foot;
    if (!firstOnly)
        Sort<int>(dataSet,
                  [&](int animationIndex) { return KeyFoot(A(animationIndex), predicates); });
    if (functionType != e_FunctionType_BallControl) {
        predicates.relativeIncomingBodyDirection = spatialState.relativeBodyDirectionVector;
        if (!firstOnly) SortByIncomingBodyDirection(dataSet);
    }
    predicates.incomingVelocity = spatialState.velocityCategory;
    if (!firstOnly)
        Sort<float>(dataSet, [&](int animationIndex) {
            return KeyIncomingVelocity(A(animationIndex), predicates);
        });
    if (command.useDesiredTripDirection) {
        predicates.relativeDesiredTripDirection =
            RotatedBy(command.desiredTripDirection, BackTurn(angle));
        if (!firstOnly)
            Sort<float>(dataSet, [&](int animationIndex) {
                return -GetDerived(animationIndex)
                            .bumpDirection.GetDotProduct(predicates.relativeDesiredTripDirection);
            });
    }
    if (functionType != e_FunctionType_Movement && !firstOnly) SortBaseAnimation(dataSet);
    if (functionType == e_FunctionType_Deflect)
        Sort<int>(dataSet, [&](int animationIndex) {
            return GetDerived(animationIndex).outgoingRetainRaw ? 0 : 1;
        });

    int selectedAnimationID = -1;
    PositionList temporaryPositions;
    int temporaryTouchFrame = -1;
    Vector3 temporaryTouchPosition;
    Vector3 temporaryActionSmuggle;
    float temporaryRotationSmuggle = 0;

    if (dataSet.empty() && functionType == e_FunctionType_Movement)
        dataSet.push_back(GetIdleMovementAnimationID());

    Vector3 desiredBodyDirectionRelative = Vector3(0, -1, 0);
    if (command.useDesiredLookAt)
        desiredBodyDirectionRelative = RotatedBy(
            (command.desiredLookAt - (spatialState.position + spatialState.movement * 0.1f))
                .GetNormalized(Vector3(0, -1, 0)),
            BackTurn(angle));

    if (firstOnly) {
        // The keys of the chain above, the last sort first. A single animation computes no key.
        const bool byTripDirection = command.useDesiredTripDirection,
                   byBase = functionType != e_FunctionType_Movement;
        const int queryIndex = dataSet.count > 1 ? IncomingBodyDirectionQuery() : 0;
        selectedAnimationID = FirstOfSortChain(
            dataSet, 5 + byTripDirection + byBase, [&](int animationIndex, double *keys) {
                int key = 0;
                if (byBase) keys[key++] = A(animationIndex).baseAnimation ? 0 : 1;
                if (byTripDirection)
                    keys[key++] =
                        -GetDerived(animationIndex)
                             .bumpDirection.GetDotProduct(predicates.relativeDesiredTripDirection);
                keys[key++] = KeyIncomingVelocity(A(animationIndex), predicates);
                keys[key++] = IncomingBodyDirectionKey(animationIndex, queryIndex);
                keys[key++] = KeyFoot(A(animationIndex), predicates);
                keys[key++] = KeyIdle(A(animationIndex), predicates);
                keys[key++] = GetDerived(animationIndex).priorityAbsolute;
            });
        Vector3 desiredMovement = command.desiredDirection * command.desiredVelocityFloat;
        CalculatePhysicsVector(selectedAnimationID, command.useDesiredMovement, desiredMovement,
                               command.useDesiredLookAt, desiredBodyDirectionRelative,
                               temporaryPositions, temporaryRotationSmuggle);
    } else if (functionType == e_FunctionType_BallControl) {
        if (NeedTouch(dataSet.at(0), command))
            selectedAnimationID = GetBestCheatableAnimationID(
                dataSet, command.useDesiredMovement, command.desiredDirection,
                command.desiredVelocityFloat, command.useDesiredLookAt,
                desiredBodyDirectionRelative, temporaryPositions, temporaryTouchFrame,
                temporaryTouchPosition, temporaryActionSmuggle, temporaryRotationSmuggle,
                preferPassAndShot);
    } else if (functionType == e_FunctionType_Trap || functionType == e_FunctionType_Interfere ||
               functionType == e_FunctionType_Deflect) {
        selectedAnimationID = GetBestCheatableAnimationID(
            dataSet, command.useDesiredMovement, command.desiredDirection,
            command.desiredVelocityFloat, command.useDesiredLookAt, desiredBodyDirectionRelative,
            temporaryPositions, temporaryTouchFrame, temporaryTouchPosition, temporaryActionSmuggle,
            temporaryRotationSmuggle, preferPassAndShot);
    } else if (functionType == e_FunctionType_ShortPass ||
               functionType == e_FunctionType_LongPass || functionType == e_FunctionType_HighPass ||
               functionType == e_FunctionType_Shot) {
        selectedAnimationID = GetBestCheatableAnimationID(
            dataSet, command.useDesiredMovement, command.desiredDirection,
            command.desiredVelocityFloat, command.useDesiredLookAt, desiredBodyDirectionRelative,
            temporaryPositions, temporaryTouchFrame, temporaryTouchPosition, temporaryActionSmuggle,
            temporaryRotationSmuggle);
    } else if (functionType == e_FunctionType_Sliding) {
        selectedAnimationID = GetBestCheatableAnimationID(
            dataSet, command.useDesiredMovement, command.desiredDirection,
            command.desiredVelocityFloat, command.useDesiredLookAt, desiredBodyDirectionRelative,
            temporaryPositions, temporaryTouchFrame, temporaryTouchPosition, temporaryActionSmuggle,
            temporaryRotationSmuggle);
        if (selectedAnimationID == -1 && !dataSet.empty()) {
            selectedAnimationID = dataSet[0];
            Vector3 desiredMovement = command.desiredDirection * command.desiredVelocityFloat;
            CalculatePhysicsVector(selectedAnimationID, command.useDesiredMovement, desiredMovement,
                                   command.useDesiredLookAt, desiredBodyDirectionRelative,
                                   temporaryPositions, temporaryRotationSmuggle);
        }
    }

    if (localInterruptAnimation == e_InterruptAnimation_ReQueue && selectedAnimationID != -1 &&
        currentAnimation.positions.count > 1 && temporaryPositions.size() > 1) {
        const animation::Animation &selected = A(selectedAnimationID);
        bool currentIdle = FloatToVelocityCategory(current.outgoingVelocity) == e_Velocity_Idle;
        bool selectedIdle = FloatToVelocityCategory(selected.outgoingVelocity) == e_Velocity_Idle;
        if (currentAnimation.functionType == functionType &&
            ((!currentIdle && current.quadrantID == selected.quadrantID) ||
             ((currentIdle && selectedIdle) &&
              fabsf(PreferredOutgoingAngle(currentAnimation.animation) -
                    PreferredOutgoingAngle(selectedAnimationID)) < 0.06f * PI)))
            selectedAnimationID = -1;
    }

    if (selectedAnimationID == -1) return SELECT_FALSE;

    humanoid.previousAnimationFrameNumber = currentAnimation.frameNumber;
    humanoid.previousAnimationFunctionType = currentAnimation.functionType;
    const animation::Animation &next = A(selectedAnimationID);
    currentAnimation.animation = selectedAnimationID;
    currentAnimation.id = selectedAnimationID;
    currentAnimation.functionType = functionType;
    currentAnimation.frameNumber = 0;
    currentAnimation.touchFrame = temporaryTouchFrame;
    currentAnimation.originatingInterrupt = localInterruptAnimation;
    currentAnimation.touchPosition = temporaryTouchPosition;
    const float maximumRotation = 0.25f * PI;
    const float rotationScale =
        currentAnimation.functionType == e_FunctionType_Movement ? 1.0f : 0.5f;
    SetRadian(currentAnimation.rotationSmuggle.begin,
              clamp(ModulateIntoRange(-PI, PI,
                                      spatialState.relativeBodyAngleNonQuantized.value() -
                                          next.incomingBodyAngle.value()) *
                        1.0f,
                    -maximumRotation * rotationScale, maximumRotation * rotationScale));
    SetRadian(currentAnimation.rotationSmuggle.end, temporaryRotationSmuggle);
    SetRadian(currentAnimation.rotationSmuggleOffset, 0);
    currentAnimation.actionSmuggle = temporaryActionSmuggle;
    currentAnimation.actionSmuggleOffset = Vector3(0);
    currentAnimation.actionSmuggleSustain = Vector3(0);
    currentAnimation.actionSmuggleSustainOffset = Vector3(0);
    currentAnimation.movementSmuggle = Vector3(0);
    currentAnimation.movementSmuggleOffset = Vector3(0);
    currentAnimation.incomingMovement = spatialState.movement;
    currentAnimation.outgoingMovement = CalculateOutgoingMovement(temporaryPositions);
    currentAnimation.positions.count = 0;
    for (const Vector3 &position : temporaryPositions)
        currentAnimation.positions.push_back(position);
    currentAnimation.positionOffset = Vector3(0);
    currentAnimation.originatingCommand = command;
    currentAnimation.movementSmuggle = CalculateMovementSmuggle();
    currentAnimation.movementSmuggleOffset = Vector3(0);
    return SELECT_TRUE;
}

}  // namespace simulation
}  // namespace cuju
