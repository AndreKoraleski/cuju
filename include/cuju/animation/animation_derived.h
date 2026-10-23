// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The values that the simulation derives from the text variables of each animation, once at load.
// Exact: the host C library's `atof` and `atoi` parse them, so they match GRF's parse at each use.
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "../math_utilities.h"
#include "../vector3.h"
#include "animation_data.h"
#include "skeleton.h"

namespace cuju {
namespace animation {

enum e_AnimationString {
    e_AnimationString_Empty = 0,
    e_AnimationString_OutgoingSpecialState = 1,
    e_AnimationString_IncomingSpecialState = 2,
    e_AnimationString_SpecialVariable1 = 3,
    e_AnimationString_SpecialVariable2 = 4,
    e_AnimationString_Type = 5,
    e_AnimationString_Trap = 6,
    e_AnimationString_Deflect = 7,
    e_AnimationString_Interfere = 8,
    e_AnimationString_Trip = 9,
    e_AnimationString_ShortPass = 10,
    e_AnimationString_LongPass = 11,
    e_AnimationString_Shot = 12,
    e_AnimationString_Sliding = 13,
    e_AnimationString_Movement = 14,
    e_AnimationString_Special = 15,
    e_AnimationString_BallControl = 16,
    e_AnimationString_HighPass = 17,
    e_AnimationString_Catch = 18,
    e_AnimationString_OutgoingRetainState = 19,
    e_AnimationString_IncomingRetainState = 20
};

inline std::vector<std::string> Tokenize(const std::string &text, const std::string &delimiters) {
    std::vector<std::string> tokens;
    std::string::size_type tokenStart = text.find_first_not_of(delimiters, 0);
    std::string::size_type tokenEnd = text.find_first_of(delimiters, tokenStart);
    while (std::string::npos != tokenEnd || std::string::npos != tokenStart) {
        tokens.push_back(text.substr(tokenStart, tokenEnd - tokenStart));
        tokenStart = text.find_first_not_of(delimiters, tokenEnd);
        tokenEnd = text.find_first_of(delimiters, tokenStart);
    }
    return tokens;
}

inline Vector3 GetVectorFromString(const std::string &text) {
    if (text.empty()) return Vector3(0.0f);
    std::vector<std::string> tokens = Tokenize(text, ",");
    Vector3 result;
    result.coordinates[0] = (float)atof(tokens.at(0).c_str());
    if (tokens.size() > 1) result.coordinates[1] = (float)atof(tokens.at(1).c_str());
    if (tokens.size() > 2) result.coordinates[2] = (float)atof(tokens.at(2).c_str());
    return result;
}

struct AnimationDerived {
    double priorityAbsolute;
    int touchFrame;
    float animationDifficultyFactor;
    bool tripTypeIsOne;
    int tripTypeRounded;
    Vector3 bumpDirection;
    Vector3 incomingBallDirection;
    Vector3 ballDirection;
    float incomingMaximumDeviation;
    float outgoingMaximumDeviation;
    bool outgoingRetainRaw;
    int32_t outgoingRetainRawValue;
    int32_t incomingRetainRawValue;
    int32_t outgoingRetainBodyPart;
    int32_t incomingRetainBodyPart;
    int forcedFootWhich;
    bool touchFootLeft;
    // 0 is `e_TouchType_Intentional_Kicked`, for a foot or a lower leg, and 1 is
    // `e_TouchType_Intentional_NonKicked`.
    int touchType;
    float touchMaximumPowerFactor;
    float touchDifficultyFactor;
    // Animation-only subexpressions of the crude selection, hoisted to load with the same
    // functions.
    Vector3 crudeOutDirectionRotated;
    float crudeTurnAngle;
    float crudeInBodyFixAbsolute;
    Vector3 crudeOutBodyDirection;
    bool crudeInBallNonzero;
    Vector3 crudeInBallAdapted;
    Vector3 crudeBallDirectionNormalized;
};

inline std::vector<AnimationDerived> DeriveAll(const AnimationStore &store) {
    const AnimationCollection &collection = store.view;
    std::vector<AnimationDerived> derived(collection.animations.size());
    for (size_t animationIndex = 0; animationIndex < collection.animations.size();
         animationIndex++) {
        auto AnimationVariable = [&](const char *key) {
            return store.GetVariable(animationIndex, key);
        };
        AnimationDerived &entry = derived[animationIndex];
        const Animation &animationData = collection.animations[animationIndex];
        entry.priorityAbsolute = fabs(atof(AnimationVariable("priority").c_str()));
        entry.touchFrame = atoi(AnimationVariable("touchframe").c_str());
        entry.animationDifficultyFactor =
            (float)atof(AnimationVariable("animdifficultyfactor").c_str());
        entry.tripTypeIsOne = AnimationVariable("triptype") == "1";
        entry.tripTypeRounded = (int)round(atof(AnimationVariable("triptype").c_str()));
        entry.bumpDirection = GetVectorFromString(AnimationVariable("bumpdirection"));
        entry.incomingBallDirection =
            GetVectorFromString(AnimationVariable("incomingballdirection"));
        entry.ballDirection = GetVectorFromString(AnimationVariable("balldirection"));
        entry.incomingMaximumDeviation =
            (float)fabs(atof(AnimationVariable("incomingballdirection_maxdeviation").c_str()) * PI);
        entry.outgoingMaximumDeviation =
            (float)fabs(atof(AnimationVariable("outgoingballdirection_maxdeviation").c_str()) * PI);
        const std::string outgoingRetainState = AnimationVariable("outgoing_retain_state"),
                          incomingRetainState = AnimationVariable("incoming_retain_state");
        entry.outgoingRetainRawValue = InternStateString(outgoingRetainState);
        entry.outgoingRetainRaw = !outgoingRetainState.empty();
        entry.incomingRetainRawValue = InternStateString(incomingRetainState);
        entry.outgoingRetainBodyPart =
            outgoingRetainState.empty() ? -1 : BodyPartFromString(outgoingRetainState);
        entry.incomingRetainBodyPart =
            incomingRetainState.empty() ? -1 : BodyPartFromString(incomingRetainState);
        std::string forcedFoot = AnimationVariable("forcedfoot");
        entry.forcedFootWhich = forcedFoot == "strong" ? 1 : forcedFoot == "weak" ? 2 : 0;
        entry.touchFootLeft = AnimationVariable("touchfoot") == "left";
        std::string touchBodyPart = AnimationVariable("touch_bodypart");
        entry.touchType = (touchBodyPart.find("foot") != std::string::npos ||
                           touchBodyPart.find("lowerleg") != std::string::npos)
                              ? 0
                              : 1;
        entry.touchMaximumPowerFactor =
            (float)atof(AnimationVariable("touch_maxpowerfactor").c_str());
        entry.touchDifficultyFactor =
            (float)atof(AnimationVariable("touch_difficultyfactor").c_str());

        entry.crudeOutDirectionRotated =
            animationData.outgoingDirection.GetRotated2D(animationData.outgoingBodyAngle.value());
        entry.crudeTurnAngle =
            entry.crudeOutDirectionRotated.GetAngle2D(animationData.incomingBodyDirection);
        // `FixAngle` of the simulation, written out.
        entry.crudeInBodyFixAbsolute = fabsf(ModulateIntoRange(
            -PI, PI, animationData.incomingBodyDirection.GetAngle2D() + 0.5f * PI));
        entry.crudeOutBodyDirection = Vector3(0, -1, 0).GetRotated2D(
            animationData.outgoingBodyAngle.value() + animationData.outgoingAngle.value());
        entry.crudeInBallNonzero = entry.incomingBallDirection.GetLength() != 0.0f;
        entry.crudeInBallAdapted = entry.incomingBallDirection;
        entry.crudeInBallAdapted.coordinates[2] *= 0.4f;
        if (entry.crudeInBallNonzero) entry.crudeInBallAdapted.Normalize();
        entry.crudeBallDirectionNormalized = entry.ballDirection;
        entry.crudeBallDirectionNormalized.Normalize(Vector3(0));
    }
    return derived;
}

}  // namespace animation
}  // namespace cuju
