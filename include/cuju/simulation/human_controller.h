// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The keyboard input of the gamers. Every `player` passed here must be under external control.
#pragma once

#include "../libm/powf.h"
#include "humanoid.h"

namespace cuju {
namespace simulation {

enum e_ButtonFunction {
    e_ButtonFunction_LongPass,
    e_ButtonFunction_HighPass,
    e_ButtonFunction_ShortPass,
    e_ButtonFunction_Shot,
    e_ButtonFunction_KeeperRush,
    e_ButtonFunction_Sliding,
    e_ButtonFunction_Pressure,
    e_ButtonFunction_TeamPressure,
    e_ButtonFunction_Switch,
    e_ButtonFunction_Sprint,
    e_ButtonFunction_Dribble,
    e_ButtonFunction_Size
};
constexpr float ANALOG_STICK_DEADZONE = 0.75f;

CUJU_FUNCTION inline state::HumanGamerState &HumanGamerOf(World &world, int player) {
    int gamerIndex = world.serialized.match.players[player].base.externalController;
    int firstTeamGamerCount = world.serialized.match.teams[0].humanGamers.count;
    return gamerIndex < firstTeamGamerCount
               ? world.serialized.match.teams[0].humanGamers[gamerIndex]
               : world.serialized.match.teams[1].humanGamers[gamerIndex - firstTeamGamerCount];
}

// GRF: the keyboard applies its mirror factor of 1 or -1 on store and again on report, so the
// stored direction is the reported one.
CUJU_FUNCTION inline Vector3 KeyboardDirection(const state::KeyboardState &keyboard) {
    return keyboard.direction;
}

CUJU_FUNCTION inline float GetLastSwitchBias(World &world,
                                             state::PlayerControllerState &controller) {
    if (!world.serialized.match.inPlay || world.serialized.match.inSetPiece)
        controller.lastSwitchTimeMilliseconds = (uint64_t)(int64_t)-10000;
    if (controller.lastSwitchTimeDurationMilliseconds > 0)
        return 1.0f - clamp((world.serialized.match.actualTimeMilliseconds -
                             controller.lastSwitchTimeMilliseconds) /
                                (float)controller.lastSwitchTimeDurationMilliseconds,
                            0.0f, 1.0f);
    return 0.0f;
}

CUJU_FUNCTION_NO_INLINE inline void HumanGetKeyboardInput(World &world, int player,
                                                          Vector3 &rawInputDirection,
                                                          float &rawInputVelocityFloat) {
    state::HumanControllerState &humanController = HumanGamerOf(world, player).controller;
    const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
    rawInputDirection = KeyboardDirection(humanController.keyboard);
    const Vector3 position = spatialState.position;
    float &directionX = rawInputDirection.coordinates[0],
          &directionY = rawInputDirection.coordinates[1];
    // Exact: each clamp equals `std::min(0.0f, x)` or `std::max(0.0f, x)`, signed zero included.
    // nvcc turns `(x < 0.0f) ? x : 0.0f` into a min that keeps an input -0, where the expression
    // gives +0. The expression never gives -0, so `+ 0.0f` matches it on host and device.
    if (position.coordinates[0] > PITCH_HALF_LENGTH)
        directionX = ((directionX < 0.0f) ? directionX : 0.0f) + 0.0f;
    if (position.coordinates[0] < -PITCH_HALF_LENGTH)
        directionX = (0.0f < directionX) ? directionX : 0.0f;
    if (position.coordinates[1] > PITCH_HALF_WIDTH)
        directionY = ((directionY < 0.0f) ? directionY : 0.0f) + 0.0f;
    if (position.coordinates[1] < -PITCH_HALF_WIDTH)
        directionY = (0.0f < directionY) ? directionY : 0.0f;
    if (rawInputDirection.GetLength() < ANALOG_STICK_DEADZONE) {
        rawInputDirection = spatialState.directionVector;
        rawInputVelocityFloat = IDLE_VELOCITY;
    } else {
        const bool *buttons = humanController.keyboard.buttonsPressed;
        if (buttons[e_ButtonFunction_Sprint])
            rawInputVelocityFloat = SPRINT_VELOCITY;
        else if (buttons[e_ButtonFunction_Dribble])
            rawInputVelocityFloat = DRIBBLE_VELOCITY;
        else if (buttons[e_ButtonFunction_Switch] &&
                 world.serialized.match.designatedPossessionPlayer == player)
            rawInputVelocityFloat = IDLE_VELOCITY;
        else
            rawInputVelocityFloat = WALK_VELOCITY;
        rawInputDirection.Normalize();
    }
    if (GetLastSwitchBias(world, humanController.base) > 0.0f) {
        float switchInfluence = 0.5f;
        float switchBias = libm::powf(GetLastSwitchBias(world, humanController.base), 0.7f);
        Vector3 currentMovement = spatialState.directionVector * PlayerFloatVelocity(world, player);
        Vector3 manualMovement = rawInputDirection * rawInputVelocityFloat;
        Vector3 resultMovement = currentMovement * switchBias * switchInfluence +
                                 manualMovement * (1.0f - switchBias * switchInfluence);
        rawInputDirection = resultMovement.GetNormalized(rawInputDirection);
        rawInputVelocityFloat = resultMovement.GetLength();
    }
}

CUJU_FUNCTION inline float HumanGetFloatVelocity(World &world, int player) {
    Vector3 direction;
    float velocity = 0;
    HumanGetKeyboardInput(world, player, direction, velocity);
    return velocity;
}

CUJU_FUNCTION inline Vector3 HumanGetDirection(World &world, int player) {
    return KeyboardDirection(HumanGamerOf(world, player).controller.keyboard)
        .GetNormalized(Humanoid(world, player).spatialState.directionVector);
}

}  // namespace simulation
}  // namespace cuju
