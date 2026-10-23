// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Fixed-capacity lists that the simulation uses in place of `std::vector`.
#pragma once

#include "../inline_vector.h"
#include "../state/environment_state.h"

namespace cuju {
namespace simulation {

constexpr int MAXIMUM_ANIMATIONS = 1536;   // The animation tables hold 1452.
constexpr int MAXIMUM_COMMAND_QUEUE = 16;  // A player queues about 10 commands at most.
// The opponents, the team mates, the ball and the posts around a player.
constexpr int MAXIMUM_FORCE_SPOTS = 64;
constexpr int MAXIMUM_POSITIONS = state::MAXIMUM_ANIMATION_POSITIONS;

using PlayerList = InlineVector<int, state::MAXIMUM_PLAYERS>;
// `int16_t` holds every index while `MAXIMUM_ANIMATIONS` stays at most 32767.
using AnimationList = InlineVector<int16_t, MAXIMUM_ANIMATIONS>;
using CommandQueue = InlineVector<state::PlayerCommand, MAXIMUM_COMMAND_QUEUE>;
using PositionList = InlineVector<Vector3, MAXIMUM_POSITIONS>;

}  // namespace simulation
}  // namespace cuju
