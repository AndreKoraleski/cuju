// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's match constants, in meters and meters per second. The pitch is centered, its length on x.
#pragma once

#include "host_device.h"

namespace cuju {

constexpr float IDLE_VELOCITY = 0.0f;
constexpr float DRIBBLE_VELOCITY = 3.5f;
constexpr float WALK_VELOCITY = 5.0f;
constexpr float SPRINT_VELOCITY = 8.0f;
// The sprint velocity that the animation tables are made for.
constexpr float ANIMATION_SPRINT_VELOCITY = 7.0f;
constexpr float IDLE_DRIBBLE_SWITCH = 1.8f;
constexpr float DRIBBLE_WALK_SWITCH = 4.2f;
constexpr float WALK_SPRINT_SWITCH = 6.0f;

constexpr uint32_t BALL_PREDICTION_SPAN_MILLISECONDS = 3000;
// Predictions past the span, one for each tick in a row that may reuse the last ones.
constexpr uint32_t CACHED_PREDICTIONS = 100;
constexpr uint32_t BALL_HISTORY_SIZE = 401;
constexpr uint32_t DEFAULT_TOUCH_OFFSET_MILLISECONDS = 80;

constexpr float PITCH_HALF_LENGTH = 55;
constexpr float PITCH_HALF_WIDTH = 36;
constexpr float GOAL_DEPTH = 2.55f;
constexpr float GOAL_HEIGHT = 2.5f;
constexpr float GOAL_HALF_WIDTH = 3.7f;
constexpr float LINE_HALF_WIDTH = 0.06f;
constexpr float DEFAULT_PLAYER_HEIGHT = 1.92f;

}  // namespace cuju
