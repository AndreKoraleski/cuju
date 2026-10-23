// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The enumerations of the simulation. A state stores them as integers, so no enumerator moves.
#pragma once

namespace cuju {
namespace simulation {

enum e_Velocity { e_Velocity_Idle, e_Velocity_Dribble, e_Velocity_Walk, e_Velocity_Sprint };
enum e_FunctionType {
    e_FunctionType_None,
    e_FunctionType_Movement,
    e_FunctionType_BallControl,
    e_FunctionType_Trap,
    e_FunctionType_ShortPass,
    e_FunctionType_LongPass,
    e_FunctionType_HighPass,
    e_FunctionType_Header,
    e_FunctionType_Shot,
    e_FunctionType_Deflect,
    e_FunctionType_Catch,
    e_FunctionType_Interfere,
    e_FunctionType_Trip,
    e_FunctionType_Sliding,
    e_FunctionType_Special
};
enum e_TouchType {
    e_TouchType_Intentional_Kicked,
    e_TouchType_Intentional_NonKicked,
    e_TouchType_Accidental,
    e_TouchType_None,
    e_TouchType_SIZE
};
enum e_PlayerRole {
    e_PlayerRole_Goalkeeper,
    e_PlayerRole_CentreBack,
    e_PlayerRole_LeftBack,
    e_PlayerRole_RightBack,
    e_PlayerRole_DefensiveMidfielder,
    e_PlayerRole_CentralMidfielder,
    e_PlayerRole_LeftMidfielder,
    e_PlayerRole_RightMidfielder,
    e_PlayerRole_AttackingMidfielder,
    e_PlayerRole_CentreForward
};
enum e_GameMode {
    e_GameMode_Normal,
    e_GameMode_KickOff,
    e_GameMode_GoalKick,
    e_GameMode_FreeKick,
    e_GameMode_Corner,
    e_GameMode_ThrowIn,
    e_GameMode_Penalty
};
enum e_StrictMovement { e_StrictMovement_False, e_StrictMovement_True, e_StrictMovement_Dynamic };
enum e_PlayerCommandModifier {
    e_PlayerCommandModifier_None = 0,
    e_PlayerCommandModifier_KnockOn = 1
};

}  // namespace simulation
}  // namespace cuju
