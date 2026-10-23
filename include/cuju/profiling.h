// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Device profiler, off unless built with `CUJU_PROFILE`. Each scope adds its cycles, callees
// included, to `scopeCycles` and its calls to `scopeCalls`. `cudaMemcpyFromSymbol` reads both.
#pragma once

#include "host_device.h"

namespace cuju {
namespace profiling {

// clang-format off
#define CUJU_PROFILE_SCOPE_IDENTIFIERS(ENTRY)                                                      \
    ENTRY(TICK) ENTRY(MATCH_PROCESS) ENTRY(MIRROR_MATCH) ENTRY(CHECK_BALL_COLLISIONS)              \
    ENTRY(REFEREE_PROCESS) ENTRY(BALL_PROCESS) ENTRY(BALL_PREDICTION) ENTRY(TEAM_UPDATE_SWITCH)    \
    ENTRY(TEAM_PROCESS) ENTRY(PLAYER_PROCESS) ENTRY(HUMANOID_PROCESS) ENTRY(REQUEST_COMMAND)       \
    ENTRY(HUMAN_REQUEST_COMMAND) ENTRY(SELECT_ANIMATION) ENTRY(BASE_SELECT_ANIMATION)              \
    ENTRY(CRUDE_SELECTION) ENTRY(STABLE_SORT) ENTRY(ANIMATION_APPLY)                               \
    ENTRY(CALCULATE_GEOMETRY_OFFSETS) ENTRY(TEAM_AI_PROCESS) ENTRY(OFFICIALS_PROCESS)              \
    ENTRY(CHECK_HUMANOID_COLLISIONS) ENTRY(POSSESSION_SCAN) ENTRY(PUT_BUFFERS)                     \
    ENTRY(UPDATE_CAMERA) ENTRY(ENVIRONMENT_GET_STATE) ENTRY(PUSH_MENTAL_IMAGE) ENTRY(TIME_NEEDED)  \
    ENTRY(CALCULATE_SPATIAL_STATE) ENTRY(BALL_TOUCH) ENTRY(KEEP_BEST_DIRECTION)                    \
    ENTRY(KEEP_BEST_BODY_DIRECTION) ENTRY(PHYSICS_VECTOR) ENTRY(PREDICTED_SITUATION)               \
    ENTRY(BEST_CHEATABLE) ENTRY(AI_CONTROLLER_REQUEST_COMMAND) ENTRY(REFEREE_REQUEST_COMMAND)      \
    ENTRY(MIRROR_HUMANOID)

enum ScopeID {
#define CUJU_PROFILE_SCOPE_ENUMERATOR(name) name,
    CUJU_PROFILE_SCOPE_IDENTIFIERS(CUJU_PROFILE_SCOPE_ENUMERATOR)
#undef CUJU_PROFILE_SCOPE_ENUMERATOR
    COUNT
};
// clang-format on

inline const char *Name(int id) {
    static const char *const names[] = {
#define CUJU_PROFILE_SCOPE_NAME(name) #name,
        CUJU_PROFILE_SCOPE_IDENTIFIERS(CUJU_PROFILE_SCOPE_NAME)
#undef CUJU_PROFILE_SCOPE_NAME
    };
    return id < COUNT ? names[id] : "?";
}

#if defined(CUJU_PROFILE) && defined(__CUDACC__)
static __device__ unsigned long long scopeCycles[COUNT];
static __device__ unsigned long long scopeCalls[COUNT];
#endif

#if defined(CUJU_PROFILE) && defined(__CUDA_ARCH__)
struct Scope {
    int id;
    long long startCycles;
    __device__ explicit Scope(int identifier) : id(identifier), startCycles(clock64()) {}
    __device__ ~Scope() {
        atomicAdd(&scopeCycles[id], (unsigned long long)(clock64() - startCycles));
        atomicAdd(&scopeCalls[id], 1ull);
    }
};
#define CUJU_PROFILE_CONCATENATE_EXPANDED(LEFT, RIGHT) LEFT##RIGHT
#define CUJU_PROFILE_CONCATENATE(LEFT, RIGHT) CUJU_PROFILE_CONCATENATE_EXPANDED(LEFT, RIGHT)
#define CUJU_PROFILE_SCOPE(id)                                             \
    ::cuju::profiling::Scope CUJU_PROFILE_CONCATENATE(cuju_profile_scope_, \
                                                      __LINE__)(::cuju::profiling::id)
#else
#define CUJU_PROFILE_SCOPE(id)
#endif

}  // namespace profiling
}  // namespace cuju
