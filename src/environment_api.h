// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The C API of `libcuju_cuda.so` and `libcuju_cpu.so`. The includer defines `BatchEnvironment`
// and `CUJU_BACKEND_NAME`. Calls are synchronous unless noted. A failure returns -1 or null, and
// `cuju_last_error` says why. A `stream` is a CUDA stream, or null for the default one.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include "cuju/batch/memory_estimate.h"
#include "cuju/batch/stacked_state.h"
#include "cuju/floating_point.h"
#include "cuju/nvtx.h"

using cuju::batch::BatchEnvironmentConfiguration;

namespace {
thread_local std::string lastError;
// Set by `cuju_memory_estimate` around its call of `cuju_environment_create`, which then counts.
thread_local unsigned long long *memoryEstimate = nullptr;
int Fail(const std::exception &exception) {
    lastError = exception.what();
    return -1;
}
}  // namespace

// Only these functions are exported. The build hides every other symbol. On Windows it exports what
// `CMakeLists.txt` finds here, each `cuju_` function whose definition starts a line.
#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {

// The message of the last failed call on this thread, or an empty string.
const char *cuju_last_error() { return lastError.c_str(); }

// `cuda` or `cpu`.
const char *cuju_backend() { return CUJU_BACKEND_NAME; }

// `cuju_environment_create` refuses a `cuju_configuration` of another version.
#define CUJU_CONFIGURATION_VERSION 2

struct cuju_configuration {
    int version;                // `CUJU_CONFIGURATION_VERSION`
    const char *const *assets;  // `[asset_count]` paths of asset bundles
    int asset_count;
    const int *assignment;   // `[environment_count]` bundle indices, or null for bundle 0
    const char *animations;  // The path of `animations.bin`
    int environment_count;
    // Environment `i` plays as GRF's `create_environment` after
    // `random.seed(seed * 2**32 + first_index + i)`.
    unsigned long long seed;
    int action_set;    // 0 `default`, 1 `v2`, 2 `full`
    int observation;   // A bit mask, 1 `simple115`, 2 `simple115v2`, 4 SMM and 8 `raw`.
    int checkpoints;   // Nonzero adds the checkpoint reward to the scoring reward
    int numpy_legacy;  // Nonzero adds the checkpoint reward as numpy before version 2 does
    int auto_reset;    // Nonzero resets every environment whose episode ended within the step
    // CPU threads, the caller included, or 0 for one per logical processor the process may use.
    int threads;
    int ticks_per_step;         // GRF's `physics_steps_per_frame`, 10 there
    int smm_width, smm_height;  // GRF's `channel_dimensions`, 96 and 72 there
    int stack;                  // 4 stacks the observations as GRF's `stacked`, else 1
    // `[environment_count]` episode seeds, as GRF's `game_engine_random_seed`, -1 to draw one and
    // null draws every seed.
    const long long *fixed_seeds;
    // `[asset_count][2]` left and right agents per bundle, 0 to the team's players or -1 to keep
    // the bundle's count. Null keeps every count.
    const int *agent_counts;
    // GRF's `episode_number`, the episode before the first, 0 there. Its parity picks the variant
    // of each later episode.
    int episode_number;
    // The index of environment 0 in the seed rule, so environment `i` plays as index
    // `first_index + i`, below 2**32.
    unsigned long long first_index;
    // On CUDA the device memory of the batch comes from `allocate(bytes, context)`, which
    // returns null when it has none, and goes back through `release(pointer, context)` once the
    // device is idle. Both null take `cudaMalloc` and `cudaFree`.
    void *(*allocate)(size_t bytes, void *context);
    void (*release)(void *pointer, void *context);
    void *allocator_context;
    // On the CPU nonzero pins each thread of the batch to a processor of the affinity mask.
    int pin_threads;
    // On the CPU nonzero groups the threads by NUMA node, each group stepping a fixed range of
    // environments whose memory its threads touch first.
    int numa;
    // On the CPU nonzero gives every buffer whole pages of its own, so that `cudaHostRegister` can
    // pin the output buffers one by one.
    int page_aligned;
};

// Creates a batch, which `cuju_environment_destroy` frees.
void *cuju_environment_create(const cuju_configuration *settings) {
    try {
        CUJU_NVTX_RANGE("create");
        // Every call computes in the default floating point environment, whatever the caller set.
        cuju::DefaultFloatingPoint floatingPoint;
        if (settings->version != CUJU_CONFIGURATION_VERSION)
            throw std::invalid_argument("expected the configuration version " +
                                        std::to_string(CUJU_CONFIGURATION_VERSION) + ", found " +
                                        std::to_string(settings->version));
        if (settings->environment_count < 1)
            throw std::invalid_argument("expected at least one environment, found " +
                                        std::to_string(settings->environment_count));
        BatchEnvironmentConfiguration configuration;
        configuration.assets.assign(settings->assets, settings->assets + settings->asset_count);
        if (settings->assignment)
            configuration.assignment.assign(settings->assignment,
                                            settings->assignment + settings->environment_count);
        configuration.animations = settings->animations;
        configuration.environmentCount = settings->environment_count;
        configuration.seed = settings->seed;
        configuration.actionSet = settings->action_set;
        configuration.observation = settings->observation;
        configuration.checkpoints = settings->checkpoints != 0;
        configuration.numpyLegacy = settings->numpy_legacy != 0;
        configuration.autoReset = settings->auto_reset != 0;
        configuration.threads = settings->threads;
        configuration.ticksPerStep = settings->ticks_per_step;
        configuration.smmWidth = settings->smm_width;
        configuration.smmHeight = settings->smm_height;
        configuration.stack = settings->stack;
        if (settings->fixed_seeds)
            configuration.fixedSeeds.assign(settings->fixed_seeds,
                                            settings->fixed_seeds + settings->environment_count);
        if (settings->agent_counts)
            configuration.agentCounts.assign(settings->agent_counts,
                                             settings->agent_counts + 2 * settings->asset_count);
        configuration.episodeNumber = settings->episode_number;
        // A larger value fails the range check rather than wrapping.
        configuration.firstIndex = (int64_t)std::min(settings->first_index, 1ULL << 62);
        configuration.allocate = settings->allocate;
        configuration.release = settings->release;
        configuration.allocatorContext = settings->allocator_context;
        configuration.pinThreads = settings->pin_threads != 0;
        configuration.numa = settings->numa != 0;
        configuration.pageAligned = settings->page_aligned != 0;
        lastError.clear();
        if (memoryEstimate) {
            using Backend =
                std::remove_reference<decltype(std::declval<BatchEnvironment &>().Device())>::type;
            *memoryEstimate = cuju::batch::EstimateMemory<Backend>(configuration);
            return nullptr;
        }
        return new BatchEnvironment(configuration);
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

// Writes into `bytes` the memory that `cuju_environment_create(settings)` requests, on the device
// for CUDA and the host for the CPU, without allocating it. It is what
// `cuju_environment_memory_bytes` reports right after that creation. It reads the bundles and the
// animation tables.
int cuju_memory_estimate(const cuju_configuration *settings, unsigned long long *bytes) {
    memoryEstimate = bytes;
    cuju_environment_create(settings);
    memoryEstimate = nullptr;
    return lastError.empty() ? 0 : -1;
}

// Writes into `bytes` the local memory that the creation of a batch adds for the kernels on the
// current CUDA device, which the driver then holds for the process, so it is 0 once a batch was
// created there, and always 0 for the CPU. Neither `cuju_memory_estimate` nor
// `cuju_environment_memory_bytes` counts it.
int cuju_local_memory_bytes(unsigned long long *bytes) {
    try {
        using Backend =
            std::remove_reference<decltype(std::declval<BatchEnvironment &>().Device())>::type;
        *bytes = Backend::LocalMemoryBytes();
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// The bytes of memory that the batch has requested so far, its tables included, on the device for
// CUDA and on the host for the CPU. The allocators round each request up, and CUDA also reserves
// the kernels' local memory once per process, which this leaves out.
unsigned long long cuju_environment_memory_bytes(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->MemoryBytes();
}

// Frees a batch and its buffers.
void cuju_environment_destroy(void *environment) {
    delete static_cast<BatchEnvironment *>(environment);
}

int cuju_environment_count(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->EnvironmentCount();
}

// The CPU threads, the caller included, or 0 for CUDA.
int cuju_environment_threads(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->Threads();
}

// Writes the left and right agents of each environment into `[environment_count]` arrays, either
// null to skip, and returns the agent axis, the largest agent count of any bundle.
int cuju_environment_agents(void *environment, int *left, int *right) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++) {
        if (left) left[index] = batch->LeftAgents(index);
        if (right) right[index] = batch->RightAgents(index);
    }
    return batch->Agents();
}

// Writes the left and right players of each environment as `cuju_environment_agents` does, and
// returns the size of each `raw` team array, the largest team of any bundle. An agent of the right
// team sees its team as the left team, so its arrays swap the two counts.
int cuju_environment_players(void *environment, int *left, int *right) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++) {
        if (left) left[index] = batch->LeftPlayers(index);
        if (right) right[index] = batch->RightPlayers(index);
    }
    return batch->Players();
}

// `difficulty` is a host `[environment_count][2]` array, from 0 to 1 for the left and right team or
// negative for the scenario's, which every later reset writes into the scenario.
int cuju_environment_set_difficulty(void *environment, const float *difficulty) {
    try {
        static_cast<BatchEnvironment *>(environment)->SetDifficulty(difficulty);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Writes the left and right difficulty of each environment into a host `[environment_count][2]`
// array, the one `cuju_environment_set_difficulty` set, else the scenario's.
void cuju_environment_difficulty(void *environment, float *difficulty) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++)
        for (int side = 0; side < 2; side++)
            difficulty[index * 2 + side] = batch->Difficulty(index, side);
}

// Starts the next episode of every environment and waits for it. `seeds` and `reverse` are host
// arrays of one entry per environment, null to draw the seeds and to follow them. Only the first
// reset of an environment reads `reverse`, a `reverseTeamProcessing` flag of 0 or 1.
int cuju_environment_reset(void *environment, const unsigned int *seeds, const int *reverse,
                           void *stream) {
    try {
        CUJU_NVTX_RANGE("reset");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->Reset(seeds, reverse, static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Starts the next episode of each environment whose `mask` byte is nonzero, as GRF's `reset()`
// does with its seed draw, and leaves the others untouched. A null `mask` takes every environment.
// `mask` is `[environment_count]`, in the batch's memory if `mask_on_device`, else on the host.
// `seeds` is a host `[environment_count]` array, -1 to draw as null does, and `reverse` is as in
// `cuju_environment_reset`. Waits only if `synchronize`, or to read back a device mask while some
// environment has never started.
int cuju_environment_reset_masked(void *environment, const unsigned char *mask, int mask_on_device,
                                  const int64_t *seeds, const int *reverse, void *stream,
                                  int synchronize) {
    try {
        CUJU_NVTX_RANGE("reset_masked");
        cuju::DefaultFloatingPoint floatingPoint;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        const BatchEnvironment::Stream batchStream = static_cast<BatchEnvironment::Stream>(stream);
        batch->ResetMasked(mask, mask_on_device != 0, seeds, reverse, batchStream);
        if (synchronize) batch->Device().Synchronize(batchStream);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Starts the next episode of every environment whose done flag is set, as a batch without
// `auto_reset` leaves it, and clears the flag. Returns without waiting for the device.
int cuju_environment_reset_done(void *environment, void *stream) {
    try {
        CUJU_NVTX_RANGE("reset_done");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->ResetDone(static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// `actions` holds `[environment_count][agents]` `int32` action indices, in the batch's memory if
// `on_device`, else on the host. CUDA waits only if `synchronize`. Needs a reset, or a state in
// every environment, first. A host action outside `[-n, n)` for n actions gives -2, as GRF raises.
int cuju_environment_step(void *environment, const int *actions, int on_device, void *stream,
                          int synchronize) {
    try {
        CUJU_NVTX_RANGE("step");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->Step(actions, on_device != 0, static_cast<BatchEnvironment::Stream>(stream),
                   synchronize != 0);
        return 0;
    } catch (const std::out_of_range &exception) {
        Fail(exception);
        return -2;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Starts a step as `cuju_environment_step` does and returns before it ends. The CPU steps in a
// thread of the batch with a copy of the actions, and every later call on the batch waits for it.
// CUDA steps on `stream` without waiting. Checks the actions first, failing with -2 as the step
// does. Each call needs a `cuju_environment_step_wait` before the next.
int cuju_environment_step_async(void *environment, const int *actions, int on_device,
                                void *stream) {
    try {
        CUJU_NVTX_RANGE("step_async");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->StepAsync(actions, on_device != 0, static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::out_of_range &exception) {
        Fail(exception);
        return -2;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Waits for the step of `cuju_environment_step_async` on the CPU, and fails as that step failed.
// Returns at once on CUDA, whose outputs follow the stream.
int cuju_environment_step_wait(void *environment) {
    try {
        CUJU_NVTX_RANGE("step_wait");
        static_cast<BatchEnvironment *>(environment)->StepWait();
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Waits for the work of the batch on `stream`, and fails as the device failed, so that a failure
// of an earlier call that did not wait surfaces here. Returns at once on the CPU.
int cuju_environment_synchronize(void *environment, void *stream) {
    try {
        static_cast<BatchEnvironment *>(environment)
            ->Device()
            .Synchronize(static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Returns the GRF engine bytes of environment `index`, valid until the next call on this thread.
// It copies outside the caller's streams. `episode` gets the score difference, the game mode and
// the ball-owning team after the previous step, then the checkpoints of each agent.
const char *cuju_environment_get_state(void *environment, int index, unsigned long long *size,
                                       int *episode) {
    thread_local std::string bytes;
    try {
        CUJU_NVTX_RANGE("get_state");
        cuju::DefaultFloatingPoint floatingPoint;
        cuju::simulation::EnvironmentCore core;
        cuju::batch::EnvironmentSlot slot;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        bytes = batch->GetState(index, core, slot);
        *size = bytes.size();
        episode[0] = core.previousScoreDifference;
        episode[1] = core.previousGameMode;
        episode[2] = core.previousBallOwnedTeam;
        for (int agent = 0; agent < batch->Agents(); agent++)
            episode[3 + agent] = slot.collected[agent];
        return bytes.data();
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

// Sets environment `index` to engine bytes from `cuju_environment_get_state` or GRF, with
// `episode` as there, and writes its outputs. It copies outside the caller's streams.
int cuju_environment_set_state(void *environment, int index, const char *bytes,
                               unsigned long long size, const int *episode) {
    try {
        CUJU_NVTX_RANGE("set_state");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->SetState(index, std::string(bytes, size), episode[0], episode[1], episode[2],
                       episode + 3);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Writes the observation stack of environment `index`, as a stacked state of GRF holds it in its
// `FrameStack`, after `cuju_environment_set_state`. `frames` holds the environment's agents, the
// oldest frame first along the last axis, `[agents][115 * stack]` `float32` for `name` `simple115`
// or `[agents][smm_height][smm_width][4 * stack]` `uint8` for `smm`.
int cuju_environment_set_stack(void *environment, int index, const char *name, const void *frames,
                               unsigned long long size) {
    try {
        cuju::batch::SetStackedFrames(*static_cast<BatchEnvironment *>(environment), index, name,
                                      frames, size);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// Returns a named output buffer, or null if the batch has none. Buffers lie in device memory for
// CUDA, host memory for the CPU, fixed until `cuju_environment_destroy`. With E environments and
// A agents, `simple115` is `float32` `[E][A][115 * stack]`.
// `smm` is `uint8` `[E][A][smm_height][smm_width][4 * stack]`.
// `raw` holds the arrays that `cuju_environment_raw_field` describes.
// `reward` is `float32` `[E][A]` and `done` is `uint8` `[E]`.
// `score_reward` is `int32` `[E]`, GRF's `info['score_reward']`, the left team's scoring reward.
// `final_simple115`, `final_smm` and `final_raw` hold the observation of the step that ended an
// episode, which the automatic reset replaces in the regular buffer.
void *cuju_environment_buffer(void *environment, const char *name) {
    const cuju::batch::EnvironmentBuffers &buffers =
        static_cast<BatchEnvironment *>(environment)->Buffers();
    const std::string bufferName = name;
    if (bufferName == "simple115") return buffers.simple115;
    if (bufferName == "final_simple115") return buffers.finalSimple115;
    if (bufferName == "smm") return buffers.smm;
    if (bufferName == "final_smm") return buffers.finalSmm;
    if (bufferName == "reward") return buffers.reward;
    if (bufferName == "done") return buffers.done;
    if (bufferName == "score_reward") return buffers.scoreReward;
    if (bufferName == "raw") return buffers.raw;
    if (bufferName == "final_raw") return buffers.finalRaw;
    return nullptr;
}

// Describes `raw` array `field`, from 0 until null. `type` is 0 `float32`, 1 `int32`, 2 `bool` or 3
// `uint8`. `dimensions` are the two after the agent axis, 0 if absent. `offset` counts bytes into
// `raw` and `final_raw`. Returns its name, or null past the last array or without `raw`.
const char *cuju_environment_raw_field(void *environment, int field, int *type, int *dimensions,
                                       unsigned long long *offset) {
    const cuju::batch::EnvironmentBuffers &buffers =
        static_cast<BatchEnvironment *>(environment)->Buffers();
    if (!buffers.raw || field < 0 || field >= cuju::batch::RAW_FIELD_COUNT) return nullptr;
    const cuju::batch::RawArray &array = buffers.rawLayout.arrays[field];
    *type = array.type;
    dimensions[0] = array.dimensions[0];
    dimensions[1] = array.dimensions[1];
    *offset = array.offset;
    return cuju::batch::RAW_FIELD_NAMES[field];
}

// Writes the current episode of environment `index`, synchronously, for GRF's dumps. `values`
// gets 17 entries, in order its seed as GRF's `game_engine_random_seed`, its
// `reverseTeamProcessing` flag, the resets so far, GRF's `episode_number` before the first, its
// bundle, the step of its last observation or -1 at a reset, its fixed seed or -1, whether its
// scenario is deterministic, GRF's `physics_steps_per_frame`, its left and right agents, its left
// and right players, the left and right team's designated player in its last observation, and
// those in the observation that ended its last episode. `difficulty`, unless null, gets the left
// and right team difficulty of the episode, then those of its scenario.
int cuju_environment_episode(void *environment, int index, long long *values, float *difficulty) {
    try {
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        if (index < 0 || index >= batch->EnvironmentCount())
            throw std::invalid_argument("expected an environment from 0 to " +
                                        std::to_string(batch->EnvironmentCount() - 1) + ", found " +
                                        std::to_string(index));
        const cuju::batch::EnvironmentBuffers &buffers = batch->Buffers();
        cuju::batch::EnvironmentSlot slot;
        batch->Device().ToHost(&slot, buffers.slots + index, sizeof(slot));
        cuju::simulation::Observation observation;
        batch->Device().ToHost(&observation, &buffers.cores[index].observation,
                               sizeof(observation));
        const cuju::batch::Assets &bundle = batch->Bundles()[slot.bundle];
        // The variant without `reverseTeamProcessing` orders the teams as the observation does.
        const cuju::state::ScenarioState &scenario = bundle.variants[0].world->serialized.scenario;
        values[0] = slot.seed;
        values[1] = slot.reverse;
        values[2] = slot.episode;
        values[3] = slot.episodeOffset;
        values[4] = slot.bundle;
        values[5] = observation.step;
        values[6] = slot.fixedSeed;
        values[7] = scenario.deterministic;
        values[8] = batch->HostContext(slot.variant).ticksPerStep;
        values[9] = slot.leftAgents;
        values[10] = slot.rightAgents;
        values[11] = scenario.leftTeamSize;
        values[12] = scenario.rightTeamSize;
        for (int team = 0; team < 2; team++) {
            values[13 + team] = observation.team[team].designatedPlayer;
            values[15 + team] = slot.finalDesignated[team];
        }
        if (difficulty) {
            const cuju::state::ScenarioState &played = buffers.worlds[index].serialized.scenario;
            batch->Device().ToHost(&difficulty[0], &played.leftTeamDifficulty, sizeof(float));
            batch->Device().ToHost(&difficulty[1], &played.rightTeamDifficulty, sizeof(float));
            const cuju::state::ScenarioState &own =
                bundle.variants[slot.variant % 4].world->serialized.scenario;
            difficulty[2] = own.leftTeamDifficulty;
            difficulty[3] = own.rightTeamDifficulty;
        }
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

// The name of the scenario that bundle `bundle` holds, or null past the last bundle.
const char *cuju_environment_scenario(void *environment, int bundle) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    if (bundle < 0 || bundle >= (int)batch->Bundles().size()) return nullptr;
    return batch->Bundles()[bundle].scenario.c_str();
}

}  // extern "C"
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
