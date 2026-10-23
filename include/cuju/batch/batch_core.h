// Copyright 2026 The cuju Authors
// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

// The batch buffers and the work on one environment, shared by the CUDA and the CPU backends.
#pragma once

#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../animation/skeleton.h"
#include "../file_path.h"
#include "../simulation/environment.h"
#include "../simulation/observation_encoding.h"
#include "../simulation/state_io.h"
#include "../simulation/static_data.h"
#include "../warp.h"
#include "agent_setup.h"

// Threads per environment. A loop from `Lane()` with this stride runs each index once.
#if CUJU_WARP_LANES
#define CUJU_ENVIRONMENT_LANES 32
#else
#define CUJU_ENVIRONMENT_LANES 1
#endif

namespace cuju {
namespace batch {

constexpr int MAXIMUM_AGENTS = 2 * simulation::MAXIMUM_TEAM_CONTROLLERS;
constexpr int SIMPLE115 = 115;

// A barrier that makes the writes of an environment's threads visible, a no-op with one thread.
CUJU_HOST_DEVICE void EnvironmentSync() {
#if CUJU_WARP_LANES
    __syncwarp();
#endif
}

enum ObservationFlags : int {
    OBSERVATION_SIMPLE115 = 1,
    OBSERVATION_SIMPLE115_VERSION2 = 2,
    OBSERVATION_SMM = 4,
    OBSERVATION_RAW = 8
};

// Each `raw` field is one array `[environmentCount][agents][dimensions]`, in this order.
// GRF: an agent sees its own team as the left team.
enum RawField : int {
    RAW_BALL,
    RAW_BALL_DIRECTION,
    RAW_BALL_ROTATION,
    RAW_BALL_OWNED_TEAM,
    RAW_BALL_OWNED_PLAYER,
    RAW_LEFT_TEAM,
    RAW_LEFT_TEAM_DIRECTION,
    RAW_LEFT_TEAM_TIRED_FACTOR,
    RAW_LEFT_TEAM_YELLOW_CARD,
    RAW_LEFT_TEAM_ACTIVE,
    RAW_LEFT_TEAM_ROLES,
    RAW_RIGHT_TEAM,
    RAW_RIGHT_TEAM_DIRECTION,
    RAW_RIGHT_TEAM_TIRED_FACTOR,
    RAW_RIGHT_TEAM_YELLOW_CARD,
    RAW_RIGHT_TEAM_ACTIVE,
    RAW_RIGHT_TEAM_ROLES,
    RAW_SCORE,
    RAW_STEPS_LEFT,
    RAW_GAME_MODE,
    RAW_ACTIVE,
    RAW_DESIGNATED,
    RAW_STICKY_ACTIONS,
    RAW_FIELD_COUNT
};
constexpr const char *RAW_FIELD_NAMES[RAW_FIELD_COUNT] = {"ball",
                                                          "ball_direction",
                                                          "ball_rotation",
                                                          "ball_owned_team",
                                                          "ball_owned_player",
                                                          "left_team",
                                                          "left_team_direction",
                                                          "left_team_tired_factor",
                                                          "left_team_yellow_card",
                                                          "left_team_active",
                                                          "left_team_roles",
                                                          "right_team",
                                                          "right_team_direction",
                                                          "right_team_tired_factor",
                                                          "right_team_yellow_card",
                                                          "right_team_active",
                                                          "right_team_roles",
                                                          "score",
                                                          "steps_left",
                                                          "game_mode",
                                                          "active",
                                                          "designated",
                                                          "sticky_actions"};
enum RawType : int { RAW_FLOAT32, RAW_INT32, RAW_BOOL, RAW_UINT8 };
CUJU_HOST_DEVICE int RawTypeBytes(int type) {
    return type == RAW_FLOAT32 || type == RAW_INT32 ? 4 : 1;
}

struct RawArray {
    int type;           // A `RawType`.
    int dimensions[2];  // The shape after the agent axis, with 0 for an absent dimension.
    int elements;
    size_t offset;  // In bytes.
};
struct RawLayout {
    int players;  // The length of each team array, that of the larger team.
    RawArray arrays[RAW_FIELD_COUNT];
    size_t bytes;
};

// Places the raw arrays of `outputs` agent views in one buffer, each at a multiple of 256 bytes.
inline RawLayout MakeRawLayout(size_t outputs, int players, int stickyActions) {
    RawLayout layout{};
    layout.players = players;
    const int teamShapes[6][3] = {{RAW_FLOAT32, players, 2}, {RAW_FLOAT32, players, 2},
                                  {RAW_FLOAT32, players, 0}, {RAW_BOOL, players, 0},
                                  {RAW_BOOL, players, 0},    {RAW_INT32, players, 0}};
    int shapes[RAW_FIELD_COUNT][3] = {{RAW_FLOAT32, 3, 0},
                                      {RAW_FLOAT32, 3, 0},
                                      {RAW_FLOAT32, 3, 0},
                                      {RAW_INT32, 0, 0},
                                      {RAW_INT32, 0, 0}};
    for (int team = 0; team < 2; team++)
        for (int field = 0; field < 6; field++)
            for (int index = 0; index < 3; index++)
                shapes[RAW_LEFT_TEAM + 6 * team + field][index] = teamShapes[field][index];
    const int matchShapes[6][3] = {{RAW_INT32, 2, 0}, {RAW_INT32, 0, 0},
                                   {RAW_INT32, 0, 0}, {RAW_INT32, 0, 0},
                                   {RAW_INT32, 0, 0}, {RAW_UINT8, stickyActions, 0}};
    for (int field = 0; field < 6; field++)
        for (int index = 0; index < 3; index++)
            shapes[RAW_SCORE + field][index] = matchShapes[field][index];
    size_t offset = 0;
    for (int field = 0; field < RAW_FIELD_COUNT; field++) {
        RawArray &array = layout.arrays[field];
        array.type = shapes[field][0];
        array.dimensions[0] = shapes[field][1];
        array.dimensions[1] = shapes[field][2];
        array.elements = (array.dimensions[0] ? array.dimensions[0] : 1) *
                         (array.dimensions[1] ? array.dimensions[1] : 1);
        array.offset = offset;
        offset += (outputs * array.elements * RawTypeBytes(array.type) + 255) / 256 * 256;
    }
    layout.bytes = offset;
    return layout;
}

// `output` is `environment * agents + agent`. A team smaller than `layout.players` pads with zeros.
CUJU_FUNCTION inline void WriteRaw(const simulation::AgentView &view, int stepsLeft,
                                   const RawLayout &layout, uint8_t *buffer, size_t output) {
    auto element = [&](int field, int index) {
        const RawArray &array = layout.arrays[field];
        return buffer + array.offset +
               ((size_t)output * array.elements + index) * RawTypeBytes(array.type);
    };
    auto setFloat = [&](int field, int index, float value) {
        *reinterpret_cast<float *>(element(field, index)) = value;
    };
    auto setInteger = [&](int field, int index, int value) {
        *reinterpret_cast<int32_t *>(element(field, index)) = value;
    };
    auto setByte = [&](int field, int index, int value) {
        *element(field, index) = (uint8_t)value;
    };
    const simulation::Observation &observation = view.observation;
    for (int index = 0; index < 3; index++) {
        setFloat(RAW_BALL, index, observation.ball[index]);
        setFloat(RAW_BALL_DIRECTION, index, observation.ballDirection[index]);
        setFloat(RAW_BALL_ROTATION, index, observation.ballRotation[index]);
    }
    setInteger(RAW_BALL_OWNED_TEAM, 0, observation.ballOwnedTeam);
    setInteger(RAW_BALL_OWNED_PLAYER, 0, observation.ballOwnedPlayer);
    for (int team = 0; team < 2; team++) {
        const simulation::TeamObservation &teamObservation = observation.team[team];
        const int first = team == 0 ? RAW_LEFT_TEAM : RAW_RIGHT_TEAM;
        for (int player = 0; player < layout.players; player++) {
            const bool present = player < teamObservation.count;
            const simulation::PlayerObservation &playerObservation =
                teamObservation.players[player];
            for (int coordinate = 0; coordinate < 2; coordinate++) {
                setFloat(first, 2 * player + coordinate,
                         present ? playerObservation.position[coordinate] : 0.0f);
                setFloat(first + 1, 2 * player + coordinate,
                         present ? playerObservation.direction[coordinate] : 0.0f);
            }
            setFloat(first + 2, player, present ? playerObservation.tiredFactor : 0.0f);
            setByte(first + 3, player, present && playerObservation.yellowCard);
            setByte(first + 4, player, present && playerObservation.active);
            setInteger(first + 5, player, present ? playerObservation.role : 0);
        }
    }
    setInteger(RAW_SCORE, 0, observation.score[0]);
    setInteger(RAW_SCORE, 1, observation.score[1]);
    setInteger(RAW_STEPS_LEFT, 0, stepsLeft);
    setInteger(RAW_GAME_MODE, 0, observation.gameMode);
    setInteger(RAW_ACTIVE, 0, view.active);
    setInteger(RAW_DESIGNATED, 0, view.designated);
    for (int index = 0; index < layout.arrays[RAW_STICKY_ACTIONS].elements; index++)
        setByte(RAW_STICKY_ACTIONS, index,
                index < view.stickyActionCount ? view.stickyActions[index] : 0);
}

struct BatchEnvironmentConfiguration {
    std::vector<std::string> assets;
    // The index into `assets` of each environment's bundle, or empty for the first bundle.
    std::vector<int32_t> assignment;
    std::string animations;  // The path of `animations.bin`.
    int environmentCount = 1;
    // GRF: environment `i` plays as `create_environment` after
    // `random.seed(seed * 2**32 + firstIndex + i)`.
    uint64_t seed = 0;
    int actionSet = 0;  // 0 `default`, 1 `v2`, 2 `full`.
    // `ObservationFlags`, with at most one of `simple115` and `simple115v2`.
    int observation = OBSERVATION_SIMPLE115_VERSION2;
    bool checkpoints = false;  // Adds the checkpoint reward to the scoring reward.
    bool numpyLegacy = true;   // The checkpoint reward adds as numpy before 2.0 does.
    // Without it, an ended episode stays at its end until a reset.
    bool autoReset = true;
    int ticksPerStep = 10;  // GRF `physics_steps_per_frame`.
    // The SMM size in cells, GRF `channel_dimensions`.
    int smmWidth = simulation::SMM_WIDTH, smmHeight = simulation::SMM_HEIGHT;
    int stack = 1;  // 4 for GRF `stacked`, else 1.
    // Per environment, GRF `game_engine_random_seed` for every episode, -1 to draw, empty for none.
    std::vector<int64_t> fixedSeeds;
    // `[assets.size()][2]` left and right agents, -1 for the bundle's. Empty keeps every bundle's.
    std::vector<int32_t> agentCounts;
    // GRF `episode_number`, the episode before the first, whose parity picks each later variant.
    int episodeNumber = 0;
    int threadsPerBlock = 64;  // CUDA only.
    // CUDA thread stack, or 0 for the driver's choice. The driver reserves it for every thread the
    // device holds, so a larger stack leaves less memory for environments.
    size_t stackBytes = 0;
    // CPU threads, the caller included, or 0 for one per processor the process may run on.
    int threads = 0;
    // The seed rule's index of the first environment, so environment `i` plays as index
    // `firstIndex + i`. Every index stays below 2**32.
    int64_t firstIndex = 0;
    // On CUDA the device memory of the batch comes from `allocate(bytes, context)`, which
    // returns null when it has none, and goes back through `release(pointer, context)` once the
    // device is idle. Null takes `cudaMalloc` and `cudaFree`.
    void *(*allocate)(size_t bytes, void *context) = nullptr;
    void (*release)(void *pointer, void *context) = nullptr;
    void *allocatorContext = nullptr;
    // On the CPU `pinThreads` pins each thread of the batch to a processor of the affinity mask,
    // and `numa` groups the threads by NUMA node, each group stepping a fixed range of
    // environments whose memory its threads touch first.
    bool pinThreads = false, numa = false;
    // On the CPU every buffer takes whole pages of its own, which `cudaHostRegister` can pin.
    bool pageAligned = false;
};

// GRF: the episode seeds come from Python's `random`, the MT19937 of CPython's C code.
struct EpisodeGenerator {
    static constexpr int WORDS = 624;
    uint32_t state[WORDS];
    int32_t next;  // `WORDS` when the state must turn over.
};

// As `random.seed(key)` for `key` with 32-bit words `words[0]`, the lowest, to `words[count - 1]`.
// CPython passes `init_by_array` the words of `abs(key)`, as few as it needs and at least one.
CUJU_FUNCTION inline void SeedEpisodeGenerator(EpisodeGenerator &generator, const uint32_t *words,
                                               int count) {
    constexpr int WORDS = EpisodeGenerator::WORDS;
    uint32_t *state = generator.state;
    state[0] = 19650218u;
    for (int index = 1; index < WORDS; index++)
        state[index] =
            1812433253u * (state[index - 1] ^ (state[index - 1] >> 30)) + (uint32_t)index;
    int index = 1, word = 0;
    for (int remaining = WORDS > count ? WORDS : count; remaining; remaining--) {
        state[index] = (state[index] ^ ((state[index - 1] ^ (state[index - 1] >> 30)) * 1664525u)) +
                       words[word] + (uint32_t)word;
        index++;
        word++;
        if (index >= WORDS) {
            state[0] = state[WORDS - 1];
            index = 1;
        }
        if (word >= count) word = 0;
    }
    for (int remaining = WORDS - 1; remaining; remaining--) {
        state[index] =
            (state[index] ^ ((state[index - 1] ^ (state[index - 1] >> 30)) * 1566083941u)) -
            (uint32_t)index;
        index++;
        if (index >= WORDS) {
            state[0] = state[WORDS - 1];
            index = 1;
        }
    }
    state[0] = 0x80000000u;
    generator.next = WORDS;
}

CUJU_FUNCTION inline void SeedEpisodeGenerator(EpisodeGenerator &generator, uint64_t seed,
                                               uint32_t index) {
    const uint32_t words[3] = {index, (uint32_t)seed, (uint32_t)(seed >> 32)};
    int count = 3;
    while (count > 1 && words[count - 1] == 0) count--;
    SeedEpisodeGenerator(generator, words, count);
}

// As CPython's `genrand_uint32`.
CUJU_FUNCTION inline uint32_t NextEpisodeWord(EpisodeGenerator &generator) {
    constexpr int WORDS = EpisodeGenerator::WORDS, SHIFT = 397;
    uint32_t *state = generator.state;
    if (generator.next >= WORDS) {
        auto twist = [&](int index, int following, int shifted) {
            const uint32_t mixed = (state[index] & 0x80000000u) | (state[following] & 0x7fffffffu);
            state[index] = state[shifted] ^ (mixed >> 1) ^ ((mixed & 1u) ? 0x9908b0dfu : 0u);
        };
        int index = 0;
        for (; index < WORDS - SHIFT; index++) twist(index, index + 1, index + SHIFT);
        for (; index < WORDS - 1; index++) twist(index, index + 1, index + SHIFT - WORDS);
        twist(WORDS - 1, 0, SHIFT - 1);
        generator.next = 0;
    }
    uint32_t value = state[generator.next++];
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    value ^= value >> 18;
    return value;
}

// GRF: `random.randint(0, 2000000000)`, which takes 31 bits until they fall below 2000000001.
CUJU_FUNCTION inline uint32_t DrawEpisodeSeed(EpisodeGenerator &generator) {
    for (;;) {
        const uint32_t bits = NextEpisodeWord(generator) >> 1;
        if (bits < 2000000001u) return bits;
    }
}

struct EnvironmentSlot {
    EpisodeGenerator generator;
    int32_t episode;  // The resets so far.
    int32_t reverse;  // The run's `reverseTeamProcessing`.
    uint32_t seed;
    int32_t variant;  // The index into `Variants`.
    int32_t bundle;
    int32_t leftAgents, rightAgents;
    int32_t collected[MAXIMUM_AGENTS];  // The checkpoints each agent collected.
    uint8_t stickyActions[MAXIMUM_AGENTS][simulation::MAXIMUM_STICKY_ACTIONS];
    int64_t fixedSeed;      // GRF `game_engine_random_seed`, or -1 to draw.
    int32_t episodeOffset;  // GRF `episode_number`, which shifts the parity of every episode.
    // Each team's designated player in the observation that ended the last episode, which an
    // automatic reset replaces. GRF's dumps hold it for a team without agents too.
    int32_t finalDesignated[2];
};

// The templates in backend memory, indexed by `bundle * 4 + reverse * 2 + parity`.
struct Variants {
    const simulation::SimulationContext *contexts;
    const simulation::World *const *templates;
};

// The batch buffers, in backend memory. Each environment has `agents` agent slots, the most agents
// of any bundle. `agentMask` marks the agents that exist.
struct EnvironmentBuffers {
    simulation::World *worlds;
    simulation::EnvironmentCore *cores;
    EnvironmentSlot *slots;
    int agents;
    int environmentCount;
    int actionSet, observation;
    bool checkpoints, numpyLegacy;
    int stack;
    float *simple115;  // `[environmentCount][agents][115 * stack]`
    uint8_t *smm;      // `[environmentCount][agents][smmHeight][smmWidth][4 * stack]`
    int smmWidth, smmHeight;
    uint8_t *raw;   // As `rawLayout` places it.
    float *reward;  // `[environmentCount][agents]`
    uint8_t *done;  // `[environmentCount]`
    // `[environmentCount]`, GRF `info['score_reward']`, the left team's before checkpoints and
    // sign.
    int32_t *scoreReward;
    // The outputs of the step that ended an episode, which an automatic reset replaces.
    float *finalSimple115;
    uint8_t *finalSmm;
    // `[environmentCount][agents][stack][1 + SMM_MARKS]`, per frame the mark count and then the
    // marks as indices into one frame.
    int32_t *smmMarks, *finalSmmMarks;
    uint8_t *finalRaw;
    RawLayout rawLayout;
    uint8_t *agentMask;  // `[environmentCount][agents]`
    // `[environmentCount][2]` from 0 to 1, set on every reset, negative or null for the scenario's.
    const float *teamDifficulty;
};

// GRF: a right agent's view negates both axes, a left agent's none.
CUJU_HOST_DEVICE int SideNegation(bool right) {
    return right ? simulation::NEGATE_X | simulation::NEGATE_Y : 0;
}

// GRF: some paths are undefined, such as a set piece without a taker reading a missing player.
// The host throws `std::runtime_error`. The device traps, and the next CUDA call reports it.
CUJU_FUNCTION inline void StopAtUndefinedPath(int environment) {
#if defined(__CUDA_ARCH__)
    __trap();
#else
    throw std::runtime_error(
        "environment " + std::to_string(environment) +
        " reached a path that GRF leaves undefined: " + simulation::UnsupportedReason());
#endif
}

CUJU_FUNCTION inline void CopyWorld(simulation::World *destination,
                                    const simulation::World *source) {
#if defined(__CUDA_ARCH__)
    static_assert(sizeof(simulation::World) % 4 == 0, "World size");
    uint32_t *destinationWords = reinterpret_cast<uint32_t *>(destination);
    const uint32_t *sourceWords = reinterpret_cast<const uint32_t *>(source);
    for (size_t word = Lane(); word < sizeof(simulation::World) / 4; word += CUJU_ENVIRONMENT_LANES)
        destinationWords[word] = sourceWords[word];
    EnvironmentSync();
#else
    memcpy(static_cast<void *>(destination), static_cast<const void *>(source),
           sizeof(simulation::World));
#endif
}

// GRF: even and odd episodes use the two team orders of the scenario.
CUJU_FUNCTION inline int VariantOf(const EnvironmentSlot &slot) {
    return slot.bundle * 4 + slot.reverse * 2 + ((slot.episode - 1 + slot.episodeOffset) & 1);
}

CUJU_FUNCTION inline void UpdateStickyActions(const EnvironmentBuffers &buffers,
                                              simulation::World &world, EnvironmentSlot &slot) {
    for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++) {
        const bool leftTeam = agent < slot.leftAgents;
        simulation::StickyActions(world, buffers.actionSet, leftTeam,
                                  leftTeam ? agent : agent - slot.leftAgents,
                                  slot.stickyActions[agent]);
    }
}

// GRF: each scenario built draws `random.randint(0, 2000000000)`. `create_environment` builds
// three, two configurations then the core's reset, and the second's parity sets the run's
// `reverseTeamProcessing`. A seed, from `seeds` or fixed, replaces every draw as
// `game_engine_random_seed` does. A negative entry of `seeds` draws, as null `seeds` does. Only the
// first reset reads `reverse`, and null follows the seeds.
CUJU_FUNCTION inline void ResetEnvironment(const EnvironmentBuffers &buffers,
                                           const Variants &variants, int environment,
                                           const int64_t *seeds, const int32_t *reverse) {
    EnvironmentSlot &slot = buffers.slots[environment];
    if (Lane() == 0) {
        const bool given = seeds && seeds[environment] >= 0;
        const bool fixed = slot.fixedSeed >= 0;
        // GRF: a deterministic scenario still draws seeds but plays its own seed and keeps team
        // order.
        const bool deterministic =
            variants.templates[slot.bundle * 4]->serialized.scenario.deterministic;
        if (slot.episode == 0) {
            uint32_t creation[3];
            for (int draw = 0; draw < 3; draw++)
                creation[draw] = given   ? (uint32_t)seeds[environment]
                                 : fixed ? (uint32_t)slot.fixedSeed
                                         : DrawEpisodeSeed(slot.generator);
            slot.reverse = reverse         ? reverse[environment]
                           : deterministic ? 0
                                           : (int32_t)(creation[1] & 1);
        }
        slot.seed = given   ? (uint32_t)seeds[environment]
                    : fixed ? (uint32_t)slot.fixedSeed
                            : DrawEpisodeSeed(slot.generator);
        slot.episode++;
        slot.variant = VariantOf(slot);
        if (deterministic)
            slot.seed = variants.templates[slot.variant]->serialized.scenario.gameEngineRandomSeed;
    }
    EnvironmentSync();
    const int variant = slot.variant;
    simulation::World &world = buffers.worlds[environment];
    CopyWorld(&world, variants.templates[variant]);
    // The difficulty goes in before the ticks that bring the match into play, which read it.
    if (buffers.teamDifficulty && Lane() == 0) {
        state::ScenarioState &scenario = world.serialized.scenario;
        const float left = buffers.teamDifficulty[environment * 2],
                    right = buffers.teamDifficulty[environment * 2 + 1];
        if (left >= 0.0f) scenario.leftTeamDifficulty = left;
        if (right >= 0.0f) scenario.rightTeamDifficulty = right;
    }
    EnvironmentSync();
    if (!simulation::EnvironmentResetFromTemplate(world, variants.contexts[variant], slot.seed) ||
        !simulation::EnvironmentCoreAfterReset(world, variants.contexts[variant],
                                               buffers.cores[environment]))
        StopAtUndefinedPath(environment);
    for (int agent = 0; agent < MAXIMUM_AGENTS; agent++) slot.collected[agent] = 0;
    UpdateStickyActions(buffers, world, slot);
}

// Runs on one thread for one agent. A step that ends an episode also writes the `final*` buffers.
CUJU_FUNCTION inline void WriteAgentOutputs(const EnvironmentBuffers &buffers, int environment,
                                            int agent, bool afterStep) {
    const size_t output = (size_t)environment * buffers.agents + agent;
    EnvironmentSlot &slot = buffers.slots[environment];
    const bool right = agent >= slot.leftAgents;
    simulation::AgentView view;
    simulation::MakeAgentView(buffers.cores[environment].observation, right, SideNegation(right),
                              right ? agent - slot.leftAgents : agent, buffers.actionSet,
                              slot.stickyActions[agent], view);
    if (afterStep && buffers.checkpoints)
        buffers.reward[output] = simulation::CheckpointReward(
            view, buffers.reward[output], slot.collected[agent], buffers.numpyLegacy);
    const bool finished = afterStep && buffers.done[environment];
    // GRF: as `FrameStack`, the last `stack` observations along the last axis, the oldest first. A
    // reset fills every frame with its observation.
    const int stack = buffers.stack;
    if (buffers.observation & (OBSERVATION_SIMPLE115 | OBSERVATION_SIMPLE115_VERSION2)) {
        const int width = SIMPLE115 * stack;
        float *stacked = buffers.simple115 + output * width;
        float *encoded = stacked + width - SIMPLE115;
        if (afterStep)
            for (int index = 0; index < width - SIMPLE115; index++)
                stacked[index] = stacked[index + SIMPLE115];
        simulation::EncodeSimple115(
            view, (buffers.observation & OBSERVATION_SIMPLE115_VERSION2) != 0, encoded);
        if (!afterStep)
            for (int index = 0; index < width - SIMPLE115; index++)
                stacked[index] = encoded[index % SIMPLE115];
        if (finished)
            for (int index = 0; index < width; index++)
                buffers.finalSimple115[output * width + index] = stacked[index];
    }
    if (buffers.observation & OBSERVATION_SMM) {
        // An SMM is zero but for a few marks, so an update rewrites only those. Each frame of a
        // stack keeps its marks as indices into one frame and lies at its own channels.
        const int markRow = 1 + simulation::SMM_MARKS;
        const size_t frameBytes =
            (size_t)buffers.smmWidth * buffers.smmHeight * simulation::SMM_LAYERS;
        const int channels = simulation::SMM_LAYERS * stack;
        auto at = [&](int mark, int frame) {
            return (size_t)(mark / simulation::SMM_LAYERS) * channels +
                   simulation::SMM_LAYERS * frame + mark % simulation::SMM_LAYERS;
        };
        auto write = [&](uint8_t *minimap, int32_t *marks, uint8_t value) {
            for (int frame = 0; frame < stack; frame++)
                for (int mark = 0; mark < marks[frame * markRow]; mark++)
                    minimap[at(marks[frame * markRow + 1 + mark], frame)] = value;
        };
        uint8_t *minimap = buffers.smm + output * frameBytes * stack;
        int32_t *marks = buffers.smmMarks + output * markRow * stack;
        write(minimap, marks, 0);
        if (afterStep)
            for (int index = 0; index < (stack - 1) * markRow; index++)
                marks[index] = marks[index + markRow];
        int32_t *newest = marks + (stack - 1) * markRow;
        newest[0] = simulation::SmmMarks(view, reinterpret_cast<int *>(newest + 1),
                                         buffers.smmWidth, buffers.smmHeight);
        if (!afterStep)
            for (int index = 0; index < (stack - 1) * markRow; index++)
                marks[index] = newest[index % markRow];
        write(minimap, marks, 255);
        if (finished) {
            uint8_t *finalMinimap = buffers.finalSmm + output * frameBytes * stack;
            int32_t *finalMarks = buffers.finalSmmMarks + output * markRow * stack;
            write(finalMinimap, finalMarks, 0);
            for (int index = 0; index < stack * markRow; index++) finalMarks[index] = marks[index];
            write(finalMinimap, finalMarks, 255);
        }
    }
    if (buffers.observation & OBSERVATION_RAW) {
        const simulation::World &world = buffers.worlds[environment];
        const int stepsLeft = world.serialized.scenario.gameDuration - view.observation.step;
        WriteRaw(view, stepsLeft, buffers.rawLayout, buffers.raw, output);
        if (finished) WriteRaw(view, stepsLeft, buffers.rawLayout, buffers.finalRaw, output);
    }
}

// One agent per lane, and the lanes diverge inside.
CUJU_FUNCTION inline void WriteEnvironmentOutputs(const EnvironmentBuffers &buffers,
                                                  int environment, bool afterStep) {
    EnvironmentSync();
    const EnvironmentSlot &slot = buffers.slots[environment];
    for (int agent = Lane(); agent < slot.leftAgents + slot.rightAgents;
         agent += CUJU_ENVIRONMENT_LANES)
        WriteAgentOutputs(buffers, environment, agent, afterStep);
    EnvironmentSync();
}

CUJU_HOST_DEVICE void ResetAndWriteEnvironment(const EnvironmentBuffers &buffers,
                                               const Variants &variants, int environment,
                                               const int64_t *seeds, const int32_t *reverse) {
    ResetEnvironment(buffers, variants, environment, seeds, reverse);
    if (Lane() == 0) buffers.done[environment] = 0;
    WriteEnvironmentOutputs(buffers, environment, false);
}

// With `autoReset`, an ended episode resets in the same call, done still set. Without it, the
// environment keeps its outputs and done flag until a reset, as GRF steps no episode past its end.
// `actions` is `[environmentCount][agents]`, left agents first. Out of range means idle.
CUJU_HOST_DEVICE void StepEnvironment(const EnvironmentBuffers &buffers, const Variants &variants,
                                      int environment, const int32_t *actions, bool autoReset) {
    EnvironmentSlot &slot = buffers.slots[environment];
    const int agents = slot.leftAgents + slot.rightAgents;
    if (!autoReset && buffers.done[environment]) {
        if (Lane() == 0) {
            for (int agent = 0; agent < buffers.agents; agent++)
                buffers.reward[(size_t)environment * buffers.agents + agent] = 0.0f;
            buffers.scoreReward[environment] = 0;
        }
        return;
    }
    int engineActions[MAXIMUM_AGENTS];
    for (int agent = 0; agent < agents; agent++) {
        const size_t index = (size_t)environment * buffers.agents + agent;
        const bool right = agent >= slot.leftAgents;
        int actionIndex = actions[index];
        const int actionCount = simulation::ActionSetSize(buffers.actionSet);
        // GRF: actions index a Python list, so an index from -count to -1 counts from the end.
        if (actionIndex < 0) actionIndex += actionCount;
        if (actionIndex < 0 || actionIndex >= actionCount) actionIndex = 0;
        // GRF: a right agent acts on the flipped pitch, so its direction turns back by both axes.
        engineActions[agent] = simulation::NegateActionAxes(
            simulation::ActionFromSet(buffers.actionSet, actionIndex), right, right);
    }
    int reward = 0;
    bool done = false;
    simulation::World &world = buffers.worlds[environment];
    if (!simulation::EnvironmentCoreStep(world, variants.contexts[slot.variant],
                                         buffers.cores[environment], engineActions, reward, done,
                                         [] { return true; }))
        StopAtUndefinedPath(environment);
    UpdateStickyActions(buffers, world, slot);
    if (Lane() == 0) {
        for (int agent = 0; agent < buffers.agents; agent++)
            buffers.reward[(size_t)environment * buffers.agents + agent] =
                agent < agents ? (float)(agent < slot.leftAgents ? reward : -reward) : 0.0f;
        buffers.done[environment] = done;
        buffers.scoreReward[environment] = reward;
    }
    WriteEnvironmentOutputs(buffers, environment, true);
    if (done && Lane() == 0)
        for (int team = 0; team < 2; team++)
            slot.finalDesignated[team] =
                buffers.cores[environment].observation.team[team].designatedPlayer;
    if (!autoReset || !done) return;
    ResetEnvironment(buffers, variants, environment, nullptr, nullptr);
    WriteEnvironmentOutputs(buffers, environment, false);
}

struct AssetVariant {
    int reverse = 0, parity = 0;
    // The keyboards no gamer owns, which the codec needs to save a state of this variant.
    int freeControllerCount = 0;
    simulation::StaticData staticData;
    std::unique_ptr<simulation::World> world;  // After a reset, in the rest frame.
    std::string scenario;  // `SaveScenario(*world)`, which identifies the variant of a state.
    std::vector<animation::Skeleton> skeletons;
};
struct Assets {
    std::string scenario;
    int leftAgents = 0, rightAgents = 0;
    AssetVariant variants[4];  // Indexed by `reverse * 2 + parity`.
};

inline std::string ReadFileBytes(const std::string &path) {
    std::ifstream file(FilePath(path), std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

// The scenario bytes of a state without the per-episode seed and the batch-set difficulties.
inline std::string SaveScenario(const simulation::World &world) {
    state::ScenarioState scenario = world.serialized.scenario;
    scenario.gameEngineRandomSeed = 0;
    scenario.leftTeamDifficulty = 0;
    scenario.rightTeamDifficulty = 0;
    state::Archive archive = state::Archive::Saver();
    scenario.Process(archive);
    return archive.data();
}

// `CUJASSET1`, the scenario name, the two agent counts, the variant count, then 4 variants of flag,
// parity, static data, reset state and skeletons. Integers are `int32_t`, blocks length-prefixed.
// `SetupAgents` handles other agent counts, where -1 keeps the bundle's. Static data and skeletons
// do not depend on the agents.
inline Assets LoadAssets(const std::string &path, int leftAgents = -1, int rightAgents = -1) {
    const std::string bytes = ReadFileBytes(path);
    size_t offset = 0;
    auto require = [&](size_t size) {
        if (offset + size > bytes.size())
            throw std::runtime_error("asset bundle truncated: " + path);
    };
    auto readInteger = [&]() {
        require(4);
        int32_t value;
        memcpy(&value, bytes.data() + offset, 4);
        offset += 4;
        return value;
    };
    auto readBlob = [&]() {
        const uint32_t size = (uint32_t)readInteger();
        require(size);
        std::string blob = bytes.substr(offset, size);
        offset += size;
        return blob;
    };
    if (bytes.compare(0, 9, "CUJASSET1") != 0)
        throw std::runtime_error("not an asset bundle: " + path);
    offset = 9;
    Assets assets;
    assets.scenario = readBlob();
    assets.leftAgents = readInteger();
    assets.rightAgents = readInteger();
    const int variantCount = readInteger();
    if (variantCount != 4) throw std::runtime_error("asset bundle: expected 4 variants");
    const int left = leftAgents == -1 ? assets.leftAgents : leftAgents;
    const int right = rightAgents == -1 ? assets.rightAgents : rightAgents;
    const bool setup = left != assets.leftAgents || right != assets.rightAgents;
    simulation::StateCodec codec;
    for (int index = 0; index < variantCount; index++) {
        const int reverse = readInteger(), parity = readInteger();
        if ((reverse != 0 && reverse != 1) || (parity != 0 && parity != 1))
            throw std::runtime_error("asset bundle: bad variant");
        AssetVariant &variant = assets.variants[reverse * 2 + parity];
        variant.reverse = reverse;
        variant.parity = parity;
        variant.staticData = simulation::ParseStatic(readBlob());
        variant.world = std::make_unique<simulation::World>();
        if (!codec.LoadFrame(readBlob(), 0, *variant.world))
            throw std::runtime_error("asset bundle: bad reset state");
        variant.freeControllerCount = codec.freeControllerCount;
        if (setup) {
            SetupAgents(*variant.world, variant.staticData, left, right, path);
            variant.freeControllerCount = variant.world->serialized.match.freeControllerCount;
        }
        variant.scenario = SaveScenario(*variant.world);
        variant.skeletons = animation::ParseSkeletons(readBlob());
        // The simulation reads one skeleton per player and per official.
        if (variant.skeletons.size() != (size_t)variant.staticData.count)
            throw std::runtime_error(
                "asset bundle: expected " + std::to_string(variant.staticData.count) +
                " skeletons, found " + std::to_string(variant.skeletons.size()));
    }
    for (const AssetVariant &variant : assets.variants)
        if (!variant.world) throw std::runtime_error("asset bundle: missing variant");
    assets.leftAgents = left;
    assets.rightAgents = right;
    return assets;
}

}  // namespace batch
}  // namespace cuju
