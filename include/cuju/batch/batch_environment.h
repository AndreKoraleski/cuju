// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A batch of environments on any backend. The host keeps the books, the backend does the work.
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../animation/animation_data.h"
#include "../animation/animation_derived.h"
#include "../animation/skeleton.h"
#include "batch_core.h"
#include "state_difference.h"
#include "tables.h"

namespace cuju {
namespace batch {

// A backend owns the buffers and provides `Stream`. `HOST_MEMORY` is true when the caller reads
// and writes the buffers directly. It provides `Configure`, called first, `Reserve`, `Threads`,
// `Allocate<T>(count)`, zeroed and at least one value, `Tables()` for `UploadTables`, `ToDevice`,
// `ToHost`, `Copy` and `Fill`, which wait, `ToDeviceAsync`, `Synchronize(stream)`,
// `SynchronizeAll()`, and `Reset`, `Step` and `Outputs`, which run `batch_core.h` over the batch.
// A step and a reset run on the caller's stream. Outputs stay in the buffers until the next call.
template <class Backend>
class BackendBatchEnvironment {
public:
    using Stream = typename Backend::Stream;

    // Throws `std::invalid_argument` for a configuration out of range and `std::runtime_error` for
    // a file that does not open or parse.
    explicit BackendBatchEnvironment(const BatchEnvironmentConfiguration &configuration)
        : configuration_(configuration) {
        if ((configuration.observation & OBSERVATION_SIMPLE115) &&
            (configuration.observation & OBSERVATION_SIMPLE115_VERSION2))
            throw std::invalid_argument("simple115 and simple115v2 are exclusive");
        if (configuration.assets.empty()) throw std::invalid_argument("expected an asset bundle");
        // GRF: a step of no ticks never brings the match into play.
        if (configuration.ticksPerStep < 1)
            throw std::invalid_argument("expected at least one tick in a step, found " +
                                        std::to_string(configuration.ticksPerStep));
        if (!configuration.fixedSeeds.empty()) {
            if ((int)configuration.fixedSeeds.size() != configuration.environmentCount)
                throw std::invalid_argument("expected one fixed seed for each environment, found " +
                                            std::to_string(configuration.fixedSeeds.size()));
            // GRF: the scenario holds its seed as a 32-bit `unsigned int`.
            for (int64_t seed : configuration.fixedSeeds)
                if (seed < -1 || seed > 4294967295LL)
                    throw std::invalid_argument(
                        "expected a fixed seed from 0 to 4294967295, found " +
                        std::to_string(seed));
        }
        // GRF: the index and the seed share one key, so an index of 2**32 would be another seed's.
        if (configuration.firstIndex < 0 ||
            configuration.firstIndex + configuration.environmentCount > (int64_t(1) << 32))
            throw std::invalid_argument(
                "expected a first index from 0 to 2**32 minus the environment count, found " +
                std::to_string(configuration.firstIndex));
        if (configuration.episodeNumber < 0)
            throw std::invalid_argument("expected an episode number of at least 0, found " +
                                        std::to_string(configuration.episodeNumber));
        // GRF: `FrameStack` holds four observations.
        if (configuration.stack != 1 && configuration.stack != 4)
            throw std::invalid_argument("expected a stack of 1 or 4 observations, found " +
                                        std::to_string(configuration.stack));
        if (configuration.stack > 1 && (configuration.observation & OBSERVATION_RAW) &&
            !(configuration.observation & ~OBSERVATION_RAW))
            throw std::invalid_argument("the raw representation has no stack");
        if (configuration.smmWidth < 1 || configuration.smmHeight < 1)
            throw std::invalid_argument("expected an SMM of at least one cell, found " +
                                        std::to_string(configuration.smmWidth) + " by " +
                                        std::to_string(configuration.smmHeight));
        const int count = configuration.environmentCount;
        std::vector<int32_t> assignment = configuration.assignment;
        if (assignment.empty()) assignment.assign(count, 0);
        if ((int)assignment.size() != count)
            throw std::invalid_argument("expected one bundle for each of the " +
                                        std::to_string(count) + " environments, found " +
                                        std::to_string(assignment.size()));
        for (int32_t bundle : assignment)
            if (bundle < 0 || bundle >= (int)configuration.assets.size())
                throw std::invalid_argument("expected a bundle from 0 to " +
                                            std::to_string(configuration.assets.size() - 1) +
                                            ", found " + std::to_string(bundle));
        const std::vector<int32_t> &agentCounts = configuration.agentCounts;
        if (!agentCounts.empty() && agentCounts.size() != 2 * configuration.assets.size())
            throw std::invalid_argument("expected a left and a right agent count for each of the " +
                                        std::to_string(configuration.assets.size()) +
                                        " bundles, found " + std::to_string(agentCounts.size()) +
                                        " counts");
        for (size_t bundle = 0; bundle < configuration.assets.size(); bundle++) {
            const int left = agentCounts.empty() ? -1 : agentCounts[2 * bundle];
            const int right = agentCounts.empty() ? -1 : agentCounts[2 * bundle + 1];
            bundles_.push_back(LoadAssets(configuration.assets[bundle], left, right));
        }
        store_ = animation::ParseAnimationTables(ReadFileBytes(configuration.animations));
        for (const animation::Animation &animation : store_->animations)
            if (animation.frameCount > state::MAXIMUM_ANIMATION_POSITIONS)
                throw std::invalid_argument("expected animations of at most " +
                                            std::to_string(state::MAXIMUM_ANIMATION_POSITIONS) +
                                            " frames, found " +
                                            std::to_string(animation.frameCount));
        derived_ = animation::DeriveAll(*store_);
        backend_.Configure(configuration);
        // Shared animation tables. Each variant uploads its own static data, skeletons and
        // template.
        auto &tables = backend_.Tables();
        UploadTables(tables, *store_, derived_, bundles_[0].variants[0].skeletons,
                     bundles_[0].variants[0].staticData);
        std::vector<simulation::SimulationContext> contexts;
        std::vector<const simulation::World *> templates;
        int agents = 0, players = 0;
        for (const Assets &bundle : bundles_) {
            // A bundle without agents plays the built-in AI on both sides.
            if (bundle.leftAgents < 0 || bundle.rightAgents < 0 ||
                bundle.leftAgents + bundle.rightAgents > MAXIMUM_AGENTS)
                throw std::runtime_error("expected at most " + std::to_string(MAXIMUM_AGENTS) +
                                         " agents in a bundle, found " +
                                         std::to_string(bundle.leftAgents + bundle.rightAgents));
            agents = std::max(agents, bundle.leftAgents + bundle.rightAgents);
            for (const AssetVariant &variant : bundle.variants) {
                simulation::SimulationContext context = tables.context;
                context.staticData = tables.Upload(&variant.staticData, 1);
                context.skeletons = tables.Upload(variant.skeletons);
                context.ticksPerStep = configuration.ticksPerStep;
                contexts.push_back(context);
                templates.push_back(tables.Upload(variant.world.get(), 1));
                const state::ScenarioState &scenario = variant.world->serialized.scenario;
                players = std::max({players, scenario.leftTeamSize, scenario.rightTeamSize});
                // GRF: the core asserts that a scenario with `control_all_players` gives each side
                // no agent or one for every controllable player.
                if (scenario.controlAllPlayers &&
                    simulation::DynamicPlayerSelection(*variant.world)) {
                    int left = 0, right = 0;
                    for (int player = 0; player < scenario.leftTeamSize; player++)
                        left += scenario.leftTeam[player].controllable ? 1 : 0;
                    for (int player = 0; player < scenario.rightTeamSize; player++)
                        right += scenario.rightTeam[player].controllable ? 1 : 0;
                    throw std::invalid_argument(
                        "expected 0 or " + std::to_string(left) + " left agents and 0 or " +
                        std::to_string(right) +
                        " right agents, all the controllable players of a side, for a scenario "
                        "with control_all_players, found " +
                        std::to_string(scenario.leftAgents) + " and " +
                        std::to_string(scenario.rightAgents));
                }
                // GRF: a player keeps one position per second of play between kick-offs, without a
                // limit.
                constexpr int HISTORY =
                    decltype(state::PlayerBaseState<
                             state::AIControllerState>::positionHistoryPerSecond)::CAPACITY;
                const int64_t seconds =
                    (int64_t)scenario.gameDuration * configuration.ticksPerStep / 100 + 1;
                if (seconds > HISTORY)
                    throw std::invalid_argument(
                        "expected a match of at most " + std::to_string(HISTORY) +
                        " seconds of play, " + "the position history of a player, found " +
                        std::to_string(scenario.gameDuration) + " steps of " +
                        std::to_string(configuration.ticksPerStep) + " ticks");
            }
        }
        variants_.contexts = tables.Upload(contexts);
        variants_.templates = tables.Upload(templates);
        players_ = players;
        buffers_.environmentCount = count;
        buffers_.agents = agents;
        buffers_.actionSet = configuration.actionSet;
        buffers_.observation = configuration.observation;
        buffers_.checkpoints = configuration.checkpoints;
        buffers_.numpyLegacy = configuration.numpyLegacy;
        buffers_.worlds = Allocate<simulation::World>(count);
        buffers_.cores = Allocate<simulation::EnvironmentCore>(count);
        buffers_.slots = Allocate<EnvironmentSlot>(count);
        buffers_.reward = Allocate<float>((size_t)count * agents);
        buffers_.done = Allocate<uint8_t>(count);
        buffers_.scoreReward = Allocate<int32_t>(count);
        const size_t stack = (size_t)configuration.stack;
        buffers_.stack = configuration.stack;
        if (configuration.observation & (OBSERVATION_SIMPLE115 | OBSERVATION_SIMPLE115_VERSION2)) {
            buffers_.simple115 = Allocate<float>((size_t)count * agents * SIMPLE115 * stack);
            buffers_.finalSimple115 = Allocate<float>((size_t)count * agents * SIMPLE115 * stack);
        }
        buffers_.smmWidth = configuration.smmWidth;
        buffers_.smmHeight = configuration.smmHeight;
        if (configuration.observation & OBSERVATION_SMM) {
            const size_t smmBytes = (size_t)configuration.smmWidth * configuration.smmHeight *
                                    simulation::SMM_LAYERS * stack;
            const size_t marks = (size_t)count * agents * (1 + simulation::SMM_MARKS) * stack;
            buffers_.smm = Allocate<uint8_t>((size_t)count * agents * smmBytes);
            buffers_.finalSmm = Allocate<uint8_t>((size_t)count * agents * smmBytes);
            buffers_.smmMarks = Allocate<int32_t>(marks);
            buffers_.finalSmmMarks = Allocate<int32_t>(marks);
        }
        if (configuration.observation & OBSERVATION_RAW) {
            buffers_.rawLayout =
                MakeRawLayout((size_t)count * agents, players,
                              simulation::StickyActionCount(configuration.actionSet));
            buffers_.raw = Allocate<uint8_t>(buffers_.rawLayout.bytes);
            buffers_.finalRaw = Allocate<uint8_t>(buffers_.rawLayout.bytes);
        }
        if (!Backend::HOST_MEMORY) actions_ = Allocate<int32_t>((size_t)count * agents);
        outputMask_ = Allocate<uint8_t>(count);
        std::vector<EnvironmentSlot> slots(count);
        std::vector<uint8_t> agentMask((size_t)count * agents);
        for (int environment = 0; environment < count; environment++) {
            EnvironmentSlot &slot = slots[environment];
            slot = EnvironmentSlot{};
            SeedEpisodeGenerator(slot.generator, configuration.seed,
                                 (uint32_t)(configuration.firstIndex + environment));
            slot.fixedSeed =
                configuration.fixedSeeds.empty() ? -1 : configuration.fixedSeeds[environment];
            slot.episodeOffset = configuration.episodeNumber;
            slot.bundle = assignment[environment];
            slot.leftAgents = bundles_[slot.bundle].leftAgents;
            slot.rightAgents = bundles_[slot.bundle].rightAgents;
            for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++)
                agentMask[(size_t)environment * agents + agent] = 1;
        }
        backend_.ToDevice(buffers_.slots, slots.data(), sizeof(EnvironmentSlot) * count);
        buffers_.agentMask = Allocate<uint8_t>(agentMask.size());
        backend_.ToDevice(buffers_.agentMask, agentMask.data(), agentMask.size());
        hostSlots_ = slots;
        started_.assign(count, false);
        unstarted_ = count;
        backend_.Reserve(buffers_, variants_);
    }
    BackendBatchEnvironment(const BackendBatchEnvironment &) = delete;
    BackendBatchEnvironment &operator=(const BackendBatchEnvironment &) = delete;

    int EnvironmentCount() const { return buffers_.environmentCount; }
    // The agent slots of each environment, the most agents of any bundle.
    int Agents() const { return buffers_.agents; }
    int LeftAgents(int environment) const { return hostSlots_[environment].leftAgents; }
    int RightAgents(int environment) const { return hostSlots_[environment].rightAgents; }
    // The length of each raw team array, that of the largest team of any bundle.
    int Players() const { return players_; }
    int LeftPlayers(int environment) const { return Scenario(environment).leftTeamSize; }
    int RightPlayers(int environment) const { return Scenario(environment).rightTeamSize; }
    int Bundle(int environment) const { return hostSlots_[environment].bundle; }
    const std::vector<Assets> &Bundles() const { return bundles_; }
    const EnvironmentBuffers &Buffers() const { return buffers_; }
    Backend &Device() { return backend_; }
    // The CPU threads, the caller included, or 0 on the GPU.
    int Threads() const { return backend_.Threads(); }
    // The bytes of backend memory that the batch has requested so far, its tables included.
    size_t MemoryBytes() const { return backend_.AllocatedBytes(); }

    // Waits. `seeds` and `reverse` are host arrays, one entry per environment, or null to draw the
    // seeds and follow them. Only a first reset reads `reverse`, which must be 0 or 1.
    void Reset(const uint32_t *seeds = nullptr, const int32_t *reverse = nullptr,
               Stream stream = Stream()) {
        const int64_t *deviceSeeds = nullptr;
        const int32_t *deviceReverse = nullptr;
        std::vector<int64_t> wideSeeds;
        if (seeds) {
            wideSeeds.assign(seeds, seeds + buffers_.environmentCount);
            deviceSeeds =
                Scratch<int64_t>(wideSeeds.data(), buffers_.environmentCount, scratchSeeds_);
        }
        if (reverse) {
            CheckReverse(reverse);
            deviceReverse = Scratch<int32_t>(reverse, buffers_.environmentCount, scratchReverse_);
        }
        backend_.Reset(buffers_, variants_, nullptr, deviceSeeds, deviceReverse, stream);
        backend_.Synchronize(stream);
        started_.assign(buffers_.environmentCount, true);
        unstarted_ = 0;
    }

    // Starts the next episode of each environment whose `mask` entry is nonzero, each as GRF's
    // `reset()` with its seed draw, and leaves the others untouched. A null `mask` takes every
    // environment. `mask` is `[environmentCount]`, in backend memory if `maskOnDevice`. `seeds`,
    // where a negative entry draws, and `reverse` are host arrays as in `Reset`, or null. Runs on
    // `stream` without waiting, so a CUDA graph can capture it with a device mask and no seeds,
    // once every environment has started. Until then it reads a device mask back.
    void ResetMasked(const uint8_t *mask, bool maskOnDevice, const int64_t *seeds,
                     const int32_t *reverse, Stream stream = Stream()) {
        const int count = buffers_.environmentCount;
        if (seeds) CheckSeeds(seeds);
        if (reverse) CheckReverse(reverse);
        const uint8_t *deviceMask =
            mask && !maskOnDevice ? AsyncScratch<uint8_t>(mask, count, maskedMask_, stream) : mask;
        const int64_t *deviceSeeds =
            seeds ? AsyncScratch<int64_t>(seeds, count, maskedSeeds_, stream) : nullptr;
        const int32_t *deviceReverse =
            reverse ? AsyncScratch<int32_t>(reverse, count, maskedReverse_, stream) : nullptr;
        if (unstarted_ > 0) {
            std::vector<uint8_t> started(count, 1);
            if (mask && maskOnDevice && !Backend::HOST_MEMORY) {
                // The caller's stream may still be writing the mask.
                backend_.Synchronize(stream);
                backend_.ToHost(started.data(), mask, count);
            } else if (mask) {
                started.assign(mask, mask + count);
            }
            for (int environment = 0; environment < count; environment++)
                if (started[environment] && !started_[environment]) {
                    started_[environment] = true;
                    unstarted_--;
                }
        }
        backend_.Reset(buffers_, variants_, deviceMask, deviceSeeds, deviceReverse, stream);
    }

    // `actions` is `[environmentCount][agents]`, in backend memory if `actionsOnDevice`. Then it
    // takes no copy, so on the GPU a CUDA graph can capture the call without `synchronize`.
    void Step(const int32_t *actions, bool actionsOnDevice, Stream stream = Stream(),
              bool synchronize = true) {
        CheckStarted();
        if (!actionsOnDevice || Backend::HOST_MEMORY) CheckActions(actions);
        const int32_t *deviceActions = actions;
        if (!actionsOnDevice && !Backend::HOST_MEMORY) {
            backend_.ToDeviceAsync(actions_, actions,
                                   sizeof(int32_t) * buffers_.environmentCount * Agents(), stream);
            deviceActions = actions_;
        }
        backend_.Step(buffers_, variants_, deviceActions, configuration_.autoReset, stream);
        if (synchronize) backend_.Synchronize(stream);
    }

    // Resets the environments whose done flag is set and clears it. It reads the flags in backend
    // memory, so on the GPU it does not wait for the device.
    void ResetDone(Stream stream = Stream()) {
        CheckStarted();
        backend_.Reset(buffers_, variants_, buffers_.done, nullptr, nullptr, stream);
    }

    // `Step` without the wait. The CPU steps in a thread of the batch, with a copy of `actions`,
    // while the caller goes on, and every later call waits for it. The GPU steps on `stream` as
    // `Step` does. Checks the actions before it returns, and `StepWait` throws what the step threw.
    void StepAsync(const int32_t *actions, bool actionsOnDevice, Stream stream = Stream()) {
        if constexpr (Backend::HOST_MEMORY) {
            CheckStarted();
            CheckActions(actions);
            const size_t count = (size_t)buffers_.environmentCount * Agents();
            if (!asyncActions_) asyncActions_ = Allocate<int32_t>(count);
            backend_.Copy(asyncActions_, actions, sizeof(int32_t) * count);
            // The copies outlive the batch's members, which its destruction frees first.
            backend_.Launch([this, buffers = buffers_, variants = variants_,
                             autoReset = configuration_.autoReset, copied = asyncActions_] {
                backend_.Step(buffers, variants, copied, autoReset, Stream());
            });
        } else {
            Step(actions, actionsOnDevice, stream, false);
        }
    }
    void StepWait() {
        if constexpr (Backend::HOST_MEMORY) backend_.Wait();
    }

    // Every later reset writes `difficulty`, a host array `[environmentCount][2]`, left then right,
    // from 0 to 1, or negative for the scenario's. Above 1 or NaN throws `std::invalid_argument`.
    // GRF: difficulty scales a team's statistics and, as it falls, slows its AI players' reactions.
    void SetDifficulty(const float *difficulty) {
        const size_t count = (size_t)buffers_.environmentCount * 2;
        for (size_t index = 0; index < count; index++)
            if (!(difficulty[index] <= 1))
                throw std::invalid_argument("expected a difficulty from 0 to 1, found " +
                                            std::to_string(difficulty[index]));
        if (!buffers_.teamDifficulty) buffers_.teamDifficulty = Allocate<float>(count);
        backend_.ToDevice(const_cast<float *>(buffers_.teamDifficulty), difficulty,
                          count * sizeof(float));
        hostDifficulty_.assign(difficulty, difficulty + count);
    }

    // The difficulty of `side`, 0 left and 1 right, that the episodes of `environment` play, the
    // one `SetDifficulty` set, else the scenario's. A scenario whose difficulty changes with the
    // parity of the episode gives the one of the environment's first episode.
    float Difficulty(int environment, int side) const {
        if (!hostDifficulty_.empty() && hostDifficulty_[environment * 2 + side] >= 0.0f)
            return hostDifficulty_[environment * 2 + side];
        // GRF: `episode_number` is the episode before the first, so its parity is the first's.
        const state::ScenarioState &scenario =
            Variant(Bundle(environment) * 4 + (configuration_.episodeNumber & 1))
                .world->serialized.scenario;
        return side == 0 ? scenario.leftTeamDifficulty : scenario.rightTeamDifficulty;
    }

    // Synchronous. Also copies out `core` and `slot`.
    std::string GetState(int environment, simulation::EnvironmentCore &core,
                         EnvironmentSlot &slot) {
        CheckEnvironment(environment);
        auto world = std::make_unique<simulation::World>();
        backend_.ToHost(world.get(), buffers_.worlds + environment, sizeof(simulation::World));
        backend_.ToHost(&core, buffers_.cores + environment, sizeof(core));
        backend_.ToHost(&slot, buffers_.slots + environment, sizeof(slot));
        simulation::StateCodec codec;
        codec.freeControllerCount = Variant(slot.variant).freeControllerCount;
        return codec.SaveFrame(*world, 0, nullptr);
    }

    // Synchronous. The state's scenario picks the variant to continue from. The reward is 0 and
    // done clear until the next step. `collected` may be null. Bytes of no state, or of another
    // scenario or difficulty, throw `std::invalid_argument`.
    void SetState(int environment, const std::string &bytes, int previousScoreDifference,
                  int previousGameMode, int previousBallOwnedTeam, const int32_t *collected) {
        CheckEnvironment(environment);
        auto world = std::make_unique<simulation::World>();
        simulation::StateCodec codec;
        codec.animationCount = (int)store_->animations.size();
        const int bundle = hostSlots_[environment].bundle;
        if (!codec.LoadFrame(bytes, 0, *world))
            throw std::invalid_argument(
                "expected the bytes of a state, found bytes that do not load" +
                LoadFailure(bundle, bytes));
        // GRF: scenarios compare without the state's `real_time`, so the bundle's stands in.
        const bool realTime = world->serialized.scenario.realTime;
        world->serialized.scenario.realTime =
            bundles_[bundle].variants[0].world->serialized.scenario.realTime;
        const std::string scenario = SaveScenario(*world);
        int parity = 0;
        while (parity < 4 && bundles_[bundle].variants[parity].scenario != scenario) parity++;
        if (parity == 4)
            throw std::invalid_argument(
                "expected a state of the scenario of the environment, found another: " +
                ScenarioDifference(world->serialized.scenario, BundleScenarios(bundle)));
        world->serialized.scenario.realTime = realTime;
        // GRF: an environment runs one scenario, so a state continues only under its difficulties.
        const state::ScenarioState &bundleScenario =
            bundles_[bundle].variants[parity].world->serialized.scenario;
        const state::ScenarioState &loaded = world->serialized.scenario;
        float expected[2] = {bundleScenario.leftTeamDifficulty, bundleScenario.rightTeamDifficulty};
        for (int side = 0; side < 2; side++)
            if (!hostDifficulty_.empty() && hostDifficulty_[environment * 2 + side] >= 0.0f)
                expected[side] = hostDifficulty_[environment * 2 + side];
        if (memcmp(&loaded.leftTeamDifficulty, &expected[0], sizeof(float)) != 0 ||
            memcmp(&loaded.rightTeamDifficulty, &expected[1], sizeof(float)) != 0)
            throw std::invalid_argument(
                "expected a state of the difficulties of the environment, " +
                std::to_string(expected[0]) + " and " + std::to_string(expected[1]) + ", found " +
                std::to_string(loaded.leftTeamDifficulty) + " and " +
                std::to_string(loaded.rightTeamDifficulty));
        const int variant = bundle * 4 + parity;
        simulation::EnvironmentCore core;
        core.previousScoreDifference = previousScoreDifference;
        core.previousGameMode = previousGameMode;
        core.previousBallOwnedTeam = previousBallOwnedTeam;
        simulation::EnvironmentGetState(*world, HostContext(variant), core.observation);
        EnvironmentSlot slot;
        backend_.ToHost(&slot, buffers_.slots + environment, sizeof(slot));
        slot.variant = variant;
        const int agents = slot.leftAgents + slot.rightAgents;
        for (int agent = 0; agent < MAXIMUM_AGENTS; agent++)
            slot.collected[agent] = collected && agent < agents ? collected[agent] : 0;
        for (int agent = 0; agent < agents; agent++) {
            const bool leftTeam = agent < slot.leftAgents;
            simulation::StickyActions(*world, buffers_.actionSet, leftTeam,
                                      leftTeam ? agent : agent - slot.leftAgents,
                                      slot.stickyActions[agent]);
        }
        // GRF: a sent-off keeper's hand-over lives in team data outside the state, so it carries
        // over.
        backend_.ToHost(world->formationOverride, buffers_.worlds[environment].formationOverride,
                        sizeof(world->formationOverride));
        backend_.ToDevice(buffers_.worlds + environment, world.get(), sizeof(simulation::World));
        backend_.ToDevice(buffers_.cores + environment, &core, sizeof(core));
        backend_.ToDevice(buffers_.slots + environment, &slot, sizeof(slot));
        backend_.Fill(buffers_.reward + (size_t)environment * Agents(), 0,
                      sizeof(float) * Agents());
        backend_.Fill(buffers_.done + environment, 0, 1);
        backend_.Fill(buffers_.scoreReward + environment, 0, sizeof(int32_t));
        backend_.Fill(outputMask_, 0, buffers_.environmentCount);
        backend_.Fill(outputMask_ + environment, 1, 1);
        backend_.Outputs(buffers_, outputMask_, false);
        backend_.SynchronizeAll();
        if (!started_[environment]) {
            started_[environment] = true;
            unstarted_--;
        }
    }

    // The host tables of `variant`, `bundle * 4 + reverse * 2 + parity`, valid until the next
    // call.
    const simulation::SimulationContext &HostContext(int variant) {
        hostContext_ = simulation::SimulationContext{&Variant(variant).staticData, &store_->view,
                                                     derived_.data(),
                                                     Variant(variant).skeletons.data(), nullptr};
        hostContext_.ticksPerStep = configuration_.ticksPerStep;
        return hostContext_;
    }

private:
    template <class T>
    T *Allocate(size_t count) {
        return backend_.template Allocate<T>(count);
    }
    template <class T>
    const T *Scratch(const T *host, int count, T *&slot) {
        if (Backend::HOST_MEMORY) return host;
        if (!slot) slot = Allocate<T>(count);
        backend_.ToDevice(slot, host, sizeof(T) * count);
        return slot;
    }
    // As `Scratch`, ordered on `stream`. The host array may go once this returns.
    template <class T>
    const T *AsyncScratch(const T *host, int count, T *&slot, Stream stream) {
        if (Backend::HOST_MEMORY) return host;
        if (!slot) {
            slot = Allocate<T>(count);
            // The zeroing of a new buffer orders on no stream of the caller's, so it ends first.
            backend_.SynchronizeAll();
        }
        backend_.ToDeviceAsync(slot, host, sizeof(T) * count, stream);
        return slot;
    }
    void CheckReverse(const int32_t *reverse) const {
        for (int environment = 0; environment < buffers_.environmentCount; environment++)
            if (reverse[environment] != 0 && reverse[environment] != 1)
                throw std::invalid_argument("the reverse flag of environment " +
                                            std::to_string(environment) + " is " +
                                            std::to_string(reverse[environment]) + ", not 0 or 1");
    }
    // GRF: the scenario holds its seed as a 32-bit `unsigned int`, and -1 draws.
    void CheckSeeds(const int64_t *seeds) const {
        for (int environment = 0; environment < buffers_.environmentCount; environment++)
            if (seeds[environment] < -1 || seeds[environment] > 4294967295LL)
                throw std::invalid_argument(
                    "expected seeds from 0 to 4294967295, or -1 to draw, found " +
                    std::to_string(seeds[environment]) + " for environment " +
                    std::to_string(environment));
    }
    const AssetVariant &Variant(int variant) const {
        return bundles_[variant / 4].variants[variant % 4];
    }
    // The variant without `reverseTeamProcessing`, whose teams come in the order of the
    // observation.
    const state::ScenarioState &Scenario(int environment) const {
        return Variant(Bundle(environment) * 4).world->serialized.scenario;
    }
    void CheckStarted() const {
        if (unstarted_ > 0)
            throw std::logic_error(
                "expected a reset of the batch, or a state for every environment, "
                "before its first step");
    }
    // GRF: each action indexes a Python list, which raises outside `[-count, count)` too.
    void CheckActions(const int32_t *actions) const {
        const int count = simulation::ActionSetSize(buffers_.actionSet);
        for (int environment = 0; environment < buffers_.environmentCount; environment++) {
            const EnvironmentSlot &slot = hostSlots_[environment];
            const int32_t *row = actions + (size_t)environment * buffers_.agents;
            for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++)
                if (row[agent] < -count || row[agent] >= count)
                    throw std::out_of_range(
                        "expected action indices from " + std::to_string(-count) + " to " +
                        std::to_string(count - 1) + ", found " + std::to_string(row[agent]));
        }
    }
    void CheckEnvironment(int environment) const {
        if (environment < 0 || environment >= buffers_.environmentCount)
            throw std::invalid_argument("expected an environment from 0 to " +
                                        std::to_string(buffers_.environmentCount - 1) + ", found " +
                                        std::to_string(environment));
    }
    std::vector<const state::ScenarioState *> BundleScenarios(int bundle) const {
        std::vector<const state::ScenarioState *> scenarios;
        for (const AssetVariant &variant : bundles_[bundle].variants)
            scenarios.push_back(&variant.world->serialized.scenario);
        return scenarios;
    }
    // Why `bytes` do not load as a state of `bundle`, another scenario or the field where they
    // stop holding one.
    std::string LoadFailure(int bundle, const std::string &bytes) const {
        auto environment = std::make_unique<state::EnvironmentStateFull>();
        try {
            state::Archive archive = state::Archive::Loader(bytes);
            state::ProcessEnvironmentScenario(archive, *environment);
        } catch (const std::exception &exception) {
            return std::string(": ") + exception.what();
        }
        environment->scenario.realTime =
            bundles_[bundle].variants[0].world->serialized.scenario.realTime;
        const std::string difference =
            ScenarioDifference(environment->scenario, BundleScenarios(bundle));
        if (!difference.empty()) return ", of another scenario: " + difference;
        try {
            state::Archive archive = state::Archive::Loader(bytes);
            state::ProcessEnvironmentState(archive, *environment,
                                           bundles_[bundle].variants[0].freeControllerCount);
            if (!archive.AtEnd())
                return ": " + std::to_string(bytes.size() - archive.position()) +
                       " bytes follow the end of the state";
        } catch (const std::exception &exception) {
            return std::string(": ") + exception.what();
        }
        return ": a player, team, gamer, keyboard or animation index lies out of range";
    }

    // Declared first, so it frees its memory after every other member is gone.
    Backend backend_;
    BatchEnvironmentConfiguration configuration_;
    std::vector<Assets> bundles_;
    // The slots as created, used only for their bundle and agent counts.
    std::vector<EnvironmentSlot> hostSlots_;
    // `[environmentCount][2]`, negative for the bundle's, or empty when the batch sets none.
    std::vector<float> hostDifficulty_;
    std::unique_ptr<animation::AnimationStore> store_;
    std::vector<animation::AnimationDerived> derived_;
    Variants variants_{};
    int players_ = 0;
    std::vector<bool> started_;
    int unstarted_ = 0;
    EnvironmentBuffers buffers_{};
    int32_t *actions_ = nullptr;
    uint8_t *outputMask_ = nullptr;  // The environment that `SetState` writes.
    int64_t *scratchSeeds_ = nullptr;
    int32_t *scratchReverse_ = nullptr;
    // `ResetMasked` copies on the caller's stream, so it keeps buffers apart from `Reset`'s.
    uint8_t *maskedMask_ = nullptr;
    int64_t *maskedSeeds_ = nullptr;
    int32_t *maskedReverse_ = nullptr;
    int32_t *asyncActions_ = nullptr;  // The CPU's copy of the actions of `StepAsync`.
    simulation::SimulationContext hostContext_{};
};

}  // namespace batch
}  // namespace cuju
