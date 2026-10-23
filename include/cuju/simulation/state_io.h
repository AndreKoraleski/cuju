// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The conversion between states and worlds on the host.
#pragma once

#include <exception>
#include <string>
#include <vector>

#include "../state/environment_state.h"
#include "access.h"
#include "world.h"

namespace cuju {
namespace simulation {

// A state does not record how many free controllers its environment has, so the codec finds the
// count on the first load and tries it first on later ones.
struct StateCodec {
    int freeControllerCount = -1;
    int animationCount = -1;  // -1 leaves the animation pointers of a loaded state unchecked.

    // GRF: every pointer of a state that it loads is -1 or indexes an element of its collection.
    bool PointersInRange(const World &world, const std::vector<state::LoadedPointer> &pointers) {
        const state::MatchState &match = world.serialized.match;
        const int sizes[state::POINTER_KINDS] = {
            match.playerCount, 2,
            match.teams[0].humanGamers.count + match.teams[1].humanGamers.count,
            state::MATCH_KEYBOARDS, animationCount};
        for (const state::LoadedPointer &pointer : pointers) {
            const int size = sizes[pointer.kind];
            if (size >= 0 && (*pointer.value < -1 || *pointer.value >= size)) return false;
        }
        return true;
    }

    // `Load` and `Save` leave the coordinate frame as it is. `LoadFrame` and `SaveFrame` convert.
    bool Load(const std::string &bytes, World &world) {
        auto tryLoad = [&](int controllerCount) {
            try {
                std::vector<state::LoadedPointer> pointers;
                state::Archive archive = state::Archive::Loader(bytes);
                archive.pointers = &pointers;
                state::ProcessEnvironmentState(archive, world.serialized, controllerCount);
                return archive.AtEnd() && PointersInRange(world, pointers);
            } catch (const std::exception &) {
                return false;
            }
        };
        if (freeControllerCount >= 0 && tryLoad(freeControllerCount)) return true;
        for (int controllerCount = 0; controllerCount <= state::MAXIMUM_CONTROLLERS;
             controllerCount++)
            if (tryLoad(controllerCount)) {
                freeControllerCount = controllerCount;
                return true;
            }
        return false;
    }

    std::string Save(World &world, std::vector<state::SchemaEntry> *schema = nullptr) {
        BallMaterialize(world.serialized.match.ball);
        for (int image = 0; image < world.serialized.match.mentalImages.count; image++)
            MentalMaterialize(world.serialized.match.mentalImages[image]);
        state::Archive archive = state::Archive::Saver();
        archive.schema = schema;
        state::ProcessEnvironmentState(archive, world.serialized, freeControllerCount);
        return archive.data();
    }

    // `mirror` names the call frame. Bit 0 mirrors team 0, bit 1 team 1 and bit 2 the ball.
    bool LoadFrame(const std::string &bytes, int mirror, World &world) {
        world = World();
        if (!Load(bytes, world)) return false;
        InitializePlayerTeams(world);
        CanonicalToRest(world);
        MirrorMatch(world, mirror & 1, mirror & 2, mirror & 4);
        return true;
    }

    // Leaves `world` in the call frame that `mirror` names, as it was.
    std::string SaveFrame(World &world, int mirror,
                          std::vector<state::SchemaEntry> *schema = nullptr) {
        MirrorMatch(world, mirror & 1, mirror & 2, mirror & 4);
        RestToCanonical(world);
        std::string bytes = Save(world, schema);
        CanonicalToRest(world);
        MirrorMatch(world, mirror & 1, mirror & 2, mirror & 4);
        return bytes;
    }
};

}  // namespace simulation
}  // namespace cuju
