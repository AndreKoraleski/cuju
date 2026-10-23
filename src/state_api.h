// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The C API that reads the layout of a match state, on the host in both libraries.
// Include after `environment_api.h`, whose `Fail` and `cuju_last_error` it shares.
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cuju/simulation/state_io.h"

#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {

// Writes 1 into `mask`, `[size]` bytes, at each byte of the match `bytes`, a state after its
// pickle, that GRF leaves uninitialized and saves as it is, and 0 elsewhere, so two states can be
// compared without them. Returns 0, or -1 if the bytes hold no match.
int cuju_state_padding(const char *bytes, unsigned long long size, unsigned char *mask) {
    try {
        const std::string match(bytes, size);
        cuju::simulation::StateCodec codec;
        auto world = std::make_unique<cuju::simulation::World>();
        if (!codec.Load(match, *world))
            throw std::invalid_argument("expected the bytes of a match");
        std::vector<cuju::state::LayoutEntry> layout;
        cuju::state::Archive archive = cuju::state::Archive::Saver();
        archive.layout = &layout;
        archive.layoutBase = &world->serialized;
        archive.layoutStorage = false;
        cuju::state::ProcessEnvironmentState(archive, world->serialized, codec.freeControllerCount);
        std::memset(mask, 0, size);
        // GRF: a `blunted::radian` saves its float, its flag and three bytes of padding, which
        // hold whatever its memory held, so two GRF environments of one match differ there.
        for (const cuju::state::LayoutEntry &entry : layout)
            if (std::strcmp(entry.type, TYPE_RADIAN) == 0 && entry.size == 8 &&
                entry.position + 8 <= (int)size)
                std::memset(mask + entry.position + 5, 1, 3);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

}  // extern "C"
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
