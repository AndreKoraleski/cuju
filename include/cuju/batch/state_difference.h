// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Names the fields in which the scenario of a state differs from an environment's, for the errors
// of `SetState`, on the host only.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../state/environment_state.h"

namespace cuju {
namespace batch {

// GRF's name of a field of the scenario, so `leftTeam[3].startPosition` is
// `left_team[3].start_position`.
inline std::string GrfFieldName(const std::string &path) {
    std::string name;
    for (char character : path) {
        if (character >= 'A' && character <= 'Z') {
            name += '_';
            name += (char)(character - 'A' + 'a');
        } else {
            name += character;
        }
    }
    return name;
}

// A field's value in `bytes`, as GRF's Python prints it.
inline std::string FieldValue(const state::LayoutEntry &field, const std::string &bytes) {
    const char *data = bytes.data() + field.position;
    const std::string type = field.type;
    char text[64];
    if (type == TYPE_BOOLEAN && field.size == 1) return data[0] ? "True" : "False";
    if (type == TYPE_FLOAT && field.size == 4) {
        float value;
        memcpy(&value, data, 4);
        snprintf(text, sizeof(text), "%.9g", value);
    } else if (type == TYPE_VECTOR3 && field.size >= 12) {
        float value[3];
        memcpy(value, data, 12);
        snprintf(text, sizeof(text), "(%.9g, %.9g, %.9g)", value[0], value[1], value[2]);
    } else if (type == TYPE_UNSIGNED_INTEGER && field.size == 4) {
        uint32_t value;
        memcpy(&value, data, 4);
        snprintf(text, sizeof(text), "%u", value);
    } else if (field.size == 4) {
        // An integer or an enumeration, such as a role.
        int32_t value;
        memcpy(&value, data, 4);
        snprintf(text, sizeof(text), "%d", value);
    } else {
        std::string hexadecimal = "0x";
        for (int byte = 0; byte < field.size; byte++) {
            snprintf(text, sizeof(text), "%02x", (unsigned)(uint8_t)data[byte]);
            hexadecimal += text;
        }
        return hexadecimal;
    }
    return text;
}

// The saved fields of a scenario without the per-episode seed and the batch-set difficulties, as
// `SaveScenario` compares them, and their bytes.
inline std::vector<state::LayoutEntry> ScenarioFields(state::ScenarioState scenario,
                                                      std::string &bytes) {
    scenario.gameEngineRandomSeed = 0;
    scenario.leftTeamDifficulty = 0;
    scenario.rightTeamDifficulty = 0;
    std::vector<state::LayoutEntry> fields;
    state::Archive archive = state::Archive::Saver();
    archive.layout = &fields;
    archive.layoutBase = &scenario;
    archive.layoutStorage = false;
    scenario.Process(archive);
    bytes = archive.data();
    return fields;
}

// The fields in which `loaded` differs from the closest of `expected`, at most three of them, or
// an empty string if one of them is the same.
inline std::string ScenarioDifference(const state::ScenarioState &loaded,
                                      const std::vector<const state::ScenarioState *> &expected) {
    std::string loadedBytes;
    const std::vector<state::LayoutEntry> loadedFields = ScenarioFields(loaded, loadedBytes);
    constexpr size_t NAMED = 3;
    std::string closest;
    size_t fewest = SIZE_MAX;
    for (const state::ScenarioState *scenario : expected) {
        std::string bytes;
        const std::vector<state::LayoutEntry> fields = ScenarioFields(*scenario, bytes);
        std::vector<std::string> named;
        size_t count = 0, index = 0;
        for (; index < fields.size() && index < loadedFields.size(); index++) {
            const state::LayoutEntry &mine = loadedFields[index], &theirs = fields[index];
            // Teams of other sizes shift every later field, which the team sizes already name.
            if (mine.path != theirs.path || mine.size != theirs.size) break;
            if (memcmp(loadedBytes.data() + mine.position, bytes.data() + theirs.position,
                       mine.size) == 0)
                continue;
            count++;
            if (named.size() < NAMED)
                named.push_back(GrfFieldName(mine.path) + " is " + FieldValue(mine, loadedBytes) +
                                " in the state and " + FieldValue(theirs, bytes) +
                                " in the environment");
        }
        count += std::max(fields.size(), loadedFields.size()) - index;
        if (count == 0) return "";
        if (count >= fewest) continue;
        fewest = count;
        closest.clear();
        for (const std::string &difference : named)
            closest += (closest.empty() ? "" : "; ") + difference;
        if (count > named.size())
            closest += (closest.empty() ? "" : "; ") + std::string("and ") +
                       std::to_string(count - named.size()) + " more fields differ";
    }
    return closest;
}

}  // namespace batch
}  // namespace cuju
