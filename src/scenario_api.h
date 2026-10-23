// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The C API that builds the asset bundle of a scenario, on the host in both libraries.
// Include after `environment_api.h`, whose `Fail` and `cuju_last_error` it shares.
#include <fstream>

#include "cuju/file_path.h"
#include "cuju/floating_point.h"

#include "cuju/assets/scenario_builder.h"

#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {

// `cuju_scenario_build_bundle` refuses a `cuju_scenario_description` of another version.
#define CUJU_SCENARIO_VERSION 1

// One `AddPlayer` call of a GRF scenario file.
struct cuju_scenario_player {
    double x, y;       // The start position, in the coordinates of the environment
    int role;          // An `e_PlayerRole`, from 0 for the goalkeeper to 9 for the centre forward
    int lazy;          // Nonzero if the AI leaves the player idle
    int controllable;  // Nonzero if a human gamer can control the player
};

// What a scenario file builds for one parity of the episode number, in the fields of GRF's
// `ScenarioConfig`. A team that the file leaves empty holds the lazy goalkeeper that GRF adds.
struct cuju_scenario_description {
    int version;  // `CUJU_SCENARIO_VERSION`
    double ball_x, ball_y;
    const cuju_scenario_player *left_team;
    int left_size;
    const cuju_scenario_player *right_team;
    int right_size;
    int use_magnet, offsides, real_time;
    double left_team_difficulty, right_team_difficulty;
    int deterministic, end_episode_on_score, end_episode_on_possession_change,
        end_episode_on_out_of_play;
    int game_duration, second_half;
    int control_all_players;
    // GRF: only a deterministic scenario keeps the seed that it sets.
    unsigned int game_engine_random_seed;
};

// Writes the bundle without agents of a scenario to `out`, from the descriptions of its even and
// its odd episodes. `name` names the scenario inside the bundle. `animations` is the path of
// `animations.bin`.
int cuju_scenario_build_bundle(const cuju_scenario_description *even,
                               const cuju_scenario_description *odd, const char *name,
                               const char *animations, const char *out) {
    try {
        // Every call computes in the default floating point environment, whatever the caller set.
        cuju::DefaultFloatingPoint floatingPoint;
        const cuju_scenario_description *sources[2] = {even, odd};
        cuju::assets::ScenarioDescription parity[2];
        for (int index = 0; index < 2; index++) {
            const cuju_scenario_description &source = *sources[index];
            if (source.version != CUJU_SCENARIO_VERSION)
                throw std::invalid_argument("expected the scenario description version " +
                                            std::to_string(CUJU_SCENARIO_VERSION) + ", found " +
                                            std::to_string(source.version));
            if (source.left_size < 0 || source.right_size < 0)
                throw std::invalid_argument("expected team sizes of zero or more");
            cuju::assets::ScenarioDescription &target = parity[index];
            target.ballX = source.ball_x;
            target.ballY = source.ball_y;
            for (int side = 0; side < 2; side++) {
                const cuju_scenario_player *players = side ? source.right_team : source.left_team;
                const int size = side ? source.right_size : source.left_size;
                std::vector<cuju::assets::ScenarioPlayer> &team =
                    side ? target.rightTeam : target.leftTeam;
                for (int player = 0; player < size; player++) {
                    cuju::assets::ScenarioPlayer entry;
                    entry.x = players[player].x;
                    entry.y = players[player].y;
                    entry.role = players[player].role;
                    entry.lazy = players[player].lazy != 0;
                    entry.controllable = players[player].controllable != 0;
                    team.push_back(entry);
                }
            }
            target.useMagnet = source.use_magnet != 0;
            target.offsides = source.offsides != 0;
            target.realTime = source.real_time != 0;
            target.leftTeamDifficulty = source.left_team_difficulty;
            target.rightTeamDifficulty = source.right_team_difficulty;
            target.deterministic = source.deterministic != 0;
            target.endEpisodeOnScore = source.end_episode_on_score != 0;
            target.endEpisodeOnPossessionChange = source.end_episode_on_possession_change != 0;
            target.endEpisodeOnOutOfPlay = source.end_episode_on_out_of_play != 0;
            target.gameDuration = source.game_duration;
            target.secondHalf = source.second_half;
            target.controlAllPlayers = source.control_all_players != 0;
            target.gameEngineRandomSeed = source.game_engine_random_seed;
        }
        std::ifstream file(cuju::FilePath(animations), std::ios::binary);
        if (!file) throw std::runtime_error(std::string("cannot open ") + animations);
        const std::string animationBytes((std::istreambuf_iterator<char>(file)),
                                         std::istreambuf_iterator<char>());
        const cuju::assets::BuilderTables tables = cuju::assets::LoadBuilderTables(animationBytes);
        const std::string bundle = cuju::assets::BuildBundle(name, parity, tables);
        std::ofstream output(cuju::FilePath(out), std::ios::binary);
        if (!output) throw std::runtime_error(std::string("cannot write ") + out);
        output.write(bundle.data(), (std::streamsize)bundle.size());
        if (!output) throw std::runtime_error(std::string("cannot write ") + out);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

}  // extern "C"
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
