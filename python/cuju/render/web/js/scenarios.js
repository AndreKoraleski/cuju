// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The scenarios that the page lists, and a short text for each. The texts take their facts from
// the scenario files and the scenarios page of the docs. The GRF MARL training variants and GRF's
// example of several agents stay playable from Python and the API, and the page lists leave them
// out.

const HIDDEN = /^(benchmark_full_game_|test_)/;

// The end of an academy episode, a goal, the ball out of play or a change of possession, else the
// given number of steps.
function academyEnd(steps) {
    return `It ends on a goal, the ball going out or lost possession, or after ${steps} steps.`;
}
const ACADEMY_END = academyEnd('401');
const ACADEMY_LEVEL = 'Difficulty 1.0 on the left and 0.6 on the right.';

export const SCENARIO_TEXT = {
    '11_vs_11_stochastic': 'Eleven a side over a full match of 3,001 steps. '
        + 'Difficulty 1.0 on the left and 0.6 on the right.',
    '11_vs_11_easy_stochastic': 'Eleven a side over a full match of 3,001 steps against a weak '
        + 'right team. Difficulty 1.0 on the left and 0.05 on the right.',
    '11_vs_11_hard_stochastic': 'Eleven a side over a full match of 3,001 steps against a strong '
        + 'right team. Difficulty 1.0 on the left and 0.95 on the right.',
    '11_vs_11_competition': 'Eleven a side over a full match of 3,001 steps, both teams at full '
        + 'strength with difficulty 1.0.',
    '11_vs_11_kaggle': 'The match of the Kaggle competition, eleven a side over 3,001 steps with '
        + 'the second half from step 1,500. Difficulty 1.0 for both teams.',
    '5_vs_5': 'Five a side over a full match of 3,001 steps, with both goalkeepers left to the AI. '
        + 'Difficulty 0.05 for both teams.',
    '1_vs_1_easy': 'One player against one over 501 steps. Difficulty 0 for both teams.',
    academy_empty_goal_close: `Drill, score from close range into an empty goal, two players `
        + `against one. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_empty_goal: `Drill, score into an empty goal from the middle of the pitch, two players `
        + `against one. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_run_to_score: `Drill, run from the middle to an empty goal with five defenders `
        + `chasing, two players against six. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_run_to_score_with_keeper: `Drill, run from the middle to score past a goalkeeper with `
        + `five defenders chasing, two players against six. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_pass_and_shoot_with_keeper: `Drill, two attackers at the edge of the box, the one with `
        + `the ball marked and the one in the centre free, three players against two. `
        + `${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_run_pass_and_shoot_with_keeper: `Drill, two attackers at the edge of the box, the one `
        + `with the ball free and the one in the centre marked, three players against two. `
        + `${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_3_vs_1_with_keeper: `Drill, three attackers against one defender and a goalkeeper, `
        + `four players against two. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_corner: 'Drill, score from a corner, eleven a side. It ends on a goal, the ball going '
        + `out or after 401 steps. ${ACADEMY_LEVEL}`,
    academy_counterattack_easy: `Drill, a counterattack of four attackers against one defender, `
        + `eleven a side. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_counterattack_hard: `Drill, a counterattack of four attackers against two defenders, `
        + `eleven a side. ${ACADEMY_END} ${ACADEMY_LEVEL}`,
    academy_single_goal_versus_lazy: 'Drill, score once against a team whose outfield players '
        + `stand still, eleven a side. ${academyEnd('3,001')} ${ACADEMY_LEVEL}`,
};

// The name of each scenario as the page shows it, in plain English.
const SCENARIO_NAMES = {
    '11_vs_11_stochastic': '11 vs 11',
    '11_vs_11_easy_stochastic': '11 vs 11, easy',
    '11_vs_11_hard_stochastic': '11 vs 11, hard',
    '11_vs_11_competition': '11 vs 11, competition',
    '11_vs_11_kaggle': '11 vs 11, Kaggle',
    '5_vs_5': '5 vs 5',
    '1_vs_1_easy': '1 vs 1, easy',
    academy_empty_goal_close: 'Academy, empty goal from close range',
    academy_empty_goal: 'Academy, empty goal',
    academy_run_to_score: 'Academy, run to score',
    academy_run_to_score_with_keeper: 'Academy, run to score past a keeper',
    academy_pass_and_shoot_with_keeper: 'Academy, pass and shoot past a keeper',
    academy_run_pass_and_shoot_with_keeper: 'Academy, run, pass and shoot past a keeper',
    academy_3_vs_1_with_keeper: 'Academy, 3 vs 1 with a keeper',
    academy_corner: 'Academy, corner',
    academy_counterattack_easy: 'Academy, counterattack, easy',
    academy_counterattack_hard: 'Academy, counterattack, hard',
    academy_single_goal_versus_lazy: 'Academy, single goal against a lazy team',
};

// The name of a scenario as the page shows it. A scenario without its own name, such as a
// training variant or a custom file, reads in words, with its underscores as spaces.
export function scenarioName(name) {
    if (!name) {
        return '';
    }
    if (SCENARIO_NAMES[name]) {
        return SCENARIO_NAMES[name];
    }
    const marl = /^benchmark_full_game_(\d+)_vs_(\d+)_(easy|medium|hard)$/.exec(name);
    if (marl) {
        return `GRF MARL ${marl[1]} vs ${marl[2]}, ${marl[3]}`;
    }
    const words = String(name).replace(/_/g, ' ').trim();
    return words.charAt(0).toUpperCase() + words.slice(1);
}

// A title as the page shows it. Titles that a batch or a recorder made by itself start with the
// name of their scenario, which reads in plain English here, so older recordings show the same.
export function titleText(title, scenario) {
    if (!title) {
        return '';
    }
    if (scenario && title.startsWith(scenario)) {
        return scenarioName(scenario) + title.slice(scenario.length);
    }
    return title.replace(/^([a-z0-9]+(?:_[a-z0-9]+)+)(?=,|$)/, (name) => scenarioName(name));
}

// The names a page list shows, in the order given.
export function listedScenarios(names) {
    return names.filter((name) => !HIDDEN.test(name));
}

// The short text of a scenario, empty for one without a text.
export function scenarioText(name) {
    return SCENARIO_TEXT[name] || '';
}
