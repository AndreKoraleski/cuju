# Changelog

All notable changes to this project are documented in this file. Every change that moves an output
is listed, as is every change to the interface. A change of outputs is a change of results, so runs
compare only within one version, or across versions whose entries say the outputs stayed the same.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Sample matches of eleven and of five a side, which the renderer's library receives on the first
    start of the server, and `cuju render sample`, which adds them again.

## [0.2.0] - 2026-11-07

With default options, every output stays as it was in 0.1.0, bit for bit, on both backends.

### Added

- Renderer MVP medal entrance and match contribution screen with grades, achievements and charts.

- Renderer Help page, resumable games from Home, team zip imports and page addresses.
- Recording fields for the engine scorer, called offside player and scenario timing, with half
    time in scenarios with a second half and separate episode and recording end cards.
- Automatic kit mapping and editable model colors in the team editor, and a soundscape and a
    crowd for each arena, with glass helmets for the players on the space station.
- Renderer teams with dynamic counts, fixed player indices or roles, goalkeeper choices and
    separate action sets for each side, and six GRF MARL training scenarios. Fixed sides can
    keep their chosen player order through set pieces by routing observations and actions.
- `cuju.state`, which reads single fields of a GRF state by path, through the C functions
    `cuju_state_layout` and `cuju_state_gather`, in the rest frame or the frame of the pitch.
- `cuju.render`, a renderer that shows matches in a browser in three dimensions with sound, from
    recordings that `Recorder` saves or live from a running batch with `CUJU_RENDER` or `watch`,
    reading states without changing them. It serves a local page with a library of recordings, a
    registry of Kaggle submission folders that play requested matches, team customization and
    custom player models, and ships art from GRF, three.js, Fabric.js, eleven KayKit characters with
    KayKit's animation clips, and its arenas' sounds.
- The renderer's page, with a Home page of live sources, recent recordings, links and credits, and
    pages to watch, browse recordings, request matches, customize teams and set the field. It
    presents matches as a broadcast does, with goal and decision replays, cards from the referee,
    restarts that ease into place, a wind-down after the final whistle, an end of match show and a
    broadcast mode without the page's controls, under a sky with a time of day, weather and five
    arenas.
- An ability from 0 to 1 for each side of a requested match, GRF's team difficulty for the players
    that the built-in AI controls, through `cuju.render.play`, `cuju render play` and the Matches
    page.
- The players that a team of the renderer's registry controls, from the active player alone up to
    every controllable player of its format, or the 10 outfield players with the goalkeeper left to
    the built-in AI, played through Kaggle's multi-agent form. `cuju.kaggle.KaggleAgent` takes
    `agents`, a list of agents that one function plays, and keeps its single agent form as it was.
    Each frame of a recording marks the players that agents control on its step.
- The renderer's page Play a game, where a person controls the active player of a side with the
    keyboard or a controller of the Xbox layout against GRF's built-in AI, at real time, through
    GRF's 19 actions, with two people on opposite sides, pause and resume, and saving to Recordings
    enabled by default.
- The environment variables `CUJU_RENDER`, `CUJU_RENDER_BACKLOG` and `CUJU_RENDER_HOME`.
- The command `cuju`, with `cuju render`, which serves the renderer's page, registers teams and
    plays requested matches, `cuju check`, the self-check of `python -m cuju.check`, `cuju replay`
    and `cuju dump-to-txt`, GRF's scripts for dumps, `cuju help` and `cuju --version`. The modules
    still run with `python -m`, and `python -m cuju check` stands in where the scripts folder of a
    user installation is missing from the path.
- The extra `render`, which names the renderer and adds no dependency beyond cuju's own.

## [0.1.0] - 2026-10-27

The first public release.

### Added

- A bit-exact port of Google Research Football 2.10.3 at commit 3d9e754. For the same seed and
    actions, every state, observation, reward and done flag equals GRF's, built by GCC for x86-64
    Linux with glibc 2.39.
- One C++ source, compiled for NVIDIA GPUs with one warp per environment and for the CPU with a
    thread pool, with the same bits on both and for any number of threads.
- All 19 of GRF's scenarios, any number of agents on either side, several scenarios in one batch,
    GRF's own scenario files without GRF, and scenarios that set `control_all_players`.
- GRF's representations `simple115`, `simple115v2`, `extracted` and `raw`, its rewards `scoring`
    and `checkpoints`, its action sets `default`, `v2` and `full`, its stacks and its difficulties.
- PyTorch tensors that view the batch's buffers, and steps that never wait for the GPU and that a
    CUDA graph or `torch.compile` can capture, through the operators `cuju::step` and
    `cuju::reset_done`.
- Resets of chosen environments with `reset(mask=)`, and seeds per environment with
    `reset(seeds=)`, as a list, a tuple, an array, a tensor or one integer, with -1 to draw.
- States in GRF's byte format in both directions, stacked or not. `set_state` rewrites the returned
    tensors in place and names the fields of a state that fails to fit.
- `batch.difficulty`, the difficulties that play, the scenario's where nothing overrides them.
- Exception classes in `cuju.errors`, each also a built-in type, and `cuju.__version__`.
- Memory freed for a batch dropped without `close()`.
- CPU batches that survive `os.fork()`, and `threads=0` within the cgroup's CPU quota.
- A clear message when a batch runs out of memory, naming the memory it requested and the memory
    that was free.
- `cuju.memory_estimate`, the memory that a batch requests at creation, counted without creating
    it, and `batch.memory_bytes`, the memory that a batch has requested.
- A warning when a GPU batch requests more than the free memory of its GPU, naming both amounts,
    which on Windows and WSL2 signals a spill into shared system memory.
- `check_actions=True`, which checks actions in GPU memory before each step and raises
    `ActionRangeError` before any environment steps, where such an action otherwise plays as `idle`.
- `debug=True`, which makes a GPU batch wait after each step and reset, so that a failure on the
    device raises at the call that caused it.
- The CMake option `CUJU_NVTX`, off by default and in the wheel, which wraps the calls of the C API
    in NVTX ranges for Nsight Systems.
- `with cuju.BatchEnvironment(...) as batch`, which closes the batch at the end of its block.
- Devices `cuda:N`, `torch.device` and `auto`, `first_index` to split a run, and one batch over
    several devices with `MultiDeviceBatch`.
- Options that change no output, `allocator='torch'`, asynchronous CPU steps with `step_async`,
    pinned threads, NUMA groups and page-locked outputs. Host inputs, page-locked ones included,
    may be reused as soon as a call returns.
- `cuju.env.create_environment` with GRF's arguments, defaults and gym 0.21 interface, Gymnasium's
    interface with `api='gymnasium'`, and the `GFootball-*-v0` ids.
- `cuju.gfootball`, GRF's import paths, so GRF code runs unchanged with its imports switched, with
    `write_dump`, custom players in `extra_players`, replays of dumps without rendering, and
    `Config.ScenarioConfig()`, `NewScenario` and `set_scenario_value`. As in GRF,
    `import cuju.gfootball` binds `env`, `scenario_builder.Scenario(config)` builds a
    configuration's scenario, `FootballEnv` plays a configuration without an `agent` through
    `step([])`, the dump options list GRF's dictionaries in `info['dumps']` under GRF's file names,
    and `control_all_players` combines with `extra_players`.
- GRF's wrapper classes in `cuju.gfootball.env.wrappers`, whose states carry the counts of
    `CheckpointRewardWrapper` and the frames of `FrameStack` as GRF's do, in the same bytes as the
    states of `create_environment` with `rewards='scoring,checkpoints'` and `stacked=True`.
- GRF's `bot` and `lazy` players in `cuju.players`, its `MultiAgentToSingleAgent` wrapper in
    `cuju.wrappers`, and its raw dictionaries in `cuju.raw`.
- Kaggle's observations and agents, and matches between Kaggle agents, in `cuju.kaggle`.
- GRF's dumps in `cuju.dumps`, which GRF replays and renders.
- `CUJU_LOG=1`, a line for each batch created, and `CUJU_BUNDLE_CACHE`, a cache of the bundles
    built from scenario files.
- `python -m cuju.check` in the wheel, with diagnostics, the shipped data against `SHA256SUMS`, and
    the digests on the CPU and every GPU.
- `cuju.digest`, action logs that replay a run bit for bit with `cuju.ActionLog` and `cuju.replay`,
    leaving every option that changes no output to the replaying machine, and
    `examples/compare_with_grf.py` for a side-by-side run with GRF.
- A quickstart notebook, `examples/cuju_quickstart.ipynb`, that runs on Colab.
- Wheels with both backends for Linux on x86-64, manylinux 2.28, and for Windows on x86-64, with
    machine code for compute capabilities 7.5, 8.0, 8.6 and 8.9. The wheel for Linux serves WSL2.
- Wheels with the CPU backend for Linux on Arm, for macOS 11 or later on Apple silicon, and for
    macOS 10.15 or later on Intel processors, with PyTorch 2.2 there.
- The same digests on every platform, from GCC and Clang on Linux, MSVC on Windows and Apple Clang
    on macOS.
- Builds from source for other GPU architectures.
- GRF's ids registered with Gymnasium, and with gym before 0.26, as soon as either is imported, so
    that `import cuju` leaves both unloaded.

[Unreleased]: https://github.com/AndreKoraleski/cuju/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AndreKoraleski/cuju/releases/tag/v0.1.0
