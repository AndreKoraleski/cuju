# Contributing

One rule shapes everything else in cuju. Its outputs equal those of Google Research Football, or
GRF, bit for bit, and every contribution keeps that rule.

## Reports

- **A difference from GRF** is the most valuable report. Use the Difference from GRF issue form,
    which asks for exactly what a reproduction needs.
- **Bugs, installation problems, performance and missing GRF features** have their own forms.
- **Questions** go to [Discussions](https://github.com/AndreKoraleski/cuju/discussions).

## Changes to the simulation

The simulation lives in `include/` and `src/`. Testing against GRF is internal. Issues are welcome.

- A change that keeps every output is welcome, such as a speed-up, a simplification or a crash fix.
- A change that moves an output is accepted only toward GRF, with a case that reproduces the
    difference.
- New behaviour beyond GRF is out of scope for now.

Open an issue before starting on a change to the simulation, so the approach is agreed first.

## Changes elsewhere

Changes to the documentation, the examples, the packaging, the build and the error messages need
only a pull request.

## Before a pull request

1. Run `pytest tests/` on the CPU, and on a GPU where one is available.
2. Run `ruff check` and `ruff format --check` for Python.
3. Run `clang-format-18 --dry-run -Werror` on changed C++ and CUDA files.
4. Keep the compiler flags as they are. The build refuses flags that change floating-point results,
    and a pull request that needs one explains why.
5. Write each commit message as one line in the imperative, such as `Port the flight of the ball`.

[AGENTS.md](AGENTS.md) holds the rules, commands and style in short form.
