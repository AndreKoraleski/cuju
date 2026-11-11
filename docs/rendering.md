# Rendering

cuju's **renderer** is a browser page that shows matches in three dimensions, with sound, from
files saved by a recorder or live from a running batch. It is the package cuju-render, which
installs into the folder of cuju and adds the module `cuju.render` and the command `cuju render`.
Its guide, with every page, command and setting, is [Rendering in cuju-render][guide].

## Installing

The extra `render` installs the renderer with a version that fits the installed cuju.

```sh
pip install 'cuju[render]'
```

`pip install cuju-render` does the same and brings cuju along when it is missing. cuju-render 0.1
runs with cuju 0.2.1 and every later release of cuju 0.2, and `pip check` names a pair that does
not fit. The renderer lives inside the folder of cuju, so `pip uninstall cuju-render cuju` removes
both together.

## Quick start

Three lines of Python show a running batch in the browser.

```python
import cuju.render
batch = cuju.BatchEnvironment('11_vs_11_stochastic_l1_r0', environment_count=8)
cuju.render.show(batch)
```

`cuju render demo` starts the server, opens the page and plays a live match of the built in AI,
with no script at all.

```sh
cuju render demo
```

## Watching and the match

The renderer reads states and never writes one. Watching a match therefore leaves every
observation, reward, done flag and state of the batch as it would be unwatched, bit for bit, and a
saved or watched match is the same match that training sees.

## Loading

cuju imports the renderer only when `cuju render` runs, a script imports `cuju.render` or
`CUJU_RENDER` asks for it, so a program that only trains loads none of it. Where cuju-render is
missing, `cuju render`, `python -m cuju.render` and `import cuju.render` each end with one message
naming `pip install 'cuju[render]'`, `cuju render` with exit status 1, and `CUJU_RENDER` warns once
per process through the logger `cuju` while every batch runs on as it would without the variable.
The [API reference](api.md#environment-variables) lists the variables of the renderer, and its
[command line](api.md#command-line) section shows how a program imports the renderer where it may
be missing, in a `try` block, since `importlib.util.find_spec('cuju.render')` raises there too.

## See also

- [Rendering in cuju-render][guide], the full guide to the renderer
- [Team presentation packages][presentation], the contract of a team's own look
- [Python reference of cuju-render][reference], the reference of `cuju.render`
- [States](states.md), the state format that the renderer reads

[guide]: https://github.com/AndreKoraleski/cuju-render/blob/main/docs/rendering.md
[presentation]: https://github.com/AndreKoraleski/cuju-render/blob/main/docs/presentation.md
[reference]: https://github.com/AndreKoraleski/cuju-render/blob/main/docs/reference.md
