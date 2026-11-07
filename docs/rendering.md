# Rendering

In cuju, the **renderer** is a browser page that shows matches in three dimensions, with the
players, the ball, the pitch and the crowd drawn from the art of Google Research Football, or GRF,
and from free characters, and with sound mixed in the browser. It watches two kinds of source.
A **recording** is a match saved to a file,
which the page plays at any speed, forwards and backwards, and a **live source** is one environment
of a running batch, which the page shows while the batch plays. The renderer ships in the package
as the module `cuju.render`. Its server needs only Python's standard library beside cuju's own
dependencies, and its page runs in the browser.

The viewer reads states. Watching a match therefore leaves every observation, reward, done flag and
state of the batch as it would be unwatched, bit for bit. GRF's own renderer runs inside the engine
and changes the matches it draws. In cuju, a saved or watched match is the same match that training
sees, which the public tests check on both backends.

## Quick start

The page runs as a local server, which listens on the loopback interface of the machine and keeps
its files in a hidden folder, `~/.cuju/render` by default. The package installs the command `cuju`,
whose subcommand `cuju render` runs the renderer, as `python -m cuju.render` does. The first command
starts the server and opens the page, and the second plays a script with one of its environments
shown live.

```sh
cuju render serve --open
CUJU_RENDER=on python train.py
```

The server prints its address, `http://127.0.0.1:8780/` by default. The option `--port` of `serve`
accepts ports from zero to 65535, with zero choosing a free port. The option `--home` of
`cuju render`, written before the subcommand, picks another folder, as in
`cuju render --home ./matches serve --port 8790`. A running batch with
`CUJU_RENDER` set appears in the page's list of live sources when the list next refreshes, every
eight seconds, and saved recordings appear on the next library refresh.

## Recordings

A recording holds the captured steps of one episode, each as the world position and rotation of
every joint
of every player, the position and rotation of the ball, the score, the clock, the game mode, the
owner of the ball, and the action and touches of every player. `cuju.render.Recorder` captures one
environment of a batch and saves one recording per episode into the library. The recorder is called
after `reset` and after each `step`, with the done flags of the step, and returns the path of a
recording when an episode ends.

```python
import cuju
import cuju.render

batch = cuju.BatchEnvironment('11_vs_11_stochastic_l1_r0', environment_count=8, device='cuda')
recorder = cuju.render.Recorder(
    batch,
    environment=3,
    title='Training run 12, environment 3',
    teams=[{'name': 'Agent', 'color': [214, 48, 49]}, {'name': 'Built-in AI'}],
)
observation = batch.reset()
recorder.capture()
for step in range(3001):
    actions = policy(observation)
    observation, reward, done, information = batch.step(actions)
    recorder.capture(done)
```

Each capture reads the state of one environment and keeps about 4 KB of it, at a cost of about
5 milliseconds on one core of a laptop processor, more than two steps of an eleven a side match on
the CPU. A batch with automatic resets already shows the start of the next episode on the step that
ends one, so the recorder closes the recording at the step before, which it marks as the end, adds
the goal of the ending step to that frame's score from the scoring signal, and opens the next
recording with the new kick off. The positions of the ending step itself are gone, so a goal shows
the step before the ball crosses the line. A batch with `auto_reset=False` keeps the last step,
and the recording ends on it.

Call `recorder.save()` to keep an episode before it ends, for example when a training run stops.
It returns the saved path, or `None` when there are no captured frames, and starts a fresh
recording on the next capture. The list `recorder.saved` holds the paths written so far.
Capture after every step to preserve consecutive motion. Each recorder takes one batch of a
`MultiDeviceBatch`, as a live source does.

A full match of eleven a side takes 3001 frames with automatic resets and 3002 without, about 12 MB
on disk, and one of five a side about 1 MB with the built in AI on both sides and up to about 5 MB
when agents keep the players moving. A full match of
eleven a side plays and records in about half a minute on one core of a laptop processor. The
library keeps each recording as a `.cjr` file beside a small summary that the page lists, and the
page deletes both together.

Two sample matches ship with the renderer, a full match of eleven a side and one of five a side,
both played by the built-in AI. The first start of the server copies each into the library as
`sample-11v11` and `sample-5v5`, where they appear on Home and in Recordings and open in Watch like
any other recording. The folder notes each sample it has received in `samples.json`, so a deleted
sample stays deleted, and a recording of the same name stays as it is. The command
`cuju render sample` adds both samples again, and `cuju render sample 11v11` or
`cuju render sample 5v5` adds one.

## Live sources

A live source streams one environment of a running batch to the page. The environment variable
`CUJU_RENDER` gives every batch of a process a source, with `on` for environment 0 or an index for
another environment, and the call `cuju.render.watch` gives one batch a source from code.

```python
source = cuju.render.watch(batch, environment=0, title='Self-play, league match')
```

A source listens on a free port of the loopback interface and announces itself with a small file in
the renderer's folder, which the server lists and relays to the page. While a viewer watches, the
source reads passages of 50 consecutive steps, five seconds of play, and starts the next passage
when the wall clock has caught up. A batch that runs faster than real time therefore reaches the
page as continuous passages, which the page joins with cuts, and pays for about ten state reads per
second of wall time, about 5 milliseconds each. A batch that runs slower than real time is shown
step by step. Each step checks whether a live source is attached. An attached source also checks
its viewers under a lock and reads states for viewers or the backlog.
On the GPU, each read also waits for the step in flight, as `get_state` does, and a GPU batch that
queues many steps ahead of the device may take tens of seconds to send its first frame.

A source keeps the last 10 seconds of the steps it read, and a viewer who joins late receives them
first, so a second viewer, or a page reopened, starts with that history. The first viewer of a quiet
source starts at the present. The variable `CUJU_RENDER_BACKLOG`, or `backlog=True` for `watch`,
makes the source read every step, watched or not, at the cost of one state read per step, so that
every viewer starts 10 seconds back.

The batch keeps stepping through any trouble of watching. A source that fails, from a full disk or
a closed port among other causes, turns itself off with one warning through the logger `cuju`, and
the batch steps on. A step
captured into a CUDA graph and a step of a compiled function under `torch.compile` run without
Python, so a source shows the steps that the batch takes outside them. Each batch of a
`MultiDeviceBatch` announces a source of its own under `CUJU_RENDER`, with the index counted within
that batch, and `watch` takes one of those batches.

## Teams and requested matches

The page keeps a **team registry**, a list of agents that can play requested matches. A team is a
submission folder in the format of Kaggle's GRF competition, a folder whose `main.py` defines the
agent as its last function, together with any modules and weights that it loads. The built in AI of
GRF is always a team as well. A team is added on the page by choosing or dropping its folder, or
from the command line, and each team carries a **format**, `5v5` or `11v11`, its size for
competition.

```sh
cuju render team add path/to/submission --name "Rule-based baseline" --format 11v11
cuju render team list
cuju render play --left t-20261106-101500-3fa9 --right builtin --seed 7 --right-ability 0.4
cuju render team export t-20261106-101500-3fa9 --out baseline.zip
```

The team add command also accepts exported zip files, keeping the team name, format, control
settings and assets. The option `--name` can rename the import.

A match requested on the page plays in a process of its own on the CPU and becomes a recording in
the library. The play command also takes a device. Each submission runs from its own folder,
with its own modules, as on Kaggle. Full matches follow the format of the team, five or eleven a
side. Academy scenarios accept either format when the requested players exist on that side.
The default scenario is the Kaggle competition scenario. The built in AI plays any scenario.

Each side of a requested match takes an **ability**, a number from 0 to 1 that sets how well the
built in AI plays the players it controls on that side, which is GRF's team difficulty under
another name. It steers the whole team of the built in AI and, in a scenario where a submission
controls one player of its side, that player's teammates as well. A side left alone plays the
scenario's own difficulty, and the recording keeps the ability that each team played with. The
options `--left-ability` and `--right-ability` of `cuju render play`, the argument `ability` of
`cuju.render.play` and the Ability controls on the Matches page set it. The page enables a slider
for each side with a checkbox, in
steps of 0.05, and leaving it unchecked keeps the scenario value. The same controls on Play
set the ability of the teammates and opponents there.

Each submission sets its count, assignment, player selection, goalkeeper and action set in the
add form or the Identity section of the editor. The count runs from one to every controllable
player of the scenario. A requested match checks both sides separately and reports the team,
scenario and conflicting setting when they cannot play together. A smaller scenario refuses an
excess count rather than reducing it. Exported teams carry these settings with them. An import
keeps the control settings from its archive unless the user edits the control form. Earlier
teams with only a count and goalkeeper setting load with the same control as before.

Dynamic assignment gives GRF the requested count of agents. With fewer agents than controllable
players, GRF switches them among players near the ball. One dynamic agent is the Kaggle case.
With every controllable player, each agent keeps a player when both sides use full counts or AI.
The scenario goalkeeper setting lets GRF decide whether the keeper can be selected. The agent
setting requires a controllable keeper, and leaves selection to GRF in dynamic assignment.

Fixed assignment takes a list of player indices starting at zero, or GRF roles, in agent order.
The roles are GK, CB, LB, RB, DM, CM, LM, RM, AM and CF. Repeated roles select successive players
of that role in lineup order. The form shows the default lineup for the format. Its player buttons
show lineup indices and roles and select fixed players in agent order. The match checks against
the actual scenario lineup, which may differ. A missing role, repeated index or
uncontrollable selected player gives an error. Agent goalkeeper control requires the keeper in
the selection. Builtin goalkeeper control requires fixed assignment and leaves the controllable
keeper outside the selection. Scenario goalkeeper control in fixed assignment requires a keeper
that the scenario makes uncontrollable.

Fixed assignment gives the engine every controllable agent of the side. On each step the runner
reads the active player of every agent from its raw observation. It sends the observations of the
selected players to the submission in the chosen order, and routes each returned action to the
agent holding that player. Agents holding players outside the selection send builtin AI, 19.
GRF clears the selected players of both sides at set pieces when either side uses dynamic
selection. A side with every controllable agent keeps every controllable player under control,
while the agents holding those players may change. The runner follows those changes on each step.
During a handover an agent may briefly report minus one as its active player. For that step the
fixed side uses the built in AI for all its players and skips the submission call. Other missing
selections receive replacement agent columns, and the submission is called in selected order even
when two agents report the same player. The built in AI then replaces its actions in the columns
of missing selected players.

The default action set has 19 actions. The v2 set adds builtin AI at index 19, so a submission may
hand any selected player to the AI for one step. Each side keeps its own action set, including
its range checks and negative indices, when the other side needs v2. Every player left to the
AI uses the ability of its side. A fixed team may meet a dynamic team, a single agent or the AI.

The renderer includes six training scenarios from GRF MARL by Light MALib. Each format has easy,
medium and hard variants, with team difficulties of 0.05, 0.6 and 1.0. They take 3001 calls to
`step`, with the second half at step 1500, swap the formations by episode parity and leave both
goalkeepers uncontrollable. Their names are shown below.

```text
benchmark_full_game_5_vs_5_easy
benchmark_full_game_5_vs_5_medium
benchmark_full_game_5_vs_5_hard
benchmark_full_game_11_vs_11_easy
benchmark_full_game_11_vs_11_medium
benchmark_full_game_11_vs_11_hard
```

Pass their names to `cuju.render.play` or to the server's match and game requests. They are
available to the renderer through scenario files and the bundle cache. The scenario lists on Matches
and Play hold the full matches, `1_vs_1_easy` and the academies, and leave out these six and
GRF's example scenario `test_example_multiagent`, which stay playable from Python and the API.

The add form also accepts a single `main.py` or a `.zip` or `.tar.gz` archive. Uploads allow up to
100 MB,
with at most 10000 files and 500 MB after unpacking. The submission needs `main.py` at its root.

A submission is Python code that runs with the permissions of the server's user, so only trusted
code belongs in the registry, and a submission that imports Kaggle's helpers needs the package
`kaggle_environments` installed beside cuju. A submission that fails ends only its own match, and
the page shows its error. A match that runs past 30 minutes ends, and the page cancels a waiting or
running match on request.

### Multi agent teams

A team trained with one network per player, such as GRF_MARL or TiZero, becomes a submission folder
whose `main.py` answers Kaggle's multi agent form. The observation `obs['players_raw']` then holds
one raw observation per controlled player, in agent order for dynamic assignment or selected
player order for fixed assignment, each with the
team as `left_team` and its own player as `active`, and `obs['controlled_players']` holds their
count. The agent returns one action per controlled player in the same order, each an index of the
action set, so a team of five that controls 4 players returns four actions and a team of eleven
that leaves its goalkeeper to the built in AI returns ten.

```python
import torch

weights = torch.load('weights.pt', map_location='cpu', weights_only=True)
networks = build_networks(weights)


def agent(obs):
    return [act(networks, raw) for raw in obs['players_raw']]
```

Here `build_networks` and `act` stand for the architecture and inference code of the training
project, which travel with `main.py`. Build the networks once when the file loads, put them in
evaluation mode and run inference without gradients. Reproduce the feature encoder, normalization,
available action masking and recurrent state handling used in training. A recurrent policy needs
separate memory for each controlled player and must clear it at the start of each episode.
A single controlled player may return one action, while several return a list of the same length
as `players_raw`. Keep `agent` as the last callable defined or imported in `main.py`.

The weights travel best as plain tensors, loaded with weights only enabled. A checkpoint saved
as a pickled module converts once inside a Docker container with the project code, then saves
its state dictionary as tensors. The submission builds the network and loads those tensors.
For one network per player, index the networks by the active player rather than the current
position of an observation in the list, as the following example does.

```python
import torch

weights = torch.load('weights.pt', map_location='cpu', weights_only=True)
networks = build_networks(weights)


def agent(obs):
    return [act(networks[raw['active']], raw) for raw in obs['players_raw']]
```

For GRF MARL training scenarios, choose fixed assignment, ten players for eleven a side or four
for five a side, indices one through ten or one through four, scenario goalkeeper control and
the action set used in training. For the Kaggle scenario with ten outfield players, choose the
builtin goalkeeper setting instead. A role selection such as CB, CB, CF keeps the first two
centre backs and the centre forward in that order. The checkpoint must use the same observation
features and action set as the submission.

### Customization

Each team has customization of its own, which the renderer alone draws, so a customized team plays
exactly
the matches it would play in its plain kit. The editor on the Teams page groups the look into nine
sections, identity, kit, roster, model and colors, fans, celebrations, sounds, sponsors and export,
with a preview of a player that idles, walks or keeps goal. The section menu and its previous
and next buttons move between sections. Changes wait for Save, except artwork and sound uploads
and removals, which save immediately. The identity holds the team name, its format and a badge of
up to three characters.

The look starts from the **player model**, which is GRF's own skinned player by default. A team may
instead take one of the characters that ship with the renderer, the skeletons and adventurers of Kay
Lousberg's KayKit packs and KayKit's mannequin, or a rigged model of its own, as described under
Custom models, and each player of the roster may take a model of its own in place of the team's.

Every model wears the team's kit. The editor reads the colors present in a model other than GRF's,
up to twelve by the share of its surface, and finds the part of the figure that each one covers from
the names of the model's meshes and materials, such as body, arm, leg, cape, hood, hat or helmet on
KayKit's characters, or shirt, pants and boots on an uploaded model. The heaviest color above the
legs takes the shirt color of the kit, the other colors of the body, capes, hoods and hats take the
trim, the upper legs take the shorts and the lower legs and boots the socks. Skin, bone, faces,
eyes, hair, metal and small accents keep the model's own colors, and a model whose names tell
nothing takes the shirt, shorts, trim and socks by surface share. Each mapped part keeps the
shading painted into the model. The goalkeeper wears the same parts in the goalkeeper's colors, and
when the right team plays in its away kit, its players wear the away kit's colors.
The viewer compares the shirts, patterns and shorts of both sides and chooses an away kit for
the right side when they clash. It tries the team's swapped kit first, then alternative sash
and plain kits, keeping the first suitable option or the most distinct available option.
Default goalkeeper kits are separated when needed, while an explicit goalkeeper kit is kept.

Under Model and colors, a swatch for each color shows the color drawn and a label, the part of the
kit that it follows, Custom for a color of the team's own, or Model color. Picking a color makes it
the team's own, Reset returns it to the kit, and Reset all to the kit clears every color of the
team's
own. The box Dress the model in the kit colors turns the kit mapping off and on for that model. A
swatch that is focused, picked or pointed at blinks its parts on the preview, and the preview shows
the model whose colors are listed, a player who wears it, so the Colors of list, which appears when
the roster uses several models, and the preview always agree. The team's own colors save with the
team and travel in its export.

On GRF's model, the team chooses the colors of its shirt, shorts and socks, a pattern for the shirt
among plain, stripes, hoops, halves, a sash and checkers, the colors of the goalkeeper's kit, a
badge of up to three characters and the color of the numbers. The roster lists five or eleven
players by the team's format, each with a name, a number from 0 to 99, a skin tone, a hairstyle and
a hair color, and each left alone keeps the look that GRF gives that position. The front and back of
the shirt are flat images, painted in an editor on the page with brushes, shapes, text and placed
images, and laid over the colors and the pattern. The pattern, the artwork, the badge and the
numbers
dress GRF's model, and the other models wear the kit's colors as described above.

A goal brings the scorer's celebration, picked from the moves that the team ticks, or from every
move when it ticks none, and each player of the roster may take a celebration of its own. The moves
include arms up, a knee slide, a jump and punch, dances, the airplane, a backflip, a salute, the
shush, heart hands, the robot, a bow and the worm, and on KayKit's characters the clips of KayKit's
animations, such as cheering, waving to the fans, jumps, taunts, push ups, an air punch, a forward
roll and sitting or lying on the grass, together with the clips of a custom model. Each move plays
in the editor before it is chosen.

A team may also bring sounds of its own, in Ogg, MP3 or WAV files of up to 2 MB, for eight moments,
a goal, a shot, a pass, a tackle, a save, any kick, the scorer's celebration and a chant that the
fans sing now and then in every arena. The page plays each one from where it happens, cut to four
seconds and brought to the level of the other sounds on the page.

A team also holds up to four sponsors, wide images made in the same editor from uploaded images and
drawing, which the advertising boards around the pitch show while the team plays. Boards alternate
between the sponsors of the two teams and fall back to GRF's boards. The fans of a team wear one to
four colors of its choosing, its kit's colors by default, and fill the stands behind its goal and
its half of the side stands.

### Custom models

A custom model is a binary glTF 2.0 file, `.glb`, with a skinned mesh, up to 30 MB, holding all its
data within the file. The page maps the joints of GRF's skeleton onto the bones of the model, by
name where they match and by the conventions of common rigs such as Mixamo's, and the mapping
remains editable. Only the pelvis, GRF's joint `body`, is required, so a simple rig whose arm is one
bone maps its shoulder and leaves the elbow out, and the bones without a joint follow their parents.

The renderer drives each mapped bone by the rotation of its joint relative to GRF's rest pose,
applied to the bone's own rest pose, and keeps the bone lengths of the model, so a model of other
proportions or another rest pose moves as GRF's player does without stretching. The pelvis follows
GRF's pelvis, scaled to the model's own height.

### Export

A team leaves the page as its submission folder, a zip archive with `main.py` and the submission's
files at its top, as Kaggle reads them, and a folder `cuju-team` that holds the team's format,
style, roster, images, sponsors and model. Kaggle ignores the extra folder, and a submission folder
that holds one brings the whole look back when it is added to a registry again, on the page or from
the command line, each file checked as an upload is. The built in AI exports its look alone.

## Page layout

The page holds eight pages, chosen from the bar along its top. The cuju logo at the left of the bar
opens Home, and tabs open Watch, Play, Recordings, Matches, Teams, Field and Help.
Settings and Fullscreen sit at the right end of the bar. On a narrow screen the tabs fold into
one menu of pages.

The address follows each page change and names what it shows, so a page or a moment reopens as a
link on the same machine. The address `#page=teams` opens the Teams page, `#match=` with the
identifier of a recording and `&t=` with a number of seconds opens that recording at that moment,
and `#live=` with the identifier of a live source opens the source. An address that names nothing
opens Home. A local file uses `#page=watch` and must be opened again after a reload.

**Home** opens with the cuju title, with a
tagline, a short description and the actions beneath it. Watch and Play a game open their pages,
alongside
Request a match, Teams and Field, and a line under the actions counts the live sources and the
recordings. The page continues in sections. The first holds Live now and Latest recordings
side by side, the sources that stream at the moment, each opening with one click, and the five
newest recordings, each with a control that plays it. While there are no sources and no
recordings, a guide of three steps takes their place, with the commands that run the page, stream a
running batch and make a recording. A montage of the viewer follows, muted at first and
started from its controls. Next come six short notes on what the viewer does and the main
keys of the Watch page in four groups, with a control for every shortcut. The page closes with an
about section and its credits beside the links to the repository, the documentation, this guide,
the issues, the changelog and PyPI, and a line with the license and the installed version. A server
that does not answer is shown as such, and a recording file still opens from the computer. On a
narrow screen the sections use one column.

**Watch** fills the page with the match. Its bar holds the Match and View menus and camera controls.
The Match menu opens a
recording file from the computer, takes an address, leads to the Recordings and Matches pages, lists
the goals of the open match, each of which jumps to six seconds before it, and lists the live
sources and the six newest recordings, with a count of the live sources on the menu's button. The
address field takes the identifier of a live source or of a recording, such as
`m-20261106-105640-3a3f`. The View menu chooses the camera, turns broadcast mode and fullscreen on
and off, sets the hour, the weather and the arena, and sets the quality and the theme. A recording
also opens when its file is dropped anywhere on the page. The view shows a spinner while it waits
for a live source to send its first step, and a bar of progress while a recording downloads.

The controls along the bottom of the view hold a timeline, the time and the length of the match,
buttons that restart, jump to the previous and next goal, skip 30 and 5 seconds either way, step
one frame either way, play backwards, pause and play, a speed of a quarter to four times real time,
and the mute button and the volume. The timeline seeks on a click or a drag, and marks each goal,
which a click on its mark reaches six seconds early. While the match plays, the controls and the
bar fade after a few seconds without input, together with the pointer, and return when the pointer
moves.

**Recordings** lists the library like a file explorer. A pane of folders holds every recording
under All recordings and the folders in a tree with their counts, where folders are made, renamed
and deleted, and recordings are dragged onto them. A folder is a label in the summary of each
recording, so deleting a folder moves its recordings to the folder above. A search field finds
recordings by title, teams, scenario and folder across every folder, filters narrow the list by
folder, scenario, team and date, and the list sorts by date, title, goals, teams, length or size.
The columns show the title, the teams, the score, the scenario, the steps, the date and the size,
and a click on a header sorts by it. Recordings are selected one at a time, several at once with
Ctrl or a range with Shift, and opened, renamed, moved or deleted, with the arrows, Enter, Delete
and F2 at hand. Deletion removes the file from the server for good, after a question.

**Matches** requests a match between two teams of the registry. Two cards pick the left team,
which plays at home, and the right team, which plays away. Each card shows the count, assignment,
fixed selection, goalkeeper and action set, and has an Ability control. The scenario list offers
full matches of the team format and academy scenarios, and a short text under it gives the chosen
scenario's players a side, length, difficulties, the events that end an episode and, for an
academy, its drill. A seed field takes a whole number or stays empty for a random one. Each
requested match checks the actual controllable players and roles on both sides before entering the
queue. Teams of different formats can meet in academy scenarios when both settings fit.
Below, the list of requested matches shows each match with its teams, state, scenario, seed and
progress, filtered by state, team, scenario and date, with controls to cancel a waiting or playing
match and watch a finished one. A message announces each match that is ready.

**Teams** lists the registry on its left, with a form that adds a team from a submission folder or
a single file, and opens the editor of the chosen team on its right, as described under
Customization.

**Field** configures the ground, as described under Environment. Its settings stay in the
browser, and a file of JSON exports and imports them, so they move between machines and browsers.
An import with any value out of place changes nothing and says which value, and a reset returns
the default afternoon in the stadium.

**Help** opens from its tab or with H or the question mark key. Escape returns to the previous page.
It lists the pages, getting started steps, Watch shortcuts, Play controls and keyboard navigation.
Its links open the full guide and repository.

## Watching a match

The keys work on the Watch page, outside the text fields, and the Help page, opened with H, lists
them all. The keys of the players follow the physical keys, so they work on every keyboard layout.

| Key | Action |
| --- | ------ |
| Space | play or pause |
| L and J | play forwards or backwards, faster on each further press |
| K | pause |
| Left and Right | one frame while paused, 5 seconds while playing, 30 seconds with Shift |
| Comma and period | the previous or next frame |
| Up and Down | a faster or slower speed |
| Home and End | the start, and the last frame or the present of a live source |
| N | the next goal, the previous one with Shift |
| R | a replay of the last goal, with more angles on Shift with R |
| Enter | skips the current sequence or the current part of a show |
| Esc | leaves the current sequence, show or player camera |
| V | the next camera in the cycle |
| B and P | follows the ball, or the player on the ball |
| 1 to 9 and 0 | the left team's outfield players, the right team's with Shift |
| G | the left team's goalkeeper, the right team's with Shift |
| Brackets | the previous or next player of the chase camera |
| W, A, S and D | fly the free camera along the pitch, faster with Shift |
| Q and E | lower and raise the free camera, C lowers it too |
| T | broadcast mode |
| F, M and H | fullscreen, mute and the help |

Menus accept the arrow keys to move between entries, Home and End to reach either end, and
Enter or Space to choose. Typing a letter finds the next entry that starts with it. Esc closes
the menu and returns focus to its button, and a click outside closes it too.

A drag of the mouse turns the free camera and orbits the cameras that follow the ball or its owner,
and the wheel sets the flying speed or the distance of those cameras. A click on the view skips the
current sequence. The setting for capturing the mouse lets a right click in the free camera capture
the pointer for looking around, and Esc releases it. The setting is off by default, and the page
releases the pointer whenever a dialog, a menu or a card opens, the window loses focus or the page
hides.

A live source plays like a digital video recorder, three seconds behind the newest step, and keeps
up to 30 minutes within an 80 MiB memory budget, about 15 minutes for eleven a side.
The view pauses, rewinds and scrubs while the stream continues. The controls show how far the view
lies behind, and a button returns to the present. A viewer who joins
late starts with the history that the source holds, up to 10 seconds, and a speed other than real
time leaves the present until the button returns to it.

### Cameras

The cameras follow the conventions of football broadcasts and video games. The key V cycles five
of them.

1. **Broadcast**, from a gantry high on the main stand, which pans with the ball, leads the play,
    widens when the play is fast and tightens near the goals.
2. **Tactical**, from high above, with the whole pitch in the window.
3. **Behind the goal**, behind whichever goal the ball is nearer.
4. **Follow player**, a chase view behind one player, looking towards the ball, whose player the
    brackets change.
5. **Free camera**, which flies from wherever the view stands.

The key B follows the ball, and P follows the player on the ball, the last owner for a moment after
the ball runs loose, and then the nearest player. Both keep the angle and the distance of the view
at the moment they start, so the drag and the wheel set them. The keys of the players put the view
on one player, over the shoulder at first. Pressing the same key again cycles the view through the
player's eyes, from the side and from the front, and the chosen view stays when the key of another
player is pressed. A name and number at the bottom of the picture say who the view follows.

Every change of camera glides, and every steered camera stays within the ground, above the grass,
under the roof line, off the seats and clear of the players' bodies. After 20 seconds without
input, the free camera returns to the broadcast view, which a setting turns off or lengthens.

## Playing a game

The **Play** page, opened from its tab or the Play a game action on Home, lets a person play a match
against the built in AI of GRF. A setup card picks the scenario, with the same short text under it
as on Matches, the side to play, left, right or both for two people, the ability of the built in AI
on each side, as on the Matches page, and whether the game goes to Recordings, which it does by
default. The person controls one player of the side, and
the built in AI plays the teammates and the other team. The side is an agent of the scenario, as a
submission that controls the active player is, so the engine hands control to the designated
player as play moves, the player nearest the ball among them, as GRF does for its own human
players. An arrow floats over the controlled player, above the ring of control.

The server plays the game in a process of its own at real time, ten steps a second, and streams
each frame to the page as soon as it exists, and the page shows the picture about a tenth of a
second behind the newest frame. The page sends the held keys and buttons on every change and twice
a second while it waits, and the server turns them into GRF's 19 default actions with GRF's own
controller logic, so a person's actions are actions like any agent's and the game is the match that
the engine plays for them. A change of direction comes first in a step, then the first button whose
state changed. The arrows and the stick stay held as GRF's sticky directions, sprint and dribble
stay pressed while held, and the passes, the shot and the slide act on the step they are pressed.
The controlled player switches automatically, by the engine's choice.

The high pass key and button send both high pass and slide inputs. GRF handles high pass first,
then slide on a later step if the input stays held, so the result also depends on possession and
the action that the player can perform. Direction changes take priority over either input.
Movement follows the screen directions of the broadcast view, including when playing on the
right side.

The keys and buttons follow the layout of Gameplay Football, the game that GRF grew from, and the
keys of the players follow the physical keys, so they work on every keyboard layout.

| Key | Action |
| --- | ------ |
| Arrow keys | move, held as a sticky direction |
| S | short pass |
| W | long pass |
| A | high pass, or a sliding tackle when defending |
| D | shot |
| E | sprint, while held |
| C | dribble, while held |
| Esc | the pause menu |
| H | the Help page |
| M and F | mute and fullscreen |

A controller plays with GRF's own Xbox layout, the layout of FIFA. Each action belongs to the
position of a button, so an Xbox, Nintendo or PlayStation controller plays the same with the same
hands, and the page names the buttons in the labels of the controller's style. A stick pushed past
half way along an axis counts as that direction, as GRF's own gamepad player reads it.

| Button, Xbox, Nintendo, PlayStation | Action |
| ------ | ------ |
| <kbd>Left stick</kbd> or <kbd>D-pad</kbd> | move, held as a sticky direction |
| the lower face button, <kbd>A</kbd>, <kbd>B</kbd>, cross | short pass |
| the right face button, <kbd>B</kbd>, <kbd>A</kbd>, circle | shot |
| the left face button, <kbd>X</kbd>, <kbd>Y</kbd>, square | high pass, or a slide when defending |
| the upper face button, <kbd>Y</kbd>, <kbd>X</kbd>, triangle | long pass |
| the right trigger, <kbd>RT</kbd>, <kbd>ZR</kbd>, <kbd>R2</kbd> | sprint, while held |
| the right bumper, <kbd>RB</kbd>, <kbd>R</kbd>, <kbd>R1</kbd> | dribble, while held |
| <kbd>Menu</kbd>, plus, Options, also called Start | the pause menu |
| <kbd>View</kbd>, minus, Create or Share, also called Back | the card of the controls |

A controller that the browser reports in the standard mapping of the Gamepad API plays at once.
Any other controller reports its own layout, and a notice offers to set it up. The setup, also
on the controls card and in the pause menu, first asks the style of the controller, Xbox,
Nintendo or PlayStation, preselected from the id the browser reports. It then asks for each
button in turn, the four face buttons, the bumpers, the triggers, the two system buttons, the left
stick and the D-pad, drawing each on a small sketch of the controller, and takes the next press,
whether the controller reports it as a button or as an axis. A trigger or button read as an axis
counts as pressed past half its travel. Skip keeps what a position had, Back returns a step, and
Cancel or Esc leaves the layout as it was. The page keeps the style and the layout in the
browser's storage for each controller id, so two people each set up their own controller, and
before the style is chosen a press on another controller moves the setup to that one. The controls
card also shows the id the browser reports and whether its mapping is standard or custom.

The page reads the keyboard and every controller at once, and whichever device was used last drives
the player. A notice announces each controller that connects or disconnects, and a card in a
corner shows the controls of the device in use at the start of the game and whenever the device
changes. For two people, the keyboard and the first controller play the left side and the second
controller the right, or with one controller, the keyboard plays the left and the controller the
right.

Esc or Start opens the pause menu, which pauses the game on the server and offers Resume, the
controls, the controller setup, Quit, Home and Quit to Home. The paused picture and the end of a
game fall silent. Home pauses the game and opens Home. Return to Play to Resume or Quit. Quit to
Home ends the game and opens Home. A page that hides pauses its game, a game that hears nothing from
its page for eight seconds waits for it and carries on by itself as soon as the page is heard again,
and after two minutes of silence the game ends. A game that ends, at full time or on Quit, lands in
Recordings when the setup asked for it, up to where it stopped, and the card at the end offers to
watch it on the Watch page with every feature of the viewer. Restart transitions, replays and
celebrations stay off during a game, so play never waits for the page, while the scoreboard, the
clock, the goal graphic and the sound remain. The server plays one game at a time, independently of
its queue of requested matches. Leaving Play for another page pauses the game, and returning
lets it resume. Losing window focus releases the keyboard inputs. The end card offers Play again
with the same setup, New game to return to setup and Home. The Play page calls a natural ending
Full time when the scenario is a full match, the final frame marks an episode end and both the
step and the clock reach the configured duration. Other natural endings show End of episode.
A saved game needs at least one played step. Its stream stays separate from the list of live
training sources.

A controller also works the setup card, the pause menu, the controls card, the end card and the
controller setup by the positions of its buttons. The D-pad or the left stick moves the focus and
repeats while held, the lower face button picks, steps a list or toggles a box, left and right
change a list or a slider, and the right face button goes back, closing a card or the setup or
resuming the game, while the steps of the setup that record a button take every press as theirs.

## Broadcast presentation

The page presents a match as a broadcast does, and covers the seams of the engine, the jumps of
players to their places at a restart, with the transitions of television. Settings turn each part
off.

**Lineups.** A recording opened at its start pauses for a pan across the ground under the team
sheets, with each team's goalkeeper, its lines from the back and its formation. Enter or a click
starts the match.

**Goals.** Playing forwards at up to two and a half times real time, the page holds play on the step
before a goal, so the kick off that follows stays unseen. The crowd roars, a graphic names the
scorer, the minute and the assist, and the camera covers the celebration in four shots, the live
view, a close tracking shot of the scorer, a slow orbit of the scorer and teammates, and a low
angle. The scorer runs towards the corner and celebrates with a move of the team's choosing, the
teammates run in to join or copy it, the scorer's goalkeeper claps, the beaten goalkeeper looks down
and the other team walks back. A replay follows under a wipe, from one angle that suits the
goal, high for a move of three passes or more and for an own goal, reversed for a shot from 18
meters or more and from inside the net otherwise, and in slow motion from just before the shot.
Shift with R queues further angles, and R replays the last goal at any time. A ball that reaches
the net during open play makes it bulge at the point of impact and spring back.

**Restarts.** Before each restart, the page holds the last step of play for a moment. The players
slow, turn towards the ball and walk towards their new places, and the ball runs on with gravity,
bounces and rolls, strikes the posts, bulges the net and stops at the boards, so a ball that leaves
the pitch keeps going in the picture. A wipe across the picture then hides the jump to the new
places, and the camera cuts to a view that suits the restart, behind the flag for a corner, beyond
the touch line for a throw in, behind the goal for a goal kick and behind the ball for a free kick
or a penalty. A pre roll of one and a half to three seconds follows, in which the players shift
their weight and glance about. The taker of a throw in picks the ball up and holds it overhead, and
the taker of another set piece bounces the ball, places it and stands behind it. Every movement of
the pre roll fades out before the whistle, so play resumes exactly on the restart's own step. The
first kick off of a match has the pre roll alone. A scenario with a second half, such as
`11_vs_11_kaggle`, stops play when the clock reaches 45 minutes and restarts with a kick off, which
brings a card for half time and the referee's double whistle. The timing follows the scenario's
second half step, so an interval configured at another step occurs at its own clock threshold.
Scenarios with continuous play carry on through 45 minutes.

**Decisions.** A foul, a card or an offside brings a replay of the decision. Play stops on the
whistle, the replay runs up to the contact, slowing at the end, and freezes on it, with a ring under
the offender, a ring under the victim and an arrow along the offender's path. A banner names the
decision and the offender. An offside draws the line of the second last defender and the line of
the attacker on the grass, with the space beyond the defence shaded. A yellow or red card adds a
scene in which the referee runs to the offender and raises the card, the offender clasps both hands
to the head, and a graphic names the card, the player and the minute. Enter or a click skips the
replay.

**The referee** runs the diagonal of the pitch through the match, keeping a distance from the ball
and the players, points at each restart, raises the whistle before it, and blows it.

**The end of a recording.** The last frame brings End of recording when the file stops during
play. An episode end flag brings End of episode when `full_match` is `false`, as in an academy
exercise, and Full time otherwise. Older files without timing therefore bring Full time when the
last frame has an episode end flag, and End of recording otherwise.

**Full time.** At the final whistle, play winds down instead of stopping. Each player slows from a
run
to a stand over two to four seconds, the ball rolls to rest, the referee blows and lowers the
whistle, and a card for full time names the winner or the draw. A recording then plays the show at
the end of the match, in five parts joined by wipes.

1. **Reactions**, in which the winners jump, dance, huddle and run to their fans, the losers slump,
    sit or lie on the grass and get up, the players of a draw shake hands, and everyone walks
    towards the stand applauding, covered by a wide shot, the winners' hero, the beaten goalkeeper
    and the stands.
2. **Facts**, a card of the score, the scorers and the share of each team in possession, shots,
    shots on target, passes, pass accuracy, tackles, saves and cards.
3. **Reel**, every goal again with its replay.
4. **Spotlight** on the player of the match, a slow loop of the player's best moment beside the
    player dancing beside an MVP medal on a ribbon in the team's two kit colors. An entrance moves
    the camera closer, flashes once and plays a fanfare, then reveals the player name and shirt
    number. Reduced motion settings keep the medal and name in place and omit the flash, the
    camera movement and the shine on the medal.
5. **Podium**, with a fanfare, the best three of the winners on the steps, the rest of the team
    celebrating behind them, confetti in the team's colors and the card of the player of the match,
    or for a draw, the best four of each team on two low stages.

The player of the match is the winners' best, or the best of both teams after a draw, by a score of
five points per goal, three per assist, two per save, one per tackle and a small share for passes
and touches, less five per own goal. The facts come from the actions and touches that every frame
holds. A shot is an intentional touch during the GRF shot animation, or an intentional header
from within 30 metres of the opposing goal that sends the ball towards it along the pitch at more
than three metres a second. A save is a goalkeeper touch that stops a shot on target. Enter or a
click moves to the next part, Esc ends the show, and the page then fades to Home. A seek during the
reactions ends the show where it stands.

The statistics screen opens on demand. The Match statistics button sits on the podium panel at
the end of the show and on the card at the end of a recording, and the Match menu offers it for
the open recording under This match. It opens over whatever is on screen, the show waits beneath
it, and Close or Esc returns to the same place. It reads from top to bottom. The
result comes first, then the awards, the player of the match and the rarest achievements and team
awards, then the comparison of the two teams. Team tabs list every player of the team with shirt
number, name, inferred role, grade medal, minutes, goals, assists, shots, passes, tackles and
saves, and the More columns button adds shots on target, key passes, pass and shot accuracy,
touches, clean sheets, own goals and cards. Column buttons sort either way. Opening a player shows
rating terms and achievement reasons. Last come three charts over twelve periods of the match,
possession, the running totals of shots and goals, and territory with possession, each with its
caption, time axis and key, and the guide to the grades opens below them. The whole screen scrolls
as one card. Focus starts on the outer dialog, where Esc closes the screen. Tab reaches the
controls, arrow keys switch team tabs, and Enter activates a focused control. Within the content,
Enter away from a control closes the screen. For a live source the screen opens from the podium
panel of the show that Match summary offers. It takes a stable copy of the latest available
episode with an explicit end flag, or the current episode otherwise. Sources with automatic
resets supply start flags, so their summary uses the current episode.

Player grades describe contributions in the available frames. Minutes use increasing game clock
time while the player is on the pitch, excluding cuts. Goalkeeper is the header goalkeeper flag.
Other roles use the initial position along the attack direction. Positions below minus fifteen
metres are defenders, above ten metres are forwards, and the rest are midfielders. These roles
are estimates rather than formation assignments.

The rating starts at five and is limited to zero through ten. Event contribution adds one point
four per goal, one per assist, zero point two five per shot on target, zero point three five per
key pass, and zero point six per tackle or recovery for a defender or goalkeeper, otherwise zero
point three five. Goalkeepers also receive zero point six five per save. This sum is multiplied
by thirty divided by the greater of ten and minutes played, capped at three. Pass accuracy adds
one point five times its difference from zero point six five after at least three attempts.
A clean sheet adds one for a defender or goalkeeper and zero point three for other roles.
Own goals subtract two each, yellow cards zero point four each and red cards one point five each.
A second yellow followed by dismissal can therefore incur both card penalties.

Grades use these lower rating thresholds. F starts at zero, D minus at two, D at three, D plus at
four, C minus at four point five, C at five, C plus at five point five, B minus at six, B at six
point five, B plus at seven, A minus at seven point five, A at eight, A plus at eight point five
and S at nine point five. S additionally requires at least one minute and sixty percent of the
available duration, event contribution of at least four point five and zero penalties. Otherwise
the highest grade is A plus. A grade requires the greater of half a minute and twenty percent of
the duration capped at ten minutes. Shorter appearances show a dash. The grade medal identifies
the grade and exposes its numeric rating on demand.

Achievements include all tied leaders when a rule names a leader. Each achievement gives its
name, symbol and numerical reason, with reasons available when opening a player.

Hat trick awards three or more goals, and Brace awards exactly two. Playmaker awards the most
assists with at least one. Creator awards the most key passes with at least two for players outside
the Playmaker award. A key pass is a completed pass to the shooter within ten seconds before
a shot, with continuous
open play, possession staying clear of the opposition and zero intentional opposition touches
between pass and shot. Each shot credits at most the latest qualifying pass.
Wall awards the most saves with at least one. Engine awards the most touches with at least five.
Ball winner awards the most tackles and recoveries with at least two. These recoveries use the
same successful steals as the facts screen, rather than a separate interception estimate.
Sharpshooter requires at least three shots with at least seventy five percent on target.
Own goal identifies every player with an own goal. Sent off identifies every player receiving
a red card in the available frames.

Player Clean sheet requires an available duration of at least one minute, participation for at
least eighty percent of it and zero opposition goals while on the pitch. Team Clean sheet
requires at least one minute, an initial opposition score of zero and a final opposition score
of zero. Comeback requires a team to win after trailing at a visible goal. Late winner requires
a win with its decisive goal in the final tenth of an episode lasting at least one minute.
The decisive goal is the goal that puts the winning total above the final opposition total.
These rules use episode duration for academy exercises and smaller games.

Three charts divide the available frames into twelve equal periods. Possession is a line of the
first team's share of owned frames around an even split at 50 percent. Above the middle line the
line and its shading take the first team's kit color, below it the second team's, so the color
changes where the line crosses the middle. Every chart uses the two kit colors, and a second team
whose shirt looks close to the first team's takes its trim color. Cumulative shots use dashed
lines and goals use solid lines. Pressure averages half the ball position along the pitch and half
the possession direction across owned frames. Above the centre favours the first team. Pressure
describes territory and possession rather than expected goals. A shared event strip marks goals
and card changes for both teams along the time axis. The axis endpoints show the available game
clock range. Available frames supply the statistics for partial sources as well as completed
episodes.

**Live sources** skip the lineups, show the card for full time alone, and offer the show at the end
through a button rather than playing it. A decision replay that falls far behind the present
returns to it under a wipe.

**Broadcast mode**, turned on with T, from the View menu or in the settings, shows the picture and
its graphics alone, as on television, with the bars, the controls, the messages and the pointer
hidden. Moving the pointer to the bottom edge of the window, or pressing Esc, brings the controls
back for a few seconds, and they stay while the pointer rests on them. T, or Esc pressed again while
the controls show, leaves broadcast mode, which works in a window and in fullscreen and stays on
across reloads.

## Environment

The ground follows the Field page and the View menu, which share their settings.

**Time of day.** A slider sets the hour, from midnight to midnight in quarter hours, and four
presets choose dawn, day, dusk and night. The sun follows a summer path over the ground, rising
behind one end, crossing behind the far stand and setting behind the other end, and the moon and
the stars come out at night. Four floodlight towers light the pitch as the sun sets.

**Weather.** The weather is clear, overcast, rain, heavy rain, snow or fog, with a strength and a
wind, which drives the clouds and the rain. Rain wets the grass over a few seconds and dries
slowly, snow settles on the pitch and leaves the lines clear, and the floodlights also come up by
day in bad weather.

**Arenas.** Five arenas surround the same pitch, lines and goals, and each has fans along both touch
lines.

| Arena | Ground | Crowd |
| ----- | ------ | ----- |
| Classic stadium | two tiers of fans under a roof, with floodlights on four towers | the full bowl |
| Beach | sand, palms and umbrellas, with the sea behind the far touch line | three rows on the sand, in deck chairs and on their feet, under umbrellas |
| City rooftop | a pitch on top of a tower, with the skyline all around, lit at night | a small terraced stand on each touch line |
| Space station | a deck in orbit above the turning Earth, under a sky of stars | a gallery on each touch line behind a pane of glass, and a glass helmet on every player |
| Neon arcade | a grid to the horizon, with pylons, mountains and a sun | figures of light on glowing steps |

The crowds of the other arenas are the same painted people as the stadium's, so the fans of each
team wear its colors on its half of the ground and stand and jump after a goal. Each is a few
hundred people, so it costs little at the low quality. On the space station each player wears a
bubble of glass sized to the head and the hat of the model, which is clear in the middle and bright
at the rim, and a camera at a player's eyes leaves that player's bubble out.

**Crowd and boards.** The stands fill with the fans of each team by default, in the team's colors,
or with a painted crowd in colors of the viewer's choosing, or with GRF's own crowd, at a density of
choice. The painted crowd sways with the play and stands and jumps after a goal. The advertising
boards show the sponsors of the two teams in turn, the boards that ship with the renderer, with the
cuju logo and the team names, or up to eight boards of the viewer's own, uploaded or drawn.

**Field page.** The page groups its settings into presets, the arena, the grass, the time of day,
the weather, the boards and the crowd. The ten presets range from a summer afternoon to a rainy
night, snow, the beach at sunset and the space station. The grass takes colors for the grass, the
lines and the nets, and a mowing pattern of stripes, checkers or circles with the width of its
bands. The view on the page is a free camera, flown with the drag and W, A, S and D, and touching a
control flies the camera to the part it changes, the grass, the lines, the nets, the boards, the
crowd, the sky or the whole ground. The camera flies once to each part, so a view set by hand stays
for further changes to the same part.

## Sound

The page mixes its sound in the browser, placed in the ground and heard from the camera, and each
arena has a soundscape of its own that rises with the play near a goal and answers goals, near
misses and the final whistle in its own way.

| Arena | Soundscape | Crowd |
| ----- | ---------- | ----- |
| Classic stadium | the crowd itself, with a fanfare at full time | the full crowd under the roof of the stands |
| Beach | uneven waves, breeze and gulls | a smaller crowd in the open air |
| City rooftop | traffic, horns, sirens and wind, quieter at night | a smaller crowd whose voices come back off the towers |
| Space station | hum, airflow, beeps, radio chirps and a chime for goals | a smaller crowd in an enclosed gallery |
| Neon arcade | the buzz of the tubes, arcade blips and goal stings | a smaller crowd on the open grid |

The referee's whistle marks restarts, decisions, half time and full time in every arena, and a near
miss, a save or a shot off the woodwork draws a reaction. Rain and wind follow the weather, with
rain drumming on the stadium's roof and hissing on the sand, and the space station has no weather.
Real recordings in the public domain or under CC0 carry the crowd, its roar for a goal, its groan at
a near miss and its applause, which every arena plays through the band, the echo and the walls of
its place and quieter than the stadium, the referee's whistle, the waves and gulls of the beach,
the traffic, siren and horn of the city, the hum, beeps and radio of the space station, the arcade's
goal sting, the rain and the wind. The rest is synthesized in the browser, and the file
`sounds.json` beside the sounds names the author, source and license of each file. A change of arena
or weather fades from one soundscape to the next over about a second. The sounds of the play follow
forward play at up to two and a half times real time, and pausing silences the page. Each arena's
own sounds play under its crowd. The arena files load with the sound system, and unavailable files
fall back to the synthesized sounds where available.

The page starts muted, since browsers start sound only after a click or a key, and M or the speaker
button turns the sound on, at half volume by default.

## Settings

The settings panel holds the parts of the presentation and the behavior of the page. Each setting
stays in the browser. The idle return interval accepts 5 to 300 seconds.

| Setting | Default | Effect |
| ------- | ------- | ------ |
| Return to broadcast when idle | on, after 20 seconds | return from the free camera |
| Ball trail | on | a line behind the ball |
| Broadcast mode | off | the picture and its graphics alone |
| Lineups before kick-off | on | the team sheets at the start of a recording |
| Celebrations after goals | on | the celebration and its cameras |
| Replays after goals | on | a replay of each goal |
| Decision replays | on | replays of fouls, cards and offsides |
| Restart transitions | on | the wind-down, wipe and pre-roll of each restart |
| Slow motion in replays | half speed | off, half speed or quarter speed |
| Post-match show | on | the show at the end of a recording |
| GRF classic pitch | off | GRF's own pitch texture in place of the drawn pitch |
| Capture the mouse | off | a right click in the free camera captures the pointer |
| Use the kits saved in the match | off | saved kits instead of current team settings |
| Theme | the system's | dark or light |

Three quality levels in the View menu set the resolution of the shadows, the pixel ratio, the
glow of bright lights, the particles of rain and snow, the detail of the crowd and the shadows of
the floodlights, and the page picks one from the device.

## Keyboard

Tab and Shift Tab move between controls on every page. Enter or Space activates a control.
The skip link moves to the main content. Menus and lists use the arrow keys, Home and End.
Esc closes a menu or panel and returns focus to its opener. Dialogs keep Tab inside them.
Sliders use the arrow keys. Color fields accept a hex value beside the color picker.

Home cards open their pages. Watch keeps its playback and camera keys while the picture has
focus. Focused controls use their own keys, and text fields keep typing within the field.
Recordings supports folder arrows and row arrows, Enter opens a row, and Space selects it.
Matches and Play setup use Tab through their forms. Teams uses arrows through its list and
section menus, with Tab through editor controls. Field uses Tab through settings and the
preview, where the camera keys move the view and arrows turn it. Play uses Esc for its pause menu,
with Home available there and on the end card. Home pauses an active game. Quit to Home ends
the game and saves it when Save to Recordings is enabled. A natural ending shows Full time
for a complete match
and End of episode otherwise.

The picture uses plus and minus to zoom, and Watch uses Alt with arrows to turn the camera.

The painter canvas uses arrow keys to move its cursor and Space to paint. Page Up and Page Down
select objects. Arrow keys move a selected object, Shift increases the distance, the plus and
minus keys resize it, and the bracket keys rotate it. Enter edits selected text.
The post match show uses Enter to advance and Esc to finish while the picture has focus.

## Recording format

Optional header fields carry scenario timing. The `duration` is the configured number of steps.
The `second_half` field is the configured step of the second half, or null when no half time occurs.
The clock threshold is `second_half` times 1800 milliseconds. The `full_match` field identifies a
scenario with a duration of at least 3000 steps and no early episode end conditions. Missing timing
leaves the recording playable, with Full time for an episode end flag and End of recording
otherwise.

Bits 0 and 1 of the frame flags mark episode end and start respectively, with values 1 and 2.
The offside flag has value 4. Player references in the frame use indices in the header's player
list, including the scorer, foul offender, victim and offside player.

The optional `engine_scorer` header flag enables bits 8 through 12 of the frame flags. They hold the
engine scorer index in the header's player list plus one, with zero meaning the scorer is
unavailable. Older pages ignore these bits.
The optional `offside_called` header flag enables bit 2 of the frame flags. When set, the offside
integer holds the last toucher penalized at a buffered free kick without a foul. Otherwise it
retains the first offside candidate. Older recordings retain their candidate based interpretation.

A recording is a gzip compressed file with the extension `.cjr`. After decompression it holds the
seven bytes `CUJREC1` followed by a newline, the length of a header as an unsigned 32 bit integer,
the
header as UTF 8 JSON padded with spaces to a multiple of four bytes, and the frames back to back,
each of the same size and aligned to four bytes. Every number is little endian.

The header records the version, 1, the title, the scenario, the steps per second, 10, the names of
the fourteen joints and the parent of each, one entry per player with its team, its index within the
team, whether it is the goalkeeper and whether an agent controls it, the name and colors of each
team, the size of a frame and the number of frames, which a live stream gives as null.

A frame opens with eight signed 32 bit integers, seven floats and four more integers, as the table
lists them.

| Field | Type | Content |
| ----- | ---- | ------- |
| step | int32 | the step of the episode, 0 at the reset |
| game mode | int32 | GRF's game mode, 0 in normal play and 1 to 6 for the restarts |
| score | int32 twice | the goals of the left team, then of the right team |
| ball owner team | int32 | the team that owns the ball, as GRF's raw observation gives it, or -1 |
| ball owner | int32 | the owner's index in the header's list of players, or -1 |
| clock | int32 | the match time in milliseconds, 90 minutes over the 3000 steps of a full match |
| flags | int32 | episode boundaries, called offside and scorer, as described above |
| ball | float32 seven times | the position of the ball, then its rotation as a quaternion |
| foul offender | int32 | the player who committed the referee's current foul, or -1 |
| foul victim | int32 | the player fouled, or -1 |
| foul kind | int32 | 0 for none, 1 for a foul, 2 for a yellow card and 3 for a red card |
| offside | int32 | the called player when flag 4 is set, otherwise the first candidate, or -1 |

The joints of every player follow, seven signed 16 bit integers per joint, fourteen joints per
player in the order of the header. The first three hold the world position in steps of 2 mm, which
spans 65 m either way of the center spot, and the last four the rotation as a quaternion, x, y, z
then w, in steps of one part in 32767. A full match of eleven a side occupies 13 MB before
compression and 12 MB in the file, half the size of the same match stored as 32 bit floats. Every
position is in the frame of the pitch, with the center spot at the origin, the left team defending
negative x and the z axis pointing up.

Three runs of one byte per player close the frame, each padded with zeros to a multiple of four
bytes. The first holds bits, 1 for a player on the pitch, 2 for a yellow card, 4 for a red card
and 8 for a player that an agent controls on the step, so a red card sets 2 and 4 together.
Recordings of an older cuju carry no 8, and the page then marks the agents' players of the header.
The second holds the kind of the player's current animation
in GRF's numbering, 8 for a shot, 4 to 6 for the passes, 9 and 10 for a deflection and a catch and
13 for a slide among them. The third holds 0 for a player who did not touch the ball during the
step, and otherwise 1 plus GRF's kind of the touch, 1 for a kick, 2 for another intended touch and
3 for an accidental one.

A live stream sends the same bytes uncompressed.

## Folder and server

The renderer's folder holds three subfolders, `matches` for the recordings, `teams` for the
submissions and `live` for the announcements of running sources. The file `samples.json` names the
sample matches that the library has received. Requested matches wait in the
server's memory and play one at a time. Restarting the server clears the queue and its request
history, while saved recordings and teams stay on disk. The variable `CUJU_RENDER_HOME`, or the
option `--home`, moves the
whole folder, so a project can keep its own matches beside its code.

The server listens on the loopback interface alone, since its API deletes files and plays the code
of submissions, and it checks the loopback host on every request. Requests that change data also
check any
Origin header against the server's own address, including the scheme and port. A server on another
machine is reached through a tunnel, such as
the port forwarding of SSH, which makes it a loopback server of the local machine.

```sh
ssh -L 8780:127.0.0.1:8780 training-server
```

## See also

- [States](states.md), the state format that the renderer reads, and the module `cuju.state`
- [Recipes](recipes.md), the Kaggle competition and the training loops that a recorder joins
- [API reference](api.md), the reference of `cuju.render` and `cuju.state`
