# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Report a vulnerability privately through GitHub's private vulnerability reporting. Open the
[Security tab](https://github.com/AndreKoraleski/cuju/security) of the repository, then choose
Report a vulnerability. Please keep vulnerabilities out of public issues and discussions.

A useful report names the version, the platform and the backend, and gives the steps or the file
that reproduce the problem.

## What to Expect

- An acknowledgment of the report within seven days.
- An assessment, and a plan for a fix when the report is confirmed, within thirty days.
- A fixed release and a published advisory, with credit to the reporter unless anonymity is asked.

## Trusted Input

Several of cuju's inputs are code or pickles. Load only the ones that come from trusted sources.

- **Scenario files are Python.** A batch or `cuju.scenarios.build_bundle` that takes a scenario file
    runs it with the permissions of the caller.
- **States are pickles.** `set_state` reads a state through a restricted unpickler that accepts only
    GRF's episode class and plain values, and for a stacked state the numpy arrays of its frames.
    It still parses bytes from outside.
- **Dumps are pickles too.** `cuju.dumps.read_dump` reads them through a restricted unpickler that
    accepts only the classes of GRF's dumps.
- **Action logs name their scenarios.** `cuju.replay` creates the batch that a log describes, so a
    log that names a scenario file runs that file.
- **Team submissions are Python.** The renderer's registry plays Kaggle submissions, so a match
    requested with `cuju render play` or on the renderer's page runs the submission's code with the
    permissions of the user who started the server. Add only trusted submissions to a registry.
- **Team folders and models come from outside.** A submission folder, an archive, a team exported
    from another registry or a custom player model is checked before it is stored. Archive members
    must land inside the team's folder, archives are capped in size and file count, images are
    checked by their bytes, and a model must be a self-contained binary glTF file. The page also
    refuses links to other places inside models and drawings.

## The Renderer's Server

`cuju render serve` listens on the loopback interface alone and serves an API that deletes files and
plays submitted code. Every request must name the server by a loopback address in its Host header.
GET requests may carry any origin. Other requests must come from the same origin, by scheme, host
and port, or carry no Origin header. Its page carries a Content Security Policy that keeps scripts,
styles, images and connections to the same origin. A server on another machine is reached through
a tunnel, such as the port forwarding of SSH, and never by listening on another interface.

A weakness in these protections, such as a way for another site to change data through the API or
for an upload to write outside the team folder, is a vulnerability and belongs in a private report.
