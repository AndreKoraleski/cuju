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
    GRF's episode class and plain values, and for a stacked state the numpy arrays of its frames. It still parses bytes from outside.
- **Dumps are pickles too.** `cuju.dumps.read_dump` reads them through a restricted unpickler that
    accepts only the classes of GRF's dumps.
- **Action logs name their scenarios.** `cuju.replay` creates the batch that a log describes, so a
    log that names a scenario file runs that file.
