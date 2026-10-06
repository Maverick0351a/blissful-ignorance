# Portable progress snapshot

The running game's front page is now the fully interactive valley. Start the
local server and open its root URL to play; choose **Research** for this report
and **The world** to return. Opening this report from disk still gives offline
charts and recorded playback, not a running Python world.
`world-preview.jpg` is an actual browser capture of a separate UI test world
with eight independent PPO residents; it contains no live save or weights.

[Actual browser validation](../PROGRESS-VALIDATION.md) covers chart filters,
recorded playback and layout checks for this presentation.

Open `index.html` in a modern browser. The page works from disk without Python,
model weights, external libraries, web fonts, an API key or a running game.
Markdown links open their source files when viewed from disk; on GitHub those
documents render normally. GitHub's repository view does not execute HTML;
download the checkout and open the page locally to use its controls.

The snapshot contains the October 6, 2026 retained-practice-amount results.
`results.json` is the public numerical record; `results.js` contains the same
record for offline browser use. `replay.js` contains the already recorded first
adjacent-food evaluation map of the first historical training group, with
starting, +16 and +64 conditions. The choice is by index, not by outcome.
The diagram shows two isolated experimental residents, not the main population.

The exporter is `python scripts/Build-Progress.py`. It requires the existing
local `runs/practice-amount-20261006/` evidence and verifies the recorded hashes
of the analysis, protocol, audit and source replay before writing these files.
It does not run a new experiment, load weights or contact the live server.

The exported fields are an explicit selection of aggregate metrics, protocol
settings, hashes, room geometry and recorded body/action state. Live saves,
private learner journals, machine paths, service records and model checkpoints
are not included. Raw experimental traces and checkpoints remain local.

This is a portable result summary and illustrative replay, not the full
training evidence package. The original audit used the native simulator and
brain plus a separately implemented replay and outcome recount. Readers can
inspect the public figures and method, but cannot repeat that exact training
audit from this snapshot alone. See [the experiment](../PRACTICE-AMOUNT.md)
for the full scope and limitations.
