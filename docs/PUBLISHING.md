# Publication status

**Blissful Ignorance is public on GitHub:**
[Maverick0351a/blissful-ignorance](https://github.com/Maverick0351a/blissful-ignorance).
The first source snapshot was published October 6, 2026 under Apache-2.0.
The project was previously called Godhood Trials; historical experiment names,
`GT-` milestone identifiers and saved-world paths remain compatible.

## Included in the first publication

The 213-file snapshot includes the playable browser world and characters,
Python simulation and independent learners, tests, Windows launchers,
architecture notes, [roadmap](MILESTONES.md), [progress report](PROGRESS.md),
and [portable interactive showcase](progress/index.html). The main app opens
on the actual game map; the research viewer is a secondary page. The showcase
also works offline without a running game or model.

Three preservation defects are fixed. Seven findings remain open in the
[implementation audit](AUDIT-2026-10-06.md). The nearby-food practice pilot
passed its development screen; navigation, sustained survival, meaningful
communication and general intelligence remain unconfirmed. Working mechanics,
measured activity and demonstrated learning are reported separately.

The clean publication copy passed **231 tests in 92.466 seconds**, including
PettingZoo's compatibility checks. Final publication edits changed documentation
and the notice only. The local review records every selected file's size and
SHA-256, checks relative links and all four JavaScript files, and compares the
public scores with the earlier browser-validation receipts. The filename and
content review excluded private paths and common credential patterns. These
checks are not a comprehensive security audit. See [validation](PROGRESS-VALIDATION.md).

## What stays local

Raw `runs/`, full experimental traces, model weights, live saves, private
resident histories, operational watch logs, machine settings, private planning
notes and existing development Git history are excluded. Publication uses a
separate source copy with new history; it does not push the development
checkout's old history. Its local `review.json` records the selected files and
documentation adjustments. Links to omitted evidence are labeled as local
evidence rather than presented as downloadable files.

The selected `docs/progress/` assets contain aggregate scores, source hashes
and one recorded experimental comparison. They are not the complete training
evidence package, and the exact full audit cannot be repeated from this
snapshot alone. The source can start a fresh PPO world with an existing Python
and PyTorch installation. Optional Laya integration needs separately installed
local model assets; no pretrained weights are bundled.

Publishing the repository does not expose or restart the continuing local
world, change its roster, import policies, or deploy a public game server.

Naming check: an [existing itch.io game](https://hoork.itch.io/blissful-ignorance)
uses the same display title. The chosen working title remains Blissful
Ignorance; no name-availability or trademark-clearance claim is made.

## Free recorded showcase

The [Hugging Face Space](https://huggingface.co/spaces/Maverick03511/blissful-ignorance)
was published October 6, 2026 with the free static SDK. It puts a screenshot
of the playable test world first, followed by interactive per-learner results
and recorded experiment playback. It does not run or train a live population.
The 11 selected files come from the already-public source snapshot; no models,
packages, paid compute, private saves or raw training traces are required.
See [publication checks and reproduction](HUGGING-FACE.md).

GitHub Pages is not deployed. GitHub's file viewer displays HTML source; use
the Space or open the downloaded research page locally to interact with it.

## Live hosting and later checkpoints

A static Space cannot execute the Python backend. A future Docker Space needs
reviewed proxy host/origin configuration, iframe-compatible headers, a
container image and Dockerfile, resource limits and an actual smoke test.

The existing `--public-demo` mode is read-only and currently starts paused
without a permitted way to resume; this is open finding AUD-09. Resolve it
before advertising a running shared demo. Per-visitor interactive worlds and
online training also need explicit session and resource designs.

No Docker image was downloaded or built for either publication. Verify current
hosting persistence and limits before relying on it for a continuing
population. Development training remains local; paid hosting is a separate
decision.

Evaluated policy checkpoints may later be published separately with training
configuration, task definitions, licenses and held-out results. Their model
cards should identify task-specific capabilities and limitations.

References checked October 6, 2026:

- [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Static HTML Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
- [Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
- [Space configuration](https://huggingface.co/docs/hub/spaces-config-reference)
