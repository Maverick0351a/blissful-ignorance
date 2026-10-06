# Free recorded showcase

[Open Blissful Ignorance on Hugging Face](https://huggingface.co/spaces/Maverick03511/blissful-ignorance).

Published October 6, 2026 as a public **static Space**, under Apache-2.0.
The page leads with a screenshot of the playable map and characters, followed
by interactive results and recorded feeding-experiment playback. The screenshot
comes from a separate UI test world; the replay uses a simplified renderer of
archived experimental states. The page does not run or train a live population.

The source game still runs locally. No paid plan, Docker image, downloaded
model, dependency install, inference API or runtime secret was needed to host
this showcase. Hugging Face's own site infrastructure is separate from the
static app, whose scripts use only bundled assets and make no network requests.

## What was published

The build reads selected assets from public GitHub commit
[`baef9bd`](https://github.com/Maverick0351a/blissful-ignorance/commit/baef9bd01ab2248766156892b2d0e6380ee0eabf).
It adapts navigation and relative document links for hosting, adds the map
introduction, and supplies static Space metadata. Recorded scores and replay
states are unchanged. Historical evidence links remain pinned to that source
commit even as later development continues.

The reviewed package contains **11 files / 439,728 bytes**: the page, style,
three JavaScript files, results JSON, map image, icon, README, license and
notice. Hugging Face also retains its generated `.gitattributes` file. Private
histories, raw traces, checkpoint weights, watch logs and credentials are absent.

The published Space revision is
[`5efc3f6`](https://huggingface.co/spaces/Maverick03511/blissful-ignorance/commit/5efc3f6b2900c69ce6f8b0470235ed2008c79611).
Its final change reserves space for Hugging Face's floating header so that it
does not cover the app navigation.

## Validation and limits

- Verified the 11-file allowlist, file hashes, 28 asset/document links, static
  metadata and all three JavaScript files' syntax.
- Matched all 14 local chart views against the aggregate results JSON.
- Inspected nine replay frames across all three practice conditions, and
  checked play and restart locally.
- Tested the published iframe: task and learner filtering, replay scrubbing,
  playback, pause and restart. Its desktop document width matched its viewport;
  images loaded and the browser reported no warnings or errors.
- The narrow-width override did not affect the existing hosted tab, so this
  publication check does not claim a separate hosted mobile validation.
- Confirmed the continuing local world stayed private and manually paused at
  tick 85,447 with the same eight PPO learners and distinct Laya controller.

The earlier source snapshot passed 231 software tests, including PettingZoo.
This publication changed packaging, presentation and documentation only; it did
not rerun training or repeat the entire simulator test suite. The recorded
217-test figure belongs to the original experiment. Neither figure proves a
new learned ability. **GT-01 remains open**, as do seven implementation audit
findings, including the separate read-only live-demo pause issue (AUD-09).

## Rebuild the package

From a checkout containing the pinned source commit and this builder:

```sh
python scripts/Build-HF-Showcase.py --source . --commit baef9bd01ab2248766156892b2d0e6380ee0eabf local/hf-showcase-review
```

The output directory must be new and inside ignored `local/`. The stdlib-only
builder writes `source/` plus a private `review.json` with sizes and SHA-256
hashes. It does not publish, read live saves, install software or run models.
Inspect the complete package before any later upload. Only `source/` belongs
in the Space; review receipts and operational files stay local. Previous Space
revisions remain available in its Git history for comparison or restoration.

Hosting references checked October 6, 2026:
[Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).
