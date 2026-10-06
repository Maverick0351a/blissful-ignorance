# Progress preview validation

October 6, 2026. Presentation work for GT-00; no new learning milestone.

## First public GitHub source snapshot

The refreshed clean publication copy passed **231 tests in 92.466 seconds**,
including PettingZoo's parallel API test and the completed preservation repairs.
Final documentation and notice updates leave the tested executable files
unchanged. All 213 selected files have recorded hashes. Local documentation and
asset links resolve, and all four JavaScript files pass syntax checks.
The published numerical snapshot still matches the fourteen previously
validated chart views and nine recorded replay-frame receipts; those receipts
are earlier browser observations, not a new browser run.

The review found no private home paths or common credential patterns in the
selected files. A Python matrix-multiplication expression was reviewed as a
false email match. Raw runs, weights, live saves, watch logs, local settings and
existing private Git history are excluded. These checks reduce accidental
disclosure; they are not a comprehensive security certification.

The repository is [public on GitHub](https://github.com/Maverick0351a/blissful-ignorance).
This publishes the source and portable progress report. GitHub Pages and Hugging
Face hosting are not deployed. Seven implementation findings remain in the
[audit](AUDIT-2026-10-06.md); publication does not advance a behavioral gate.

## Map-first follow-up

- The root page now contains the fully functioning game renderer and controls.
  The research viewer is a secondary same-origin page. Its files are explicitly
  allowlisted; private saves and watch logs are not served.
- **11 focused server tests passed**, including the added check that reading
  research assets neither exposes unlisted documents nor wakes a paused world.
  JavaScript syntax and Git whitespace checks passed.
- Browser checks used a separate real recurrent-PPO population: selecting Moss
  changed the followed character; individual sight hid the minimap; Everyone
  restored overview; expanding/restoring the map hid/restored the inspector.
  Keyboard movement and one step produced one step; touch movement produced
  the second. The construction palette exposed all six existing choices.
  Save at tick 1, a brief 5x run to tick 8, then load restored tick 1 and the
  paused state. A touch move and step then reached tick 2.
- At 390px the header, map, touch controls and time controls fit; document width
  did not exceed the viewport. Desktop composition, research navigation and
  return navigation passed, with no browser console errors or warnings.
  `progress/world-preview.jpg` captures this separate UI test world.
- The main world was saved and backed up before a graceful restart. It returned
  at tick **84,935**, manually paused and private, with eight PPO residents and
  Laya's distinct NPU model. Resident state, exposed learning metrics and tick
  matched before/after; controller error was null. A recursive comparison of
  the complete saved checkpoint payload matched **86,727 leaves**, including
  tensors, memories and optimizer state. Serialized file bytes changed after
  the ordinary autosave, so preservation was verified by payload comparison,
  not byte-identical archive hashes.

These UI checks do not demonstrate a new learned skill. The 217-test full-suite
result below belongs to the earlier publication snapshot; this follow-up ran
the focused 11-test server suite and the actual browser checks above.

## Earlier progress-report preview

- The export verified the recorded SHA-256 hashes of the practice analysis,
  protocol, separate audit and selected source replay. Source runs were read
  without modification.
- All **14** chart views (two tasks × aggregate/six individuals) displayed
  their expected counts in a real browser. Three practice conditions displayed
  their recorded states at ticks **0, 4 and 512**; playback advanced and reset.
- The browser reported no errors or warnings on the showcase. Desktop and
  390-pixel layouts were inspected; neither had horizontal document overflow.
  The temporary viewport override was reset afterward.
- Screenshots in `progress/preview.jpg` and `progress/replay-preview.jpg` are
  actual browser captures. The replay is a diagram of an archived evaluation,
  not a screenshot of the main game renderer.
- The **10 existing server tests** passed after the presentation edits.
  JavaScript syntax checks passed for the viewer and exported data. These are
  distinct from the earlier 217-test practice-experiment verification.
- The final clean publication copy then passed **all 217 tests in 104.49
  seconds**, including the optional PettingZoo parallel API check. It contains
  208 explicitly selected files; all file hashes and 244 local documentation
  links passed verification. An initial missing-adapter packaging error was
  fixed before this successful run; the failed attempt remains local.

No training, architecture, reward or live policy change was part of this
preview. Internal service IDs and saved-world paths keep their original names
for compatibility with existing residents and launchers.

At the earlier preview stage the showcase was a local artifact; no repository
or hosted site had been created. The later public source snapshot above was
prepared separately from the development checkout. Its local manifest records
every included file and documentation adjustment.
