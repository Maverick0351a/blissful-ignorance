# ALIEN local trial

October 5, 2026. User requested a hands-on look at
[chrxh/alien](https://github.com/chrxh/alien) and approved the named 61.5 MB
portable download and launch. ALIEN is installed separately and launched.

## Verified result

- Package: `<user-home>/Downloads/alien-develop-20261005.zip`, exactly
  61,489,415 bytes. SHA-256:
  `66439945e11eb453e7f6b26f57ebc2a9d7fd9043c0aab8f9ec3e967e8b59bd59`.
- Archive CRC and path checks passed: 145 entries, 137,636,881 unpacked bytes,
  no path traversal, absolute paths, alternate streams, symlinks or duplicate
  normalized names. Extracted only to a new directory.
- Directory: `<user-home>/Applications/ALIEN-2026-10-05`.
  Desktop shortcut: `ALIEN (portable).lnk`, with the correct working directory.
- `alien.exe` SHA-256:
  `ec74fbd0f1f399971e66d045b48f84c56e4664bbd580e4cf7f2167e7c5fdb7e0`.
  The binary is not Authenticode signed; the hash identifies this downloaded
  copy and does not prove correspondence to source.
- A responsive GUI process (PID 48276 at launch) and its log confirmed
  **v5.0.0-alpha.36**, RTX 5080 compute capability 12.0, a loaded simulation,
  3,180 MB allocated GPU memory and working CUDA/OpenGL interoperability.
  The normal launch worked; no graphics system setting was changed.
- First launch automatically retrieved public default resource id 52 from the
  project's server. The earlier expectation of a bundled starting world was
  incorrect. No account, upload or additional manual preset download was used.
- `LOCAL-TRIAL.json` in the installation directory records these facts. No
  built-in MCP listener or Codex registration was enabled by this task.

Verification used process metadata and the program log. We did not capture or
visually inspect the native window, or benchmark evolution quality.

## Fit

ALIEN simulates interacting particle/cell organisms with small neural networks,
cell memory, energy competition, genomes, reproduction and mutations. Its
documented adaptation is mainly evolutionary selection across generations;
stateful cellular memory is not evidence of gradient learning during a life.
This is a useful separate experiment for the evolving-ecosystem goal. It does
not replace the existing Godhood Trials bodies or controllers.

## Package selection

- Official portable Windows nightly: `alien-develop.zip`.
- Source: <https://alien-project.org/files/alien-develop.zip>, linked from the
  author's repository installation instructions.
- HEAD at preparation: **61,489,415 bytes**, Last-Modified
  `Mon, 05 Oct 2026 03:31:33 GMT`. The URL is mutable; record the actual byte
  count and SHA-256 before extracting an approved download.
- Launch `alien.exe` from its own extracted directory with its `resources`
  directory. The nightly packaging workflow includes CUDA and MSVC runtimes.
- GitHub's latest tagged release is v4.12.3, published December 29, 2024. Its
  MSI is 85,457,920 bytes. Current documentation recommends the nightly ZIP.
- Local hardware readback: RTX 5080 Laptop GPU, driver 616.64, 16,303 MiB VRAM.
  This meets the documented NVIDIA RTX 20-series-or-newer requirement. The
  launch subsequently confirmed working GPU/rendering interoperability.

## Focused source review

Reviewed upstream commit `840aaafdb1faa25d09a06b1f97888578695370c9`:
GUI entry point, startup checks, nightly packaging workflow, settings storage,
network operations, MCP listener and controller settings. This is a focused
review, not exhaustive verification or proof the nightly binary matches source.

The archive is intended to run without an installer. Windows app preferences
are stored under the current user's `SOFTWARE\alien` registry key. Network
features include the public simulation browser and optional authenticated
uploads; the trial needs no account or upload. The built-in MCP server is off
by default and, when enabled, binds to loopback. Its default port 8765 may
conflict with another local project, so any later agent connection should use
a checked free port. No MCP configuration has been changed.

The GUI has a documented `--no-interop` source option, useful if CUDA/OpenGL
interoperability fails on a hybrid laptop. Start normally first and diagnose
the log before using it; no system GPU setting change is planned.

## First interaction

Use the automatically loaded starting simulation. Space runs/pauses, the wheel zooms,
middle-button drag pans, Alt+O shows cell types, and F1 opens documentation.
Additional community/preset downloads can be considered after the first launch.

## Sources

- [Installation and source](https://github.com/chrxh/alien)
- [Cellular networks and memory](https://github.com/chrxh/alien/blob/develop/resources/docs/neural-networks.md)
- [Evolution mechanisms](https://github.com/chrxh/alien/blob/develop/resources/docs/evolution.md)
- [Getting started](https://github.com/chrxh/alien/blob/develop/resources/docs/getting-started.md)
- [Optional local agent connection](https://github.com/chrxh/alien/blob/develop/resources/docs/ai-agents.md)
