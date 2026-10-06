# Clef Flash versus the installed Laya

Research and local comparison completed October 5, 2026 (America/Los_Angeles). The user approved the three downloads and temporarily unloading/restoring idle Gemma. All files passed size and SHA-256 verification. Clef ran locally in a separate portable runtime; no resident/controller was changed.

Clef Flash is a promising candidate for stronger pretrained decisions. Its published results do not establish better Godhood Trials survival, independent learning, or a replacement for the existing Laya resident. Keeping Laya separately supported was the preference at the time of this comparison.

October 6 update: the user has proposed retiring Laya in favor of an alternative.
Replacement selection is pending; no live replacement or model deletion has
occurred. The original comparison and its limits below remain unchanged.

## Local result: stronger decisions, unresolved deadline errors

Completed 48 locally authored cases across six rule-based template families, each with three cyclic relabellings of its choices: 144 scored requests per model. Cases, answer labels and the evaluation runner were hashed before inference. Three separate warmup requests preceded timing. Both models received the same state string, instruction and options, with fixed weights. Laya requests used 95-153 tokens without truncation; Clef server logs also report no truncation.

| Measured result | Clef Flash Q4_K_M, GPU | Installed typed-decisions Laya, NPU |
| --- | ---: | ---: |
| Correct choices | 128/144 (88.9%) | 70/144 (48.6%) |
| Cases correct in all three orders | 42/48 | 20/48 |
| Cases whose answer changed with option order | 1/48 | 7/48 |
| Errors with top-choice probability at least 0.9 | 15 | 0 |
| Mean multiclass Brier score; lower is better | 0.2255 | 0.6430 |
| Warm median decision time | 80.5 ms | 220.1 ms |
| Warm p95 decision time | 104.2 ms | 223.8 ms |

This compares the actual selected runtimes and devices, including Clef's loopback HTTP overhead. It does not isolate architecture, precision or hardware effects. The 144 requests are repeated presentations of 48 cases from only six templates, not 144 independent trials. These synthetic supplied-rule tasks are easier and narrower than persistent-world behavior. No training or game actions occurred in this comparison.

| Category | Clef correct / 24 | Laya correct / 24 |
| --- | ---: | ---: |
| Immediate food availability | 24 | 12 |
| Harmful fruit from supplied personal outcome history | 24 | 13 |
| Farming deadlines | 12 | 12 |
| Revival preconditions | 20 | 9 |
| Supplied tone-code interpretation | 24 | 15 |
| Visible versus unknown evidence | 24 | 9 |

The cleanest failure is farming: both models chose to farm in all four cases where converting, planting, waiting and harvesting could not finish by the deadline. Clef made those twelve repeated errors with probabilities of 0.967-0.981. For example, three turns were available but the stated sequence required four; keeping the original fruit was the only plan retaining edible inventory at the deadline. Clef did not demonstrate the needed temporal planning on these cases.

A post-run wording audit flagged `revival-05`: its question presupposes an unconscious neighbor while its state says the neighbor is awake. Preserve all original scores, but treat the three corresponding high-confidence Clef errors as ambiguous evidence. Excluding only that case as a sensitivity check gives Clef 128/141 and Laya 70/141, with 12 versus 0 high-confidence errors. This is a disclosed post-hoc check, not a changed primary result. No prompt repair or retest followed the scores.

Clef loaded to healthy status in 6.88 seconds; Laya engine setup took 2.51 seconds with existing local caches, not a fresh-machine cold-start measurement. Global GPU usage rose from 6,956 MiB after unloading Gemma to a sampled peak of 13,047 MiB during Clef: approximately 5.95 GiB additional usage, not an isolated per-process allocation measurement. The Clef server reached a sampled peak working set of 6.46 GiB and private committed memory of 7.45 GiB. Initial NPU memory telemetry captured only its small Windows Python launcher and is **not a valid Laya memory measurement**; the supervisor was subsequently corrected and tested with an owned dummy child. Model decisions were not rerun.

The first Gemma restore command hit an LM Studio CLI incompatibility: `--no-engine-config-file` was rejected by this backend. Retrying without that flag succeeded in nine seconds. Verified restoration includes the same model/identifier, context length 131,072, parallelism four and disabled engine-config-file mode. The original failure record is retained with a separate successful recovery audit. The Clef server and benchmark workers are stopped. Main-world health retained PID 69328 and the same eight PPO IDs plus `laya: laya-npu`; its own live time advanced from 63,849 to 63,911 between read-only checks.

Evidence: `runs/clef-laya-20261005/` contains the frozen cases, preregistration, both raw traces/summaries, server log, telemetry, restoration records, `audit.json` and `measurement-notes.json`. Case SHA-256: `a77cd352dcdaff6beef70d93e73bd9e00ca75f1ac5ace48b0a3acbc46c5f955d`. Scored runner SHA-256: `ea86a2ba974d9c33a7bfa3c2c6b9836c37fc23413a367e781d18eb32d83b6c76`.

Recommendation: retain Laya and test Clef as a separate candidate in matched, bounded food/farming lives. Include deadlines and resource preconditions. Faster, more accurate supplied-rule answers justify that next experiment; they do not establish improved survival, discovery, learned communication or lifelong adaptation.

## Published evidence

Cloudflare describes Clef Flash as a 9B model derived from Qwen3.5-9B, with a joint decision head, text/JSON/image/video inputs and typed probability outputs. It does not generate free-form answers. The release is Apache-2.0. These features could support richer private observations, but installing the checkpoint does not create an online learning loop. [Official model card](https://huggingface.co/Cloudflare/clef-flash)

Cloudflare's own comparison reports:

| Metric | Clef Flash | Published Laya comparator |
| --- | ---: | ---: |
| BFCL case-exact accuracy | 98.76% | 38.13% |
| API-Bank accuracy | 93.11% | 11.41% |
| BANKING77 macro-F1 | 90.93% | 14.29% |
| Median request latency | 38.8 ms | 5.8 ms |

Source: [Cloudflare announcement](https://blog.cloudflare.com/clef-decision-models/). These are publisher results, not measurements on this laptop. The article links its Laya comparator to the root `convaiinnovations/laya` repository; it does not establish a match to our installed typed-decisions checkpoint and OpenVINO adapter.

The complete model-card table also contains losses: Clef Flash scores 34.2 versus Laya 42.4 on SGD/SGD-X, and 35.6 versus 48.8 on RAGTruth. Neither the benchmark coverage nor quantized accuracy has been independently reproduced here. [Full results](https://huggingface.co/Cloudflare/clef-flash#results)

## Local baseline and hardware

Verified from `agents/laya_resident.py`, the installed checkpoint configuration, and local device queries:

- Installed Laya is the fine-tuned ModernBERT-large typed-decisions model, approximately 421M parameters. Its weight file is 842,609,220 bytes. The checkpoint configuration allows 1,024 tokens, while our deployed NPU IR uses a 512-token input and 16 option slots.
- The resident adapter groups candidates in shuffled groups of eight, selects winners through a tournament, and supplies three recent action/consequence entries. Weights are frozen. The compact observation drops the tile grid, visual recall sketches and several other fields. A larger input or one-pass choice could help, but that is a hypothesis about the adapter as well as the model.
- RTX 5080 Laptop: 16,303 MiB VRAM; 14,075 MiB occupied at the check. Driver 616.64 reports CUDA UMD 13.4. RAM: 31.43 GiB total, approximately 2.53 GiB available. C: approximately 20.45 GiB free. These are transient readings, not resource reservations.
- A roughly 19 GB original release is unsuitable for a wholly GPU-resident load here. The proposed quantized weights are 6.49 GB; that makes a GPU trial plausible once memory is available. Actual peak memory, speed and accuracy remain unverified. No Clef NPU path has been validated.

The earlier [Laya farming diagnostic](LAYA-RECIPE-TRIAL.md) is a useful task lead, not a fresh comparison: Laya failed that frozen checkpoint/prompt trial despite an explicit recipe. Re-run a matched diagnostic before attributing any difference to Clef.

## Local runtime route

Use the [ggml-org Clef GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF), which declares the Clef architecture and native decision head. Current upstream [llama.cpp v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0) includes Clef text and vision support; its Windows assets are supplied under b11429. Preserve existing LM Studio and Python installations.

The actual endpoint is `/v1/systemone`, not chat completions. The [pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api) describes typed questions, choice probabilities, and the requirement that the entire Clef request fit the physical batch. Returned confidence is not guaranteed to be calibrated on our tasks. Vision additionally requires a projector, which is excluded from the first proposed download.

The upstream Clef loader, joint-head model source, server decision parsing/scoring and startup path were reviewed before execution. This was a scoped integration review, not a comprehensive security audit. Both downloaded ZIPs passed CRC and path/link checks before extraction; 55 extracted files total 745,559,661 bytes. The executable identifies itself as 0.6.0-dev, build 11429, commit d81235049. The trial used a local model path, offline mode, loopback port 8793, one slot, 1,024 context tokens and a 512-token physical batch. UI, agent tools and MCP proxy were disabled. Existing LM Studio and Python installations were preserved.

## Original comparison protocol

First run a bounded text decision test against the actual NPU Laya, with neither model controlling the live world:

1. Prepare 48 locally authored cases covering food availability, harmful resources, farming conversions and delays, helping unconscious neighbors, local social observations, and insufficient evidence. Use explicit rules where an outcome is scored as correct; exclude ambiguous action preferences from accuracy. Freeze case labels, acceptable answers and file hashes before inference.
2. Give both models identical compact state, legal options and consequence history. Check that every complete request fits Laya's input and head budgets. Apply three fixed option permutations to each case. Limit this first comparison to choice sets both adapters can score in one call; a later full-action trial must preserve and report Laya's tournament cost.
3. Record correctness, per-category counts, paired errors, choice sensitivity to order, Brier score and errors with top-choice probability at least 0.9. Also record warm median/p95 wall latency, model calls per decision, timeouts, peak RAM/VRAM and cold load time. Preserve raw answers and distinguish model probabilities from endpoint confidence fields.
4. Use a ten-minute inference budget per model after loading, request timeouts and an owned-process cleanup path. An interrupted comparison stays incomplete. No silent scripted fallback, model substitution or post-result prompt tuning.

This measures supplied-knowledge decision quality. It cannot establish learning from experience or sustained survival. If it is promising, preregister matched farming/survival lives with the same initial bodies, resources, rules and seeds. Report ordinary versus amber meals, zero-food and unconscious time, successful farming sequences, exploration and social outcomes separately. Keep those benchmark lives separate from named residents.

Test expanded context and private visual observations as separate conditions later. A cropped view must honor that resident's senses and occlusion. An improvement from extra perception must not be reported as a pure model improvement. Neither screenshots nor an experience journal alone demonstrate changing weights or learned competence.

## Approved and verified download manifest

All sizes are bytes before extraction and matched the downloaded files. Model revision: `4a192915ef971886004b5b13294f2b4c7a7fc39d`. Runtime release: `b11429`. Files are outside the repository at `<user-home>/AI/clef-flash/`, alongside the download, archive-inspection and extracted-file hash manifests. The initial Windows curl request was reset; a standard-library HTTPS retry completed without disabling certificate verification.

| File and pinned source | Bytes | Published SHA-256 |
| --- | ---: | --- |
| [Clef-Flash-Q4_K_M.gguf](https://huggingface.co/ggml-org/Clef-Flash-GGUF/resolve/4a192915ef971886004b5b13294f2b4c7a7fc39d/Clef-Flash-Q4_K_M.gguf) | 6,486,448,288 | `fd3e90605e8103307dca37cb5a8cdb036267e2fe3cb2d908d80a8ceb9ec0638c` |
| [llama-b11429-bin-win-cuda-13.4-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-win-cuda-13.4-x64.zip) | 153,089,864 | `76ddc6eff2389570789ed608881efc6977751a722015d8e8c94f302224ff1a3a` |
| [cudart-llama-bin-win-cuda-13.4-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b11429/cudart-llama-bin-win-cuda-13.4-x64.zip) | 423,535,356 | `738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668` |

Total downloaded: **7,063,073,508 bytes (7.06 GB / 6.58 GiB)**, plus extracted runtime space and local evidence. Inference stayed local and used no paid API. The user's supplied AGENTS.md download and execution approvals were obtained. The population and Laya's permanent model role were preserved.
