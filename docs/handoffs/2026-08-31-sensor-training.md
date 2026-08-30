# TouchTrace Sensor Training Handoff

Generated: 2026-08-31

## Working Context

- Repository: TouchTrace
- Branch: `main`
- Parent commit before this work: `cdb43b1` (`Unroll scheduled sampling so ΔIMU training sees compounded AR error.`)
- This handoff is committed together with the Sensor training/evaluation implementation it describes. After pulling the handoff commit on another device, expect these sources to be present rather than reconstructing them from prose.
- No formal model release was made. `inference/public` and the tracked Sensor H5/ONNX/norm assets were deliberately left unchanged.
- The current execution blocker is Google Colab GPU usage quota. The previous Colab browser bridge/runtime should be treated as expired; establish a fresh bridge and T4 runtime after quota recovers.

## Objective

Finish the controlled Sensor realism experiment:

1. Train one user-grouped epoch-20 warmup.
2. Branch the same warmup into `correction`, `natural`, and `hybrid` scheduled-sampling targets.
3. Select checkpoint, temperature, and innovation correlation on held-out validation users using three fixed stochastic seeds.
4. Evaluate the selected setup once on final test users.
5. Only then decide whether to add correlated-innovation parity to TypeScript and package a release candidate.

Do not overwrite release assets during experimentation.

## Sources of Truth

Read these files rather than reconstructing implementation details from this handoff:

- Training pipeline and CLI: `train/sensor/train.py`
- Experiment defaults and guards: `train/sensor/config.py`
- Fixed user split implementation: `train/sensor/split.py`
- Fixed 40/5/5 split manifest: `train/sensor/sensor_split.json`
- Multi-checkpoint, multi-seed selector: `train/sensor/sweep.py`
- Structured rollout metrics and grouped partition evaluation: `train/sensor/eval.py`
- Optional correlated innovation generator: `train/sensor/generate.py`
- Isolated ONNX conversion and matching norm support: `train/sensor/convert.py` and `train/touch/convert.py`
- Colab workflow and current command examples: `notebooks/train_sensor_colab.ipynb`, especially sections 6-9
- User-facing command summary: `README.md`
- Tests: `train/sensor/test_train.py`, `test_split.py`, `test_sweep.py`, `test_eval.py`, `test_generate.py`, and `test_features.py`
- Full implementation delta: inspect the handoff commit with `git show --stat` and `git show`.

## Completed Work

### Historical checkpoint sweep

Saved checkpoints from run `20260830T025434Z` were evaluated with a fixed 500-swipe stratified sweep. Under independent sampling, `ss-p080-e060` at temperature `0.30` was the best saved checkpoint, but gyro p50 remained below the release threshold. The deterministic-AR-selected hold epoch 116 collapsed gyro diversity further.

The old checkpoint bundles and detailed reports remain local to the original device under `~/Downloads`; they are generated experiment artifacts and were not committed. They are not required to start the new grouped run. Do not reuse their warmup for the new ablation because those models used a trajectory-random split.

### User-grouped validation

`train/sensor/sensor_split.json` fixes the tracked dataset into:

- 40 train users / 25,899 rows
- 5 validation users / 3,118 rows
- 5 final-test users / 3,010 rows

Training fits norm and initial-state distributions from train users only. Existing manifests are reused; a changed dataset user set fails instead of silently redrawing test users. The tracked `train/sensor/sensor_data.jsonl.gz` is available after cloning/pulling the repository.

### Target ablation implementation

The code supports:

- `correction`: bounded legacy correction target
- `natural`: human natural-delta target even when the prior state is synthetic
- `hybrid`: natural-delta MDN NLL plus a low-weight bounded mixture-mean Huber correction on synthetic-input steps

All modes keep the ONNX `13 -> 65` contract. A shared warmup can be resumed with `--initial-weights` and `--initial-epoch 20`; `--run-name` isolates weights, checkpoints, and norm files.

Deterministic AR MAE is now a severe-divergence guard only. It no longer selects the best checkpoint, reduces LR, or early-stops a healthy plateau. A rollout candidate is saved every five epochs for offline stochastic selection.

### Correlated-innovation diagnostic

On the legacy random-split `ss-p080-e060` model only, `temp=0.26, rho=0.65` passed the current numeric gates for seeds 42/43/44 with 500 swipes per seed:

- Worst-condition real gyro ratio across the three runs: at least about 83%
- Worst AR/teacher-forced gyro ratio: at least about 90%
- Temporal MARE: below 9.3%
- Accel-p90 MARE: below 0.9%
- Drift MARE: below 1.5%

This is a promising diagnostic, not a release result, because the underlying model did not use the fixed unseen-user split. Retest `rho=0` and `rho=0.65` on every new grouped branch.

## Verification Completed Before Commit

- `.venv/bin/pytest -q`: **86 passed**
- `.venv/bin/python -m compileall -q train`: passed
- Notebook JSON validation: passed
- `git diff --check`: passed
- `npm --prefix inference run build`: passed
- Isolated TensorFlow/TFP training smoke crossed grouped warmup, resume, hybrid ramp, and hold. It verified manifest reuse, periodic candidates, independent norm output, and the AR divergence guard.
- `npm --prefix inference run typecheck` still fails in the pre-existing `tsup` declaration surface because `types.cts` and optional `@swc/core` declarations are unavailable. Do not attribute that failure to the Sensor changes without new evidence.

## Continuation Checklist

1. Pull `main`, inspect this commit, and run `.venv/bin/pytest -q` before editing.
2. Once Colab quota permits, create a fresh browser bridge/tab and confirm a T4 with `nvidia-smi`. Do not assume a visible old notebook tab still owns its runtime.
3. Clone/pull this commit into Colab and verify the current Sensor sources plus `sensor_split.json`. Do not expose credentials or bridge tokens in notebook output or repository files.
4. Train a new grouped epoch-20 warmup and download/back it up immediately. Colab runtimes have previously reset and removed the workspace; Google Drive mount is not a verified backup path for the controlled kernel.
5. Before launching the three branches, lock one identical total epoch budget for all targets. **This remains an open decision.** The code default is 250 and AR early-stop is intentionally disabled; do not silently treat 250 as the approved ablation budget. A shorter fixed budget such as 120 may be practical, but confirm it from runtime/cost evidence before launch.
6. Train `correction`, `natural`, and `hybrid` sequentially from the exact same epoch-20 warmup with `ss_unroll_hops=4`, `ss_max=1.0`, and identical split, norm, and data settings. Preserve every five-epoch candidate and each branch norm.
7. On validation users only, run `sensor.sweep` over candidate checkpoints with at least 500 stratified swipes and seeds `42,43,44`. Include both `rho=0` and `rho=0.65`; scan relevant temperatures around `0.20-0.30`. Runs below 500 are diagnostic screening only.
8. Selection gates are enforced per worst seed/condition: real gyro p50 >=80%, AR/teacher-forced gyro >=80%, temporal MARE <=10%, accel-p90 MARE <=10%, and drift MARE <=10%. Distribution score ranks only candidates that pass all gates.
9. Freeze target, checkpoint, temperature, and rho based on validation. Evaluate that one setup once with `--partition test`; do not repeatedly tune against test users.
10. If correlated innovation remains selected, implement and test equivalent residual state in the TypeScript/public Sensor generation path before release. Keep default inference temperature at `0.2` until a complete grouped release candidate passes. If `rho=0` wins, remove or leave the correlated path explicitly optional rather than changing defaults.
11. Convert candidates with explicit matching `--weights`, `--norm`, `--output`, and `--no-publish`. Only publish/package after approval of the final test result.
12. Re-run all verification commands and provide a manifest with hashes for any final candidate package.

## Known Operational Constraints

- The Colab MCP surface does not provide generic file upload/download operations; browser downloads or notebook-managed transfer are required.
- A prior `drive.mount` attempt failed credential propagation in the controlled kernel. Treat Drive backup as unverified.
- Colab may reclaim the runtime while leaving notebook cells visible, deleting the workspace and returning to CPU.
- Establish a new bridge rather than reusing an old connection. Never record bridge tokens or local bridge URLs in logs or documentation.
- Formal Sensor assets and `inference/public` must remain untouched during all ablations.
- Generated H5 files and training ONNX outputs are ignored by Git. Back up required run artifacts outside the ephemeral runtime.

## Suggested Skills

- **`tdd`**: invoke before further behavior changes, especially selector gates, resume behavior, generator state, or TypeScript parity.
- **`codebase-design`**: invoke if correlated residual state is promoted into the public Python/TypeScript API, to keep the ONNX contract stable and place state ownership cleanly.
- **`uv-package-manager`**: invoke only if the Python environment must be recreated or synchronized; the existing project setup was working when this handoff was written.
- No web-research skill is needed for the next experiment. Use Colab/MCP directly; only invoke `grok-search` if the Colab integration itself changes or requires current external documentation.

## Stop Conditions

Stop and report rather than improvising if:

- GPU quota remains unavailable and only CPU training is possible.
- The current data user set does not exactly match `sensor_split.json`.
- A branch cannot resume the shared warmup with identical norm/split settings.
- Validation produces no candidate passing all hard gates across all three seeds.
- Colab artifact backup cannot be verified before a long branch run.
- A command would overwrite tracked release assets.
