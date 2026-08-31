# TouchTrace Sensor e120 Experiment Handoff

Generated: 2026-08-31

## Status

- Repository: TouchTrace
- Branch: `main`
- Parent commit for this handoff: `b31abb83286730b84d3a9c153771aab32bf7aac6`
- The controlled grouped-user Sensor experiment is complete.
- Release is blocked because the validation-selected configuration failed its single held-out test evaluation.
- No file under `inference/public` was changed.
- The committed H5 checkpoint is an experimental forensic artifact, not a release candidate.

## Committed Experiment Artifact

The validation-selected model and its matching train-only normalization file are committed at:

- `train/sensor/experiments/e120-failed-test/sensor-correction-e055.h5`
- `train/sensor/experiments/e120-failed-test/sensor-correction-e055_norm.json`

Hashes:

- H5 SHA-256: `ae8fac36030290a47a916fb1667186b316f553eade27bdd19eb0aa240ad79583`
- Norm SHA-256: `47cf02b8862692f767fdad2a2d2cf07a0f90c23a66d5e0c052d091a2e73b3242`

The H5 is intentionally stored outside the normal release paths. If it must be inspected as ONNX, use an isolated output and do not publish it:

```bash
cd train
python -m sensor.convert \
  --weights sensor/experiments/e120-failed-test/sensor-correction-e055.h5 \
  --norm sensor/experiments/e120-failed-test/sensor-correction-e055_norm.json \
  --output /tmp/sensor-correction-e055.onnx \
  --no-publish
```

## Controlled Training Run

The tracked `sensor_split.json` was verified against the dataset before training:

- Train: 40 users / 25,899 rows
- Validation: 5 users / 3,118 rows
- Test: 5 users / 3,010 rows

All branches used the same:

- Epoch-20 grouped warmup
- Train-only norm and initial-state distributions
- `ss_unroll_hops=4`
- `ss_max=1.0`
- Scheduled-sampling ramp from epoch 20 to 70
- Fixed total budget of 120 epochs
- Candidate checkpoint every five epochs

The branches completed sequentially on a Colab T4:

| Branch | Runtime | Final AR stability |
| --- | ---: | ---: |
| correction | 73.2 minutes | 0.40320 |
| natural | 72.4 minutes | 0.37207 |
| hybrid | 73.4 minutes | 0.37025 |

Training completion and final AR stability were not used for model selection.

## Validation Selection

All 60 branch checkpoints were evaluated locally with the same ONNX rollout path:

- Partition: validation users only
- 500 stratified swipes per seed
- Seeds: `42,43,44`
- Temperatures: `0.20,0.24,0.26,0.27,0.28,0.30`
- Innovation rho: `0,0.65`
- Hard gates retained the worst seed and condition

Only two candidates passed every validation gate:

| Checkpoint | Temp | Rho | Score | Min gyro | Min AR/TF gyro | Temporal MARE | Accel p90 MARE | Drift MARE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| correction-e055 | 0.27 | 0.65 | 0.1017 | 81.4% | 85.6% | 9.9% | 2.5% | 1.0% |
| correction-e065 | 0.26 | 0.65 | 0.1223 | 81.7% | 87.3% | 8.0% | 1.8% | 0.6% |

`correction-e055`, temperature `0.27`, rho `0.65` was frozen before accessing test users because it had the lower feasible validation distribution score.

The selected point had little validation margin. At the same checkpoint:

- Temp `0.26`, rho `0.65` failed minimum gyro at 78.67%.
- Temp `0.27`, rho `0.65` passed with temporal MARE 9.94% and minimum gyro 81.37%.
- Temp `0.28`, rho `0.65` failed temporal MARE at 11.76%.

Natural and hybrid produced no validation-feasible checkpoint.

## Single Held-Out Test Evaluation

The frozen validation-selected configuration was evaluated exactly once on test users with the same three seeds and 500 swipes per seed.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Distribution score | 0.19294 | ranking only |
| Minimum gyro ratio | 79.72% | at least 80% |
| Minimum AR/TF gyro ratio | 83.68% | at least 80% |
| Temporal MARE | 19.58% | at most 10% |
| Accel p90 MARE | 3.10% | at most 10% |
| Drift MARE | 0.43% | at most 10% |

The candidate failed both the temporal and minimum gyro gates. Do not test `correction-e065` or other configurations on these test users as a follow-up selection step.

## Failure Analysis

The evidence indicates unseen-user distribution shift and validation selection fragility, not an execution or artifact failure:

- The selected point only narrowly passed validation after comparing 60 checkpoints and 12 temp/rho combinations on five validation users.
- Test seated real gyro p50 was only about 45% to 56% of the validation cohort value, while generated gyro fell less and became 1.78x to 2.09x the test real value.
- Test seated generated step changes exceeded real changes by roughly 24% to 45%.
- Test stress generated step changes were generally about 9% to 20% below real changes.
- Test walking minimum gyro reached only 79.72% for the worst seed.
- Five hundred swipes per seed reduce rollout noise but do not increase the number of independent users.
- The test stress partition contains rows from four users, while validation stress contains all five validation users.

The Colab and local rollout paths were cross-checked on `correction-e025` and produced identical metrics. The split, norm hashes, checkpoint hashes, and 86-test suite were also verified. There is no evidence that MCP disconnection, CPU evaluation, norm leakage, or an incomplete training run caused the failure.

## Uncommitted Full Experiment Bundle

The complete local experiment directory remains at:

- `~/Downloads/touchtrace-sensor-e120-b31abb8/`

It contains all 60 validation JSON/Markdown/log reports, the final test report, branch archives, scripts, and outcome manifests. A self-contained results bundle is at:

- `~/Downloads/touchtrace-sensor-e120-results-b31abb8-20260831.tar.gz`
- SHA-256: `34b1f507f1fff270aee370f05b8b8ded3178bdf37675c9221b30a8dbe44db960`

Training archive hashes:

- Warmup: `bb3162a8eb28f7cac9f987e1e5a4f4653367f11279c14c74155640b33cc1429d`
- Correction: `33e8338062734036559135d9253efe98e38e4560110657ea65812b28a6e4b18a`
- Natural: `8abc8b53289337cdae9b063e4d8f46e46858420c4fb11415c64efd26d60b6d4b`
- Hybrid: `efd51feb847622bb61ef07e0112ab0e00b803acc764fda6113d776d9ebc5f390`

These local files are not required to identify the failed candidate because the selected H5, matching norm, metrics, and hashes are committed here.

## Next Experiment Rules

1. Treat the current test cohort as consumed. Its metrics must not drive checkpoint, temperature, rho, branch, or objective tuning while it is still called a final test set.
2. Develop only with train users and grouped validation or grouped cross-validation folds.
3. Prefer worst-user or worst-fold constraints and require meaningful validation margin, rather than accepting values directly against the 80% and 10% gates.
4. Investigate condition-specific temporal dynamics and user-level amplitude variation using train/validation data only.
5. Acquire or lock a fresh, unseen user cohort before the next formal release evaluation.
6. Keep the committed failed-test H5 isolated. Never copy it to `inference/public` or use it as a release default.

## Verification Before Commit

- Repository was clean before adding this handoff and experimental artifact.
- Local `.venv` test suite: 86 tests passed.
- Local grouped-validation sweep: 60 JSON, 60 Markdown, and 60 log reports; no failures.
- Selected configuration evaluated once on the grouped test partition.
- H5 and norm hashes match the downloaded, verified correction branch archive.
- The committed H5 loaded with the tracked full Sensor architecture and completed an isolated `--no-publish` ONNX conversion.
- `inference/public` remains unchanged.
