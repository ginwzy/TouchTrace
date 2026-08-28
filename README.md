# TouchTrace

Synthesize human-like touch trajectories and synchronized sensor data using deep learning.

TouchTrace generates organically varied mobile swipe paths—and optionally IMU readings—as JSON output. No device injection; pure generation for testing, simulation, and research.

> [!WARNING]
> Experimental software. For educational and research purposes only.

---

## Status

This project is under active development. See [TOUCH_EXTENSION_PLAN.md](./TOUCH_EXTENSION_PLAN.md) for the full roadmap.

| Phase | Scope | Status |
|-------|--------|--------|
| Phase 0 | SwipeMotionDB exploration | Done — see plan appendix C |
| Phase 1 | Touch trajectory JSON output | Working — remaining-frame LSTM+MDN, no-backtrack generation |
| Phase 2 | Touch + sensor bundle output | Data ready — IMU LSTM+MDN training (accel+gyro, touch frozen) |

The repository currently includes the **LSTM + MDN** training and inference pipeline (originally from [mousecrack](https://github.com/puffinsoft/mousecrack)), which serves as the foundation for touch model training.

---

## How it works

TouchTrace treats movement prediction as a **multivariate time series** problem:

- **Input**: previous step in remaining-aligned `(tangent, normal, dt)` plus remaining distance to the target
- **Output**: next step `(dx, dy, dt)` sampled from a Mixture Density Network (MDN)
- **Generation**: autoregressive loop until the path reaches the target

Using an MDN avoids [mode collapse](https://en.wikipedia.org/wiki/Mode_collapse)—each run produces a different, human-like trajectory.

Training data for the touch model comes from [CSD4CA](https://doi.org/10.5281/zenodo.17931118) (SwipeMotionDB v2: touch + accelerometer + gyroscope + magnetometer). Phase 2 trains a separate IMU head on accel+gyro interpolated onto touch timestamps.

---

## Installation

```bash
npm i -g touchtrace
```

---

## Usage

Output-only API:

```js
import { touchSteps } from 'touchtrace';

const path = await touchSteps(
  { x: 100, y: 800 },
  { x: 540, y: 1200 }
);
// [{ x: 100, y: 800, t: 0 }, { x: 105, y: 798, t: 16.2 }, ...]
```

CLI:

```bash
touchtrace touch-steps 100 800 540 1200
```

### Legacy mouse API

The existing mouse pipeline remains available during migration:

```js
import { steps, ModelType } from 'touchtrace';

await steps({ x: 100, y: 200 }, { x: 200, y: 400 }, ModelType.STANDARD);
```

```bash
touchtrace steps 100 200 200 400
```

---

## Models

Two model sizes are supported:

| Model | File | Notes |
|-------|------|-------|
| **Touch** | `touch.onnx` | Remaining-frame LSTM+MDN, no-backtrack. In `inference/public/` |
| **Mouse** | `model.onnx` / `model_lite.onnx` | Legacy mouse pipeline |

---

## Development

```bash
cd inference
npm install
npm run build
```

Training (Python):

```bash
cd train
pip install tensorflow tensorflow-probability tf-keras tf2onnx
python train.py
python convert.py
```

Touch model (`inference/public/touch.onnx`; training data is `train/touch/touch_data.jsonl.gz`):

```bash
# uv venv (project root)
uv venv --python 3.12
source .venv/bin/activate
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
uv sync

cd train
python -m pytest -q
python -m touch.train            # remaining-frame LSTM+MDN
python -m touch.convert          # writes train/touch/touch.onnx and inference/public/touch.onnx
python -m touch.eval             # real vs raw vs no-backtrack: bow, duration, step size
python -m touch.preview --from-data
python -m touch.preview --from-data --raw

# Sensor model (training data is train/sensor/sensor_data.jsonl.gz)
python convert_swipemotiondb.py --sensors   # needs data/raw/CSD4CA/*.csv
python -m sensor.eval                       # |a| / |gyro| by seated/walking/stress
python -m sensor.train                      # frozen-touch IMU LSTM+MDN
python -m sensor.convert                    # writes sensor.onnx + copies sensor_norm.json
```

**Google Colab (GPU):**

- Touch: [`notebooks/train_touch_colab.ipynb`](notebooks/train_touch_colab.ipynb)
- Sensor: [`notebooks/train_sensor_colab.ipynb`](notebooks/train_sensor_colab.ipynb)

Set runtime to GPU, run all cells.

---

## Attribution

TouchTrace builds on ideas and code from [mousecrack](https://github.com/puffinsoft/mousecrack) by ColonelParrot (MIT License).

Touch training data will use [SwipeMotionDB](https://doi.org/10.5281/zenodo.17171888) (Naji & Bouzidi, ENSIAS).

---

*TouchTrace* is open source software, licensed under the [MIT](LICENSE) license.
