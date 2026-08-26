# TouchTrace 开发计划

> 触摸轨迹 + 传感器数据生成（仅输出，不执行）  
> 状态：Phase 0 完成；Phase 1–2 尚未实现  
> 日期：2026-08-26（附录 C：2026-08-27）  
> 仓库：https://github.com/ginwzy/TouchTrace

---

## 1. 背景与目标

### 1.1 项目现状

TouchTrace 当前基于 deep learning 的轨迹合成框架（源自 [mousecrack](https://github.com/puffinsoft/mousecrack)）：

- **训练**：Python + TensorFlow，LSTM + Mixture Density Network（MDN）
- **推理**：TypeScript + ONNX Runtime + robotjs
- **数据**：`train/data.jsonl`，约 25,000 条 `(x, y, timestamp)` 轨迹
- **输出**：自回归生成 `(dx, dy, dt)` 序列，最终得到带时间戳的路径点

核心抽象与输入设备无关：模型学习的是**相对位移 + 到目标距离 + 时间间隔**，而非绝对屏幕坐标。

### 1.2 扩展目标

将项目扩展为支持**移动端触摸轨迹**及**同步传感器数据**的生成器，且：

| 要求 | 说明 |
|------|------|
| **只输出结果** | 生成 JSON 格式的轨迹/传感器序列，不调用 robotjs，不注入真实触摸或传感器 |
| **不自动执行** | 移除对硬件控制的依赖，推理层为纯函数 |
| **复用现有架构** | 尽可能保留 LSTM + MDN + 自回归生成框架 |
| **第一版数据源** | 使用 [CSD4CA v2](https://doi.org/10.5281/zenodo.17931118)（SwipeMotionDB 的官方清洗版） |

### 1.3 非目标（第一版不做）

- 真机触摸/传感器注入（Android Accessibility、iOS 等）
- 多指手势（pinch、rotate）
- 统一 mouse + touch 的单模型（后续可选）
- tap、long-press 等非 swipe 手势

---

## 2. 可行性结论

### 2.1 总体判断

**可行，且比「带执行」方案简单得多。**

现有 `steps(from, to)` 接口本质上已是「纯生成」；扩展只需：

1. 用触摸数据重新训练模型（或增加 touch 专用模型）
2. 扩展输出格式（pressure、传感器序列）
3. 去掉 `robotjs` 依赖（对 touch 模块而言）

### 2.2 训练数据

SwipeMotionDB **足够支撑第一版 MVP**：

- 50 参与者 × 2 session × 3 场景（seated / walking / stress）
- 同步 touch（坐标、时间戳、pressure、area）+ IMU（accelerometer、gyroscope、magnetometer）
- Zenodo 免费下载（176 MB 压缩包）
- 格式接近 mousecrack 所需，转换成本可控

若转换后轨迹数偏少（< 5,000 条），可合并同作者的 [CSD4CA](https://doi.org/10.5281/zenodo.17931118) 或补充 [AITouch](https://doi.org/10.17632/9v7bxv3dcc)（仅 touch）。

---

## 3. 数据源：SwipeMotionDB

### 3.1 基本信息

**训练请用 v2（CSD4CA）**。v1 与 v2 是同一条 Zenodo concept record 的两个版本，不是两套可合并的数据。详见附录 C。

| 属性 | v1 SwipeMotionDB | v2 CSD4CA（推荐） |
|------|------------------|-------------------|
| DOI | [10.5281/zenodo.17171888](https://doi.org/10.5281/zenodo.17171888) | [10.5281/zenodo.17931118](https://doi.org/10.5281/zenodo.17931118) |
| 作者 | Naji, Zakaria; Bouzidi, Driss (ENSIAS) | 同左 |
| 许可 | CC-BY-4.0 | CC-BY-4.0 |
| 参与者 | 58 人（含编号变体场景） | **50 人**（v1 的清洗子集） |
| 设备 | Pixel 6a，1080×2142，density 2.625 | 同左 |
| Session | 1 / 2（另有 3 条 session=3） | 仅 1 和 2，数量接近均衡 |
| 场景标签 | `Normal*` / `Walking*` / `Stressful*`（含编号后缀） | 仅 `Normal` / `Walking` / `Stressful` |
| 压缩包 | 176 MB `.rar` | 126.5 MB `.rar` |
| 原始 swipe | 42,242 | **34,417**（全部是 v1 的子集） |

### 3.2 包含的数据流

本地路径：`data/raw/CSD4CA/`（已 gitignore）。四个 CSV：`touch_data.csv`、`acc_data.csv`、`gyro_data.csv`、`magneto_data.csv`。

**Touch 事件（已按 `id_swipe` 切段，无需 down/move/up 再切）：**

- 坐标 `(x, y)`，像素，屏宽 1080、屏高 2142
- `time`：开机后毫秒（`elapsedRealtime` 量级），需在 swipe 内归零
- `pressure`、`touch_major` / `touch_minor`、`finger_size`

**Motion 传感器（按同一 `id_swipe` 对齐，不是同一绝对时钟）：**

- Accelerometer（含重力，`|a|` 中位数 ≈ 9.81 m/s²）
- Gyroscope（rad/s）
- Magnetometer（µT 量级）
- `time`：纳秒（CSV 为 float64 科学计数，约 1 ms 精度）

### 3.3 与 Mousecrack 格式的映射

Mousecrack 当前训练格式（`data.jsonl`，每行一条轨迹）：

```json
{
  "length": 15,
  "target": { "x": 1511, "y": 340 },
  "path": [
    { "x": 1256, "y": 359, "timestamp": 0 },
    { "x": 1281, "y": 357, "timestamp": 22 }
  ]
}
```

**转换规则（已对照 v2 实测）：**

1. 按 `id_swipe` 分组（不要按 down/move/up 再切；原始文件也未提供 action 字段）
2. 同一毫秒多点：按时间排序后 **collapse（保留最后一点）**，否则约 17% 相邻点 `dt==0`
3. `target` = 该 swipe 最后一个点的 `(x, y)`
4. `timestamp` 归零为相对毫秒（touch 与 sensor **各自**以本 swipe 首点为 0）
5. 过滤：collapse 后点数 < 3、总时长 < 50ms、或仍有 `dt > 500ms` 的样本
6. 场景映射：`Normal` → `seated`，`Walking` → `walking`，`Stressful` → `stress`
7. 可选保留 `pressure`、`touch_major`（作 area）
8. Phase 2：用 `id_swipe` 取 sensor，把 sensor 时间 `/ 1e6` 成毫秒后按相对时间插值到 touch 点。**不要**把 touch 的 ms 与 sensor 的 ns 当同一时钟做绝对对齐

**扩展格式（touch + sensor，供联合训练使用）：**

```json
{
  "target": { "x": 540, "y": 1200 },
  "meta": {
    "condition": "seated",
    "session": 1,
    "user_id": "P01"
  },
  "path": [
    { "x": 100, "y": 800, "timestamp": 0, "pressure": 0.52, "area": 1.2 }
  ],
  "sensors": [
    {
      "timestamp": 0,
      "accel": [0.1, 9.8, 0.2],
      "gyro": [0.01, 0.02, 0.0],
      "mag": [20.1, 5.3, 45.2]
    }
  ]
}
```

### 3.4 数据量（实测）

CSD4CA v2：**34,417** 条原始 swipe。按上节规则 collapse + 过滤后约 **33,058** 条可用（Normal 12,130 / Stressful 9,973 / Walking 10,955）。

多于 mousecrack 的 25,000 条鼠标轨迹，**不需要合并其他数据集**。v1 多出来的 7,825 条是被 v2 丢掉的 8 名用户 + 编号场景后缀 + session 3，不要再并回去。

### 3.5 已知限制

| 限制 | 影响 | 缓解 |
|------|------|------|
| 单一设备（Pixel 6a） | 换机型时 pressure/area 尺度可能不一致 | 训练时 z-score 归一化；输出时附带 `deviceProfile` |
| 仅 50 用户 | 轨迹风格多样性有限 | MVP 可接受；后期加 AITouch / HuMIdb |
| 仅 swipe | 不支持 tap、长按 | 第一版 scope 明确为 swipe |
| 需格式转换 | 不能零成本直接使用 | 一次性转换脚本（按 `id_swipe` 分组） |
| 同毫秒重复 touch 点 | 直接算 `dt` 会大量 ≤ 0 | collapse 后再过滤 |
| touch 与 sensor 时钟不同 | 绝对时间对不上 | swipe 内相对时间 + `id_swipe` |
| 加速度含重力 | 与线性加速度数据集不可混用 | 文档声明 `TYPE_ACCELEROMETER` |

---

## 4. 技术方案

### 4.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      数据采集（已有）                         │
│              SwipeMotionDB → convert_*.py                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      训练（Python）                           │
│  touch_data.jsonl → train_touch.py → touch.onnx             │
│  touch_sensor.jsonl → train_multimodal.py → touch_sensor.onnx│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      推理（TypeScript）                       │
│  core/generatePath.ts  ← 设备无关自回归 + MDN 采样          │
│  touchSteps()          ← 输出 touch 路径 JSON               │
│  touchSensorSteps()    ← 输出 touch + sensors bundle        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1：仅 Touch 轨迹

**模型**：复用现有架构，与 mouse 模型相同。

| 参数 | 值 |
|------|-----|
| input_dims | 5 — `[dx_prev, dy_prev, dt_prev, dist_x, dist_y]` |
| output_dims | 3 — `[dx, dy, dt]` |
| LSTM | 2 × 128（Standard）/ 2 × 64（Lite） |
| MDN components | 5 |

**推理逻辑**：与 `inference/index.ts` 中 `generatePath()` 相同，替换模型文件为 `touch.onnx`。

**输出接口：**

```typescript
interface TouchStep {
  x: number;
  y: number;
  t: number;        // 毫秒，相对起点
  pressure?: number; // Phase 1 可选：从训练数据分布采样或省略
}

function touchSteps(
  start: { x: number; y: number },
  end: { x: number; y: number },
  type?: ModelType
): Promise<TouchStep[]>;
```

**CLI（可选）：**

```bash
mousecrack touch-steps 100 800 540 1200
# 输出 JSON 到 stdout
```

### 4.3 Phase 2：Touch + Sensor 联合输出

两种实现路径，推荐 **方案 B**。

#### 方案 A：扩展 MDN 输出维度

```
output_dims: 12  →  [dx, dy, dt, ax, ay, az, gx, gy, gz, mx, my, mz]
```

- 优点：单模型、端到端
- 缺点：touch 与 sensor 采样率不同，对齐复杂；12 维 MDN 训练更难

#### 方案 B：双头模型（推荐）

```
LSTM backbone (shared)
    ├── MDN head → touch  (dx, dy, dt)
    └── MDN head → sensor (ax, ay, az, gx, gy, gz)  // 9 维，mag 可选
```

- 优点：touch 与 sensor 可独立调参；sensor 头可在 touch 轨迹确定后按时间步生成
- 缺点：需改训练代码；推理需两次 MDN 采样或串联

**Sensor 时间对齐：**

- 训练：以 touch 时间戳为锚，对每个 touch 点取最近（或插值）的 sensor 读数作为 label
- 推理：每生成一个 touch 步，同步生成对应 sensor 读数

**输出接口：**

```typescript
interface SensorReading {
  t: number;
  accel: [number, number, number];
  gyro: [number, number, number];
  mag?: [number, number, number];
}

interface TouchSensorBundle {
  touch: TouchStep[];
  sensors: SensorReading[];
  meta?: {
    device: "pixel-6a";
    condition?: "seated" | "walking" | "stress";
  };
}

function touchSensorSteps(
  start: { x: number; y: number },
  end: { x: number; y: number },
  options?: { condition?: string; includeMag?: boolean }
): Promise<TouchSensorBundle>;
```

### 4.4 与现有 Mouse 模块的关系

| 策略 | 说明 |
|------|------|
| **独立模块（推荐）** | `touch.onnx` / `touch_sensor.onnx` 与现有 `model.onnx` 并存；API 分 `steps()` 与 `touchSteps()` |
| 统一模型 | 输入加 modality embedding，一个模型服务 mouse + touch；数据量和工程复杂度更高，不建议第一版 |

现有 mouse 功能保持不变；touch 扩展为新增模块，不破坏向后兼容。

### 4.5 依赖变更

| 包 | Phase 1 Touch | Phase 2 Sensor | 说明 |
|----|---------------|----------------|------|
| `onnxruntime-node` | 保留 | 保留 | 推理引擎 |
| `robotjs` | touch 模块不使用 | 不使用 | mouse 的 `move()` 仍可保留 |
| `commander` | 保留 | 保留 | CLI |
| TensorFlow + TFP | 训练侧 | 训练侧 | 无变化 |

Touch 模块可打包为**零 native 依赖**的纯 JSON 生成库（若去掉 mouse 的 robotjs，整体更轻）。

---

## 5. 目录结构规划

实现阶段建议的目录布局（当前尚未创建）：

```
touchtrace/
├── TOUCH_EXTENSION_PLAN.md      # 本文档
├── train/
│   ├── data.jsonl               # 现有 mouse 数据
│   ├── train.py                 # 现有 mouse 训练
│   ├── touch/
│   │   ├── convert_swipemotiondb.py   # SwipeMotionDB → jsonl
│   │   ├── touch_data.jsonl           # 转换后的 touch 数据
│   │   ├── touch_sensor_data.jsonl    # touch + sensor 融合数据
│   │   ├── config_touch.py
│   │   ├── train_touch.py
│   │   └── train_touch_sensor.py
│   └── ...
├── inference/
│   ├── index.ts                 # 现有 mouse API
│   ├── touch.ts                 # touchSteps(), touchSensorSteps()
│   ├── cli.ts                   # 新增 touch-steps 子命令
│   ├── config.ts                # 扩展 ModelType 或 TouchModelType
│   ├── public/
│   │   ├── model.onnx
│   │   ├── model_lite.onnx
│   │   ├── touch.onnx           # 新增
│   │   └── touch_sensor.onnx    # 新增
│   └── ...
└── skills/
    └── move-mouse/              # 现有
    └── touch-swipe/             # 可选：Agent skill
        └── SKILL.md
```

---

## 6. 数据转换流程

### 6.1 步骤

```
1. 使用已下载的 data/raw/CSD4CA/（不要用 v1 再切一遍）
2. 运行 convert_swipemotiondb.py：
   a. 按 id_swipe 分组 touch
   b. 按 time 排序并 collapse 同一毫秒
   c. 时间戳归零、异常过滤
   d. （Phase 2）按 id_swipe 取 sensor，相对时间插值
   e. 写出 touch_data.jsonl / touch_sensor_data.jsonl
3. 输出统计报告：总轨迹数、每场景分布、平均路径长度、dt 分布
```

### 6.2 转换脚本职责（`convert_swipemotiondb.py`）

| 功能 | 说明 |
|------|------|
| `parse_touch_events(path)` | 读取 CSD4CA `touch_data.csv` |
| `group_swipes(rows)` | 按 `id_swipe` 分组（无 action 字段） |
| `collapse_timestamps(path)` | 同 ms 保留最后一点，再排序 |
| `normalize_timestamps(path)` | 相对毫秒 |
| `merge_sensors(touch_path, sensor_dir)` | 按 `id_swipe` + 相对时间插值 |
| `filter_trajectory(traj)` | 长度、dt、坐标合法性 |
| `write_jsonl(trajectories, out_path)` | 输出训练格式 |
| `print_stats(trajectories)` | 打印数据集摘要 |

### 6.3 数据质量检查

转换完成后应验证：

- [ ] 轨迹数 ≥ 3,000（v2 预期约 33k，远超此线）
- [ ] collapse 之后无 `dt <= 0` 的相邻点对
- [ ] `target` 与 `path` 末点一致（或误差 < 2px）
- [ ] seated / walking / stress 三类均有足够样本
- [ ] Phase 2：每个 touch 点均有对应 sensor 读数（或插值标记）

---

## 7. 训练计划

### 7.1 Phase 1：Touch 轨迹模型

| 项 | 值 |
|----|-----|
| 脚本 | `train/touch/train_touch.py`（由 `train.py` 改编） |
| 数据 | `touch_data.jsonl` |
| 配置 | 与 mouse 相同：input=5, output=3, LSTM 2×128, MDN 5 components |
| Epochs | 200（与 mouse 一致，可按 validation loss 早停） |
| Batch size | 64 |
| Validation split | 10% |
| 输出 | `touch/model.h5` → convert → `inference/public/touch.onnx` |

**训练数据加载**：复用 `load_mouse_data` 逻辑，仅改数据源路径；若 path 含 `pressure`，Phase 1 仍只取 `(x, y, timestamp)`。

### 7.2 Phase 2：Touch + Sensor 模型

| 项 | 值 |
|----|-----|
| 脚本 | `train/touch/train_touch_sensor.py` |
| 数据 | `touch_sensor_data.jsonl` |
| 架构 | Shared LSTM + 双 MDN head |
| Touch head output | 3 — dx, dy, dt |
| Sensor head output | 9 — ax, ay, az, gx, gy, gz, mx, my, mz（或 6 维不含 mag） |
| 归一化 | 传感器各轴 z-score（在训练集上 fit，推理时反归一化） |
| 输出 | `touch_sensor.onnx` |

### 7.3 评估指标

**Touch 轨迹：**

- 路径长度分布 vs 真实数据
- 平均曲率、jerk（加速度变化率）分布
- 到达目标误差（末点与 target 距离，应 ≈ 0）
- 生成耗时（步数 × 单次 ONNX 推理）

**Sensor（Phase 2）：**

- 各轴均值/方差与训练集对比
- Touch 步与 sensor 读数的时间对齐误差
- seated vs walking vs stress 条件下 sensor 模式是否合理（定性 + 简单统计）

**不做**：与 mouse 模型的交叉评测（模态不同）。

---

## 8. 推理 API 设计

### 8.1 核心原则

- **纯函数**：`(start, end, options?) → JSON`，无副作用
- **不依赖 robotjs**（touch 模块）
- **与 mouse API 平行**，不破坏现有 `steps()` / `move()`

### 8.2  proposed API

```typescript
// inference/touch.ts

export interface TouchStep {
  x: number;
  y: number;
  t: number;
  pressure?: number;
}

export interface SensorReading {
  t: number;
  accel: [number, number, number];
  gyro: [number, number, number];
  mag?: [number, number, number];
}

export interface TouchSensorBundle {
  touch: TouchStep[];
  sensors: SensorReading[];
  meta?: {
    device: string;
    model: "standard" | "lite";
  };
}

/** Phase 1：仅 touch 路径 */
export function touchSteps(
  start: { x: number; y: number },
  end: { x: number; y: number },
  type?: ModelType
): Promise<TouchStep[]>;

/** Phase 2：touch + sensor */
export function touchSensorSteps(
  start: { x: number; y: number },
  end: { x: number; y: number },
  options?: {
    type?: ModelType;
    includeMag?: boolean;
    condition?: "seated" | "walking" | "stress";  // 可选条件采样
  }
): Promise<TouchSensorBundle>;
```

### 8.3 CLI 扩展

```bash
# Phase 1
mousecrack touch-steps <fromX> <fromY> <toX> <toY> [standard|lite]

# Phase 2
mousecrack touch-sensor-steps <fromX> <fromY> <toX> <toY> [standard|lite] [--no-mag]
```

输出为 JSON，可直接 pipe 到文件或下游系统。

### 8.4 生成算法（与 mouse 一致）

```
1. curr = start, sequence = []
2. loop (max 500 steps):
   a. if distance(curr, end) < 3px: break
   b. input = [dx_prev, dy_prev, dt_prev, end.x - curr.x, end.y - curr.y]
   c. append to sequence, run ONNX
   d. sample (dx, dy, dt) from MDN output
   e. curr += (dx, dy), append to path
   f. (Phase 2) sample sensor from sensor head at same step
3. append exact end point to path
4. (Phase 1) optional: smoothPath(path, 7)
5. return JSON
```

---

## 9. 实施阶段与里程碑

### Phase 0：准备（预估 1–2 天）

- [x] 下载并解压 SwipeMotionDB（v1）以及同记录的 CSD4CA（v2）
- [x] 文档化实际文件结构与字段名
- [x] 统计原始 swipe 条数与场景分布
- [x] 确认是否需合并 CSD4CA → **否**；v2 即清洗版，作为训练源

**交付物**：附录 C

---

### Phase 1：Touch 轨迹 MVP（预估 1–2 周）

| 任务 | 优先级 |
|------|--------|
| `convert_swipemotiondb.py` | P0 |
| `train_touch.py` + `touch_data.jsonl` | P0 |
| 导出 `touch.onnx` | P0 |
| `inference/touch.ts` — `touchSteps()` | P0 |
| CLI `touch-steps` 子命令 | P1 |
| 基础评估脚本（路径统计对比） | P1 |
| 文档更新（README 触摸章节） | P2 |

**验收标准：**

- 给定 `(start, end)`，输出合理 swipe 路径 JSON
- 路径末点精确落在 target
- 轨迹在视觉/统计上接近真实 swipe（非直线）

---

### Phase 2：Touch + Sensor（预估 2–3 周）

| 任务 | 优先级 |
|------|--------|
| 融合 touch + sensor 的 jsonl | P0 |
| `train_touch_sensor.py` 双头模型 | P0 |
| 导出 `touch_sensor.onnx` | P0 |
| `touchSensorSteps()` API | P0 |
| CLI `touch-sensor-steps` | P1 |
| Sensor 归一化/反归一化 | P0 |
| 评估：sensor 分布对比 | P1 |

**验收标准：**

- 输出 bundle 中 touch 与 sensors 时间对齐
- sensor 读数在量级上与训练集一致
- walking 场景下 accel/gyro 方差高于 seated（定性）

---

### Phase 3： polish（可选）

- [ ] Agent Skill：`skills/touch-swipe/SKILL.md`
- [ ] `onnxruntime-web` 浏览器端推理
- [ ] pressure 作为 MDN 第 4 输出维
- [ ] 合并 AITouch / HuMIdb 扩充数据
- [ ] npm 包分 export：`touchtrace/touch`

---

## 10. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| SwipeMotionDB 实际条数 < 3k | ~~中~~ 已排除 | — | v2 过滤后仍约 33k 条 |
| 文件格式与文档不符 | 中 | 转换脚本需重写 | Phase 0 先探索再编码 |
| 单设备导致泛化差 | 中 | 换机型输出失真 | 文档声明 device profile；后期多设备数据 |
| Sensor 双头训练不稳定 | 中 | Phase 2 延期 | 先独立训 sensor 头；或降维至 6 维 |
| ONNX 双头导出复杂 | 低 | 推理需拆两个模型 | 可导出两个 onnx，推理时串联 |
| 与 mouse 代码耦合 | 低 | 维护成本 | 抽取 `core/generatePath` 共享逻辑 |

---

## 11. 开放问题（实现前需确认）

1. **Pressure 是否纳入 Phase 1 输出？**  
   - 选项 A：Phase 1 只输出 x, y, t  
   - 选项 B：从训练集 pressure 分布独立采样，不经过 MDN  

2. **Phase 2 是否包含 magnetometer？**  
   - mag 受环境干扰大，可先只做 accel + gyro（6 维）

3. **condition（seated/walking/stress）是否作为生成条件？**  
   - 选项 A：默认混合分布  
   - 选项 B：API 允许指定 condition，需 conditioning 向量或分模型  

4. **mouse 的 `move()` 是否保留？**  
   - 建议保留，touch 模块完全独立

5. **npm 包名与版本策略？**  
   - 选项 A：同包 `mousecrack@0.3.0` 新增 touch export  
   - 选项 B：子路径 `touchtrace/touch`

---

## 12. 参考资料

| 资源 | 链接 |
|------|------|
| SwipeMotionDB | https://doi.org/10.5281/zenodo.17171888 |
| CSD4CA（同类扩展数据） | https://doi.org/10.5281/zenodo.17931118 |
| AITouch（仅 touch 备选） | https://doi.org/10.17632/9v7bxv3dcc |
| HuMIdb（大规模 touch + 14 sensors） | https://github.com/BiDAlab/HuMIdb |
| Mousecrack 训练数据格式 | `train/data.jsonl`, `train/train.py` |
| MDN 采样实现 | `inference/util.ts` — `sampleFromMDN()` |

---

## 附录 A：Mousecrack 现有训练特征（供对照）

**输入（5 维）：**

| 字段 | 含义 |
|------|------|
| dx_prev | 上一步水平位移 |
| dy_prev | 上一步垂直位移 |
| dt_prev | 上一步时间间隔（ms） |
| dist_x | 当前点到目标的水平距离 |
| dist_y | 当前点到目标的垂直距离 |

**输出（3 维）：**

| 字段 | 含义 |
|------|------|
| dx | 下一步水平位移 |
| dy | 下一步垂直位移 |
| dt | 下一步时间间隔（ms） |

Touch 扩展 Phase 1 可直接沿用，无需改模型结构。

---

## 附录 B：预估工作量汇总

| 阶段 | 内容 | 预估时间 |
|------|------|----------|
| Phase 0 | 数据下载与探索 | 1–2 天 |
| Phase 1 | Touch 轨迹 MVP | 1–2 周 |
| Phase 2 | Touch + Sensor | 2–3 周 |
| Phase 3 | polish（可选） | 1–2 周 |

**总计（至 Phase 2 可用）：** 约 3–5 周（单人兼职估算）。

---

## 附录 C：Phase 0 数据探索笔记（2026-08-27）

### C.1 下载

| 文件 | URL | 大小 | md5 |
|------|-----|------|-----|
| `SwipeMotionDB.rar` | https://zenodo.org/api/records/17171888/files/SwipeMotionDB.rar/content | 167.8 MiB | `ed7dead38de41a4f32f00bea74c300b0` |
| `CSD4CA.rar` | https://zenodo.org/api/records/17931118/files/CSD4CA.rar/content | 120.6 MiB | `0283426398f3d71f185e98d811a60ed3` |

Zenodo 页面提示 v1「有更新版本」：concept record `17171887` 的 latest 就是 CSD4CA v2。v2 的 alternative title 仍是 SwipeMotionDB。解压后放在 `data/raw/`（已 gitignore）。

### C.2 目录与字段

**v1** `data/raw/SwipeMotionDB/`

```
acc_data.csv      430 MB   4,572,054 行
gyro_data.csv     458 MB   4,616,641 行
magneto_data.csv   91 MB   1,012,145 行
touch_data.csv    217 MB   2,063,239 行
users_info.csv    2.6 KB   58 行
```

v1 touch 列：`session, activity, scenario, user_id, used_hand, age, gender, id_swipe, time, x, y, touch_major, touch_minor, pressure, finger_size`（另有 unnamed index）。`activity` 恒为 `facebook`。

**v2** `data/raw/CSD4CA/`：同上四个 CSV，**无** `users_info.csv`，**无** `activity` 与 unnamed index。sensor 的 `accuracy` 写成 `1.0` 这类 float。

`users_info`（仅 v1）：`user_id, age, usedHand, gender, xdpi, ydpi, density, heightPixels, widthPixels`。全部用户屏幕相同：1080×2142，density 2.625，xdpi 428.625（Pixel 6a）。利手 r=53 / l=5。

### C.3 条数与分布

| | v1 | v2（训练源） |
|--|----|----------------|
| 用户 | 58（id 4–69，有空号） | 50（去掉 8, 12, 25, 43, 44, 56, 63, 68） |
| 原始 swipe | 42,242 | 34,417（⊂ v1，user/session 全对得上） |
| session | 1: 23,208；2: 19,031；3: 3 | 1: 17,155；2: 17,262 |
| 场景 | Normal / Walking / Stressful + 编号后缀 | 仅三个主标签 |
| collapse+过滤后 | 40,521（95.9%） | **33,058**（96.0%） |
| 过滤后场景 | seated 14,557 / stress 13,786 / walking 12,178 | Normal 12,130 / Stressful 9,973 / Walking 10,955 |

v2 丢掉的 8 人含 v1 中 age=11 的 user 68。v1 里 age=13 的 user 69 在 v2 中年龄改为 31（与同 id 轨迹仍重叠），v2 无 18 岁以下标注。

v2 过滤后轨迹：点数 p50=31、时长 p50≈193 ms、位移 p50≈406 px。每人约 300–900 条 swipe。

**结论：不合并、不用 v1 补数据。** 33k 已超过 mouse 的 25k。

### C.4 时间戳与对齐

- **Touch `time`**：毫秒。例 `36949252`，同一 swipe 跨度中位数 ~190 ms。相邻 `dt` 均值约 5 ms（~200 Hz）。
- **Sensor `time`**：纳秒。例 `7.851710827e+14`。与 touch 跨度之比中位数 ≈ 9.75×10⁵（即 1e6，ns vs ms）。
- 两个时钟**原点不同**，不能 `sensor_ns/1e6` 去对 touch 的绝对 ms。按 `id_swipe` 分组后，各自减首点再插值。
- v2 中 34,417 条 swipe 几乎都有三路 sensor；v1 四模态齐全 41,965 / 42,242。
- Acc 有极少数跨度异常（v1 有一条 span ~367,320 s），转换时应丢掉 sensor 跨度 ≫ touch 时长的 swipe 的 sensor 流。
- Sensor CSV 存在空 `time` / 空 `x`（v1 acc 约 17k 空时间），跳过即可。

### C.5 `dt<=0` 不是坏轨迹

v1 相邻 touch 点：`dt>0` 1,661,535；`dt==0` 358,621；`dt<0` 359。

`dt==0` 里约 78k 为同坐标重复，约 280k 为**同一毫秒的不同坐标**（Android 把多次 MOVE 打到同一 ms）。若按「整条轨迹有 dt≤0 就丢」，会丢掉 83% 样本。

正确做法：按 `time` 排序 → 同一 `time` 只留最后一点 → 再应用点数 / 时长 / `dt>500` 过滤。之后 `dt<=0` 为 0，v2 可留 33,058 条。

### C.6 对转换脚本的含义

`convert_swipemotiondb.py` 应读 **`data/raw/CSD4CA/*.csv`**：

| 原设想 | 实际 |
|--------|------|
| `segment_swipes`（down/move/up） | 不需要；用 `id_swipe` |
| 估计 4.5k–15k 条 | 直接用 v2 的 34k |
| 合并 CSD4CA | 禁止；那就是 v2 本身 |
| 按绝对时间融合 sensor | 按 `id_swipe` + 相对时间 |
| `area` 字段 | 用 `touch_major`（中位数 ~176 px）或 `finger_size`（~0.076） |

`deviceProfile` 可写死：`pixel-6a`，1080×2142，xdpi 428.625。

---

*文档版本：1.1 | 最后更新：2026-08-27*
