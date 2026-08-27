---
name: touch-trace
description: Synthesize human-like touch trajectories and synchronized sensor data.
---

Use the `touchtrace` CLI to generate human-like mobile swipe paths as JSON.

```bash
touchtrace touch-steps <fromX> <fromY> <toX> <toY>
```

This outputs a JSON array of touch points with timestamps (milliseconds):

```json
[
  { "x": 100, "y": 800, "t": 0 },
  { "x": 105, "y": 798, "t": 16.2 }
]
```

For touch + sensor bundle (when implemented):

```bash
touchtrace touch-sensor-steps <fromX> <fromY> <toX> <toY>
```

SDK usage:

```js
import { touchSteps } from 'touchtrace';

const path = await touchSteps({ x: 100, y: 800 }, { x: 540, y: 1200 });
```

Generation is not instant—it requires model inference time per step.

See [TOUCH_EXTENSION_PLAN.md](../../TOUCH_EXTENSION_PLAN.md) for the full roadmap.
