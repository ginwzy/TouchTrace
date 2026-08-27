import ort from 'onnxruntime-node';
import { join } from 'path';

import { ModelType } from './config.ts';
import { model_config, touch_model_config } from './config.ts';
import { sampleFromMDN, smoothPath, paramsSize } from './util.ts';
import type { Position, Step } from './index.ts';

const touchSessions: Partial<Record<ModelType, Promise<ort.InferenceSession>>> = {};

function getTouchSession(type?: ModelType) {
    const realType = type ?? ModelType.STANDARD;
    if (!touchSessions[realType]) {
        const modelPath = join(import.meta.dirname, touch_model_config.modelPaths[realType]);
        touchSessions[realType] = ort.InferenceSession.create(modelPath);
    }
    return touchSessions[realType]!;
}

function toRemainingFrame(
    dx: number,
    dy: number,
    remX: number,
    remY: number,
): { tan: number; nrm: number; rem: number } {
    const rem = Math.hypot(remX, remY);
    if (rem < 1e-6) {
        return { tan: dx, nrm: dy, rem };
    }
    const c = remX / rem;
    const s = remY / rem;
    return { tan: c * dx + s * dy, nrm: -s * dx + c * dy, rem };
}

function fromRemainingFrame(tan: number, nrm: number, remX: number, remY: number): { dx: number; dy: number } {
    const rem = Math.hypot(remX, remY);
    if (rem < 1e-6) {
        return { dx: tan, dy: nrm };
    }
    const c = remX / rem;
    const s = remY / rem;
    return { dx: c * tan - s * nrm, dy: s * tan + c * nrm };
}

interface GuidedOptions {
    maxSteps?: number;
    minStepPx?: number;
    maxStepPx?: number;
    snapPx?: number;
    avgStepPx?: number;
    smooth?: boolean;
}

function applyGuidedStep(
    cx: number,
    cy: number,
    end: Position,
    dx: number,
    dy: number,
    minStepPx: number,
    maxStepPx: number,
    snapPx: number,
): { dx: number; dy: number; snapped: boolean } {
    const dist = Math.hypot(end.x - cx, end.y - cy);
    if (dist < snapPx) {
        return { dx: end.x - cx, dy: end.y - cy, snapped: true };
    }

    const tx = (end.x - cx) / dist;
    const ty = (end.y - cy) / dist;
    let mag = Math.hypot(dx, dy);
    const dot = dx * tx + dy * ty;

    if (dot < 0 || mag < minStepPx) {
        mag = Math.min(Math.max(minStepPx, mag), maxStepPx, dist * 0.35);
        dx = tx * mag;
        dy = ty * mag;
    } else if (mag > Math.max(dist, maxStepPx)) {
        const step = Math.min(dist, maxStepPx);
        dx = tx * step;
        dy = ty * step;
    }

    return { dx, dy, snapped: false };
}

async function generateTouchPath(
    session: ort.InferenceSession,
    start: Position,
    end: Position,
    options: GuidedOptions = {},
): Promise<Step[]> {
    const {
        maxSteps = 500,
        minStepPx = 3,
        maxStepPx = 35,
        snapPx = 15,
        avgStepPx = 13,
        smooth = true,
    } = options;

    let currX = start.x;
    let currY = start.y;
    let dxPrev = 0;
    let dyPrev = 0;
    let dtPrev = 0;
    let elapsedMs = 0;
    let lastDtStep = model_config.minDelayMs;

    const path: Step[] = [{ x: Math.round(currX), y: Math.round(currY), t: elapsedMs }];
    const sequence: number[][] = [];

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const dist0 = Math.hypot(end.x - start.x, end.y - start.y);
    const stepBudget = Math.min(maxSteps, Math.max(20, Math.floor((dist0 / avgStepPx) * 3)));

    for (let step = 0; step < stepBudget; step++) {
        const dist = Math.hypot(end.x - currX, end.y - currY);
        if (dist < 3) break;

        const remX = end.x - currX;
        const remY = end.y - currY;
        const prev = toRemainingFrame(dxPrev, dyPrev, remX, remY);
        sequence.push([prev.tan, prev.nrm, dtPrev, prev.rem, 0]);

        const seqLen = sequence.length;
        const inputData = new Float32Array(seqLen * model_config.inputDims);
        for (let i = 0; i < seqLen; i++) {
            inputData.set(sequence[i], i * model_config.inputDims);
        }

        const tensor = new ort.Tensor('float32', inputData, [1, seqLen, model_config.inputDims]);
        const results = await session.run({ [inputName]: tensor });
        const outputData = results[outputName].data as Float32Array;
        const lastStepParams = outputData.slice(outputData.length - paramsSize);
        const sampled = sampleFromMDN(lastStepParams);
        let { dx, dy } = fromRemainingFrame(sampled.dx, sampled.dy, remX, remY);
        const dt = sampled.dt;
        const dtStep = dt > 0 ? dt : model_config.minDelayMs;

        const guided = applyGuidedStep(currX, currY, end, dx, dy, minStepPx, maxStepPx, snapPx);
        dx = guided.dx;
        dy = guided.dy;

        if (guided.snapped) {
            currX = end.x;
            currY = end.y;
            elapsedMs += Math.max(dtStep, model_config.minDelayMs);
            path.push({ x: Math.round(currX), y: Math.round(currY), t: elapsedMs });
            break;
        }

        currX += dx;
        currY += dy;
        elapsedMs += dtStep;
        path.push({ x: Math.round(currX), y: Math.round(currY), t: elapsedMs });

        dxPrev = dx;
        dyPrev = dy;
        dtPrev = dtStep;
        lastDtStep = dtStep;
    }

    elapsedMs += lastDtStep;
    path.push({ x: Math.round(end.x), y: Math.round(end.y), t: elapsedMs });

    return smooth ? smoothPath(path, 7) : path;
}

export async function touchSteps(
    start: Position,
    end: Position,
    type?: ModelType,
    options?: GuidedOptions,
): Promise<Step[]> {
    const session = await getTouchSession(type);
    return generateTouchPath(session, start, end, options);
}
