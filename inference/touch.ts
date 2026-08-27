import ort from 'onnxruntime-node';
import { join } from 'path';

import { ModelType } from './config.ts';
import { model_config, touch_model_config } from './config.ts';
import { sampleFromMDN, paramsSize } from './util.ts';
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

const ARRIVE_PX = 3;

export interface TouchGenerateOptions {
    maxSteps?: number;
    minStepPx?: number;
    maxStepPx?: number;
    avgStepPx?: number;
}

function applyNoBacktrackStep(
    remX: number,
    remY: number,
    dist: number,
    dx: number,
    dy: number,
    minStepPx: number,
): { dx: number; dy: number } {
    if (dist < 1) {
        return { dx, dy };
    }
    const tx = remX / dist;
    const ty = remY / dist;
    if (dx * tx + dy * ty >= 0) {
        return { dx, dy };
    }
    const mag = Math.max(Math.hypot(dx, dy), minStepPx);
    return { dx: tx * mag, dy: ty * mag };
}

function clampStepMag(dx: number, dy: number, maxStepPx: number): { dx: number; dy: number } {
    const mag = Math.hypot(dx, dy);
    if (mag > maxStepPx && maxStepPx > 0) {
        const scale = maxStepPx / mag;
        return { dx: dx * scale, dy: dy * scale };
    }
    return { dx, dy };
}

async function generateTouchPath(
    session: ort.InferenceSession,
    start: Position,
    end: Position,
    options: TouchGenerateOptions = {},
): Promise<Step[]> {
    const {
        maxSteps = 500,
        minStepPx = 3,
        maxStepPx = 35,
        avgStepPx = 13,
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
        if (dist < ARRIVE_PX) break;

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

        ({ dx, dy } = applyNoBacktrackStep(remX, remY, dist, dx, dy, minStepPx));
        ({ dx, dy } = clampStepMag(dx, dy, maxStepPx));

        currX += dx;
        currY += dy;
        elapsedMs += dtStep;
        path.push({ x: Math.round(currX), y: Math.round(currY), t: elapsedMs });

        dxPrev = dx;
        dyPrev = dy;
        dtPrev = dtStep;
        lastDtStep = dtStep;
    }

    if (Math.hypot(path[path.length - 1].x - end.x, path[path.length - 1].y - end.y) >= ARRIVE_PX) {
        elapsedMs += lastDtStep;
        path.push({ x: Math.round(end.x), y: Math.round(end.y), t: elapsedMs });
    }

    return path;
}

export async function touchSteps(
    start: Position,
    end: Position,
    type?: ModelType,
    options?: TouchGenerateOptions,
): Promise<Step[]> {
    const session = await getTouchSession(type);
    return generateTouchPath(session, start, end, options);
}
