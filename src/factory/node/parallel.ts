/**
 * Data-parallel transformer training on several CPU cores (Node only).
 *
 * Every mini-batch is split over replica threads (trainingWorker.ts). Each replica runs the
 * forward and backward passes of its windows and writes the gradients into its block of shared
 * memory; the coordinator adds them in replica order, takes the optimizer step (learning-rate
 * schedule, clipping and optimizer state stay in one place) and writes the new parameters into
 * shared memory for the next batch. Evaluation is split the same way and combined into exactly
 * the quantities `model.evaluate` reports.
 *
 * - Windows, batches, learning-rate steps and the epoch plan (stride alignment and shuffling,
 *   drawn from the coordinator's random stream) are the same as in single-thread training.
 * - Dropout and DropConnect noise comes from each replica's own seeded stream, so results differ
 *   from single-thread training (as they would with any other seed) but are reproducible for a
 *   given seed and number of workers. Without dropout the result equals single-thread training up
 *   to floating-point summation order.
 */

import { Worker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import {
  ProNeuralLM,
  type EvaluationResult,
  type EvaluationSums,
  type RandomState
} from '../../lib/ProNeuralLM';
import {
  TransformerLM,
  addWindowStats,
  emptyWindowStats,
  type TrainingWindows,
  type WindowStats
} from '../../lib/TransformerLM';
import {
  inProcessBackend,
  type TrainingBackend,
  type TrainingBackendFactory,
  type TrainingBackendSetup
} from '../pipeline';
import type { ReplicaInit, ReplicaRequest } from './trainingWorker';

/** Cores available to this process (at most 8: beyond that the coordinator becomes the limit). */
export function defaultWorkerCount(): number {
  const cores =
    typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
  return Math.max(1, Math.min(8, cores));
}

/** `--workers` value: a positive integer, or "auto" for defaultWorkerCount(). */
export function parseWorkerCount(value: string | undefined): number {
  if (value === undefined || value === 'auto') return defaultWorkerCount();
  const n = Math.floor(Number(value));
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`--workers must be "auto" or a number ≥ 1 (got ${value})`);
  }
  return n;
}

/** Split `items` into at most `parts` contiguous chunks whose sizes differ by at most one. */
export function splitEvenly<T>(items: T[], parts: number): T[][] {
  const n = Math.max(1, Math.min(parts, items.length));
  const chunks: T[][] = [];
  let start = 0;
  for (let k = 0; k < n; k++) {
    const size = Math.floor(items.length / n) + (k < items.length % n ? 1 : 0);
    chunks.push(items.slice(start, start + size));
    start += size;
  }
  return chunks;
}

/**
 * Start a replica thread. It registers tsx's CommonJS loader before loading the TypeScript
 * sources, which works on every supported Node version, with or without a loader in this thread.
 */
function startReplicaThread(init: ReplicaInit): Worker {
  const require = createRequire(import.meta.url);
  const loader = require.resolve('tsx/cjs');
  const entry = fileURLToPath(new URL('./trainingWorker.ts', import.meta.url));
  const code = `require(${JSON.stringify(loader)}); require(${JSON.stringify(entry)});`;
  return new Worker(code, { eval: true, workerData: init });
}

type ReplicaReply = { nl: string; id?: number; message?: string } & Record<string, unknown>;

class Replica {
  private waiting: {
    kind: string;
    id?: number;
    resolve: (m: ReplicaReply) => void;
    reject: (e: Error) => void;
  } | null = null;
  private failure: Error | null = null;

  constructor(
    readonly index: number,
    private readonly thread: Worker,
    readonly grads: Float64Array
  ) {
    thread.on('message', (m: ReplicaReply) => {
      const w = this.waiting;
      if (m.nl === 'error') {
        this.fail(new Error(`Training worker ${index} failed: ${m.message ?? 'unknown error'}`));
      } else if (w && m.nl === w.kind && (w.id === undefined || m.id === w.id)) {
        this.waiting = null;
        w.resolve(m);
      }
    });
    thread.on('error', (error) => this.fail(error));
    thread.on('exit', (code) => {
      if (code !== 0) this.fail(new Error(`Training worker ${index} exited with code ${code}`));
    });
  }

  private fail(error: Error): void {
    this.failure ??= error;
    const w = this.waiting;
    this.waiting = null;
    w?.reject(error);
  }

  /** Wait for the next message of the given kind (and id). */
  private next(kind: string, id?: number): Promise<ReplicaReply> {
    if (this.failure) return Promise.reject(this.failure);
    return new Promise((resolve, reject) => {
      this.waiting = { kind, id, resolve, reject };
    });
  }

  ready(): Promise<ReplicaReply> {
    return this.next('ready');
  }

  request(message: ReplicaRequest, kind: string): Promise<ReplicaReply> {
    const reply = this.next(kind, message.id);
    if (!this.failure) this.thread.postMessage(message);
    return reply;
  }

  async stop(): Promise<void> {
    await this.thread.terminate();
  }
}

export interface DataParallelOptions {
  workers: number;
}

export class DataParallelTrainer implements TrainingBackend {
  readonly description: string;
  private requestId = 0;
  private version = 0;
  private readonly validationWindows: number;

  private constructor(
    private readonly model: TransformerLM,
    private readonly data: TrainingWindows | null,
    private readonly replicas: Replica[],
    private readonly params: Float64Array,
    private readonly control: Int32Array,
    validationText: string
  ) {
    this.description = `${replicas.length} worker threads`;
    this.validationWindows = validationText ? model.countTrainingWindows(validationText) : 0;
  }

  /**
   * Start `options.workers` replicas of `setup.model`. An empty training text gives an
   * evaluation-only pool.
   */
  static async start(
    setup: TrainingBackendSetup & { model: TransformerLM },
    options: DataParallelOptions
  ): Promise<DataParallelTrainer> {
    const { model, trainText, validationText, trainMask, validationMask, seed, randomStates } =
      setup;
    const data = trainText ? model.prepareTrainingWindows(trainText, trainMask) : null;
    const json = model.toJSON() as unknown as Record<string, unknown>;
    const paramsBuffer = new SharedArrayBuffer(model.parameterCount() * 8);
    const workers = Math.max(1, Math.floor(options.workers));
    if (randomStates && randomStates.length !== workers) {
      throw new Error(
        `The checkpoint was made with ${randomStates.length} workers: resume with --workers ${randomStates.length}`
      );
    }
    const controlBuffer = new SharedArrayBuffer(4 * (1 + workers));
    const gradSize = model.gradientLayoutSize();
    const replicas = Array.from({ length: workers }, (_, k) => {
      const grads = new SharedArrayBuffer(gradSize * 8);
      const thread = startReplicaThread({
        model: json,
        trainText,
        validationText,
        ...(trainMask ? { trainMask } : {}),
        ...(validationMask ? { validationMask } : {}),
        seed: (Math.imul(seed >>> 0, 0x9e3779b1) + k + 1) >>> 0,
        ...(randomStates ? { randomState: randomStates[k] } : {}),
        index: k,
        params: paramsBuffer,
        grads,
        control: controlBuffer
      });
      return new Replica(k, thread, new Float64Array(grads));
    });
    try {
      await Promise.all(replicas.map((r) => r.ready()));
    } catch (error) {
      await Promise.all(replicas.map((r) => r.stop()));
      throw error;
    }
    return new DataParallelTrainer(
      model,
      data,
      replicas,
      new Float64Array(paramsBuffer),
      new Int32Array(controlBuffer),
      validationText
    );
  }

  /** Publish the coordinator's parameters to the replicas (read before their next request). */
  private publishParameters(): void {
    this.model.writeParameters(this.params);
    Atomics.store(this.control, 0, ++this.version);
  }

  async trainEpoch(
    epoch: number,
    onProgress?: (done: number, total: number, epoch: number) => void
  ): Promise<{ loss: number }> {
    const model = this.model;
    if (!this.data) throw new Error('This worker pool has no training text');
    const items = model.planEpoch(this.data);
    const batchSize = Math.max(1, Math.floor(model.getTransformerInfo().batchSize));
    const total = emptyWindowStats();
    // The replicas start from the model they were created with, which may since have changed.
    this.publishParameters();

    // Batches as in single-thread training: up to and including the `batchSize`-th window that
    // has something to learn (windows without a target do not count).
    let next = 0;
    while (next < items.length) {
      const batch: number[] = [];
      let learning = 0;
      while (next < items.length && learning < batchSize) {
        const item = items[next++];
        batch.push(item.index);
        if (model.windowLearns(item, this.data.lossMask)) learning++;
      }
      const shares = splitEvenly(batch, this.replicas.length);
      const replies = await Promise.all(
        shares.map((indices, k) =>
          this.replicas[k].request({ nl: 'batch', id: ++this.requestId, indices }, 'grads')
        )
      );
      let learned = 0;
      replies.forEach((reply, k) => {
        // Pairs with the replica's Atomics.store: its gradient writes are visible from here on.
        if (Atomics.load(this.control, 1 + k) !== reply.id) {
          throw new Error(`Training worker ${k} answered out of order`);
        }
        const stats = reply.stats as WindowStats;
        model.addGradients(this.replicas[k].grads, reply.tokens as Int32Array);
        addWindowStats(total, stats);
        learned += stats.learned;
      });
      if (learned > 0) model.applyGradientStep(learned);
      this.publishParameters();
      onProgress?.(next, items.length, epoch);
    }

    const loss = total.masked > 0 ? total.loss / total.masked : 0;
    model.recordEpoch(loss, total.correct / Math.max(1, items.length));
    return { loss };
  }

  async evaluate(): Promise<EvaluationResult> {
    this.publishParameters();
    const indices = Array.from({ length: this.validationWindows }, (_, i) => i);
    const ranges = splitEvenly(indices, this.replicas.length).filter((r) => r.length > 0);
    const parts = await Promise.all(
      ranges.map((range, k) =>
        this.replicas[k].request(
          { nl: 'eval', id: ++this.requestId, start: range[0], end: range[range.length - 1] + 1 },
          'sums'
        )
      )
    );
    return ProNeuralLM.evaluationResult(parts.map((p) => p.sums as EvaluationSums));
  }

  /** Positions of the replicas' random streams (for checkpoints). */
  async randomStates(): Promise<RandomState[]> {
    const replies = await Promise.all(
      this.replicas.map((r) => r.request({ nl: 'rng', id: ++this.requestId }, 'rng'))
    );
    return replies.map((reply) => reply.state as RandomState);
  }

  async dispose(): Promise<void> {
    await Promise.all(this.replicas.map((r) => r.stop()));
  }
}

/**
 * Backend factory for the model factory: data-parallel for transformers when more than one worker
 * is requested, in-process otherwise.
 */
export function dataParallelBackend(options: DataParallelOptions): TrainingBackendFactory {
  return async (setup) =>
    setup.model instanceof TransformerLM && options.workers > 1
      ? DataParallelTrainer.start({ ...setup, model: setup.model }, options)
      : inProcessBackend(setup);
}
