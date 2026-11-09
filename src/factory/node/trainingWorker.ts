/**
 * Replica thread for data-parallel training (see parallel.ts). Holds a copy of the transformer,
 * computes gradients for its share of each mini-batch and partial sums for evaluation.
 *
 * Parameters and gradients travel through shared memory: the coordinator writes the parameters
 * into `params` after every optimizer step and then stores a new version in `control[0]`; this
 * replica writes its gradients into its own `grads` block and then stores the request id in
 * `control[1 + index]`. These atomic stores and loads order the plain reads and writes of the
 * buffers. Messages only carry window indices and statistics.
 * The thread is started by parallel.ts after registering a TypeScript loader.
 */

import { parentPort, workerData } from 'node:worker_threads';
import { TransformerLM } from '../../lib/TransformerLM';
import type { RandomState } from '../../lib/ProNeuralLM';

export interface ReplicaInit {
  model: Record<string, unknown>;
  trainText: string;
  validationText: string;
  /** Per-token loss masks (Q&A training). */
  trainMask?: number[];
  validationMask?: number[];
  seed: number;
  /** Continue this random stream instead of seeding (resuming a run). */
  randomState?: RandomState;
  /** This replica's position (its control word is control[1 + index]). */
  index: number;
  /** Shared parameters (parameterCount() float64 values). */
  params: SharedArrayBuffer;
  /** This replica's gradient block (gradientLayoutSize() float64 values). */
  grads: SharedArrayBuffer;
  /** Int32 control words: [0] parameter version, [1 + k] last request answered by replica k. */
  control: SharedArrayBuffer;
}

export type ReplicaRequest =
  | { nl: 'batch'; id: number; indices: number[] }
  | { nl: 'eval'; id: number; start: number; end: number }
  | { nl: 'rng'; id: number };

const init = workerData as ReplicaInit;
const model = TransformerLM.fromJSON(init.model);
// Each replica draws its own dropout / DropConnect noise.
if (init.randomState) model.setRandomState(init.randomState);
else model.reseed(init.seed);
const data = init.trainText ? model.prepareTrainingWindows(init.trainText, init.trainMask) : null;
const params = new Float64Array(init.params);
const grads = new Float64Array(init.grads);
const control = new Int32Array(init.control);
let version = 0;

/** Load the coordinator's parameters if they changed since the last request. */
function syncParameters(): void {
  const current = Atomics.load(control, 0);
  if (current !== version) {
    model.importParameters(params);
    version = current;
  }
}

async function handle(m: ReplicaRequest): Promise<void> {
  syncParameters();
  if (m.nl === 'batch') {
    if (!data) throw new Error('This replica has no training text');
    const stats = model.accumulateWindows(data, m.indices);
    const tokens = model.exportGradients(grads);
    Atomics.store(control, 1 + init.index, m.id);
    parentPort!.postMessage({ nl: 'grads', id: m.id, stats, tokens }, [
      tokens.buffer as ArrayBuffer
    ]);
  } else if (m.nl === 'eval') {
    const sums = await model.evaluateSums(init.validationText, m.start, m.end, init.validationMask);
    parentPort!.postMessage({ nl: 'sums', id: m.id, sums });
  } else {
    parentPort!.postMessage({ nl: 'rng', id: m.id, state: model.getRandomState() });
  }
}

// Requests are handled strictly in order (an evaluation is asynchronous).
let queue: Promise<void> = Promise.resolve();
parentPort!.on('message', (m: ReplicaRequest) => {
  queue = queue
    .then(() => handle(m))
    .catch((error: unknown) => {
      parentPort!.postMessage({
        nl: 'error',
        id: m.id,
        message: error instanceof Error ? (error.stack ?? error.message) : String(error)
      });
    });
});
parentPort!.postMessage({ nl: 'ready', parameters: model.parameterCount() });
