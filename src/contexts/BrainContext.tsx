import React, { createContext, useCallback, useContext } from 'react';

import type { Optimizer } from '../lib/ProNeuralLM';
import type { Architecture } from '../components/TrainingPanel';

type TrainRunStage = 'start' | 'progress' | 'complete' | 'stop';
type GenRunStage = 'start' | 'complete';

export interface TrainRunPayload {
  stage: TrainRunStage;
  timestamp: number;
  projectId?: string | null;
  runId?: string | null;
  corpusLength?: number;
  vocabSize?: number;
  epoch?: number;
  totalEpochs?: number;
  config?: {
    architecture: Architecture;
    hiddenSize: number;
    epochs: number;
    learningRate: number;
    optimizer: Optimizer;
    contextSize: number;
    useGPU: boolean;
    useAdvanced: boolean;
  };
  metrics?: {
    loss?: number;
    accuracy?: number;
    perplexity?: number;
    tokensPerSec?: number;
    trainingDurationMs?: number;
  };
}

export interface GenRunPayload {
  stage: GenRunStage;
  timestamp: number;
  prompt: string;
  sampling: {
    temperature: number;
    samplingMode: 'off' | 'topk' | 'topp';
    topK: number;
    topP: number;
    useBeamSearch: boolean;
    beamWidth: number;
    useBayesian: boolean;
  };
  result?: {
    text: string;
    confidence?: number | null;
  };
}

export type BrainEvent =
  | {
      type: 'TRAIN_RUN';
      payload: TrainRunPayload;
    }
  | {
      type: 'GEN_RUN';
      payload: GenRunPayload;
    };

interface BrainContextValue {
  dispatchBrain: (event: BrainEvent) => void;
}

const BrainContext = createContext<BrainContextValue>({
  dispatchBrain: () => {}
});

export function BrainProvider({ children, onEvent }: { children: React.ReactNode; onEvent?: (event: BrainEvent) => void }) {
  const dispatchBrain = useCallback(
    (event: BrainEvent) => {
      console.debug('[BrainEvent]', event);
      onEvent?.(event);
    },
    [onEvent]
  );

  return <BrainContext.Provider value={{ dispatchBrain }}>{children}</BrainContext.Provider>;
}

export function useBrain() {
  const context = useContext(BrainContext);
  if (!context) {
    throw new Error('useBrain must be used within a BrainProvider');
  }
  return context;
}
