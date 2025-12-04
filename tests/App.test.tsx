import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, beforeEach, beforeAll, vi } from 'vitest';

import App from '../src/App';
import { DEFAULT_TRAINING_TEXT, STORAGE_KEYS } from '../src/config/constants';
import { BrainProvider } from '../src/contexts/BrainContext';
import { ProjectProvider } from '../src/contexts/ProjectContext';
import type { TrainingConfig } from '../src/types/project';
import { createDecisionLedger, createProject, createRun } from '../src/types/project';

const UI_SETTINGS_KEY = STORAGE_KEYS.UI_SETTINGS;
const TOKENIZER_STORAGE_KEY = STORAGE_KEYS.TOKENIZER_CONFIG;
const ONBOARDING_STORAGE_KEY = STORAGE_KEYS.ONBOARDING_DISMISSED;
const PROJECTS_KEY = 'neuro-lingua-projects-v1';
const RUNS_KEY = 'neuro-lingua-runs-v1';
const ACTIVE_PROJECT_KEY = 'neuro-lingua-active-project-v1';
const ACTIVE_RUN_KEY = 'neuro-lingua-active-run-v1';

function createTestConfig(): TrainingConfig {
  return {
    architecture: 'feedforward',
    hiddenSize: 64,
    epochs: 1,
    learningRate: 0.08,
    optimizer: 'momentum',
    momentum: 0.9,
    dropout: 0.1,
    contextSize: 3,
    seed: 42,
    tokenizerConfig: { mode: 'unicode' },
    useAdvanced: false,
    useGPU: false
  };
}

// Wrapper component that includes BrainProvider and ProjectProvider
const AppWithProvider = () => (
  <BrainProvider>
    <ProjectProvider>
      <App />
    </ProjectProvider>
  </BrainProvider>
);

describe('Neuro-Lingua App UI', () => {
  beforeAll(() => {
    const canvas2DContext = {
      fillRect: vi.fn(),
      clearRect: vi.fn(),
      getImageData: vi.fn(),
      putImageData: vi.fn(),
      createImageData: vi.fn(),
      setTransform: vi.fn(),
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      stroke: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      rotate: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      measureText: vi.fn(() => ({ width: 0, actualBoundingBoxAscent: 0, actualBoundingBoxDescent: 0 })),
      transform: vi.fn(),
      setLineDash: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      getExtension: vi.fn()
    } as unknown as CanvasRenderingContext2D;

    const webgpuContext = {
      __brand: 'GPUCanvasContext',
      canvas: document.createElement('canvas'),
      configure: vi.fn(),
      unconfigure: vi.fn(),
      getConfiguration: vi.fn(),
      getCurrentTexture: vi.fn(() => ({
        createView: vi.fn()
      }))
    } as unknown as GPUCanvasContext;

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
      ((contextId: string) => {
        if (contextId === 'webgpu') {
          return webgpuContext;
        }
        return canvas2DContext;
      }) as unknown as typeof HTMLCanvasElement.prototype.getContext
    );
  });

  beforeEach(() => {
    localStorage.clear();
  });

  it('restores saved settings on mount', async () => {
    const saved = {
      trainingText: 'Saved corpus sample',
      hiddenSize: 96,
      epochs: 12,
      lr: 0.05,
      optimizer: 'adam',
      momentum: 0.8,
      dropout: 0.2,
      contextSize: 4,
      temperature: 0.95,
      topK: 15,
      topP: 0.75,
      samplingMode: 'topk' as const,
      seed: 777,
      resume: false,
      tokenizerConfig: { mode: 'ascii' }
    };
    localStorage.setItem(UI_SETTINGS_KEY, JSON.stringify(saved));

    render(<AppWithProvider />);

    await waitFor(() => {
      expect(screen.getByLabelText('Hidden size')).toHaveValue(96);
      expect(screen.getByLabelText('Training corpus')).toHaveValue(DEFAULT_TRAINING_TEXT);
      expect(screen.getByLabelText('Tokenizer mode')).toHaveValue('ascii');
      const stored = localStorage.getItem(UI_SETTINGS_KEY);
      expect(stored).toBeTruthy();
      expect(JSON.parse(stored!).trainingText).toBeUndefined();
    });
  });

  it('persists updated hyperparameters to localStorage', async () => {
    render(<AppWithProvider />);

    fireEvent.change(screen.getByLabelText('Hidden size'), { target: { value: '72' } });
    fireEvent.change(screen.getByLabelText('Training corpus'), {
      target: { value: 'Hello persistent world' }
    });

    await waitFor(() => {
      const raw = localStorage.getItem(UI_SETTINGS_KEY);
      expect(raw).toBeTruthy();
      const parsed = JSON.parse(raw!);
      expect(parsed.hiddenSize).toBe(72);
      expect(parsed.trainingText).toBeUndefined();
    });
  });

  it('allows onboarding panel dismissal and records preference', async () => {
    render(<AppWithProvider />);

    expect(screen.getByText(/welcome!/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Got it'));

    await waitFor(() => {
      expect(screen.queryByText(/welcome!/i)).not.toBeInTheDocument();
      expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe('true');
    });
  });

  it('saves tokenizer selections, including custom patterns', async () => {
    render(<AppWithProvider />);

    fireEvent.change(screen.getByLabelText('Tokenizer mode'), { target: { value: 'ascii' } });

    await waitFor(() => {
      const raw = localStorage.getItem(TOKENIZER_STORAGE_KEY);
      expect(raw).toBeTruthy();
      expect(JSON.parse(raw!).mode).toBe('ascii');
    });

    fireEvent.change(screen.getByLabelText('Tokenizer mode'), { target: { value: 'custom' } });
    fireEvent.change(screen.getByLabelText('Custom tokenizer pattern'), {
      target: { value: '[^a-z]+' }
    });

    await waitFor(() => {
      const raw = localStorage.getItem(TOKENIZER_STORAGE_KEY);
      expect(raw).toBeTruthy();
      const parsed = JSON.parse(raw!);
      expect(parsed.mode).toBe('custom');
      expect(parsed.pattern).toBe('[^a-z]+');
    });
  });

  function seedGovernedRun(ledger: ReturnType<typeof createDecisionLedger>) {
    const project = createProject('Governed Project', 'Compliance enforced');
    const run = createRun(project.id, 'Governed Run', createTestConfig(), 'sample corpus', ledger);

    project.runIds.push(run.id);

    localStorage.setItem(PROJECTS_KEY, JSON.stringify([project]));
    localStorage.setItem(RUNS_KEY, JSON.stringify([run]));
    localStorage.setItem(ACTIVE_PROJECT_KEY, JSON.stringify(project.id));
    localStorage.setItem(ACTIVE_RUN_KEY, JSON.stringify(run.id));

    return { projectId: project.id, runId: run.id };
  }

  it('blocks training when decision ledger status is ESCALATE', async () => {
    const ledger = createDecisionLedger('', '');
    seedGovernedRun(ledger);

    render(<AppWithProvider />);

    fireEvent.click(screen.getByRole('button', { name: /start model training/i }));

    await waitFor(() => {
      expect(
        screen.getByText('🚨 Cannot train: Decision Ledger requires review (missing rationale/witness).')
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /start model training/i })).toHaveTextContent(
        '🚀 Train model'
      );
    });
  });

  it('blocks training when decision ledger status is HOLD', async () => {
    const ledger = createDecisionLedger('Expired authorization', 'observer', '2000-01-01T00:00:00Z');
    seedGovernedRun(ledger);

    render(<AppWithProvider />);

    fireEvent.click(screen.getByRole('button', { name: /start model training/i }));

    await waitFor(() => {
      expect(
        screen.getByText('⏸️ Cannot train: Run is on HOLD (expired or paused).')
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /start model training/i })).toHaveTextContent(
        '🚀 Train model'
      );
    });
  });
});
