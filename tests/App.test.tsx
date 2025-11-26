import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, beforeEach } from 'vitest';

import App from '../src/App';
import { DEFAULT_TRAINING_TEXT, STORAGE_KEYS } from '../src/config/constants';
import { BrainProvider } from '../src/contexts/BrainContext';
import { ProjectProvider } from '../src/contexts/ProjectContext';

const UI_SETTINGS_KEY = STORAGE_KEYS.UI_SETTINGS;
const TOKENIZER_STORAGE_KEY = STORAGE_KEYS.TOKENIZER_CONFIG;
const ONBOARDING_STORAGE_KEY = STORAGE_KEYS.ONBOARDING_DISMISSED;

// Wrapper component that includes BrainProvider and ProjectProvider
const AppWithProvider = () => (
  <BrainProvider>
    <ProjectProvider>
      <App />
    </ProjectProvider>
  </BrainProvider>
);

describe('Neuro-Lingua App UI', () => {
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
});
