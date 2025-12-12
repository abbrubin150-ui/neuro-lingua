import { describe, it, expect, beforeEach } from 'vitest';

import {
  calculateDiversityScore,
  calculateHeavinessScore,
  checkBrainNeeds,
  createBrain,
  getBrainStatusMessage,
  loadBrain,
  reduceBrain,
  saveBrain
} from '../../src/lib/BrainEngine';
import type { BrainStats } from '../../src/lib/BrainEngine';

describe('BrainEngine quality', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('persists and reloads brain state through StorageManager', () => {
    const brain = createBrain('abc123456789', 'QA Brain');
    brain.creativity = 44;
    brain.stability = 61;

    expect(saveBrain(brain)).toBe(true);

    const stored = localStorage.getItem('nl_brain_state_abc123456789');
    expect(stored).toBeTruthy();

    const loaded = loadBrain('abc123456789');
    expect(loaded).toMatchObject({
      id: 'abc123456789',
      label: 'QA Brain',
      creativity: 44,
      stability: 61
    });
  });

  it('applies training events, updates vitals, and logs diary entries', () => {
    const brain = createBrain('train');

    const next = reduceBrain(brain, {
      type: 'TRAIN_RUN',
      timestamp: Date.now(),
      payload: { steps: 5, tokens: 120, vocabDelta: 8, avgLoss: 1.0 }
    });

    expect(next.totalTrainSteps).toBe(5);
    expect(next.totalTokensSeen).toBe(120);
    expect(next.vocabSize).toBe(8);
    expect(next.creativity).toBeCloseTo(23, 5);
    expect(next.stability).toBeCloseTo(52, 5);
    expect(next.diary[next.diary.length - 1].type).toBe('TRAIN');
  });

  it('promotes focused mood when creativity and stability are high', () => {
    const initial: BrainStats = {
      ...createBrain('mood'),
      creativity: 82,
      stability: 75,
      // v4.3: Set lastMoodChangeTime in the past to allow mood transition (hysteresis)
      lastMoodChangeTime: Date.now() - 120000 // 2 minutes ago
    };

    const result = reduceBrain(initial, { type: 'IDLE_TICK', timestamp: Date.now() });

    expect(result.mood).toBe('FOCUSED');
    expect(result.diary.some((entry) => entry.type === 'MOOD_SHIFT')).toBe(true);
    const moodEntry = result.diary.find((entry) => entry.type === 'MOOD_SHIFT');
    expect(moodEntry?.message).toContain('Mood shifted from CALM to FOCUSED');
  });

  it('handles feeding heavy corpus with creativity lift and stability tradeoff', () => {
    const brain = createBrain('feed');

    const result = reduceBrain(brain, {
      type: 'FEED',
      timestamp: Date.now(),
      payload: { newWordsCount: 10, heavinessScore: 0.8, summary: 'dense research paper' }
    });

    expect(result.vocabSize).toBe(10);
    expect(result.lastFeedSummary).toBe('dense research paper');
    expect(result.creativity).toBeCloseTo(22, 2);
    expect(result.stability).toBeCloseTo(47.6, 2);
  });

  it('caps diary length to prevent uncontrolled growth', () => {
    let state = createBrain('cap');

    for (let i = 0; i < 105; i++) {
      state = reduceBrain(state, {
        type: 'TRAIN_RUN',
        timestamp: Date.now() + i,
        payload: { steps: 1, tokens: 10, vocabDelta: 0, avgLoss: 1.5 }
      });
    }

    expect(state.diary.length).toBeLessThanOrEqual(100);
  });

  it('prioritizes needs and status messaging based on vitals and recency', () => {
    const staleTimestamp = new Date(Date.now() - 20 * 60000).toISOString();
    const brain: BrainStats = {
      ...createBrain('needs'),
      creativity: 35,
      stability: 25,
      mood: 'AGITATED',
      updatedAt: staleTimestamp
    };

    const needs = checkBrainNeeds(brain);
    // v4.3: Priority-scored needs system - stability is critical concern
    expect(needs.needsTraining).toBe(true);
    // Message now reflects the most urgent need
    expect(needs.message).not.toBeNull();

    const status = getBrainStatusMessage(brain);
    expect(status).toContain('unstable');
  });

  it('combines generation parameters into bounded diversity score and heaviness', () => {
    const diverse = calculateDiversityScore(1.6, 120, 0.95);
    expect(diverse).toBeLessThanOrEqual(1);
    expect(diverse).toBeGreaterThan(0.8);

    const heavy = calculateHeavinessScore(
      'This sentence, although brief, uses punctuation; therefore, it feels formal.'
    );
    const light = calculateHeavinessScore('Short casual text.');
    expect(heavy).toBeGreaterThan(light);
  });
});
