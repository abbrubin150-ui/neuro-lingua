/**
 * BrainEngine - Core module for brain history and autonomous behavior
 *
 * This module tracks the "life" of the neural model through events:
 * - Training runs (increases stability, tokens seen)
 * - Generation runs (affects creativity)
 * - Feeding new corpus (adds vocabulary, changes mood)
 * - Idle ticks (natural decay over time)
 *
 * The brain has a mood system that responds to these events in predictable ways,
 * but NEVER performs heavy operations autonomously - only suggests actions via UI.
 */

import { StorageManager } from './storage';

// ============================================================================
// Types
// ============================================================================

/**
 * Mood states representing the brain's current disposition
 */
export type Mood =
  | 'CALM' // Balanced, stable state
  | 'FOCUSED' // High stability + good creativity
  | 'AGITATED' // Low stability, needs training
  | 'DREAMY' // High creativity, low stability
  | 'BURNT_OUT'; // Low creativity + high stability (overtrained)

/**
 * Core brain statistics and state
 */
export interface BrainStats {
  id: string; // Linked to model artifact/name
  label: string; // Pet name (e.g., "Avi-LM-Pet #1")
  createdAt: string; // ISO timestamp
  updatedAt: string; // ISO timestamp

  // Training metrics
  totalTrainSteps: number; // Total epochs completed
  totalTokensSeen: number; // Total tokens processed in training
  vocabSize: number; // Current vocabulary size

  // Brain vitals (0-100)
  creativity: number; // Ability to generate diverse outputs
  stability: number; // Training stability/convergence

  // State
  mood: Mood; // Current mood
  lastFeedSummary: string | null; // Summary of last feeding
  autonomyEnabled: boolean; // Whether autonomous mode is active

  // Diary - log of major events
  diary: DiaryEntry[];
}

/**
 * Diary entry for the brain's "memory"
 */
export interface DiaryEntry {
  timestamp: number;
  type: 'TRAIN' | 'GEN' | 'FEED' | 'MOOD_SHIFT' | 'SUGGESTION';
  message: string;
  metadata?: Record<string, any>;
}

/**
 * Event types that can affect the brain
 */
export type BrainEventType =
  | 'TRAIN_RUN' // Training epoch completed
  | 'GEN_RUN' // Text generation performed
  | 'FEED' // New corpus fed to the model
  | 'IDLE_TICK' // Time passing without activity
  | 'MOOD_OVERRIDE'; // Manual mood change

/**
 * Brain event payload
 */
export interface BrainEvent {
  type: BrainEventType;
  timestamp: number;
  payload?: any;
}

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY_PREFIX = 'nl_brain_state_';
const MAX_DIARY_ENTRIES = 100; // Keep diary manageable

// Thresholds for mood determination
const MOOD_THRESHOLDS = {
  FOCUSED: { stability: 70, creativity: 60 },
  DREAMY: { creativity: 80, stability: 50 },
  AGITATED: { stability: 30 },
  BURNT_OUT: { creativity: 20, stability: 60 }
};

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Create a new brain state with default values
 */
export function createBrain(id: string, label?: string): BrainStats {
  const now = new Date().toISOString();
  return {
    id,
    label: label || `Brain-${id.slice(0, 8)}`,
    createdAt: now,
    updatedAt: now,
    totalTrainSteps: 0,
    totalTokensSeen: 0,
    vocabSize: 0,
    creativity: 20,
    stability: 50,
    mood: 'CALM',
    lastFeedSummary: null,
    autonomyEnabled: false,
    diary: []
  };
}

/**
 * Load brain state from localStorage
 */
export function loadBrain(id: string): BrainStats {
  const saved = StorageManager.get<BrainStats | null>(STORAGE_KEY_PREFIX + id, null);

  if (saved) {
    return saved;
  }

  return createBrain(id);
}

/**
 * Save brain state to localStorage
 */
export function saveBrain(state: BrainStats): boolean {
  return StorageManager.set(STORAGE_KEY_PREFIX + state.id, state);
}

/**
 * Pure reducer function - applies an event to brain state
 * This is the ONLY function that modifies brain state.
 * All changes are deterministic and traceable.
 */
export function reduceBrain(state: BrainStats, event: BrainEvent): BrainStats {
  // Clone state with updated timestamp
  const next: BrainStats = {
    ...state,
    updatedAt: new Date(event.timestamp).toISOString(),
    diary: [...state.diary]
  };

  switch (event.type) {
    case 'TRAIN_RUN': {
      const { steps, tokens, vocabDelta = 0, avgLoss } = event.payload;

      next.totalTrainSteps += steps;
      next.totalTokensSeen += tokens;
      next.vocabSize += vocabDelta;

      // Creativity increases with significant loss improvements
      if (avgLoss < 1.2) {
        next.creativity = Math.min(100, next.creativity + 3);
      }

      // Stability increases with training
      next.stability = Math.min(100, next.stability + 2);

      // Add to diary
      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'TRAIN',
        message: `Trained for ${steps} steps, saw ${tokens} tokens (avg loss: ${avgLoss.toFixed(3)})`,
        metadata: { steps, tokens, avgLoss }
      });

      break;
    }

    case 'GEN_RUN': {
      const { diversityScore = 0.5, tokensGenerated = 0 } = event.payload;

      // High diversity generation increases creativity
      if (diversityScore > 0.7) {
        next.creativity = Math.min(100, next.creativity + 1);
        next.stability = Math.max(0, next.stability - 1);
      }

      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'GEN',
        message: `Generated ${tokensGenerated} tokens (diversity: ${diversityScore.toFixed(2)})`,
        metadata: { diversityScore, tokensGenerated }
      });

      break;
    }

    case 'FEED': {
      const { newWordsCount = 0, heavinessScore = 0.5, summary = '' } = event.payload;

      next.vocabSize += newWordsCount;
      next.lastFeedSummary = summary;

      // New vocabulary increases creativity
      next.creativity = Math.min(100, next.creativity + newWordsCount * 0.2);

      // Heavy/complex text reduces stability temporarily
      next.stability = Math.max(0, next.stability - heavinessScore * 3);

      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'FEED',
        message: `Fed new corpus: ${summary} (${newWordsCount} new words)`,
        metadata: { newWordsCount, heavinessScore, summary }
      });

      break;
    }

    case 'IDLE_TICK': {
      // Natural decay over time - "getting bored"
      next.creativity = Math.max(0, next.creativity - 0.3);
      next.stability = Math.max(0, next.stability - 0.1);
      break;
    }

    case 'MOOD_OVERRIDE': {
      const { mood } = event.payload;
      next.mood = mood;

      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'MOOD_SHIFT',
        message: `Mood manually set to ${mood}`,
        metadata: { mood }
      });

      break;
    }

    default:
      return state;
  }

  // Determine new mood based on vitals
  const oldMood = next.mood;
  next.mood = determineMood(next);

  // Log mood changes
  if (next.mood !== oldMood && event.type !== 'MOOD_OVERRIDE') {
    addDiaryEntry(next, {
      timestamp: event.timestamp,
      type: 'MOOD_SHIFT',
      message: `Mood shifted from ${oldMood} to ${next.mood}`,
      metadata: { oldMood, newMood: next.mood }
    });
  }

  return next;
}

/**
 * Determine mood based on creativity and stability scores
 */
function determineMood(state: BrainStats): Mood {
  const { creativity, stability } = state;

  // Focused: high stability + good creativity
  if (
    stability > MOOD_THRESHOLDS.FOCUSED.stability &&
    creativity > MOOD_THRESHOLDS.FOCUSED.creativity
  ) {
    return 'FOCUSED';
  }

  // Dreamy: high creativity, low stability
  if (
    creativity > MOOD_THRESHOLDS.DREAMY.creativity &&
    stability < MOOD_THRESHOLDS.DREAMY.stability
  ) {
    return 'DREAMY';
  }

  // Agitated: very low stability
  if (stability < MOOD_THRESHOLDS.AGITATED.stability) {
    return 'AGITATED';
  }

  // Burnt out: low creativity + high stability (overtrained)
  if (
    creativity < MOOD_THRESHOLDS.BURNT_OUT.creativity &&
    stability > MOOD_THRESHOLDS.BURNT_OUT.stability
  ) {
    return 'BURNT_OUT';
  }

  // Default: calm
  return 'CALM';
}

/**
 * Add entry to diary, keeping it under MAX_DIARY_ENTRIES
 */
function addDiaryEntry(state: BrainStats, entry: DiaryEntry): void {
  state.diary.push(entry);

  // Keep diary size manageable
  if (state.diary.length > MAX_DIARY_ENTRIES) {
    state.diary.shift();
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate diversity score from generation parameters
 */
export function calculateDiversityScore(temperature: number, topK?: number, topP?: number): number {
  let score = 0;

  // Temperature contribution (0-1 normalized)
  score += Math.min(temperature / 2.0, 1.0) * 0.5;

  // Top-k contribution
  if (topK !== undefined) {
    score += (Math.min(topK, 100) / 100) * 0.25;
  }

  // Top-p contribution
  if (topP !== undefined) {
    score += topP * 0.25;
  }

  return Math.min(score, 1.0);
}

/**
 * Calculate "heaviness" of text - complex/formal text scores higher
 */
export function calculateHeavinessScore(text: string): number {
  const sentences = text.split(/[.!?]+/);
  const avgSentenceLength =
    sentences.reduce((sum, s) => sum + s.trim().split(/\s+/).length, 0) / sentences.length;

  // Longer sentences = heavier
  const lengthScore = Math.min(avgSentenceLength / 30, 1.0);

  // More punctuation = heavier (formal/complex)
  const punctuationCount = (text.match(/[;:,—–()]/g) || []).length;
  const punctuationScore = Math.min((punctuationCount / text.length) * 50, 1.0);

  return lengthScore * 0.7 + punctuationScore * 0.3;
}

/**
 * Get a descriptive status message for the current brain state
 */
export function getBrainStatusMessage(state: BrainStats): string {
  const { mood, creativity, stability } = state;

  switch (mood) {
    case 'FOCUSED':
      return `Feeling sharp and creative! (Creativity: ${creativity.toFixed(0)}, Stability: ${stability.toFixed(0)})`;
    case 'DREAMY':
      return `Highly creative but a bit unstable. Time for some training? (Creativity: ${creativity.toFixed(0)}, Stability: ${stability.toFixed(0)})`;
    case 'AGITATED':
      return `Feeling unstable and need guidance. Please train me! (Stability: ${stability.toFixed(0)})`;
    case 'BURNT_OUT':
      return `Overtrained and lacking creativity. Feed me something new! (Creativity: ${creativity.toFixed(0)})`;
    default: // CALM
      return `In a balanced, calm state. (Creativity: ${creativity.toFixed(0)}, Stability: ${stability.toFixed(0)})`;
  }
}

/**
 * Check if brain needs attention (for autonomous suggestions)
 */
export function checkBrainNeeds(state: BrainStats): {
  needsFeeding: boolean;
  needsTraining: boolean;
  message: string | null;
} {
  const now = Date.now();
  const lastUpdate = new Date(state.updatedAt).getTime();
  const minutesSinceUpdate = (now - lastUpdate) / 60000;

  // Needs feeding if creativity is low or idle for long
  const needsFeeding = state.creativity < 40 || minutesSinceUpdate > 15;

  // Needs training if stability is low
  const needsTraining = state.stability < 30;

  let message: string | null = null;

  if (needsFeeding && needsTraining) {
    message = "I'm hungry and unstable! Please feed me new text AND run some training.";
  } else if (needsFeeding) {
    message = "I'm craving something new to learn. Feed me interesting text!";
  } else if (needsTraining) {
    message = 'Feeling unstable. A short training session would help.';
  }

  return { needsFeeding, needsTraining, message };
}
