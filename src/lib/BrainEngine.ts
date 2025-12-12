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
 *
 * v4.3 Enhancements:
 * - Adaptive decay rates based on mood and activity history
 * - Priority-scored need detection
 * - Comfort zone tracking
 * - Activity pattern analysis
 * - Enhanced mood transitions with hysteresis
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
 * Activity pattern tracking
 */
export interface ActivityPattern {
  lastTrainTime: number;
  lastGenTime: number;
  lastFeedTime: number;
  trainFrequency: number; // avg ms between training
  genFrequency: number; // avg ms between generation
  feedFrequency: number; // avg ms between feeding
  recentActivityCount: number; // activities in last hour
}

/**
 * Comfort zone tracking - optimal ranges for this brain
 */
export interface ComfortZone {
  creativityRange: { min: number; max: number };
  stabilityRange: { min: number; max: number };
  preferredMood: Mood;
  moodHistory: { mood: Mood; duration: number }[];
}

/**
 * Priority-scored need assessment
 */
export interface BrainNeed {
  type: 'FEED' | 'TRAIN' | 'REST' | 'EXPLORE';
  priority: number; // 0-100, higher = more urgent
  reason: string;
  suggestedAction: string;
  estimatedImpact: { creativity: number; stability: number };
}

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

  // v4.3: Enhanced autonomy tracking
  activityPattern: ActivityPattern;
  comfortZone: ComfortZone;
  consecutiveMoodCount: number; // How long in current mood
  lastMoodChangeTime: number;
  autoPilotLevel: 'off' | 'suggestions' | 'adaptive' | 'full'; // Autonomy level

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

// v4.3: Adaptive decay constants
const DECAY_CONSTANTS = {
  BASE_CREATIVITY_DECAY: 0.3,
  BASE_STABILITY_DECAY: 0.1,
  MIN_DECAY_MULTIPLIER: 0.2,
  MAX_DECAY_MULTIPLIER: 3.0,
  PROTECTION_FLOOR: 15, // Don't decay below this
  IDLE_THRESHOLD_MS: 300000 // 5 minutes
};

// v4.3: Need priority weights
const NEED_WEIGHTS = {
  STABILITY_CRITICAL: 90,
  CREATIVITY_CRITICAL: 80,
  BURNT_OUT_THRESHOLD: 70,
  AGITATED_THRESHOLD: 75,
  IDLE_LONG_THRESHOLD: 50,
  MODERATE_NEED: 40,
  LOW_NEED: 20
};

// v4.3: Mood hysteresis - prevent rapid mood swings
const MOOD_HYSTERESIS = {
  MIN_MOOD_DURATION_MS: 60000, // Stay in mood for at least 1 minute
  TRANSITION_DAMPENING: 5 // Points of dampening for mood transitions
};

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Create default activity pattern
 */
function createDefaultActivityPattern(): ActivityPattern {
  const now = Date.now();
  return {
    lastTrainTime: now,
    lastGenTime: now,
    lastFeedTime: now,
    trainFrequency: 0,
    genFrequency: 0,
    feedFrequency: 0,
    recentActivityCount: 0
  };
}

/**
 * Create default comfort zone
 */
function createDefaultComfortZone(): ComfortZone {
  return {
    creativityRange: { min: 40, max: 80 },
    stabilityRange: { min: 50, max: 85 },
    preferredMood: 'CALM',
    moodHistory: []
  };
}

/**
 * Create a new brain state with default values
 */
export function createBrain(id: string, label?: string): BrainStats {
  const now = new Date().toISOString();
  const nowMs = Date.now();
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
    // v4.3: Enhanced autonomy fields
    activityPattern: createDefaultActivityPattern(),
    comfortZone: createDefaultComfortZone(),
    consecutiveMoodCount: 0,
    lastMoodChangeTime: nowMs,
    autoPilotLevel: 'off',
    diary: []
  };
}

/**
 * Migrate old brain state to new format (v4.3)
 */
function migrateBrainState(saved: Partial<BrainStats>, id: string): BrainStats {
  const nowMs = Date.now();
  return {
    ...createBrain(id),
    ...saved,
    // Ensure new fields exist with defaults
    activityPattern: saved.activityPattern || createDefaultActivityPattern(),
    comfortZone: saved.comfortZone || createDefaultComfortZone(),
    consecutiveMoodCount: saved.consecutiveMoodCount ?? 0,
    lastMoodChangeTime: saved.lastMoodChangeTime ?? nowMs,
    autoPilotLevel: saved.autoPilotLevel ?? 'off'
  };
}

/**
 * Load brain state from localStorage
 */
export function loadBrain(id: string): BrainStats {
  const saved = StorageManager.get<Partial<BrainStats> | null>(STORAGE_KEY_PREFIX + id, null);

  if (saved) {
    // Migrate old format if needed
    return migrateBrainState(saved, id);
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
 * Update activity pattern tracking
 */
function updateActivityPattern(
  pattern: ActivityPattern,
  eventType: 'TRAIN' | 'GEN' | 'FEED',
  timestamp: number
): ActivityPattern {
  const updated = { ...pattern };

  switch (eventType) {
    case 'TRAIN':
      if (pattern.lastTrainTime > 0) {
        const interval = timestamp - pattern.lastTrainTime;
        // Exponential moving average for frequency
        updated.trainFrequency =
          pattern.trainFrequency === 0
            ? interval
            : pattern.trainFrequency * 0.7 + interval * 0.3;
      }
      updated.lastTrainTime = timestamp;
      break;
    case 'GEN':
      if (pattern.lastGenTime > 0) {
        const interval = timestamp - pattern.lastGenTime;
        updated.genFrequency =
          pattern.genFrequency === 0
            ? interval
            : pattern.genFrequency * 0.7 + interval * 0.3;
      }
      updated.lastGenTime = timestamp;
      break;
    case 'FEED':
      if (pattern.lastFeedTime > 0) {
        const interval = timestamp - pattern.lastFeedTime;
        updated.feedFrequency =
          pattern.feedFrequency === 0
            ? interval
            : pattern.feedFrequency * 0.7 + interval * 0.3;
      }
      updated.lastFeedTime = timestamp;
      break;
  }

  // Track recent activity count (activities in last hour)
  updated.recentActivityCount = Math.min(100, updated.recentActivityCount + 1);

  return updated;
}

/**
 * Calculate adaptive decay based on mood and activity
 */
function calculateAdaptiveDecay(
  state: BrainStats,
  timestamp: number
): { creativityDecay: number; stabilityDecay: number } {
  const lastActivity = Math.max(
    state.activityPattern.lastTrainTime,
    state.activityPattern.lastGenTime,
    state.activityPattern.lastFeedTime
  );
  const idleTime = timestamp - lastActivity;

  // Base decay rates
  let creativityDecay = DECAY_CONSTANTS.BASE_CREATIVITY_DECAY;
  let stabilityDecay = DECAY_CONSTANTS.BASE_STABILITY_DECAY;

  // Mood-based modulation
  switch (state.mood) {
    case 'FOCUSED':
      creativityDecay *= 0.5;
      stabilityDecay *= 0.5;
      break;
    case 'AGITATED':
      creativityDecay *= 0.3; // Slow creativity decay to help recovery
      stabilityDecay *= 0.3;
      break;
    case 'BURNT_OUT':
      creativityDecay *= 1.5; // Faster creativity decay when burnt out
      stabilityDecay *= 0.5;
      break;
    case 'DREAMY':
      creativityDecay *= 1.2;
      stabilityDecay *= 1.0;
      break;
  }

  // Scale by idle time (more decay if idle longer, capped)
  if (idleTime > DECAY_CONSTANTS.IDLE_THRESHOLD_MS) {
    const idleMultiplier = Math.min(
      DECAY_CONSTANTS.MAX_DECAY_MULTIPLIER,
      1 + (idleTime - DECAY_CONSTANTS.IDLE_THRESHOLD_MS) / 600000 // +1x per 10 min
    );
    creativityDecay *= idleMultiplier;
    stabilityDecay *= idleMultiplier;
  }

  // Protection floor - reduce decay if vitals are already low
  if (state.creativity < DECAY_CONSTANTS.PROTECTION_FLOOR) {
    creativityDecay *= DECAY_CONSTANTS.MIN_DECAY_MULTIPLIER;
  }
  if (state.stability < DECAY_CONSTANTS.PROTECTION_FLOOR) {
    stabilityDecay *= DECAY_CONSTANTS.MIN_DECAY_MULTIPLIER;
  }

  return { creativityDecay, stabilityDecay };
}

/**
 * Update comfort zone based on current state
 */
function updateComfortZone(zone: ComfortZone, state: BrainStats): ComfortZone {
  const updated = { ...zone, moodHistory: [...zone.moodHistory] };

  // Update mood history
  const lastEntry = updated.moodHistory[updated.moodHistory.length - 1];
  if (lastEntry && lastEntry.mood === state.mood) {
    lastEntry.duration += 30000; // Approximate idle tick interval
  } else {
    updated.moodHistory.push({ mood: state.mood, duration: 30000 });
    // Keep last 50 entries
    if (updated.moodHistory.length > 50) {
      updated.moodHistory.shift();
    }
  }

  // Calculate preferred mood from history
  const moodDurations: Record<Mood, number> = {
    CALM: 0,
    FOCUSED: 0,
    AGITATED: 0,
    DREAMY: 0,
    BURNT_OUT: 0
  };
  for (const entry of updated.moodHistory) {
    moodDurations[entry.mood] += entry.duration;
  }
  updated.preferredMood = (Object.keys(moodDurations) as Mood[]).reduce((a, b) =>
    moodDurations[a] > moodDurations[b] ? a : b
  );

  // Adapt comfort ranges based on where vitals typically stabilize
  // Only adjust if we have enough history
  if (updated.moodHistory.length > 10) {
    // Slowly adjust ranges toward current values when in a good mood
    if (state.mood === 'CALM' || state.mood === 'FOCUSED') {
      updated.creativityRange.min =
        updated.creativityRange.min * 0.95 + Math.max(30, state.creativity - 20) * 0.05;
      updated.creativityRange.max =
        updated.creativityRange.max * 0.95 + Math.min(90, state.creativity + 20) * 0.05;
      updated.stabilityRange.min =
        updated.stabilityRange.min * 0.95 + Math.max(40, state.stability - 15) * 0.05;
      updated.stabilityRange.max =
        updated.stabilityRange.max * 0.95 + Math.min(95, state.stability + 15) * 0.05;
    }
  }

  return updated;
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
    diary: [...state.diary],
    activityPattern: { ...state.activityPattern },
    comfortZone: { ...state.comfortZone, moodHistory: [...state.comfortZone.moodHistory] }
  };

  switch (event.type) {
    case 'TRAIN_RUN': {
      const { steps, tokens, vocabDelta = 0, avgLoss } = event.payload;

      next.totalTrainSteps += steps;
      next.totalTokensSeen += tokens;
      next.vocabSize += vocabDelta;

      // v4.3: Adaptive creativity gain based on mood
      const creativityGain = avgLoss < 1.2 ? (state.mood === 'BURNT_OUT' ? 5 : 3) : 1;
      next.creativity = Math.min(100, next.creativity + creativityGain);

      // v4.3: Adaptive stability gain
      const stabilityGain = state.mood === 'AGITATED' ? 4 : 2;
      next.stability = Math.min(100, next.stability + stabilityGain);

      // v4.3: Update activity pattern
      next.activityPattern = updateActivityPattern(state.activityPattern, 'TRAIN', event.timestamp);

      // Add to diary
      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'TRAIN',
        message: `Trained for ${steps} steps, saw ${tokens} tokens (avg loss: ${avgLoss.toFixed(3)})`,
        metadata: { steps, tokens, avgLoss, creativityGain, stabilityGain }
      });

      break;
    }

    case 'GEN_RUN': {
      const { diversityScore = 0.5, tokensGenerated = 0 } = event.payload;

      // v4.3: Enhanced diversity-based updates
      if (diversityScore > 0.7) {
        const creativityBoost = state.mood === 'DREAMY' ? 0.5 : 1;
        next.creativity = Math.min(100, next.creativity + creativityBoost);
        // Only reduce stability if already high (prevents death spiral)
        if (state.stability > 40) {
          next.stability = Math.max(0, next.stability - 0.5);
        }
      } else if (diversityScore < 0.3) {
        // Low diversity generation indicates repetitive output
        next.creativity = Math.max(0, next.creativity - 0.5);
      }

      // v4.3: Update activity pattern
      next.activityPattern = updateActivityPattern(state.activityPattern, 'GEN', event.timestamp);

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

      // v4.3: Enhanced feeding effects based on mood
      const creativityBoost =
        state.mood === 'BURNT_OUT'
          ? newWordsCount * 0.4 // Extra boost when burnt out
          : newWordsCount * 0.2;
      next.creativity = Math.min(100, next.creativity + creativityBoost);

      // Heavy/complex text reduces stability temporarily (less impact if already stable)
      const stabilityPenalty = state.stability > 70 ? heavinessScore * 1.5 : heavinessScore * 3;
      next.stability = Math.max(0, next.stability - stabilityPenalty);

      // v4.3: Update activity pattern
      next.activityPattern = updateActivityPattern(state.activityPattern, 'FEED', event.timestamp);

      addDiaryEntry(next, {
        timestamp: event.timestamp,
        type: 'FEED',
        message: `Fed new corpus: ${summary} (${newWordsCount} new words)`,
        metadata: { newWordsCount, heavinessScore, summary, creativityBoost }
      });

      break;
    }

    case 'IDLE_TICK': {
      // v4.3: Adaptive decay based on mood and activity
      const { creativityDecay, stabilityDecay } = calculateAdaptiveDecay(state, event.timestamp);

      next.creativity = Math.max(0, next.creativity - creativityDecay);
      next.stability = Math.max(0, next.stability - stabilityDecay);

      // v4.3: Update comfort zone
      next.comfortZone = updateComfortZone(state.comfortZone, state);

      // Decay recent activity count
      next.activityPattern.recentActivityCount = Math.max(
        0,
        state.activityPattern.recentActivityCount - 0.5
      );

      break;
    }

    case 'MOOD_OVERRIDE': {
      const { mood } = event.payload;
      next.mood = mood;
      next.lastMoodChangeTime = event.timestamp;
      next.consecutiveMoodCount = 0;

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

  // v4.3: Determine new mood with hysteresis
  const oldMood = next.mood;
  const candidateMood = determineMood(next);

  // Apply hysteresis - resist rapid mood changes
  const timeSinceMoodChange = event.timestamp - state.lastMoodChangeTime;
  if (candidateMood !== oldMood) {
    if (timeSinceMoodChange >= MOOD_HYSTERESIS.MIN_MOOD_DURATION_MS) {
      next.mood = candidateMood;
      next.lastMoodChangeTime = event.timestamp;
      next.consecutiveMoodCount = 0;
    } else {
      // Keep old mood but track pressure
      next.consecutiveMoodCount++;
    }
  } else {
    next.consecutiveMoodCount++;
  }

  // Log mood changes
  if (next.mood !== oldMood && event.type !== 'MOOD_OVERRIDE') {
    addDiaryEntry(next, {
      timestamp: event.timestamp,
      type: 'MOOD_SHIFT',
      message: `Mood shifted from ${oldMood} to ${next.mood}`,
      metadata: { oldMood, newMood: next.mood, timeSincePrevious: timeSinceMoodChange }
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
 * Legacy function - use assessBrainNeeds for priority-scored needs
 */
export function checkBrainNeeds(state: BrainStats): {
  needsFeeding: boolean;
  needsTraining: boolean;
  message: string | null;
} {
  const needs = assessBrainNeeds(state);
  const topNeed = needs[0];

  const needsFeeding = needs.some((n) => n.type === 'FEED' && n.priority > 30);
  const needsTraining = needs.some((n) => n.type === 'TRAIN' && n.priority > 30);

  let message: string | null = null;

  if (topNeed && topNeed.priority > 30) {
    message = topNeed.suggestedAction;
  }

  return { needsFeeding, needsTraining, message };
}

/**
 * v4.3: Priority-scored need assessment
 * Returns sorted list of needs by priority (highest first)
 */
export function assessBrainNeeds(state: BrainStats): BrainNeed[] {
  const now = Date.now();
  const needs: BrainNeed[] = [];

  // Calculate time since last activities
  const timeSinceTraining = now - state.activityPattern.lastTrainTime;
  const timeSinceFeeding = now - state.activityPattern.lastFeedTime;
  const timeSinceAnyActivity = Math.min(
    timeSinceTraining,
    now - state.activityPattern.lastGenTime,
    timeSinceFeeding
  );

  // Check if outside comfort zone
  const belowCreativityComfort = state.creativity < state.comfortZone.creativityRange.min;
  const belowStabilityComfort = state.stability < state.comfortZone.stabilityRange.min;

  // 1. Critical stability need
  if (state.stability < 15) {
    needs.push({
      type: 'TRAIN',
      priority: NEED_WEIGHTS.STABILITY_CRITICAL,
      reason: 'Critical stability - model is highly unstable',
      suggestedAction: 'Urgent: Run training immediately to stabilize the model!',
      estimatedImpact: { creativity: 2, stability: 8 }
    });
  } else if (state.stability < 30) {
    needs.push({
      type: 'TRAIN',
      priority: state.mood === 'AGITATED' ? NEED_WEIGHTS.AGITATED_THRESHOLD : NEED_WEIGHTS.MODERATE_NEED + 15,
      reason: 'Low stability - model needs training',
      suggestedAction: 'Training session recommended to improve stability.',
      estimatedImpact: { creativity: 1, stability: 5 }
    });
  }

  // 2. Critical creativity need
  if (state.creativity < 15) {
    needs.push({
      type: 'FEED',
      priority: NEED_WEIGHTS.CREATIVITY_CRITICAL,
      reason: 'Critical creativity - model output may be repetitive',
      suggestedAction: 'Urgent: Feed new diverse content to restore creativity!',
      estimatedImpact: { creativity: 10, stability: -2 }
    });
  } else if (state.creativity < 30 || state.mood === 'BURNT_OUT') {
    needs.push({
      type: 'FEED',
      priority: state.mood === 'BURNT_OUT' ? NEED_WEIGHTS.BURNT_OUT_THRESHOLD : NEED_WEIGHTS.MODERATE_NEED + 10,
      reason: state.mood === 'BURNT_OUT' ? 'Burnt out - needs fresh content' : 'Low creativity',
      suggestedAction: 'Feed interesting text to boost creativity.',
      estimatedImpact: { creativity: 6, stability: -1 }
    });
  }

  // 3. Long idle time need
  if (timeSinceAnyActivity > 900000) {
    // 15 minutes
    needs.push({
      type: 'EXPLORE',
      priority: NEED_WEIGHTS.IDLE_LONG_THRESHOLD,
      reason: 'Extended idle period detected',
      suggestedAction: "It's been a while! Consider generating some text or adding new content.",
      estimatedImpact: { creativity: 2, stability: 0 }
    });
  }

  // 4. Comfort zone recovery needs
  if (belowCreativityComfort && state.creativity >= 30) {
    needs.push({
      type: 'FEED',
      priority: NEED_WEIGHTS.LOW_NEED + 10,
      reason: 'Below preferred creativity range',
      suggestedAction: `Creativity (${state.creativity.toFixed(0)}) below comfort zone. Feed new content.`,
      estimatedImpact: { creativity: 4, stability: 0 }
    });
  }

  if (belowStabilityComfort && state.stability >= 30) {
    needs.push({
      type: 'TRAIN',
      priority: NEED_WEIGHTS.LOW_NEED + 5,
      reason: 'Below preferred stability range',
      suggestedAction: `Stability (${state.stability.toFixed(0)}) below comfort zone. Train to improve.`,
      estimatedImpact: { creativity: 1, stability: 3 }
    });
  }

  // 5. Rest need (if highly active recently)
  if (state.activityPattern.recentActivityCount > 20 && state.mood === 'DREAMY') {
    needs.push({
      type: 'REST',
      priority: NEED_WEIGHTS.LOW_NEED,
      reason: 'High activity with dreamlike state',
      suggestedAction: 'Consider pausing to let the model stabilize naturally.',
      estimatedImpact: { creativity: -1, stability: 2 }
    });
  }

  // Sort by priority (highest first)
  needs.sort((a, b) => b.priority - a.priority);

  return needs;
}

/**
 * Get the most urgent need
 */
export function getUrgentNeed(state: BrainStats): BrainNeed | null {
  const needs = assessBrainNeeds(state);
  return needs.length > 0 && needs[0].priority >= NEED_WEIGHTS.MODERATE_NEED ? needs[0] : null;
}

/**
 * Check if brain is in a healthy state
 */
export function isBrainHealthy(state: BrainStats): boolean {
  return (
    state.mood === 'CALM' ||
    state.mood === 'FOCUSED' ||
    (state.creativity >= state.comfortZone.creativityRange.min &&
      state.stability >= state.comfortZone.stabilityRange.min)
  );
}

/**
 * Get brain health score (0-100)
 */
export function getBrainHealthScore(state: BrainStats): number {
  // Base score from vitals (0-50)
  const vitalsScore = (state.creativity + state.stability) / 4;

  // Mood bonus (0-30)
  const moodBonus: Record<Mood, number> = {
    FOCUSED: 30,
    CALM: 20,
    DREAMY: 10,
    AGITATED: -5,
    BURNT_OUT: 0
  };

  // Comfort zone bonus (0-20)
  const inCreativityZone =
    state.creativity >= state.comfortZone.creativityRange.min &&
    state.creativity <= state.comfortZone.creativityRange.max;
  const inStabilityZone =
    state.stability >= state.comfortZone.stabilityRange.min &&
    state.stability <= state.comfortZone.stabilityRange.max;
  const comfortBonus = (inCreativityZone ? 10 : 0) + (inStabilityZone ? 10 : 0);

  return Math.min(100, Math.max(0, vitalsScore + moodBonus[state.mood] + comfortBonus));
}
