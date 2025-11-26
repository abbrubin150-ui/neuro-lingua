/**
 * BrainContext - React Context for Brain History state management
 *
 * Provides:
 * - Brain state access and updates
 * - Event dispatching system
 * - Autonomous mode management with idle ticks
 * - Suggestion system for user interactions
 *
 * Safety guarantees:
 * 1. No heavy operations run autonomously (only suggestions)
 * 2. All autonomous actions are visible in the diary
 * 3. Everything is localStorage-scoped (no network/disk access)
 */

import React, { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react';
import {
  BrainStats,
  BrainEvent,
  reduceBrain,
  loadBrain,
  saveBrain,
  getBrainStatusMessage,
  checkBrainNeeds,
  createBrain
} from '../lib/BrainEngine';

// ============================================================================
// Types
// ============================================================================

export interface BrainSuggestion {
  id: string;
  message: string;
  action: 'FEED' | 'TRAIN' | 'NONE';
  timestamp: number;
  dismissed: boolean;
}

interface BrainContextValue {
  // State
  brain: BrainStats;
  suggestions: BrainSuggestion[];
  statusMessage: string;

  // Actions
  dispatchBrain: (event: BrainEvent) => void;
  setAutonomyEnabled: (enabled: boolean) => void;
  dismissSuggestion: (id: string) => void;
  resetBrain: () => void;
  setBrainLabel: (label: string) => void;

  // Utilities
  isAutonomous: boolean;
}

// ============================================================================
// Context
// ============================================================================

const BrainContext = createContext<BrainContextValue | null>(null);

const IDLE_TICK_INTERVAL = 30000; // 30 seconds
const SUGGESTION_CHECK_INTERVAL = 60000; // 1 minute
const SUGGESTION_DISPLAY_DURATION = 300000; // 5 minutes before auto-dismiss

// ============================================================================
// Provider
// ============================================================================

export function BrainProvider({ children }: { children: React.ReactNode }) {
  // Determine brain ID from current model (you might want to make this configurable)
  const brainId = 'default'; // TODO: Link to active model ID

  const [brain, setBrain] = useState<BrainStats>(() => loadBrain(brainId));
  const [suggestions, setSuggestions] = useState<BrainSuggestion[]>([]);

  // ========================================================================
  // Event Dispatching
  // ========================================================================

  const dispatchBrain = useCallback((event: BrainEvent) => {
    setBrain((prev) => {
      const next = reduceBrain(prev, event);
      saveBrain(next);
      return next;
    });
  }, []);

  // ========================================================================
  // Autonomous Mode - Idle Ticks
  // ========================================================================

  useEffect(() => {
    if (!brain.autonomyEnabled) return;

    const interval = setInterval(() => {
      dispatchBrain({
        type: 'IDLE_TICK',
        timestamp: Date.now()
      });
    }, IDLE_TICK_INTERVAL);

    return () => clearInterval(interval);
  }, [brain.autonomyEnabled, dispatchBrain]);

  // ========================================================================
  // Autonomous Mode - Suggestion Generation
  // ========================================================================

  useEffect(() => {
    if (!brain.autonomyEnabled) return;

    const checkAndSuggest = () => {
      const needs = checkBrainNeeds(brain);

      if (needs.message) {
        const newSuggestion: BrainSuggestion = {
          id: `suggestion-${Date.now()}`,
          message: needs.message,
          action: needs.needsTraining ? 'TRAIN' : needs.needsFeeding ? 'FEED' : 'NONE',
          timestamp: Date.now(),
          dismissed: false
        };

        setSuggestions((prev) => {
          // Don't duplicate suggestions
          const hasSimilar = prev.some(
            (s) =>
              !s.dismissed &&
              s.action === newSuggestion.action &&
              Date.now() - s.timestamp < SUGGESTION_DISPLAY_DURATION
          );

          if (hasSimilar) return prev;

          return [...prev, newSuggestion];
        });

        // Log suggestion to diary
        dispatchBrain({
          type: 'MOOD_OVERRIDE',
          timestamp: Date.now(),
          payload: { mood: brain.mood }
        });
      }
    };

    const interval = setInterval(checkAndSuggest, SUGGESTION_CHECK_INTERVAL);
    checkAndSuggest(); // Check immediately on mount

    return () => clearInterval(interval);
  }, [brain.autonomyEnabled, brain, dispatchBrain]);

  // ========================================================================
  // Auto-dismiss old suggestions
  // ========================================================================

  useEffect(() => {
    const cleanup = setInterval(() => {
      const now = Date.now();
      setSuggestions((prev) =>
        prev.map((s) =>
          !s.dismissed && now - s.timestamp > SUGGESTION_DISPLAY_DURATION
            ? { ...s, dismissed: true }
            : s
        )
      );
    }, 30000); // Check every 30 seconds

    return () => clearInterval(cleanup);
  }, []);

  // ========================================================================
  // Actions
  // ========================================================================

  const setAutonomyEnabled = useCallback((enabled: boolean) => {
    setBrain((prev) => {
      const next = { ...prev, autonomyEnabled: enabled };
      saveBrain(next);
      return next;
    });

    // Clear all suggestions when disabling autonomy
    if (!enabled) {
      setSuggestions([]);
    }
  }, []);

  const dismissSuggestion = useCallback((id: string) => {
    setSuggestions((prev) => prev.map((s) => (s.id === id ? { ...s, dismissed: true } : s)));
  }, []);

  const resetBrain = useCallback(() => {
    const newBrain = createBrain(brainId, brain.label);
    setBrain(newBrain);
    saveBrain(newBrain);
    setSuggestions([]);
  }, [brainId, brain.label]);

  const setBrainLabel = useCallback((label: string) => {
    setBrain((prev) => {
      const next = { ...prev, label };
      saveBrain(next);
      return next;
    });
  }, []);

  // ========================================================================
  // Computed Values
  // ========================================================================

  const statusMessage = useMemo(() => getBrainStatusMessage(brain), [brain]);
  const isAutonomous = brain.autonomyEnabled;

  const value: BrainContextValue = {
    brain,
    suggestions: suggestions.filter((s) => !s.dismissed),
    statusMessage,
    dispatchBrain,
    setAutonomyEnabled,
    dismissSuggestion,
    resetBrain,
    setBrainLabel,
    isAutonomous
  };

  return <BrainContext.Provider value={value}>{children}</BrainContext.Provider>;
}

// ============================================================================
// Hook
// ============================================================================

/**
 * Hook to access BrainContext
 * @throws Error if used outside BrainProvider
 */
export function useBrain() {
  const context = useContext(BrainContext);
  if (!context) {
    throw new Error('useBrain must be used within a BrainProvider');
  }
  return context;
}

/**
 * Hook for easy event dispatching from training code
 */
export function useBrainTraining() {
  const { dispatchBrain } = useBrain();

  return useCallback(
    (steps: number, tokens: number, avgLoss: number, vocabDelta: number = 0) => {
      dispatchBrain({
        type: 'TRAIN_RUN',
        timestamp: Date.now(),
        payload: { steps, tokens, avgLoss, vocabDelta }
      });
    },
    [dispatchBrain]
  );
}

/**
 * Hook for easy event dispatching from generation code
 */
export function useBrainGeneration() {
  const { dispatchBrain } = useBrain();

  return useCallback(
    (diversityScore: number, tokensGenerated: number) => {
      dispatchBrain({
        type: 'GEN_RUN',
        timestamp: Date.now(),
        payload: { diversityScore, tokensGenerated }
      });
    },
    [dispatchBrain]
  );
}

/**
 * Hook for easy event dispatching from feed code
 */
export function useBrainFeed() {
  const { dispatchBrain } = useBrain();

  return useCallback(
    (newWordsCount: number, heavinessScore: number, summary: string) => {
      dispatchBrain({
        type: 'FEED',
        timestamp: Date.now(),
        payload: { newWordsCount, heavinessScore, summary }
      });
    },
    [dispatchBrain]
  );
}
