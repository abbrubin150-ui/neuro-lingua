/**
 * BrainContext - React Context for Brain History state management
 *
 * Provides:
 * - Brain state access and updates
 * - Event dispatching system
 * - Autonomous mode management with idle ticks
 * - Suggestion system for user interactions
 * - v4.3: Auto-pilot integration with BrainGovernanceBridge
 * - v4.3: Priority-scored need assessment
 * - v4.3: Recovery plan management
 *
 * Safety guarantees:
 * 1. No heavy operations run autonomously (only suggestions)
 * 2. All autonomous actions are visible in the diary
 * 3. Everything is localStorage-scoped (no network/disk access)
 * 4. Auto-pilot actions are bounded and reversible
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef
} from 'react';
import {
  BrainStats,
  BrainEvent,
  BrainNeed,
  reduceBrain,
  loadBrain,
  saveBrain,
  getBrainStatusMessage,
  assessBrainNeeds,
  isBrainHealthy,
  getBrainHealthScore,
  createBrain
} from '../lib/BrainEngine';
import {
  BrainGovernanceBridge,
  AutonomousAction,
  RecoveryPlan,
  getBridge
} from '../lib/BrainGovernanceBridge';

// ============================================================================
// Types
// ============================================================================

export interface BrainSuggestion {
  id: string;
  message: string;
  action: 'FEED' | 'TRAIN' | 'REST' | 'EXPLORE' | 'NONE';
  priority: number; // v4.3: Priority score 0-100
  timestamp: number;
  dismissed: boolean;
  source: 'brain' | 'bridge' | 'governance'; // v4.3: Where the suggestion came from
}

/**
 * v4.3: Auto-pilot level configuration
 */
export type AutoPilotLevel = 'off' | 'suggestions' | 'adaptive' | 'full';

interface BrainContextValue {
  // State
  brain: BrainStats;
  suggestions: BrainSuggestion[];
  statusMessage: string;

  // v4.3: Enhanced state
  needs: BrainNeed[];
  healthScore: number;
  isHealthy: boolean;
  recoveryPlan: RecoveryPlan | null;
  pendingActions: AutonomousAction[];
  autoPilotLevel: AutoPilotLevel;

  // Actions
  dispatchBrain: (event: BrainEvent) => void;
  setAutonomyEnabled: (enabled: boolean) => void;
  dismissSuggestion: (id: string) => void;
  actOnSuggestion: (id: string) => void;
  resetBrain: () => void;
  setBrainLabel: (label: string) => void;
  exportBrainHistory: () => string;

  // v4.3: Enhanced actions
  setAutoPilotLevel: (level: AutoPilotLevel) => void;
  executeAction: (actionId: string) => void;
  cancelAction: (actionId: string, reason?: string) => void;
  refreshNeeds: () => void;
  getRecoveryPlan: () => RecoveryPlan | null;

  // Utilities
  isAutonomous: boolean;
  bridge: BrainGovernanceBridge;
}

// ============================================================================
// Context
// ============================================================================

const BrainContext = createContext<BrainContextValue | null>(null);

const DEFAULT_IDLE_TICK_INTERVAL = 30000; // 30 seconds
const SUGGESTION_DISPLAY_DURATION = 300000; // 5 minutes before auto-dismiss
const NEEDS_REFRESH_INTERVAL = 15000; // v4.3: Refresh needs every 15 seconds

// ============================================================================
// Provider
// ============================================================================

interface BrainProviderProps {
  children: React.ReactNode;
  modelId?: string; // Optional model ID to link brain state
}

export function BrainProvider({ children, modelId }: BrainProviderProps) {
  // Determine brain ID from current model or use default
  const brainId = modelId || 'default';

  const [brain, setBrain] = useState<BrainStats>(() => loadBrain(brainId));
  const [suggestions, setSuggestions] = useState<BrainSuggestion[]>([]);

  // v4.3: Enhanced state
  const [needs, setNeeds] = useState<BrainNeed[]>([]);
  const [recoveryPlan, setRecoveryPlan] = useState<RecoveryPlan | null>(null);
  const [pendingActions, setPendingActions] = useState<AutonomousAction[]>([]);
  const [autoPilotLevel, setAutoPilotLevelState] = useState<AutoPilotLevel>(() => {
    return brain.autoPilotLevel || 'off';
  });

  // v4.3: Bridge instance
  const bridgeRef = useRef<BrainGovernanceBridge>(getBridge());

  // ========================================================================
  // Model ID Change Handler
  // ========================================================================

  useEffect(() => {
    // Load brain state for new model ID
    const newBrain = loadBrain(brainId);
    setBrain(newBrain);
    // Clear suggestions when switching models
    setSuggestions([]);
    // v4.3: Reset needs and update auto-pilot level
    setNeeds(assessBrainNeeds(newBrain));
    setAutoPilotLevelState(newBrain.autoPilotLevel || 'off');
    setRecoveryPlan(null);
    setPendingActions([]);
  }, [brainId]);

  // ========================================================================
  // v4.3: Needs Refresh
  // ========================================================================

  const refreshNeeds = useCallback(() => {
    const currentNeeds = assessBrainNeeds(brain);
    setNeeds(currentNeeds);

    // Update recovery plan if needed
    const plan = bridgeRef.current.generateRecoveryPlan(brain);
    setRecoveryPlan(plan);

    // Update pending actions from bridge
    setPendingActions(bridgeRef.current.getPendingActions());
  }, [brain]);

  // Auto-refresh needs periodically
  useEffect(() => {
    const interval = setInterval(refreshNeeds, NEEDS_REFRESH_INTERVAL);
    refreshNeeds(); // Initial refresh
    return () => clearInterval(interval);
  }, [refreshNeeds]);

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
  // Autonomous Mode - Idle Ticks (v4.3: Adaptive interval)
  // ========================================================================

  useEffect(() => {
    if (!brain.autonomyEnabled && autoPilotLevel === 'off') return;

    // v4.3: Use mood-adaptive interval from bridge
    const idleInterval = bridgeRef.current.getSuggestionInterval(brain.mood) / 2;
    const effectiveInterval = Math.max(DEFAULT_IDLE_TICK_INTERVAL, idleInterval);

    const interval = setInterval(() => {
      dispatchBrain({
        type: 'IDLE_TICK',
        timestamp: Date.now()
      });
    }, effectiveInterval);

    return () => clearInterval(interval);
  }, [brain.autonomyEnabled, brain.mood, autoPilotLevel, dispatchBrain]);

  // ========================================================================
  // Autonomous Mode - Suggestion Generation (v4.3: Priority-based)
  // ========================================================================

  useEffect(() => {
    if (!brain.autonomyEnabled && autoPilotLevel === 'off') return;

    const checkAndSuggest = () => {
      // v4.3: Use priority-scored needs
      const currentNeeds = assessBrainNeeds(brain);
      const urgentNeed = currentNeeds.length > 0 ? currentNeeds[0] : null;

      if (urgentNeed && urgentNeed.priority > 30) {
        const actionMap: Record<string, BrainSuggestion['action']> = {
          FEED: 'FEED',
          TRAIN: 'TRAIN',
          REST: 'REST',
          EXPLORE: 'EXPLORE'
        };

        const newSuggestion: BrainSuggestion = {
          id: `suggestion-${Date.now()}`,
          message: urgentNeed.suggestedAction,
          action: actionMap[urgentNeed.type] || 'NONE',
          priority: urgentNeed.priority,
          timestamp: Date.now(),
          dismissed: false,
          source: 'brain'
        };

        setSuggestions((prev) => {
          // Don't duplicate suggestions with similar action
          const hasSimilar = prev.some(
            (s) =>
              !s.dismissed &&
              s.action === newSuggestion.action &&
              Date.now() - s.timestamp < SUGGESTION_DISPLAY_DURATION
          );

          if (hasSimilar) return prev;

          // v4.3: Sort by priority when adding new suggestion
          const newList = [...prev, newSuggestion];
          newList.sort((a, b) => b.priority - a.priority);
          return newList;
        });

        // v4.3: Also generate bridge suggestions for adaptive mode
        if (autoPilotLevel === 'adaptive' || autoPilotLevel === 'full') {
          const bridgeSuggestions = bridgeRef.current.generatePrioritizedSuggestions(brain);
          bridgeSuggestions.forEach((action) => {
            // Queue actions that don't require confirmation in adaptive mode
            if (!action.requiresConfirmation || autoPilotLevel === 'full') {
              bridgeRef.current.queueAction(action);
            }
          });
          setPendingActions(bridgeRef.current.getPendingActions());
        }
      }
    };

    // v4.3: Adaptive suggestion interval based on mood
    const suggestionInterval = bridgeRef.current.getSuggestionInterval(brain.mood);

    const interval = setInterval(checkAndSuggest, suggestionInterval);
    checkAndSuggest(); // Check immediately on mount

    return () => clearInterval(interval);
  }, [brain.autonomyEnabled, brain, autoPilotLevel, dispatchBrain]);

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

  // v4.3: Set auto-pilot level
  const setAutoPilotLevel = useCallback((level: AutoPilotLevel) => {
    setAutoPilotLevelState(level);
    setBrain((prev) => {
      const next = { ...prev, autoPilotLevel: level };
      saveBrain(next);
      return next;
    });

    // Configure bridge based on level
    switch (level) {
      case 'off':
        bridgeRef.current.disableAutoPilot();
        setSuggestions([]);
        break;
      case 'suggestions':
        bridgeRef.current.setAutoPilotConfig({
          enabled: false,
          allowedPriorities: ['informational']
        });
        break;
      case 'adaptive':
        bridgeRef.current.setAutoPilotConfig({
          enabled: true,
          allowedPriorities: ['informational', 'low'],
          requireConfirmationFor: ['critical', 'high', 'medium']
        });
        break;
      case 'full':
        bridgeRef.current.setAutoPilotConfig({
          enabled: true,
          allowedPriorities: ['informational', 'low', 'medium'],
          requireConfirmationFor: ['critical', 'high']
        });
        break;
    }
  }, []);

  const dismissSuggestion = useCallback((id: string) => {
    setSuggestions((prev) => prev.map((s) => (s.id === id ? { ...s, dismissed: true } : s)));
  }, []);

  const actOnSuggestion = useCallback(
    (id: string) => {
      const suggestion = suggestions.find((s) => s.id === id);
      if (!suggestion) return;

      // Log action to diary
      dispatchBrain({
        type: 'MOOD_OVERRIDE',
        timestamp: Date.now(),
        payload: { mood: brain.mood }
      });

      // Dismiss the suggestion after acting
      dismissSuggestion(id);

      // Note: Actual action (FEED/TRAIN) should be triggered by the UI component
      // This just marks the suggestion as acted upon
    },
    [suggestions, brain.mood, dispatchBrain, dismissSuggestion]
  );

  // v4.3: Execute a pending autonomous action
  const executeAction = useCallback((actionId: string) => {
    const success = bridgeRef.current.executeAction(actionId);
    if (success) {
      setPendingActions(bridgeRef.current.getPendingActions());
    }
  }, []);

  // v4.3: Cancel a pending action
  const cancelAction = useCallback((actionId: string, reason?: string) => {
    const success = bridgeRef.current.cancelAction(actionId, reason || 'User cancelled');
    if (success) {
      setPendingActions(bridgeRef.current.getPendingActions());
    }
  }, []);

  // v4.3: Get current recovery plan
  const getRecoveryPlan = useCallback(() => {
    return bridgeRef.current.generateRecoveryPlan(brain);
  }, [brain]);

  const resetBrain = useCallback(() => {
    const newBrain = createBrain(brainId, brain.label);
    setBrain(newBrain);
    saveBrain(newBrain);
    setSuggestions([]);
    // v4.3: Reset bridge and enhanced state
    bridgeRef.current.reset();
    setNeeds([]);
    setRecoveryPlan(null);
    setPendingActions([]);
    setAutoPilotLevelState('off');
  }, [brainId, brain.label]);

  const exportBrainHistory = useCallback(() => {
    const exportData = {
      brain,
      suggestions,
      needs,
      recoveryPlan,
      bridgeState: bridgeRef.current.exportState(),
      exportedAt: new Date().toISOString(),
      version: '4.3.0' // v4.3: Updated version
    };
    return JSON.stringify(exportData, null, 2);
  }, [brain, suggestions, needs, recoveryPlan]);

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
  const isAutonomous = brain.autonomyEnabled || autoPilotLevel !== 'off';

  // v4.3: Computed health metrics
  const healthScore = useMemo(() => getBrainHealthScore(brain), [brain]);
  const isHealthy = useMemo(() => isBrainHealthy(brain), [brain]);

  const value: BrainContextValue = {
    brain,
    suggestions: suggestions.filter((s) => !s.dismissed),
    statusMessage,
    // v4.3: Enhanced state
    needs,
    healthScore,
    isHealthy,
    recoveryPlan,
    pendingActions,
    autoPilotLevel,
    // Actions
    dispatchBrain,
    setAutonomyEnabled,
    dismissSuggestion,
    actOnSuggestion,
    resetBrain,
    setBrainLabel,
    exportBrainHistory,
    // v4.3: Enhanced actions
    setAutoPilotLevel,
    executeAction,
    cancelAction,
    refreshNeeds,
    getRecoveryPlan,
    // Utilities
    isAutonomous,
    bridge: bridgeRef.current
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

/**
 * v4.3: Hook for auto-pilot mode management
 */
export function useBrainAutoPilot() {
  const {
    autoPilotLevel,
    setAutoPilotLevel,
    pendingActions,
    executeAction,
    cancelAction,
    recoveryPlan,
    getRecoveryPlan,
    healthScore,
    isHealthy,
    needs,
    bridge
  } = useBrain();

  return {
    // State
    level: autoPilotLevel,
    pendingActions,
    recoveryPlan,
    healthScore,
    isHealthy,
    needs,

    // Actions
    setLevel: setAutoPilotLevel,
    executeAction,
    cancelAction,
    refreshRecoveryPlan: getRecoveryPlan,

    // Bridge access for advanced use
    bridge,

    // Convenience methods
    isActive: autoPilotLevel !== 'off',
    canAutoExecute: autoPilotLevel === 'adaptive' || autoPilotLevel === 'full',
    hasPendingActions: pendingActions.length > 0,
    hasRecoveryPlan: recoveryPlan !== null
  };
}

/**
 * v4.3: Hook for brain health monitoring
 */
export function useBrainHealth() {
  const { brain, healthScore, isHealthy, needs, refreshNeeds, recoveryPlan } = useBrain();

  return {
    // Health metrics
    healthScore,
    isHealthy,
    mood: brain.mood,
    creativity: brain.creativity,
    stability: brain.stability,

    // Comfort zone
    comfortZone: brain.comfortZone,
    inComfortZone:
      brain.creativity >= brain.comfortZone.creativityRange.min &&
      brain.creativity <= brain.comfortZone.creativityRange.max &&
      brain.stability >= brain.comfortZone.stabilityRange.min &&
      brain.stability <= brain.comfortZone.stabilityRange.max,

    // Needs
    needs,
    urgentNeed: needs.length > 0 && needs[0].priority >= 50 ? needs[0] : null,
    refreshNeeds,

    // Recovery
    recoveryPlan,
    needsRecovery: recoveryPlan !== null
  };
}
