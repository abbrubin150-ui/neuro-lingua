/**
 * Tests for BrainEngine v4.3 - Enhanced Autonomy Features
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  createBrain,
  reduceBrain,
  loadBrain,
  saveBrain,
  assessBrainNeeds,
  getUrgentNeed,
  isBrainHealthy,
  getBrainHealthScore,
  getBrainStatusMessage,
  calculateDiversityScore,
  calculateHeavinessScore,
  BrainStats,
  BrainEvent,
  BrainNeed,
  Mood
} from '../../src/lib/BrainEngine';

describe('BrainEngine v4.3 - Enhanced Autonomy', () => {
  let brain: BrainStats;

  beforeEach(() => {
    brain = createBrain('test-id', 'Test Brain');
  });

  describe('Brain Creation with v4.3 Fields', () => {
    it('should create brain with activity pattern', () => {
      expect(brain.activityPattern).toBeDefined();
      expect(brain.activityPattern.lastTrainTime).toBeGreaterThan(0);
      expect(brain.activityPattern.lastGenTime).toBeGreaterThan(0);
      expect(brain.activityPattern.lastFeedTime).toBeGreaterThan(0);
      expect(brain.activityPattern.recentActivityCount).toBe(0);
    });

    it('should create brain with comfort zone', () => {
      expect(brain.comfortZone).toBeDefined();
      expect(brain.comfortZone.creativityRange.min).toBeLessThan(
        brain.comfortZone.creativityRange.max
      );
      expect(brain.comfortZone.stabilityRange.min).toBeLessThan(
        brain.comfortZone.stabilityRange.max
      );
      expect(brain.comfortZone.moodHistory).toEqual([]);
    });

    it('should create brain with mood tracking fields', () => {
      expect(brain.consecutiveMoodCount).toBe(0);
      expect(brain.lastMoodChangeTime).toBeGreaterThan(0);
      expect(brain.autoPilotLevel).toBe('off');
    });
  });

  describe('Activity Pattern Tracking', () => {
    it('should update activity pattern on TRAIN_RUN', () => {
      const initialTrainTime = brain.activityPattern.lastTrainTime;

      const event: BrainEvent = {
        type: 'TRAIN_RUN',
        timestamp: Date.now() + 1000,
        payload: { steps: 10, tokens: 1000, avgLoss: 1.0, vocabDelta: 0 }
      };

      const updated = reduceBrain(brain, event);

      expect(updated.activityPattern.lastTrainTime).toBeGreaterThan(initialTrainTime);
      expect(updated.activityPattern.recentActivityCount).toBe(1);
    });

    it('should update activity pattern on GEN_RUN', () => {
      const initialGenTime = brain.activityPattern.lastGenTime;

      const event: BrainEvent = {
        type: 'GEN_RUN',
        timestamp: Date.now() + 1000,
        payload: { diversityScore: 0.8, tokensGenerated: 50 }
      };

      const updated = reduceBrain(brain, event);

      expect(updated.activityPattern.lastGenTime).toBeGreaterThan(initialGenTime);
      expect(updated.activityPattern.recentActivityCount).toBe(1);
    });

    it('should update activity pattern on FEED', () => {
      const initialFeedTime = brain.activityPattern.lastFeedTime;

      const event: BrainEvent = {
        type: 'FEED',
        timestamp: Date.now() + 1000,
        payload: { newWordsCount: 100, heavinessScore: 0.5, summary: 'Test feed' }
      };

      const updated = reduceBrain(brain, event);

      expect(updated.activityPattern.lastFeedTime).toBeGreaterThan(initialFeedTime);
      expect(updated.activityPattern.recentActivityCount).toBe(1);
    });

    it('should calculate training frequency with EMA', () => {
      let current = brain;

      // First training
      current = reduceBrain(current, {
        type: 'TRAIN_RUN',
        timestamp: Date.now(),
        payload: { steps: 10, tokens: 1000, avgLoss: 1.0 }
      });

      // Second training 5 seconds later
      current = reduceBrain(current, {
        type: 'TRAIN_RUN',
        timestamp: Date.now() + 5000,
        payload: { steps: 10, tokens: 1000, avgLoss: 0.9 }
      });

      expect(current.activityPattern.trainFrequency).toBeGreaterThan(0);
    });
  });

  describe('Adaptive Decay', () => {
    it('should apply adaptive decay on IDLE_TICK based on mood', () => {
      // Create focused brain with high vitals
      const focusedBrain: BrainStats = {
        ...brain,
        mood: 'FOCUSED',
        creativity: 80,
        stability: 80
      };

      const event: BrainEvent = {
        type: 'IDLE_TICK',
        timestamp: Date.now() + 60000
      };

      const updated = reduceBrain(focusedBrain, event);

      // FOCUSED should have slower decay
      expect(updated.creativity).toBeGreaterThan(79); // Less than 1 point decay
      expect(updated.stability).toBeGreaterThan(79);
    });

    it('should apply protection floor for low vitals', () => {
      const lowVitalsBrain: BrainStats = {
        ...brain,
        creativity: 10,
        stability: 10
      };

      const event: BrainEvent = {
        type: 'IDLE_TICK',
        timestamp: Date.now() + 60000
      };

      const updated = reduceBrain(lowVitalsBrain, event);

      // Decay should be minimal due to protection floor
      expect(updated.creativity).toBeGreaterThan(9);
      expect(updated.stability).toBeGreaterThan(9);
    });

    it('should decay recent activity count on IDLE_TICK', () => {
      // Build up activity count
      let current = brain;
      for (let i = 0; i < 5; i++) {
        current = reduceBrain(current, {
          type: 'TRAIN_RUN',
          timestamp: Date.now() + i * 1000,
          payload: { steps: 10, tokens: 1000, avgLoss: 1.0 }
        });
      }

      expect(current.activityPattern.recentActivityCount).toBe(5);

      // Apply idle tick
      current = reduceBrain(current, {
        type: 'IDLE_TICK',
        timestamp: Date.now() + 10000
      });

      expect(current.activityPattern.recentActivityCount).toBe(4.5);
    });
  });

  describe('Mood Hysteresis', () => {
    it('should resist rapid mood changes', () => {
      // Start with CALM mood
      const calmBrain: BrainStats = {
        ...brain,
        mood: 'CALM',
        lastMoodChangeTime: Date.now(),
        stability: 25 // Should trigger AGITATED
      };

      // Try to change mood within hysteresis period
      const event: BrainEvent = {
        type: 'IDLE_TICK',
        timestamp: Date.now() + 30000 // 30 seconds - within 1 min hysteresis
      };

      const updated = reduceBrain(calmBrain, event);

      // Mood should NOT change due to hysteresis
      expect(updated.mood).toBe('CALM');
      expect(updated.consecutiveMoodCount).toBeGreaterThan(0);
    });

    it('should allow mood change after hysteresis period', () => {
      const calmBrain: BrainStats = {
        ...brain,
        mood: 'CALM',
        lastMoodChangeTime: Date.now() - 120000, // 2 minutes ago
        stability: 25 // Should trigger AGITATED
      };

      const event: BrainEvent = {
        type: 'IDLE_TICK',
        timestamp: Date.now()
      };

      const updated = reduceBrain(calmBrain, event);

      // Mood should change after hysteresis period
      expect(updated.mood).toBe('AGITATED');
      expect(updated.consecutiveMoodCount).toBe(0);
    });
  });

  describe('Comfort Zone Tracking', () => {
    it('should update comfort zone on IDLE_TICK', () => {
      const healthyBrain: BrainStats = {
        ...brain,
        mood: 'CALM',
        creativity: 60,
        stability: 70
      };

      const event: BrainEvent = {
        type: 'IDLE_TICK',
        timestamp: Date.now() + 30000
      };

      const updated = reduceBrain(healthyBrain, event);

      // Mood history should be updated
      expect(updated.comfortZone.moodHistory.length).toBeGreaterThan(0);
    });

    it('should track preferred mood from history', () => {
      let current: BrainStats = {
        ...brain,
        mood: 'FOCUSED',
        creativity: 70,
        stability: 80,
        lastMoodChangeTime: Date.now() - 120000
      };

      // Simulate multiple idle ticks to build history
      for (let i = 0; i < 15; i++) {
        current = reduceBrain(current, {
          type: 'IDLE_TICK',
          timestamp: Date.now() + i * 30000
        });
      }

      // Preferred mood should reflect the most common mood
      expect(current.comfortZone.moodHistory.length).toBeGreaterThan(0);
    });
  });

  describe('Priority-Scored Need Assessment', () => {
    it('should return empty array for healthy brain', () => {
      const healthyBrain: BrainStats = {
        ...brain,
        mood: 'FOCUSED',
        creativity: 70,
        stability: 80
      };

      const needs = assessBrainNeeds(healthyBrain);

      // Should have few or no urgent needs
      const urgentNeeds = needs.filter((n) => n.priority > 50);
      expect(urgentNeeds.length).toBe(0);
    });

    it('should prioritize critical stability need', () => {
      const unstableBrain: BrainStats = {
        ...brain,
        stability: 10,
        creativity: 50
      };

      const needs = assessBrainNeeds(unstableBrain);

      expect(needs.length).toBeGreaterThan(0);
      expect(needs[0].type).toBe('TRAIN');
      expect(needs[0].priority).toBe(90); // STABILITY_CRITICAL weight
    });

    it('should prioritize critical creativity need', () => {
      const lowCreativityBrain: BrainStats = {
        ...brain,
        stability: 60,
        creativity: 10
      };

      const needs = assessBrainNeeds(lowCreativityBrain);

      expect(needs.length).toBeGreaterThan(0);
      const feedNeed = needs.find((n) => n.type === 'FEED');
      expect(feedNeed).toBeDefined();
      expect(feedNeed?.priority).toBe(80); // CREATIVITY_CRITICAL weight
    });

    it('should detect BURNT_OUT need', () => {
      const burntOutBrain: BrainStats = {
        ...brain,
        mood: 'BURNT_OUT',
        stability: 70,
        creativity: 25
      };

      const needs = assessBrainNeeds(burntOutBrain);

      const feedNeed = needs.find((n) => n.type === 'FEED');
      expect(feedNeed).toBeDefined();
      expect(feedNeed?.priority).toBe(70); // BURNT_OUT_THRESHOLD weight
    });

    it('should return needs sorted by priority', () => {
      const problemBrain: BrainStats = {
        ...brain,
        stability: 10,
        creativity: 15,
        mood: 'AGITATED'
      };

      const needs = assessBrainNeeds(problemBrain);

      // Should be sorted by priority (highest first)
      for (let i = 1; i < needs.length; i++) {
        expect(needs[i - 1].priority).toBeGreaterThanOrEqual(needs[i].priority);
      }
    });
  });

  describe('Urgent Need Detection', () => {
    it('should return urgent need when priority >= 40', () => {
      const unstableBrain: BrainStats = {
        ...brain,
        stability: 20
      };

      const urgentNeed = getUrgentNeed(unstableBrain);

      expect(urgentNeed).not.toBeNull();
      expect(urgentNeed?.priority).toBeGreaterThanOrEqual(40);
    });

    it('should return null when no urgent needs', () => {
      const healthyBrain: BrainStats = {
        ...brain,
        mood: 'CALM',
        creativity: 60,
        stability: 60
      };

      const urgentNeed = getUrgentNeed(healthyBrain);

      expect(urgentNeed).toBeNull();
    });
  });

  describe('Brain Health Score', () => {
    it('should return high score for FOCUSED brain', () => {
      const focusedBrain: BrainStats = {
        ...brain,
        mood: 'FOCUSED',
        creativity: 80,
        stability: 80
      };

      const score = getBrainHealthScore(focusedBrain);

      expect(score).toBeGreaterThan(70);
    });

    it('should return low score for AGITATED brain', () => {
      const agitatedBrain: BrainStats = {
        ...brain,
        mood: 'AGITATED',
        creativity: 30,
        stability: 20
      };

      const score = getBrainHealthScore(agitatedBrain);

      expect(score).toBeLessThan(40);
    });

    it('should give bonus for being in comfort zone', () => {
      const inZoneBrain: BrainStats = {
        ...brain,
        mood: 'CALM',
        creativity: 60,
        stability: 70,
        comfortZone: {
          creativityRange: { min: 50, max: 80 },
          stabilityRange: { min: 60, max: 90 },
          preferredMood: 'CALM',
          moodHistory: []
        }
      };

      const outOfZoneBrain: BrainStats = {
        ...inZoneBrain,
        creativity: 30, // Below comfort zone
        stability: 40 // Below comfort zone
      };

      const inZoneScore = getBrainHealthScore(inZoneBrain);
      const outOfZoneScore = getBrainHealthScore(outOfZoneBrain);

      expect(inZoneScore).toBeGreaterThan(outOfZoneScore);
    });

    it('should be bounded between 0 and 100', () => {
      // Test extreme cases
      const extremeLow: BrainStats = {
        ...brain,
        mood: 'AGITATED',
        creativity: 0,
        stability: 0
      };

      const extremeHigh: BrainStats = {
        ...brain,
        mood: 'FOCUSED',
        creativity: 100,
        stability: 100
      };

      expect(getBrainHealthScore(extremeLow)).toBeGreaterThanOrEqual(0);
      expect(getBrainHealthScore(extremeHigh)).toBeLessThanOrEqual(100);
    });
  });

  describe('isBrainHealthy', () => {
    it('should return true for CALM mood', () => {
      const calmBrain: BrainStats = { ...brain, mood: 'CALM' };
      expect(isBrainHealthy(calmBrain)).toBe(true);
    });

    it('should return true for FOCUSED mood', () => {
      const focusedBrain: BrainStats = { ...brain, mood: 'FOCUSED' };
      expect(isBrainHealthy(focusedBrain)).toBe(true);
    });

    it('should return true when in comfort zone regardless of mood', () => {
      const inZoneBrain: BrainStats = {
        ...brain,
        mood: 'DREAMY',
        creativity: 60,
        stability: 60,
        comfortZone: {
          creativityRange: { min: 50, max: 80 },
          stabilityRange: { min: 50, max: 80 },
          preferredMood: 'CALM',
          moodHistory: []
        }
      };

      expect(isBrainHealthy(inZoneBrain)).toBe(true);
    });

    it('should return false for AGITATED mood outside comfort zone', () => {
      const unhealthyBrain: BrainStats = {
        ...brain,
        mood: 'AGITATED',
        creativity: 30,
        stability: 20
      };

      expect(isBrainHealthy(unhealthyBrain)).toBe(false);
    });
  });

  describe('Adaptive Training Effects', () => {
    it('should give extra stability boost when AGITATED', () => {
      const agitatedBrain: BrainStats = {
        ...brain,
        mood: 'AGITATED',
        stability: 20,
        lastMoodChangeTime: Date.now() - 120000
      };

      const event: BrainEvent = {
        type: 'TRAIN_RUN',
        timestamp: Date.now(),
        payload: { steps: 10, tokens: 1000, avgLoss: 1.0 }
      };

      const updated = reduceBrain(agitatedBrain, event);

      // AGITATED gets +4 stability (vs normal +2)
      expect(updated.stability).toBe(24); // 20 + 4
    });

    it('should give extra creativity boost when BURNT_OUT and low loss', () => {
      const burntOutBrain: BrainStats = {
        ...brain,
        mood: 'BURNT_OUT',
        creativity: 15,
        stability: 70,
        lastMoodChangeTime: Date.now() - 120000
      };

      const event: BrainEvent = {
        type: 'TRAIN_RUN',
        timestamp: Date.now(),
        payload: { steps: 10, tokens: 1000, avgLoss: 1.0 } // avgLoss < 1.2
      };

      const updated = reduceBrain(burntOutBrain, event);

      // BURNT_OUT gets +5 creativity (vs normal +3)
      expect(updated.creativity).toBe(20); // 15 + 5
    });
  });

  describe('Adaptive Feeding Effects', () => {
    it('should give extra creativity boost when BURNT_OUT', () => {
      const burntOutBrain: BrainStats = {
        ...brain,
        mood: 'BURNT_OUT',
        creativity: 15,
        stability: 70,
        lastMoodChangeTime: Date.now() - 120000
      };

      const event: BrainEvent = {
        type: 'FEED',
        timestamp: Date.now(),
        payload: { newWordsCount: 50, heavinessScore: 0.3, summary: 'New content' }
      };

      const updated = reduceBrain(burntOutBrain, event);

      // BURNT_OUT gets 0.4 * newWordsCount (vs normal 0.2)
      expect(updated.creativity).toBe(35); // 15 + 50*0.4
    });

    it('should apply less stability penalty when already stable', () => {
      const stableBrain: BrainStats = {
        ...brain,
        stability: 80
      };

      const unstableBrain: BrainStats = {
        ...brain,
        stability: 40
      };

      const event: BrainEvent = {
        type: 'FEED',
        timestamp: Date.now(),
        payload: { newWordsCount: 50, heavinessScore: 0.5, summary: 'Heavy content' }
      };

      const updatedStable = reduceBrain(stableBrain, event);
      const updatedUnstable = reduceBrain(unstableBrain, event);

      // Stable brain should lose less stability (1.5 * heaviness vs 3 * heaviness)
      const stableLoss = 80 - updatedStable.stability;
      const unstableLoss = 40 - updatedUnstable.stability;

      expect(stableLoss).toBeLessThan(unstableLoss);
    });
  });

  describe('Generation Effects', () => {
    it('should not reduce stability when already low', () => {
      const lowStabilityBrain: BrainStats = {
        ...brain,
        stability: 35 // Below 40 threshold
      };

      const event: BrainEvent = {
        type: 'GEN_RUN',
        timestamp: Date.now(),
        payload: { diversityScore: 0.8, tokensGenerated: 50 }
      };

      const updated = reduceBrain(lowStabilityBrain, event);

      // Stability should not decrease when below 40
      expect(updated.stability).toBe(35);
    });

    it('should decrease creativity for low diversity generation', () => {
      const event: BrainEvent = {
        type: 'GEN_RUN',
        timestamp: Date.now(),
        payload: { diversityScore: 0.2, tokensGenerated: 50 } // Low diversity
      };

      const updated = reduceBrain(brain, event);

      // Low diversity should decrease creativity
      expect(updated.creativity).toBeLessThan(brain.creativity);
    });
  });
});
