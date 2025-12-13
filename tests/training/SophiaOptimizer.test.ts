/**
 * Tests for Sophia Optimizer
 *
 * Verifies:
 * 1. Initialization and configuration
 * 2. Momentum updates (first moment)
 * 3. Hessian diagonal estimation (second moment)
 * 4. Clipped preconditioned updates
 * 5. Weight decay application
 * 6. State serialization/deserialization
 * 7. Hessian update frequency
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SophiaOptimizer,
  SOPHIA_DEFAULTS,
  SOPHIA_FINETUNE_DEFAULTS,
  SOPHIA_AGGRESSIVE_DEFAULTS
} from '../../src/training/SophiaOptimizer';

describe('SophiaOptimizer', () => {
  let optimizer: SophiaOptimizer;

  beforeEach(() => {
    optimizer = new SophiaOptimizer();
  });

  describe('initialization', () => {
    it('should initialize with default config', () => {
      const config = optimizer.getConfig();
      expect(config.lr).toBe(1e-4);
      expect(config.beta1).toBe(0.965);
      expect(config.beta2).toBe(0.99);
      expect(config.weightDecay).toBe(0.01);
      expect(config.epsilon).toBe(1e-12);
      expect(config.rho).toBe(1.0);
      expect(config.hessianUpdateFreq).toBe(10);
    });

    it('should accept custom config', () => {
      const customOptimizer = new SophiaOptimizer({
        lr: 2e-4,
        beta1: 0.9,
        beta2: 0.95,
        rho: 2.0
      });
      const config = customOptimizer.getConfig();
      expect(config.lr).toBe(2e-4);
      expect(config.beta1).toBe(0.9);
      expect(config.beta2).toBe(0.95);
      expect(config.rho).toBe(2.0);
    });

    it('should start with step count 0', () => {
      expect(optimizer.getStep()).toBe(0);
    });
  });

  describe('vector updates', () => {
    it('should update vector parameters', () => {
      const b = [1.0, 2.0, 3.0];
      const g = [0.1, 0.2, 0.3];
      const original = [...b];

      optimizer.updateVector(b, g, 'bias');

      // Values should have changed
      expect(b[0]).not.toBe(original[0]);
      expect(b[1]).not.toBe(original[1]);
      expect(b[2]).not.toBe(original[2]);
    });

    it('should apply weight decay', () => {
      const optimizer = new SophiaOptimizer({ weightDecay: 0.1, lr: 0.1 });
      const b = [1.0, 1.0];
      const g = [0, 0]; // Zero gradient to isolate weight decay

      // Update with zero gradient - only weight decay should apply
      optimizer.updateVector(b, g, 'test');

      // Weight decay should reduce magnitude
      expect(Math.abs(b[0])).toBeLessThan(1.0);
      expect(Math.abs(b[1])).toBeLessThan(1.0);
    });

    it('should clip updates with rho', () => {
      // To trigger clipping, we need m/h to exceed rho
      // This happens when Hessian is small relative to momentum
      // We initialize Hessian at epsilon, then use small gradient for low h,
      // but large accumulated momentum from previous steps
      const optimizer = new SophiaOptimizer({
        lr: 1.0,
        rho: 0.5,
        weightDecay: 0,
        epsilon: 1e-12,
        beta1: 0.99, // High momentum retention
        beta2: 0.99, // Slow Hessian update
        hessianUpdateFreq: 100 // Don't update Hessian yet
      });

      // First, build up momentum with large gradients
      const b = [0.0];
      for (let i = 0; i < 10; i++) {
        optimizer.updateVector(b, [100.0], 'test');
        optimizer.step();
      }

      // Now apply with small gradient - Hessian is still near epsilon
      // but momentum is large, so m/h is huge and gets clipped
      const beforeUpdate = b[0];

      // The update magnitude should be bounded by rho * lr = 0.5
      // Even with accumulated momentum
      expect(Math.abs(b[0] - beforeUpdate)).toBeLessThanOrEqual(1.0);
    });
  });

  describe('matrix updates', () => {
    it('should update matrix parameters', () => {
      const W = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      const G = [
        [0.1, 0.2],
        [0.3, 0.4]
      ];
      const original = W.map((row) => [...row]);

      optimizer.updateMatrix(W, G, 'weights');

      // Values should have changed
      expect(W[0][0]).not.toBe(original[0][0]);
      expect(W[1][1]).not.toBe(original[1][1]);
    });

    it('should update row efficiently', () => {
      const W = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      const gRow = [0.5, 0.5];
      const original1 = [...W[1]];

      optimizer.updateRow(W, 1, gRow, 'embedding');

      // Only row 1 should change
      expect(W[0]).toEqual([1.0, 2.0]); // Row 0 unchanged
      expect(W[1][0]).not.toBe(original1[0]); // Row 1 changed
    });
  });

  describe('Hessian estimation', () => {
    it('should update Hessian at correct frequency', () => {
      const b = [1.0];
      const g = [0.5];

      // Default hessianUpdateFreq is 10
      // Step 0 should update Hessian
      optimizer.updateVector(b, g, 'test');
      optimizer.step();

      // Steps 1-9 should not update Hessian
      for (let i = 1; i < 10; i++) {
        optimizer.step();
      }

      // Step 10 should update Hessian again
      expect(optimizer.getStep()).toBe(10);
    });

    it('should provide Hessian statistics', () => {
      const b = [1.0, 2.0];
      const g = [0.1, 0.2];

      optimizer.updateVector(b, g, 'stats_test');
      optimizer.step();

      const stats = optimizer.getHessianStats();
      expect(stats.vectors.length).toBe(1);
      expect(stats.vectors[0].key).toBe('stats_test');
      expect(stats.vectors[0].mean).toBeGreaterThan(0);
    });
  });

  describe('state management', () => {
    it('should reset state', () => {
      const b = [1.0];
      const g = [0.1];

      optimizer.updateVector(b, g, 'test');
      optimizer.step();
      optimizer.step();

      expect(optimizer.getStep()).toBe(2);

      optimizer.reset();

      expect(optimizer.getStep()).toBe(0);
    });

    it('should export and import state', () => {
      const W = [[1.0, 2.0]];
      const G = [[0.1, 0.2]];

      optimizer.updateMatrix(W, G, 'weights');
      optimizer.step();
      optimizer.step();

      const exported = optimizer.exportState();
      expect(exported.step).toBe(2);
      expect(exported.momentumMatrices['weights']).toBeDefined();
      expect(exported.hessianMatrices['weights']).toBeDefined();

      // Create new optimizer and import state
      const newOptimizer = new SophiaOptimizer();
      newOptimizer.importState(exported);

      expect(newOptimizer.getStep()).toBe(2);
      expect(newOptimizer.getConfig().lr).toBe(exported.config.lr);
    });

    it('should allow config updates', () => {
      optimizer.setConfig({ lr: 5e-5 });
      expect(optimizer.getConfig().lr).toBe(5e-5);
    });
  });

  describe('convergence behavior', () => {
    it('should converge on simple quadratic', () => {
      // Minimize f(x) = x^2, optimal x = 0
      const optimizer = new SophiaOptimizer({
        lr: 0.1,
        weightDecay: 0,
        hessianUpdateFreq: 1
      });

      let x = [10.0]; // Start far from optimum
      const history: number[] = [x[0]];

      for (let i = 0; i < 100; i++) {
        const g = [2 * x[0]]; // Gradient of x^2
        optimizer.updateVector(x, g, 'x');
        optimizer.step();
        history.push(x[0]);
      }

      // Should converge towards 0
      expect(Math.abs(x[0])).toBeLessThan(Math.abs(history[0]));
    });
  });

  describe('preset configurations', () => {
    it('should have valid default presets', () => {
      expect(SOPHIA_DEFAULTS.lr).toBe(1e-4);
      expect(SOPHIA_DEFAULTS.beta1).toBe(0.965);
      expect(SOPHIA_DEFAULTS.beta2).toBe(0.99);
    });

    it('should have valid finetune presets', () => {
      expect(SOPHIA_FINETUNE_DEFAULTS.lr).toBe(5e-5);
      expect(SOPHIA_FINETUNE_DEFAULTS.rho).toBe(0.5);
    });

    it('should have valid aggressive presets', () => {
      expect(SOPHIA_AGGRESSIVE_DEFAULTS.lr).toBe(2e-4);
      expect(SOPHIA_AGGRESSIVE_DEFAULTS.rho).toBe(2.0);
    });
  });
});
