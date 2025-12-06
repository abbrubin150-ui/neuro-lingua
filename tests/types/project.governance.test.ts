/**
 * Governance Execution Status Tests
 *
 * These tests verify that the Decision Ledger governance system
 * correctly blocks training on HOLD and ESCALATE statuses.
 */

import { describe, it, expect } from 'vitest';
import { computeExecutionStatus, type DecisionLedger } from '../../src/types/project';

describe('Governance Execution Status', () => {
  describe('EXECUTE status', () => {
    it('should return EXECUTE when all requirements are met', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale for this decision',
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should return EXECUTE when expiry is in the future', () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 7); // 7 days from now

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: 'JohnDoe',
        expiry: futureDate.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should return EXECUTE with all fields populated', () => {
      const ledger: DecisionLedger = {
        rationale: 'Approved decision',
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'archive',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });
  });

  describe('HOLD status - Expiry validation', () => {
    it('should return HOLD when expiry date has passed', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1); // Yesterday

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: 'JohnDoe',
        expiry: pastDate.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });

    it('should return HOLD when expiry was hours ago', () => {
      const pastDate = new Date();
      pastDate.setHours(pastDate.getHours() - 2); // 2 hours ago

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: 'JohnDoe',
        expiry: pastDate.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });

    it('should return HOLD even with valid rationale and witness if expired', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 30); // 30 days ago

      const ledger: DecisionLedger = {
        rationale: 'Very detailed and valid rationale',
        witness: 'JohnDoe',
        expiry: pastDate.toISOString(),
        rollback: 'archive',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('ESCALATE status - Missing rationale', () => {
    it('should return ESCALATE when rationale is missing', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when rationale is only whitespace', () => {
      const ledger: DecisionLedger = {
        rationale: '   \n  \t  ',
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when rationale is undefined', () => {
      const ledger: DecisionLedger = {
        rationale: undefined as any,
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });
  });

  describe('ESCALATE status - Missing witness', () => {
    it('should return ESCALATE when witness is missing', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: '',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when witness is only whitespace', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: '   \t\n   ',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when witness is undefined', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: undefined as any,
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });
  });

  describe('ESCALATE status - Multiple missing fields', () => {
    it('should return ESCALATE when both rationale and witness are missing', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        witness: '',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should check expiry first before checking witness', () => {
      // Even if witness is missing, expired ledger returns HOLD
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: '', // Missing witness
        expiry: pastDate.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      // Expiry is checked first, so should return HOLD
      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('Priority of statuses', () => {
    it('should prioritize HOLD (expiry) over ESCALATE (missing fields)', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger: DecisionLedger = {
        rationale: '', // Missing
        witness: '', // Missing
        expiry: pastDate.toISOString(), // Expired
        rollback: 'keep',
        createdAt: Date.now()
      };

      // Should return HOLD, not ESCALATE, because expiry is checked first
      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('Edge cases', () => {
    it('should handle ledger with minimal valid fields', () => {
      const ledger: DecisionLedger = {
        rationale: 'R',
        witness: 'W',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should handle ledger with all fields populated', () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 30);

      const ledger: DecisionLedger = {
        rationale: 'Comprehensive rationale',
        witness: 'JohnDoe',
        expiry: futureDate.toISOString(),
        rollback: 'archive',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should handle exact expiry boundary (current time)', () => {
      const now = new Date();

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: 'JohnDoe',
        expiry: now.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      // At exact boundary, depends on millisecond precision
      // This should be HOLD as now > expiryDate once a millisecond passes
      const status = computeExecutionStatus(ledger);
      expect(['HOLD', 'EXECUTE']).toContain(status);
    });
  });

  describe('Documentation of training blocking behavior', () => {
    it('documents that HOLD status should prevent training', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger: DecisionLedger = {
        rationale: 'Expired decision',
        witness: 'JohnDoe',
        expiry: pastDate.toISOString(),
        rollback: 'keep',
        createdAt: Date.now()
      };

      const status = computeExecutionStatus(ledger);

      // This test documents the requirement:
      // Training should be blocked when status is HOLD
      expect(status).toBe('HOLD');
      expect(status).not.toBe('EXECUTE');
    });

    it('documents that ESCALATE status should prevent training', () => {
      const ledger: DecisionLedger = {
        rationale: '', // Missing rationale
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      const status = computeExecutionStatus(ledger);

      // This test documents the requirement:
      // Training should be blocked when status is ESCALATE
      expect(status).toBe('ESCALATE');
      expect(status).not.toBe('EXECUTE');
    });

    it('documents that only EXECUTE status should allow training', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid and complete rationale',
        witness: 'JohnDoe',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      const status = computeExecutionStatus(ledger);

      // This test documents the requirement:
      // Training should only proceed when status is EXECUTE
      expect(status).toBe('EXECUTE');
    });
  });
});
