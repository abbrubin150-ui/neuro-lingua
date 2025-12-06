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
        alternatives: [],
        decision: 'Proceed with training',
        witness: 'JohnDoe',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should return EXECUTE when expiry is in the future', () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 7); // 7 days from now

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: futureDate.toISOString()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should return EXECUTE with approved status', () => {
      const ledger: DecisionLedger = {
        rationale: 'Approved decision',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        approvals: ['Manager1', 'Manager2']
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
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: pastDate.toISOString()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });

    it('should return HOLD when expiry was hours ago', () => {
      const pastDate = new Date();
      pastDate.setHours(pastDate.getHours() - 2); // 2 hours ago

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: pastDate.toISOString()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });

    it('should return HOLD even with valid rationale and witness if expired', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 30); // 30 days ago

      const ledger: DecisionLedger = {
        rationale: 'Very detailed and valid rationale',
        alternatives: ['Option A', 'Option B'],
        decision: 'Proceed with training',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: pastDate.toISOString(),
        approvals: ['Manager1', 'Manager2']
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('ESCALATE status - Missing rationale', () => {
    it('should return ESCALATE when rationale is missing', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when rationale is only whitespace', () => {
      const ledger: DecisionLedger = {
        rationale: '   \n  \t  ',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when rationale is undefined', () => {
      const ledger: DecisionLedger = {
        rationale: undefined as any,
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });
  });

  describe('ESCALATE status - Missing witness', () => {
    it('should return ESCALATE when witness is missing', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: '',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when witness is only whitespace', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: '   \t\n   ',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should return ESCALATE when witness is undefined', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: undefined as any,
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });
  });

  describe('ESCALATE status - Multiple missing fields', () => {
    it('should return ESCALATE when both rationale and witness are missing', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        alternatives: [],
        decision: 'Proceed',
        witness: '',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should check expiry first before checking witness', () => {
      // Even if witness is missing, expired ledger returns HOLD
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: '', // Missing witness
        timestamp: Date.now(),
        expiry: pastDate.toISOString()
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
        alternatives: [],
        decision: 'Proceed',
        witness: '', // Missing
        timestamp: Date.now(),
        expiry: pastDate.toISOString() // Expired
      };

      // Should return HOLD, not ESCALATE, because expiry is checked first
      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('Edge cases', () => {
    it('should handle ledger with minimal valid fields', () => {
      const ledger: DecisionLedger = {
        rationale: 'R',
        alternatives: [],
        decision: 'D',
        witness: 'W',
        timestamp: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should handle ledger with all optional fields populated', () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 30);

      const ledger: DecisionLedger = {
        rationale: 'Comprehensive rationale',
        alternatives: ['Alt1', 'Alt2', 'Alt3'],
        decision: 'Proceed with Option 1',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: futureDate.toISOString(),
        approvals: ['Manager1', 'Manager2', 'Manager3'],
        linkedRunIds: ['run-1', 'run-2']
      };

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should handle exact expiry boundary (current time)', () => {
      const now = new Date();

      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: now.toISOString()
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
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now(),
        expiry: pastDate.toISOString()
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
        alternatives: [],
        decision: 'Proceed',
        witness: 'JohnDoe',
        timestamp: Date.now()
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
        alternatives: ['Option A', 'Option B'],
        decision: 'Proceed with Option A',
        witness: 'JohnDoe',
        timestamp: Date.now()
      };

      const status = computeExecutionStatus(ledger);

      // This test documents the requirement:
      // Training should only proceed when status is EXECUTE
      expect(status).toBe('EXECUTE');
    });
  });
});
