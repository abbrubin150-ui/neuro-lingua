/**
 * Tests for the Triadic Operator
 */

import { describe, it, expect } from 'vitest';
import {
  triadicOperator,
  verifyTriadicVector,
  triadicVectorToString,
  generateTriadicTruthTable,
} from '../src/lib/triadicOperator';

describe('Triadic Operator', () => {
  describe('triadicOperator', () => {
    it('should return complete triad (strong) for all true inputs', () => {
      const result = triadicOperator(true, true, true);
      expect(result.weak).toBe(true);
      expect(result.strong).toBe(true);
      expect(result.tension).toBe(false);
      expect(result.null).toBe(false);
    });

    it('should return null for all false inputs', () => {
      const result = triadicOperator(false, false, false);
      expect(result.weak).toBe(false);
      expect(result.strong).toBe(false);
      expect(result.tension).toBe(false);
      expect(result.null).toBe(true);
    });

    it('should return tension for A=true, B=true, C=false', () => {
      const result = triadicOperator(true, true, false);
      expect(result.weak).toBe(true);
      expect(result.strong).toBe(false);
      expect(result.tension).toBe(true);
      expect(result.null).toBe(false);
    });

    it('should return tension for A=false, B=true, C=true', () => {
      const result = triadicOperator(false, true, true);
      expect(result.weak).toBe(true);
      expect(result.strong).toBe(false);
      expect(result.tension).toBe(true);
      expect(result.null).toBe(false);
    });

    it('should return weak (but not strong or tension) for A=true, B=false, C=true', () => {
      const result = triadicOperator(true, false, true);
      expect(result.weak).toBe(true);
      expect(result.strong).toBe(false);
      expect(result.tension).toBe(false);
      expect(result.null).toBe(false);
    });
  });

  describe('verifyTriadicVector', () => {
    it('should verify that strong implies weak', () => {
      const result = triadicOperator(true, true, true);
      expect(verifyTriadicVector(result)).toBe(true);
      expect(result.strong).toBe(true);
      expect(result.weak).toBe(true);
    });

    it('should verify that null implies not weak', () => {
      const result = triadicOperator(false, false, false);
      expect(verifyTriadicVector(result)).toBe(true);
      expect(result.null).toBe(true);
      expect(result.weak).toBe(false);
    });

    it('should verify that strong and tension are mutually exclusive', () => {
      // Test all 8 combinations
      for (let a = 0; a <= 1; a++) {
        for (let b = 0; b <= 1; b++) {
          for (let c = 0; c <= 1; c++) {
            const result = triadicOperator(Boolean(a), Boolean(b), Boolean(c));
            expect(verifyTriadicVector(result)).toBe(true);
            if (result.strong) {
              expect(result.tension).toBe(false);
            }
          }
        }
      }
    });

    it('should reject invalid vectors', () => {
      // Strong but not weak (invalid)
      const invalid1 = { weak: false, strong: true, tension: false, null: false };
      expect(verifyTriadicVector(invalid1)).toBe(false);

      // Null and weak (invalid)
      const invalid2 = { weak: true, strong: false, tension: false, null: true };
      expect(verifyTriadicVector(invalid2)).toBe(false);

      // Strong and tension (invalid)
      const invalid3 = { weak: true, strong: true, tension: true, null: false };
      expect(verifyTriadicVector(invalid3)).toBe(false);
    });
  });

  describe('triadicVectorToString', () => {
    it('should format complete triad correctly', () => {
      const result = triadicOperator(true, true, true);
      const str = triadicVectorToString(result);
      expect(str).toBe('⟨W,S⟩');
    });

    it('should format tension correctly', () => {
      const result = triadicOperator(true, true, false);
      const str = triadicVectorToString(result);
      expect(str).toBe('⟨W,T⟩');
    });

    it('should format null correctly', () => {
      const result = triadicOperator(false, false, false);
      const str = triadicVectorToString(result);
      expect(str).toBe('⟨N⟩');
    });

    it('should format weak-only correctly', () => {
      const result = triadicOperator(true, false, true);
      const str = triadicVectorToString(result);
      expect(str).toBe('⟨W⟩');
    });
  });

  describe('generateTriadicTruthTable', () => {
    it('should generate 8 rows (2^3 combinations)', () => {
      const table = generateTriadicTruthTable();
      expect(table).toHaveLength(8);
    });

    it('should have all valid vectors', () => {
      const table = generateTriadicTruthTable();
      table.forEach((row) => {
        expect(row.valid).toBe(true);
      });
    });

    it('should cover all boolean combinations', () => {
      const table = generateTriadicTruthTable();
      const combinations = table.map((row) => `${row.a ? 1 : 0}${row.b ? 1 : 0}${row.c ? 1 : 0}`);
      const expected = ['000', '001', '010', '011', '100', '101', '110', '111'];
      expect(combinations.sort()).toEqual(expected.sort());
    });
  });

  describe('Invariants across all inputs', () => {
    it('should maintain all invariants for every possible input', () => {
      for (let a = 0; a <= 1; a++) {
        for (let b = 0; b <= 1; b++) {
          for (let c = 0; c <= 1; c++) {
            const aVal = Boolean(a);
            const bVal = Boolean(b);
            const cVal = Boolean(c);
            const result = triadicOperator(aVal, bVal, cVal);

            // Verify the vector is valid
            expect(verifyTriadicVector(result)).toBe(true);

            // Strong implies weak
            if (result.strong) {
              expect(result.weak).toBe(true);
            }

            // Null implies not weak
            if (result.null) {
              expect(result.weak).toBe(false);
            }

            // Strong implies not null
            if (result.strong) {
              expect(result.null).toBe(false);
            }

            // Strong and tension are mutually exclusive
            if (result.strong) {
              expect(result.tension).toBe(false);
            }
            if (result.tension) {
              expect(result.strong).toBe(false);
            }
          }
        }
      }
    });
  });

  describe('Known truth table values', () => {
    it('should match expected values for key combinations', () => {
      // (0,0,0) -> null only
      expect(triadicVectorToString(triadicOperator(false, false, false))).toBe('⟨N⟩');

      // (1,1,1) -> weak and strong
      expect(triadicVectorToString(triadicOperator(true, true, true))).toBe('⟨W,S⟩');

      // (1,1,0) -> weak and tension
      expect(triadicVectorToString(triadicOperator(true, true, false))).toBe('⟨W,T⟩');

      // (0,1,1) -> weak and tension
      expect(triadicVectorToString(triadicOperator(false, true, true))).toBe('⟨W,T⟩');

      // (1,0,1) -> weak only
      expect(triadicVectorToString(triadicOperator(true, false, true))).toBe('⟨W⟩');

      // (0,0,1) -> weak only
      expect(triadicVectorToString(triadicOperator(false, false, true))).toBe('⟨W⟩');

      // (1,0,0) -> weak only
      expect(triadicVectorToString(triadicOperator(true, false, false))).toBe('⟨W⟩');

      // (0,1,0) -> weak only
      expect(triadicVectorToString(triadicOperator(false, true, false))).toBe('⟨W⟩');
    });
  });
});
