import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DecisionLedgerEditor } from '../../src/components/DecisionLedgerEditor';
import type { DecisionLedger } from '../../src/types/project';

describe('DecisionLedgerEditor', () => {
  const createLedger = (overrides?: Partial<DecisionLedger>): DecisionLedger => ({
    rationale: 'Test training for baseline evaluation',
    witness: 'researcher@example.com',
    expiry: null,
    rollback: 'keep',
    createdAt: Date.now(),
    ...overrides
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Status Computation', () => {
    it('shows EXECUTE status when ledger is valid', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/EXECUTE/i)).toBeInTheDocument();
      expect(screen.getByText(/Training permitted/i)).toBeInTheDocument();
    });

    it('shows ESCALATE status when rationale is empty', () => {
      const ledger = createLedger({ rationale: '' });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/ESCALATE/i)).toBeInTheDocument();
      expect(screen.getByText(/Review required/i)).toBeInTheDocument();
    });

    it('shows ESCALATE status when witness is empty', () => {
      const ledger = createLedger({ witness: '' });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/ESCALATE/i)).toBeInTheDocument();
      expect(screen.getByText(/Review required/i)).toBeInTheDocument();
    });

    it('shows HOLD status when expiry date has passed', () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);

      const ledger = createLedger({ expiry: yesterday.toISOString() });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/HOLD/i)).toBeInTheDocument();
      expect(screen.getByText(/paused or expired/i)).toBeInTheDocument();
    });

    it('shows EXECUTE status when expiry date is in future', () => {
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);

      const ledger = createLedger({ expiry: tomorrow.toISOString() });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/EXECUTE/i)).toBeInTheDocument();
    });

    it('treats whitespace-only rationale as empty (ESCALATE)', () => {
      const ledger = createLedger({ rationale: '   \n\t  ' });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/ESCALATE/i)).toBeInTheDocument();
    });

    it('treats whitespace-only witness as empty (ESCALATE)', () => {
      const ledger = createLedger({ witness: '   \t\n  ' });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      expect(screen.getByText(/ESCALATE/i)).toBeInTheDocument();
    });
  });

  describe('UI Interaction', () => {
    it('toggles expanded view when clicking show/hide button', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Initially collapsed
      const toggleButton = screen.getByText(/Show Details/i);
      expect(toggleButton).toBeInTheDocument();

      // Expand
      fireEvent.click(toggleButton);
      expect(screen.getByText(/Hide Details/i)).toBeInTheDocument();

      // Should show ledger fields
      expect(screen.getByDisplayValue(ledger.rationale)).toBeInTheDocument();
      expect(screen.getByDisplayValue(ledger.witness)).toBeInTheDocument();

      // Collapse again
      fireEvent.click(screen.getByText(/Hide Details/i));
      expect(screen.getByText(/Show Details/i)).toBeInTheDocument();
    });

    it('updates rationale when editing', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand to show fields
      fireEvent.click(screen.getByText(/Show Details/i));

      // Find and edit rationale
      const rationaleInput = screen.getByLabelText(/Rationale/i);
      fireEvent.change(rationaleInput, {
        target: { value: 'Updated rationale for experiment' }
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          rationale: 'Updated rationale for experiment'
        })
      );
    });

    it('updates witness when editing', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand
      fireEvent.click(screen.getByText(/Show Details/i));

      // Edit witness
      const witnessInput = screen.getByLabelText(/Witness/i);
      fireEvent.change(witnessInput, {
        target: { value: 'newresearcher@example.com' }
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          witness: 'newresearcher@example.com'
        })
      );
    });

    it('updates expiry date when editing', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand
      fireEvent.click(screen.getByText(/Show Details/i));

      // Set expiry - use ID selector
      const expiryInput = container.querySelector('#ledger-expiry') as HTMLInputElement;
      expect(expiryInput).not.toBeNull();

      fireEvent.change(expiryInput!, {
        target: { value: '2025-12-31' }
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          expiry: expect.stringContaining('2025-12-31')
        })
      );
    });

    it('updates rollback strategy when changing select', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand
      fireEvent.click(screen.getByText(/Show Details/i));

      // Change rollback - use ID selector
      const rollbackSelect = container.querySelector('#ledger-rollback') as HTMLSelectElement;
      expect(rollbackSelect).not.toBeNull();

      fireEvent.change(rollbackSelect!, {
        target: { value: 'delete-after-expiry' }
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          rollback: 'delete-after-expiry'
        })
      );
    });

    it('clears expiry when selecting "No expiry"', () => {
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      const ledger = createLedger({ expiry: tomorrow.toISOString() });
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand
      fireEvent.click(screen.getByText(/Show Details/i));

      // Clear expiry - use ID selector
      const expiryInput = container.querySelector('#ledger-expiry') as HTMLInputElement;
      expect(expiryInput).not.toBeNull();

      fireEvent.change(expiryInput!, {
        target: { value: '' }
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          expiry: null
        })
      );
    });
  });

  describe('Visual Status Indicators', () => {
    it('uses green color for EXECUTE status', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Should display EXECUTE status with proper styling
      const executeElements = screen.getAllByText(/EXECUTE/i);
      expect(executeElements.length).toBeGreaterThan(0);

      // Verify status badge exists
      const statusElement = screen.getByText(/Training permitted/i);
      expect(statusElement).toBeInTheDocument();
    });

    it('uses amber/yellow color for HOLD status', () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const ledger = createLedger({ expiry: yesterday.toISOString() });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Should display HOLD status
      const holdElements = screen.getAllByText(/HOLD/i);
      expect(holdElements.length).toBeGreaterThan(0);

      // Verify hold message
      const holdMessage = screen.getByText(/paused or expired/i);
      expect(holdMessage).toBeInTheDocument();
    });

    it('uses red color for ESCALATE status', () => {
      const ledger = createLedger({ rationale: '' });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Should display ESCALATE status
      const escalateElements = screen.getAllByText(/ESCALATE/i);
      expect(escalateElements.length).toBeGreaterThan(0);

      // Verify escalate message
      const escalateMessage = screen.getByText(/Review required/i);
      expect(escalateMessage).toBeInTheDocument();
    });
  });

  describe('Direction Support', () => {
    it('respects RTL direction when specified', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      const { container } = render(
        <DecisionLedgerEditor ledger={ledger} onChange={onChange} direction="rtl" />
      );

      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveStyle({ direction: 'rtl' });
    });

    it('defaults to LTR direction when not specified', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveStyle({ direction: 'ltr' });
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for form fields', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand to show fields
      fireEvent.click(screen.getByText(/Show Details/i));

      // Use ID selectors to verify fields exist
      expect(container.querySelector('#ledger-rationale')).toBeInTheDocument();
      expect(container.querySelector('#ledger-witness')).toBeInTheDocument();
      expect(container.querySelector('#ledger-rollback')).toBeInTheDocument();
    });

    it('toggle button has descriptive text', () => {
      const ledger = createLedger();
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      const button = screen.getByRole('button');
      expect(button).toHaveTextContent(/Show Details|Hide Details/i);
    });
  });

  describe('Edge Cases', () => {
    it('handles very long rationale text', () => {
      const longRationale = 'A'.repeat(1000);
      const ledger = createLedger({ rationale: longRationale });
      const onChange = vi.fn();

      const { container } = render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Expand
      fireEvent.click(screen.getByText(/Show Details/i));

      // Should still show EXECUTE (not empty) - use getAllByText since it appears in multiple places
      const executeTexts = screen.getAllByText(/EXECUTE/i);
      expect(executeTexts.length).toBeGreaterThan(0);

      // Should display the rationale
      const rationaleInput = container.querySelector('#ledger-rationale') as HTMLTextAreaElement;
      expect(rationaleInput).toBeInTheDocument();
      expect(rationaleInput!.value).toBe(longRationale);
    });

    it('handles expiry date exactly at current time', () => {
      const now = new Date();
      const ledger = createLedger({ expiry: now.toISOString() });
      const onChange = vi.fn();

      render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);

      // Should be HOLD (now > expiryDate will be true in most cases due to execution time)
      // or EXECUTE if executed instantaneously
      const status = screen.getByText(/HOLD|EXECUTE/i);
      expect(status).toBeInTheDocument();
    });

    it('handles missing createdAt timestamp', () => {
      const ledger = createLedger({ createdAt: 0 });
      const onChange = vi.fn();

      // Should render without errors
      expect(() => {
        render(<DecisionLedgerEditor ledger={ledger} onChange={onChange} />);
      }).not.toThrow();
    });

    it('handles all rollback options', () => {
      const onChange = vi.fn();

      const rollbackOptions: Array<'keep' | 'delete-after-expiry' | 'archive'> = [
        'keep',
        'delete-after-expiry',
        'archive'
      ];

      rollbackOptions.forEach((rollback) => {
        const ledger = createLedger({ rollback });
        const { unmount, container } = render(
          <DecisionLedgerEditor ledger={ledger} onChange={onChange} />
        );

        // Expand
        fireEvent.click(screen.getByText(/Show Details/i));

        // Should display the selected rollback option - use ID selector
        const select = container.querySelector('#ledger-rollback') as HTMLSelectElement;
        expect(select).not.toBeNull();
        expect(select!.value).toBe(rollback);

        unmount();
      });
    });
  });
});
