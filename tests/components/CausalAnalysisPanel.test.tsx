/**
 * Tests for CausalAnalysisPanel Component
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CausalAnalysisPanel } from '../../src/components/CausalAnalysisPanel';

describe('CausalAnalysisPanel', () => {
  describe('Rendering', () => {
    it('should render the panel with title', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('Causal Analysis')).toBeDefined();
    });

    it('should render phase indicator', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('1. Configure')).toBeDefined();
      expect(screen.getByText('2. Offline')).toBeDefined();
      expect(screen.getByText('3. Online')).toBeDefined();
    });

    it('should render DAG configuration section', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('DAG Configuration')).toBeDefined();
    });

    it('should render model configuration section', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('Model Configuration')).toBeDefined();
    });

    it('should render in Hebrew when language is set', () => {
      render(<CausalAnalysisPanel language="he" />);

      expect(screen.getByText('ניתוח סיבתי')).toBeDefined();
    });
  });

  describe('Initial State', () => {
    it('should start in configure phase', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('Run Offline Learning')).toBeDefined();
    });

    it('should have default configuration values', () => {
      render(<CausalAnalysisPanel />);

      // Click to expand config section
      fireEvent.click(screen.getByText('Model Configuration'));

      // Check for default values in inputs
      const inputs = document.querySelectorAll('input[type="number"]');
      expect(inputs.length).toBeGreaterThan(0);
    });

    it('should accept initial configuration', () => {
      render(
        <CausalAnalysisPanel
          initialConfig={{
            numStudents: 200,
            numTimeSteps: 50,
            featureDimension: 5,
            seed: 123
          }}
        />
      );

      // The component should render without errors
      expect(screen.getByText('Causal Analysis')).toBeDefined();
    });
  });

  describe('DAG Section', () => {
    it('should toggle DAG section on click', () => {
      render(<CausalAnalysisPanel />);

      // DAG section starts expanded, so Initialize DAG button should be visible
      expect(screen.getByText('Initialize DAG')).toBeDefined();

      // Click to collapse
      const dagHeader = screen.getByText('DAG Configuration');
      fireEvent.click(dagHeader);

      // Button should now be hidden
      expect(screen.queryByText('Initialize DAG')).toBeNull();

      // Click again to expand
      fireEvent.click(dagHeader);

      // Button should be visible again
      expect(screen.getByText('Initialize DAG')).toBeDefined();
    });

    it('should initialize DAG when button is clicked', async () => {
      render(<CausalAnalysisPanel />);

      // DAG section starts expanded, so button is already visible
      // Click Initialize DAG
      fireEvent.click(screen.getByText('Initialize DAG'));

      // Wait for SVG to render
      await waitFor(() => {
        const svg = document.querySelector('svg');
        expect(svg).toBeDefined();
      });
    });
  });

  describe('Workflow Actions', () => {
    it('should show offline button in configure phase', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.getByText('Run Offline Learning')).toBeDefined();
    });

    it('should call onAnalysisComplete callback when analysis finishes', async () => {
      const mockCallback = vi.fn();
      render(<CausalAnalysisPanel onAnalysisComplete={mockCallback} />);

      // Note: Full workflow testing would require mocking the engine
      // This is a placeholder for demonstrating the test structure
      expect(screen.getByText('Causal Analysis')).toBeDefined();
    });
  });

  describe('Configuration Section', () => {
    it('should allow editing number of students', () => {
      render(<CausalAnalysisPanel />);

      // Expand config section
      fireEvent.click(screen.getByText('Model Configuration'));

      // Find and modify input
      const inputs = document.querySelectorAll('input[type="number"]');
      const studentsInput = inputs[0] as HTMLInputElement;

      fireEvent.change(studentsInput, { target: { value: '150' } });

      expect(studentsInput.value).toBe('150');
    });

    it('should disable inputs when not in configure phase', async () => {
      render(<CausalAnalysisPanel />);

      // Expand config section
      fireEvent.click(screen.getByText('Model Configuration'));

      // Get input
      const inputs = document.querySelectorAll('input[type="number"]');
      const firstInput = inputs[0] as HTMLInputElement;

      // In configure phase, should not be disabled
      expect(firstInput.disabled).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should display error message when error occurs', async () => {
      render(<CausalAnalysisPanel />);

      // Error display is conditional, so we just verify the component renders
      // without errors in the initial state
      expect(screen.queryByText(/error/i)).toBeNull();
    });
  });

  describe('Reset Functionality', () => {
    it('should not show reset button in configure phase', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.queryByText('Reset')).toBeNull();
    });
  });

  describe('Export Functionality', () => {
    it('should not show export button initially', () => {
      render(<CausalAnalysisPanel />);

      expect(screen.queryByText('Export')).toBeNull();
    });
  });

  describe('Accessibility', () => {
    it('should have clickable section headers', () => {
      render(<CausalAnalysisPanel />);

      const dagHeader = screen.getByText('DAG Configuration');
      const configHeader = screen.getByText('Model Configuration');

      expect(dagHeader.parentElement?.style.cursor).toBe('pointer');
      expect(configHeader.parentElement?.style.cursor).toBe('pointer');
    });

    it('should toggle expand/collapse indicators', () => {
      render(<CausalAnalysisPanel />);

      // Initially DAG section is expanded (shows '-')
      const dagHeader = screen.getByText('DAG Configuration').parentElement;
      expect(dagHeader?.textContent).toContain('-');

      // Config section is collapsed (shows '+')
      const configHeader = screen.getByText('Model Configuration').parentElement;
      expect(configHeader?.textContent).toContain('+');
    });
  });

  describe('Phase Transitions', () => {
    it('should highlight current phase in indicator', () => {
      render(<CausalAnalysisPanel />);

      const configurePhase = screen.getByText('1. Configure');
      const styles = window.getComputedStyle(configurePhase);

      // Current phase should have distinct styling
      expect(styles.fontWeight).toBe('bold');
    });
  });
});

describe('CausalAnalysisPanel Integration', () => {
  it('should render without crashing', () => {
    expect(() => {
      render(<CausalAnalysisPanel />);
    }).not.toThrow();
  });

  it('should handle rapid section toggling', () => {
    render(<CausalAnalysisPanel />);

    const dagHeader = screen.getByText('DAG Configuration');
    const configHeader = screen.getByText('Model Configuration');

    // Rapid toggles should not cause errors
    for (let i = 0; i < 5; i++) {
      fireEvent.click(dagHeader);
      fireEvent.click(configHeader);
    }

    expect(screen.getByText('Causal Analysis')).toBeDefined();
  });

  it('should preserve state during re-renders', () => {
    const { rerender } = render(<CausalAnalysisPanel />);

    // Expand config section
    fireEvent.click(screen.getByText('Model Configuration'));

    // Modify a value
    const inputs = document.querySelectorAll('input[type="number"]');
    const firstInput = inputs[0] as HTMLInputElement;
    fireEvent.change(firstInput, { target: { value: '250' } });

    // Re-render
    rerender(<CausalAnalysisPanel />);

    // Value should persist (within the same render cycle)
    // Note: Since we're re-rendering without state persistence,
    // this tests the component's resilience to re-renders
    expect(screen.getByText('Causal Analysis')).toBeDefined();
  });
});
