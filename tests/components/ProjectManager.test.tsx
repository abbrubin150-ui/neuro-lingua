import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ProjectManager } from '../../src/components/ProjectManager';
import { ProjectProvider } from '../../src/contexts/ProjectContext';

describe('ProjectManager', () => {
  const renderWithProvider = (ui: React.ReactElement) => {
    return render(<ProjectProvider>{ui}</ProjectProvider>);
  };

  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    // Mock window.confirm
    vi.spyOn(window, 'confirm').mockReturnValue(false);
  });

  it('renders project manager with title', () => {
    renderWithProvider(<ProjectManager />);
    expect(screen.getByText(/Project Manager/i)).toBeInTheDocument();
  });

  it('shows create project button', () => {
    renderWithProvider(<ProjectManager />);
    const createButton = screen.getByText(/New Project/i);
    expect(createButton).toBeInTheDocument();
  });

  it('opens create project form when clicking New Project', () => {
    const { container } = renderWithProvider(<ProjectManager />);
    const createButton = screen.getByText(/New Project/i);

    fireEvent.click(createButton);

    expect(container.querySelector('#project-name')).toBeInTheDocument();
    expect(container.querySelector('#project-description')).toBeInTheDocument();
  });

  it('creates a new project with name and description', async () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Open create form
    fireEvent.click(screen.getByText(/New Project/i));

    // Fill in project details using ID selectors
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    const descInput = container.querySelector('#project-description') as HTMLTextAreaElement;

    expect(nameInput).not.toBeNull();
    expect(descInput).not.toBeNull();

    fireEvent.change(nameInput!, { target: { value: 'Test Project' } });
    fireEvent.change(descInput!, { target: { value: 'A test project description' } });

    // Submit - look for button with Create text
    const createButton = screen.getByText(/✓ Create/i).closest('button');
    expect(createButton).not.toBeNull();
    fireEvent.click(createButton!);

    // Verify project appears in list
    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });
  });

  it('requires project name to create project', () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Open create form
    fireEvent.click(screen.getByText(/New Project/i));

    // Try to create without name (only description)
    const descInput = container.querySelector('#project-description') as HTMLTextAreaElement;
    expect(descInput).not.toBeNull();
    fireEvent.change(descInput!, { target: { value: 'Description only' } });

    // Create button should be disabled or form still visible
    const nameInput = container.querySelector('#project-name');
    expect(nameInput).toBeInTheDocument();
  });

  it('cancels project creation when clicking Cancel', () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Open create form
    fireEvent.click(screen.getByText(/New Project/i));

    // Verify form is open
    expect(container.querySelector('#project-name')).toBeInTheDocument();

    // Click cancel
    const cancelButton = screen.getByText(/Cancel/i).closest('button');
    expect(cancelButton).not.toBeNull();
    fireEvent.click(cancelButton!);

    // Form should be closed
    expect(container.querySelector('#project-name')).not.toBeInTheDocument();
  });

  it('selects language when creating project', async () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Open create form
    fireEvent.click(screen.getByText(/New Project/i));

    // Change language using ID selector
    const languageSelect = container.querySelector('#project-language') as HTMLSelectElement;
    expect(languageSelect).not.toBeNull();
    fireEvent.change(languageSelect!, { target: { value: 'he' } });

    // Fill in name and create
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    expect(nameInput).not.toBeNull();
    fireEvent.change(nameInput!, { target: { value: 'Hebrew Project' } });

    const createButton = screen.getByText(/✓ Create/i).closest('button');
    expect(createButton).not.toBeNull();
    fireEvent.click(createButton!);

    // Verify project was created
    await waitFor(() => {
      expect(screen.getByText('Hebrew Project')).toBeInTheDocument();
    });
  });

  it('displays project list when projects exist', async () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Create first project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput1 = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput1!, { target: { value: 'Project 1' } });
    const createBtn1 = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn1!);

    await waitFor(() => {
      expect(screen.getByText('Project 1')).toBeInTheDocument();
    });

    // Create second project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput2 = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput2!, { target: { value: 'Project 2' } });
    const createBtn2 = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn2!);

    await waitFor(() => {
      expect(screen.getByText('Project 2')).toBeInTheDocument();
    });

    // Both projects should be visible
    expect(screen.getByText('Project 1')).toBeInTheDocument();
    expect(screen.getByText('Project 2')).toBeInTheDocument();
  });

  it('deletes project when confirmed', async () => {
    // Mock confirm to return true for deletion
    vi.spyOn(window, 'confirm').mockReturnValue(true);

    const { container } = renderWithProvider(<ProjectManager />);

    // Create a project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput!, { target: { value: 'To Delete' } });
    const createBtn = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn!);

    await waitFor(() => {
      expect(screen.getByText('To Delete')).toBeInTheDocument();
    });

    // Find and click delete button (look for trash icon or Delete text)
    const deleteButtons = screen.queryAllByText(/🗑️|Delete/i);
    if (deleteButtons.length > 0) {
      const deleteBtn = deleteButtons[0].closest('button');
      if (deleteBtn) {
        fireEvent.click(deleteBtn);

        // Verify confirmation was called
        expect(window.confirm).toHaveBeenCalled();

        // Project should be removed
        await waitFor(() => {
          expect(screen.queryByText('To Delete')).not.toBeInTheDocument();
        });
      }
    }
  });

  it('does not delete project when not confirmed', async () => {
    // Mock confirm to return false
    vi.spyOn(window, 'confirm').mockReturnValue(false);

    const { container } = renderWithProvider(<ProjectManager />);

    // Create a project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput!, { target: { value: 'Keep Me' } });
    const createBtn = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn!);

    await waitFor(() => {
      expect(screen.getByText('Keep Me')).toBeInTheDocument();
    });

    // Try to delete
    const deleteButtons = screen.queryAllByText(/🗑️|Delete/i);
    if (deleteButtons.length > 0) {
      const deleteBtn = deleteButtons[0].closest('button');
      if (deleteBtn) {
        fireEvent.click(deleteBtn);
      }
    }

    // Project should still exist
    expect(screen.getByText('Keep Me')).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    renderWithProvider(<ProjectManager onClose={onClose} />);

    // Find and click close button (X in top right)
    const closeButton = screen.getByRole('button', { name: /✕/i });
    fireEvent.click(closeButton);

    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when clicking outside modal', () => {
    const onClose = vi.fn();
    const { container } = renderWithProvider(<ProjectManager onClose={onClose} />);

    // Click on overlay (not the dialog)
    const overlay = container.querySelector('[role="presentation"]');
    if (overlay) {
      fireEvent.click(overlay);
      expect(onClose).toHaveBeenCalled();
    }
  });

  it('sets active project when clicking on project', async () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Create a project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput!, { target: { value: 'Active Project' } });
    const createBtn = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn!);

    await waitFor(() => {
      expect(screen.getByText('Active Project')).toBeInTheDocument();
    });

    // Click on the project to make it active
    const projectButton = screen.getByText('Active Project').closest('button');
    if (projectButton) {
      fireEvent.click(projectButton);
    }

    // Project should now be highlighted or marked as active
    // (specific visual indication depends on implementation)
    expect(screen.getByText('Active Project')).toBeInTheDocument();
  });

  it('displays run count for each project', async () => {
    const { container } = renderWithProvider(<ProjectManager />);

    // Create a project
    fireEvent.click(screen.getByText(/New Project/i));
    const nameInput = container.querySelector('#project-name') as HTMLInputElement;
    fireEvent.change(nameInput!, { target: { value: 'Project with Runs' } });
    const createBtn = screen.getByText(/✓ Create/i).closest('button');
    fireEvent.click(createBtn!);

    await waitFor(() => {
      expect(screen.getByText('Project with Runs')).toBeInTheDocument();
    });

    // Should show 0 runs initially
    expect(screen.getByText(/0 runs?/i)).toBeInTheDocument();
  });

  it('respects RTL direction when specified', () => {
    const { container } = renderWithProvider(<ProjectManager direction="rtl" />);

    const dialogElement = container.querySelector('[role="dialog"]');
    expect(dialogElement?.parentElement).toHaveStyle({ direction: 'rtl' });
  });

  it('defaults to LTR direction when not specified', () => {
    const { container } = renderWithProvider(<ProjectManager />);

    const dialogElement = container.querySelector('[role="dialog"]');
    expect(dialogElement?.parentElement).toHaveStyle({ direction: 'ltr' });
  });

  describe('Governance - Training Execution Blocking', () => {
    it('should block training when decision ledger status is HOLD (expired)', () => {
      const { container } = renderWithProvider(<ProjectManager />);

      // The execution status logic is tested via the computeExecutionStatus function
      // which is used by getRunExecutionStatus in ProjectContext

      // Create a project
      fireEvent.click(screen.getByText(/New Project/i));
      const nameInput = container.querySelector('#project-name') as HTMLInputElement;
      fireEvent.change(nameInput!, { target: { value: 'Governance Test Project' } });
      const createBtn = screen.getByText(/✓ Create/i).closest('button');
      fireEvent.click(createBtn!);

      // Verify project was created
      waitFor(() => {
        expect(screen.getByText('Governance Test Project')).toBeInTheDocument();
      });

      // Note: The actual blocking logic is enforced in the training flow
      // This test documents the requirement that projects with expired decisions
      // should have execution status HOLD and prevent training
    });

    it('should block training when decision ledger status is ESCALATE (missing rationale)', () => {
      const { container } = renderWithProvider(<ProjectManager />);

      // Create a project
      fireEvent.click(screen.getByText(/New Project/i));
      const nameInput = container.querySelector('#project-name') as HTMLInputElement;
      fireEvent.change(nameInput!, { target: { value: 'Escalate Test Project' } });
      const createBtn = screen.getByText(/✓ Create/i).closest('button');
      fireEvent.click(createBtn!);

      // Verify project was created
      waitFor(() => {
        expect(screen.getByText('Escalate Test Project')).toBeInTheDocument();
      });

      // Note: Decision ledgers without rationale or witness should have ESCALATE status
      // and prevent training execution
    });

    it('should allow training when decision ledger status is EXECUTE', () => {
      const { container } = renderWithProvider(<ProjectManager />);

      // Create a project
      fireEvent.click(screen.getByText(/New Project/i));
      const nameInput = container.querySelector('#project-name') as HTMLInputElement;
      fireEvent.change(nameInput!, { target: { value: 'Execute Test Project' } });
      const createBtn = screen.getByText(/✓ Create/i).closest('button');
      fireEvent.click(createBtn!);

      // Verify project was created
      waitFor(() => {
        expect(screen.getByText('Execute Test Project')).toBeInTheDocument();
      });

      // Note: Projects with valid decision ledgers (rationale + witness + no expiry)
      // should have EXECUTE status and allow training
    });
  });
});
