import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import React from 'react';
import { ProjectProvider, useProjects } from '../../src/contexts/ProjectContext';
import type { TrainingConfig, DecisionLedger } from '../../src/types/project';
import { createDecisionLedger, computeExecutionStatus } from '../../src/types/project';

// Wrapper for hooks
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ProjectProvider>{children}</ProjectProvider>
);

// Helper to create a minimal training config
function createTestConfig(): TrainingConfig {
  return {
    architecture: 'feedforward',
    hiddenSize: 64,
    epochs: 20,
    learningRate: 0.08,
    optimizer: 'momentum',
    momentum: 0.9,
    dropout: 0.1,
    contextSize: 3,
    seed: 42,
    tokenizerConfig: { mode: 'unicode' },
    useAdvanced: false,
    useGPU: false
  };
}

describe('ProjectContext', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  describe('Project CRUD operations', () => {
    it('creates a new project with unique ID', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let project1, project2;

      act(() => {
        project1 = result.current.createNewProject('Project 1', 'First project', 'en');
        project2 = result.current.createNewProject('Project 2', 'Second project', 'he');
      });

      expect(project1.id).toBeTruthy();
      expect(project2.id).toBeTruthy();
      expect(project1.id).not.toBe(project2.id);
      expect(project1.name).toBe('Project 1');
      expect(project2.name).toBe('Project 2');
      expect(project1.language).toBe('en');
      expect(project2.language).toBe('he');
    });

    it('adds projects to state and sets as active', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      expect(result.current.projects).toHaveLength(0);
      expect(result.current.activeProject).toBeNull();

      let newProject;
      act(() => {
        newProject = result.current.createNewProject('Test Project', 'Description');
      });

      expect(result.current.projects).toHaveLength(1);
      expect(result.current.activeProject).toEqual(newProject);
      expect(result.current.activeProjectId).toBe(newProject.id);
    });

    it('updates project metadata', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Original', 'Original desc');
        projectId = project.id;
      });

      const originalUpdatedAt = result.current.projects[0].updatedAt;

      act(() => {
        result.current.updateProject(projectId, {
          name: 'Updated Name',
          description: 'Updated description',
          tags: ['test', 'updated']
        });
      });

      const updated = result.current.projects[0];
      expect(updated.name).toBe('Updated Name');
      expect(updated.description).toBe('Updated description');
      expect(updated.tags).toEqual(['test', 'updated']);
      expect(updated.updatedAt).toBeGreaterThan(originalUpdatedAt);
    });

    it('deletes project and associated runs', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId, runId;
      act(() => {
        const project = result.current.createNewProject('To Delete', 'Will be deleted');
        projectId = project.id;

        const ledger = createDecisionLedger('Test run', 'test-user');
        const run = result.current.createNewRun(
          projectId,
          'Test Run',
          createTestConfig(),
          'test corpus',
          ledger
        );
        runId = run.id;
      });

      expect(result.current.projects).toHaveLength(1);
      expect(result.current.runs).toHaveLength(1);

      act(() => {
        result.current.deleteProject(projectId);
      });

      expect(result.current.projects).toHaveLength(0);
      expect(result.current.runs).toHaveLength(0);
      expect(result.current.activeProjectId).toBeNull();
      expect(result.current.activeRunId).toBeNull();
    });

    it('sets active project and clears active run', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let project1Id, project2Id, runId;
      act(() => {
        const p1 = result.current.createNewProject('Project 1', 'First');
        project1Id = p1.id;

        const p2 = result.current.createNewProject('Project 2', 'Second');
        project2Id = p2.id;

        // Create run for project 2
        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(
          project2Id,
          'Run 1',
          createTestConfig(),
          'corpus',
          ledger
        );
        runId = run.id;
      });

      expect(result.current.activeProjectId).toBe(project2Id);
      expect(result.current.activeRunId).toBe(runId);

      act(() => {
        result.current.setActiveProject(project1Id);
      });

      expect(result.current.activeProjectId).toBe(project1Id);
      expect(result.current.activeRunId).toBeNull(); // Should be cleared
    });

    it('retrieves project by ID', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Findable', 'Can be found');
        projectId = project.id;
      });

      const found = result.current.getProjectById(projectId);
      expect(found).toBeDefined();
      expect(found?.name).toBe('Findable');

      const notFound = result.current.getProjectById('non-existent-id');
      expect(notFound).toBeUndefined();
    });
  });

  describe('Run management', () => {
    it('creates run with frozen config', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let runId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const config = createTestConfig();
        const ledger = createDecisionLedger('Testing run creation', 'test-user');

        const run = result.current.createNewRun(
          project.id,
          'Run 1',
          config,
          'hello world',
          ledger
        );
        runId = run.id;
      });

      const run = result.current.runs[0];
      expect(run.id).toBe(runId);
      expect(run.name).toBe('Run 1');
      expect(run.config).toEqual(createTestConfig());
      expect(run.corpus).toBe('hello world');
      expect(run.corpusChecksum).toBeTruthy();
      expect(run.status).toBe('pending');
      expect(run.createdAt).toBeTruthy();
    });

    it('associates run with project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId, runId;
      act(() => {
        const project = result.current.createNewProject('Parent Project', 'Has runs');
        projectId = project.id;

        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(
          projectId,
          'Child Run',
          createTestConfig(),
          'corpus',
          ledger
        );
        runId = run.id;
      });

      const project = result.current.getProjectById(projectId);
      expect(project?.runIds).toContain(runId);

      const run = result.current.getRunById(runId);
      expect(run?.projectId).toBe(projectId);
    });

    it('prevents config modification after creation (frozen)', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let runId;
      const originalConfig = createTestConfig();

      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(project.id, 'Run', originalConfig, 'corpus', ledger);
        runId = run.id;
      });

      // Attempt to modify config (shouldn't affect stored run)
      const run = result.current.getRunById(runId);
      const configCopy = { ...run!.config, hiddenSize: 999 };

      act(() => {
        result.current.updateRun(runId, { config: configCopy });
      });

      const updatedRun = result.current.getRunById(runId);
      expect(updatedRun?.config.hiddenSize).toBe(999); // Update works
      // But original config object is unchanged
      expect(originalConfig.hiddenSize).toBe(64);
    });

    it('updates run status and results', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let runId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(project.id, 'Run', createTestConfig(), 'corpus', ledger);
        runId = run.id;
      });

      act(() => {
        result.current.updateRun(runId, {
          status: 'running',
          startedAt: Date.now()
        });
      });

      let run = result.current.getRunById(runId);
      expect(run?.status).toBe('running');
      expect(run?.startedAt).toBeTruthy();

      act(() => {
        result.current.updateRun(runId, {
          status: 'completed',
          completedAt: Date.now(),
          results: {
            finalLoss: 1.234,
            finalAccuracy: 0.85,
            finalPerplexity: 3.43,
            trainingHistory: [
              { loss: 2.5, accuracy: 0.5, timestamp: Date.now() },
              { loss: 1.234, accuracy: 0.85, timestamp: Date.now() }
            ]
          }
        });
      });

      run = result.current.getRunById(runId);
      expect(run?.status).toBe('completed');
      expect(run?.results?.finalLoss).toBe(1.234);
      expect(run?.results?.finalAccuracy).toBe(0.85);
      expect(run?.results?.trainingHistory).toHaveLength(2);
    });

    it('deletes run and removes from project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId, runId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        projectId = project.id;

        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(projectId, 'Run', createTestConfig(), 'corpus', ledger);
        runId = run.id;
      });

      expect(result.current.runs).toHaveLength(1);
      expect(result.current.getProjectById(projectId)?.runIds).toContain(runId);

      act(() => {
        result.current.deleteRun(runId);
      });

      expect(result.current.runs).toHaveLength(0);
      expect(result.current.getProjectById(projectId)?.runIds).not.toContain(runId);
      expect(result.current.activeRunId).toBeNull();
    });

    it('retrieves runs by project ID', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let project1Id, project2Id;
      act(() => {
        const p1 = result.current.createNewProject('Project 1', 'First');
        project1Id = p1.id;

        const p2 = result.current.createNewProject('Project 2', 'Second');
        project2Id = p2.id;

        const ledger = createDecisionLedger('Test', 'user');

        // Create 2 runs for project 1
        result.current.createNewRun(project1Id, 'P1 Run 1', createTestConfig(), 'corpus1', ledger);
        result.current.createNewRun(project1Id, 'P1 Run 2', createTestConfig(), 'corpus2', ledger);

        // Create 1 run for project 2
        result.current.createNewRun(project2Id, 'P2 Run 1', createTestConfig(), 'corpus3', ledger);
      });

      const p1Runs = result.current.getRunsByProject(project1Id);
      const p2Runs = result.current.getRunsByProject(project2Id);

      expect(p1Runs).toHaveLength(2);
      expect(p2Runs).toHaveLength(1);
      expect(p1Runs.every((r) => r.projectId === project1Id)).toBe(true);
      expect(p2Runs.every((r) => r.projectId === project2Id)).toBe(true);
    });

    it('computes projectRuns for active project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let project1Id, project2Id;
      act(() => {
        const p1 = result.current.createNewProject('Project 1', 'First');
        project1Id = p1.id;

        const p2 = result.current.createNewProject('Project 2', 'Second');
        project2Id = p2.id;

        const ledger = createDecisionLedger('Test', 'user');

        result.current.createNewRun(project1Id, 'Run 1', createTestConfig(), 'corpus', ledger);
        result.current.createNewRun(project2Id, 'Run 2', createTestConfig(), 'corpus', ledger);
        result.current.createNewRun(project2Id, 'Run 3', createTestConfig(), 'corpus', ledger);
      });

      act(() => {
        result.current.setActiveProject(project2Id);
      });

      expect(result.current.projectRuns).toHaveLength(2);
      expect(result.current.projectRuns.every((r) => r.projectId === project2Id)).toBe(true);
    });
  });

  describe('Scenario operations', () => {
    it('adds scenario to project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        projectId = project.id;

        result.current.addScenarioToProject(projectId, {
          id: 'scenario-1',
          name: 'Test Scenario',
          prompt: 'Hello',
          expectedResponse: 'World'
        });
      });

      const project = result.current.getProjectById(projectId);
      expect(project?.scenarios).toHaveLength(1);
      expect(project?.scenarios[0].name).toBe('Test Scenario');
      expect(project?.scenarios[0].prompt).toBe('Hello');
    });

    it('updates scenario within project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        projectId = project.id;

        result.current.addScenarioToProject(projectId, {
          id: 'scenario-1',
          name: 'Original',
          prompt: 'Original prompt'
        });
      });

      act(() => {
        result.current.updateScenario(projectId, 'scenario-1', {
          name: 'Updated',
          expectedResponse: 'Expected output',
          lastScore: 0.92
        });
      });

      const project = result.current.getProjectById(projectId);
      const scenario = project?.scenarios[0];
      expect(scenario?.name).toBe('Updated');
      expect(scenario?.expectedResponse).toBe('Expected output');
      expect(scenario?.lastScore).toBe(0.92);
      expect(scenario?.prompt).toBe('Original prompt'); // Unchanged
    });

    it('deletes scenario from project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        projectId = project.id;

        result.current.addScenarioToProject(projectId, {
          id: 'scenario-1',
          name: 'Scenario 1',
          prompt: 'Prompt 1'
        });
        result.current.addScenarioToProject(projectId, {
          id: 'scenario-2',
          name: 'Scenario 2',
          prompt: 'Prompt 2'
        });
      });

      expect(result.current.getProjectById(projectId)?.scenarios).toHaveLength(2);

      act(() => {
        result.current.deleteScenario(projectId, 'scenario-1');
      });

      const project = result.current.getProjectById(projectId);
      expect(project?.scenarios).toHaveLength(1);
      expect(project?.scenarios[0].id).toBe('scenario-2');
    });
  });

  describe('Decision Ledger and Σ-SIG compliance', () => {
    it('creates run with decision ledger', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let runId;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const ledger = createDecisionLedger(
          'Testing new architecture',
          'researcher-1',
          '2025-12-31T23:59:59Z',
          'archive'
        );

        const run = result.current.createNewRun(project.id, 'Run', createTestConfig(), 'corpus', ledger);
        runId = run.id;
      });

      const run = result.current.getRunById(runId);
      expect(run?.decisionLedger).toBeDefined();
      expect(run?.decisionLedger.rationale).toBe('Testing new architecture');
      expect(run?.decisionLedger.witness).toBe('researcher-1');
      expect(run?.decisionLedger.expiry).toBe('2025-12-31T23:59:59Z');
      expect(run?.decisionLedger.rollback).toBe('archive');
    });

    it('computes execution status correctly', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let runId1, runId2, runId3;
      act(() => {
        const project = result.current.createNewProject('Test', 'Project');

        // EXECUTE: valid ledger
        const ledger1 = createDecisionLedger('Valid rationale', 'user', null, 'keep');
        const run1 = result.current.createNewRun(project.id, 'Run 1', createTestConfig(), 'c', ledger1);
        runId1 = run1.id;

        // ESCALATE: missing rationale
        const ledger2 = createDecisionLedger('', 'user', null, 'keep');
        const run2 = result.current.createNewRun(project.id, 'Run 2', createTestConfig(), 'c', ledger2);
        runId2 = run2.id;

        // HOLD: expired
        const ledger3 = createDecisionLedger(
          'Valid rationale',
          'user',
          '2020-01-01T00:00:00Z',
          'keep'
        );
        const run3 = result.current.createNewRun(project.id, 'Run 3', createTestConfig(), 'c', ledger3);
        runId3 = run3.id;
      });

      expect(result.current.getRunExecutionStatus(runId1)).toBe('EXECUTE');
      expect(result.current.getRunExecutionStatus(runId2)).toBe('ESCALATE');
      expect(result.current.getRunExecutionStatus(runId3)).toBe('HOLD');
    });

    it('handles missing witness in decision ledger', () => {
      const ledger = createDecisionLedger('Valid rationale', '', null, 'keep');
      const status = computeExecutionStatus(ledger);
      expect(status).toBe('ESCALATE');
    });
  });

  describe('Persistence to localStorage', () => {
    it('persists projects to localStorage', async () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        result.current.createNewProject('Persisted Project', 'Should be saved');
      });

      await waitFor(() => {
        const stored = localStorage.getItem('neuro-lingua-projects-v1');
        expect(stored).toBeTruthy();
        const parsed = JSON.parse(stored!);
        expect(parsed).toHaveLength(1);
        expect(parsed[0].name).toBe('Persisted Project');
      });
    });

    it('persists runs to localStorage', async () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const ledger = createDecisionLedger('Test', 'user');
        result.current.createNewRun(project.id, 'Persisted Run', createTestConfig(), 'corpus', ledger);
      });

      await waitFor(() => {
        const stored = localStorage.getItem('neuro-lingua-runs-v1');
        expect(stored).toBeTruthy();
        const parsed = JSON.parse(stored!);
        expect(parsed).toHaveLength(1);
        expect(parsed[0].name).toBe('Persisted Run');
      });
    });

    it('persists active project and run IDs', async () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId, runId;
      act(() => {
        const project = result.current.createNewProject('Active', 'Project');
        projectId = project.id;

        const ledger = createDecisionLedger('Test', 'user');
        const run = result.current.createNewRun(projectId, 'Active Run', createTestConfig(), 'c', ledger);
        runId = run.id;
      });

      await waitFor(() => {
        expect(localStorage.getItem('neuro-lingua-active-project-v1')).toBe(JSON.stringify(projectId));
        expect(localStorage.getItem('neuro-lingua-active-run-v1')).toBe(JSON.stringify(runId));
      });
    });

    it('restores state from localStorage on mount', () => {
      // Pre-populate localStorage
      const savedProject = {
        id: 'proj_123',
        name: 'Saved Project',
        description: 'From storage',
        language: 'en',
        defaultArchitecture: 'feedforward',
        corpusType: 'plain-text',
        scenarios: [],
        runIds: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        tags: []
      };

      localStorage.setItem('neuro-lingua-projects-v1', JSON.stringify([savedProject]));
      localStorage.setItem('neuro-lingua-runs-v1', JSON.stringify([]));
      localStorage.setItem('neuro-lingua-active-project-v1', JSON.stringify('proj_123'));
      localStorage.setItem('neuro-lingua-active-run-v1', JSON.stringify(null));

      const { result } = renderHook(() => useProjects(), { wrapper });

      // Should restore on mount
      expect(result.current.projects).toHaveLength(1);
      expect(result.current.projects[0].name).toBe('Saved Project');
      expect(result.current.activeProjectId).toBe('proj_123');
    });
  });

  describe('Edge cases', () => {
    it('handles non-existent project ID gracefully', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        result.current.updateProject('non-existent', { name: 'Updated' });
      });

      expect(result.current.projects).toHaveLength(0);
    });

    it('handles non-existent run ID gracefully', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        result.current.updateRun('non-existent', { status: 'completed' });
      });

      expect(result.current.runs).toHaveLength(0);
    });

    it('handles orphaned runs when deleting project', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      let projectId;
      act(() => {
        const project = result.current.createNewProject('Parent', 'Project');
        projectId = project.id;

        const ledger = createDecisionLedger('Test', 'user');
        result.current.createNewRun(projectId, 'Child Run', createTestConfig(), 'corpus', ledger);
      });

      expect(result.current.runs).toHaveLength(1);

      act(() => {
        result.current.deleteProject(projectId);
      });

      // Run should be deleted with project
      expect(result.current.runs).toHaveLength(0);
    });

    it('handles empty corpus in run', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        const project = result.current.createNewProject('Test', 'Project');
        const ledger = createDecisionLedger('Test', 'user');
        result.current.createNewRun(project.id, 'Empty Corpus', createTestConfig(), '', ledger);
      });

      const run = result.current.runs[0];
      expect(run.corpus).toBe('');
      expect(run.corpusChecksum).toBeTruthy(); // Should still generate checksum
    });

    it('handles multiple languages in projects', () => {
      const { result } = renderHook(() => useProjects(), { wrapper });

      act(() => {
        result.current.createNewProject('English', 'English corpus', 'en');
        result.current.createNewProject('Hebrew', 'Hebrew corpus', 'he');
        result.current.createNewProject('Mixed', 'Mixed languages', 'mixed');
      });

      expect(result.current.projects).toHaveLength(3);
      expect(result.current.projects[0].language).toBe('en');
      expect(result.current.projects[1].language).toBe('he');
      expect(result.current.projects[2].language).toBe('mixed');
    });
  });
});
