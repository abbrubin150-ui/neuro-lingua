/**
 * ProjectContext - Centralized state management for Projects and Runs
 * Provides CRUD operations and persistence to localStorage
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type {
  Project,
  Run,
  Scenario,
  DecisionLedger,
  TrainingConfig,
  ExecutionStatus
} from '../types/project';
import type { ExperimentComparison, DecisionEntry } from '../types/experiment';
import {
  createProject,
  createRun,
  createScenario,
  createDecisionLedger,
  createExperimentComparison,
  createDecisionEntry,
  computeExecutionStatus
} from '../types/project';
import { StorageManager } from '../lib/storage';

const STORAGE_KEYS = {
  PROJECTS: 'neuro-lingua-projects-v1',
  RUNS: 'neuro-lingua-runs-v1',
  COMPARISONS: 'neuro-lingua-comparisons-v1',
  DECISIONS: 'neuro-lingua-decisions-v1',
  ACTIVE_PROJECT: 'neuro-lingua-active-project-v1',
  ACTIVE_RUN: 'neuro-lingua-active-run-v1'
} as const;

interface ProjectContextValue {
  // State
  projects: Project[];
  runs: Run[];
  comparisons: ExperimentComparison[];
  decisions: DecisionEntry[];
  activeProjectId: string | null;
  activeRunId: string | null;

  // Computed
  activeProject: Project | null;
  activeRun: Run | null;
  projectRuns: Run[];
  projectComparisons: ExperimentComparison[];
  projectDecisions: DecisionEntry[];

  // Project operations
  createNewProject: (
    name: string,
    description: string,
    language?: 'en' | 'he' | 'mixed'
  ) => Project;
  updateProject: (id: string, updates: Partial<Project>) => void;
  deleteProject: (id: string) => void;
  setActiveProject: (id: string | null) => void;

  // Run operations
  createNewRun: (
    projectId: string,
    name: string,
    config: TrainingConfig,
    corpus: string,
    decisionLedger: DecisionLedger
  ) => Run;
  updateRun: (id: string, updates: Partial<Run>) => void;
  deleteRun: (id: string) => void;
  setActiveRun: (id: string | null) => void;
  getRunExecutionStatus: (runId: string) => ExecutionStatus;

  // Scenario operations
  addScenarioToProject: (projectId: string, scenario: Scenario) => void;
  updateScenario: (projectId: string, scenarioId: string, updates: Partial<Scenario>) => void;
  deleteScenario: (projectId: string, scenarioId: string) => void;

  // Comparison operations
  createNewComparison: (
    projectId: string,
    name: string,
    description: string,
    runIds: string[]
  ) => ExperimentComparison;
  updateComparison: (id: string, updates: Partial<ExperimentComparison>) => void;
  deleteComparison: (id: string) => void;
  getComparisonById: (id: string) => ExperimentComparison | undefined;
  getComparisonsByProject: (projectId: string) => ExperimentComparison[];

  // Decision operations
  createNewDecision: (
    projectId: string,
    problem: string,
    alternatives: string[],
    decision: string,
    kpi: string,
    affectedRunIds: string[],
    witness?: string,
    category?: string
  ) => DecisionEntry;
  updateDecision: (id: string, updates: Partial<DecisionEntry>) => void;
  deleteDecision: (id: string) => void;
  getDecisionById: (id: string) => DecisionEntry | undefined;
  getDecisionsByProject: (projectId: string) => DecisionEntry[];

  // Utility
  getProjectById: (id: string) => Project | undefined;
  getRunById: (id: string) => Run | undefined;
  getRunsByProject: (projectId: string) => Run[];
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [comparisons, setComparisons] = useState<ExperimentComparison[]>([]);
  const [decisions, setDecisions] = useState<DecisionEntry[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const savedProjects = StorageManager.get<Project[]>(STORAGE_KEYS.PROJECTS, []);
    const savedRuns = StorageManager.get<Run[]>(STORAGE_KEYS.RUNS, []);
    const savedComparisons = StorageManager.get<ExperimentComparison[]>(STORAGE_KEYS.COMPARISONS, []);
    const savedDecisions = StorageManager.get<DecisionEntry[]>(STORAGE_KEYS.DECISIONS, []);
    const savedActiveProject = StorageManager.get<string | null>(STORAGE_KEYS.ACTIVE_PROJECT, null);
    const savedActiveRun = StorageManager.get<string | null>(STORAGE_KEYS.ACTIVE_RUN, null);

    setProjects(savedProjects);
    setRuns(savedRuns);
    setComparisons(savedComparisons);
    setDecisions(savedDecisions);
    setActiveProjectId(savedActiveProject);
    setActiveRunId(savedActiveRun);
  }, []);

  // Persist to localStorage on changes
  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.PROJECTS, projects);
  }, [projects]);

  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.RUNS, runs);
  }, [runs]);

  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.COMPARISONS, comparisons);
  }, [comparisons]);

  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.DECISIONS, decisions);
  }, [decisions]);

  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.ACTIVE_PROJECT, activeProjectId);
  }, [activeProjectId]);

  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.ACTIVE_RUN, activeRunId);
  }, [activeRunId]);

  // Computed values
  const activeProject = projects.find((p) => p.id === activeProjectId) ?? null;
  const activeRun = runs.find((r) => r.id === activeRunId) ?? null;
  const projectRuns = activeProjectId ? runs.filter((r) => r.projectId === activeProjectId) : [];
  const projectComparisons = activeProjectId ? comparisons.filter((c) => c.projectId === activeProjectId) : [];
  const projectDecisions = activeProjectId ? decisions.filter((d) => d.projectId === activeProjectId) : [];

  // Project operations
  const createNewProject = useCallback(
    (name: string, description: string, language: 'en' | 'he' | 'mixed' = 'en') => {
      const newProject = createProject(name, description, language);
      setProjects((prev) => [...prev, newProject]);
      setActiveProjectId(newProject.id);
      return newProject;
    },
    []
  );

  const updateProject = useCallback((id: string, updates: Partial<Project>) => {
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, ...updates, updatedAt: Date.now() } : p))
    );
  }, []);

  const deleteProject = useCallback(
    (id: string) => {
      // Delete all runs associated with this project
      setRuns((prev) => prev.filter((r) => r.projectId !== id));
      // Delete the project
      setProjects((prev) => prev.filter((p) => p.id !== id));
      // Clear active selections if this was active
      if (activeProjectId === id) {
        setActiveProjectId(null);
        setActiveRunId(null);
      }
    },
    [activeProjectId]
  );

  const setActiveProject = useCallback((id: string | null) => {
    setActiveProjectId(id);
    // Clear active run when switching projects
    setActiveRunId(null);
  }, []);

  // Run operations
  const createNewRun = useCallback(
    (
      projectId: string,
      name: string,
      config: TrainingConfig,
      corpus: string,
      decisionLedger: DecisionLedger
    ) => {
      const newRun = createRun(projectId, name, config, corpus, decisionLedger);
      setRuns((prev) => [...prev, newRun]);

      // Add run ID to project
      setProjects((prev) =>
        prev.map((p) =>
          p.id === projectId ? { ...p, runIds: [...p.runIds, newRun.id], updatedAt: Date.now() } : p
        )
      );

      setActiveRunId(newRun.id);
      return newRun;
    },
    []
  );

  const updateRun = useCallback((id: string, updates: Partial<Run>) => {
    setRuns((prev) => prev.map((r) => (r.id === id ? { ...r, ...updates } : r)));
  }, []);

  const deleteRun = useCallback(
    (id: string) => {
      const run = runs.find((r) => r.id === id);
      if (run) {
        // Remove run ID from project
        setProjects((prev) =>
          prev.map((p) =>
            p.id === run.projectId
              ? { ...p, runIds: p.runIds.filter((rid) => rid !== id), updatedAt: Date.now() }
              : p
          )
        );
      }
      // Delete the run
      setRuns((prev) => prev.filter((r) => r.id !== id));
      // Clear active run if this was active
      if (activeRunId === id) {
        setActiveRunId(null);
      }
    },
    [runs, activeRunId]
  );

  const setActiveRun = useCallback((id: string | null) => {
    setActiveRunId(id);
  }, []);

  const getRunExecutionStatus = useCallback(
    (runId: string): ExecutionStatus => {
      const run = runs.find((r) => r.id === runId);
      if (!run) return 'HOLD';
      return computeExecutionStatus(run.decisionLedger);
    },
    [runs]
  );

  // Scenario operations
  const addScenarioToProject = useCallback((projectId: string, scenario: Scenario) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === projectId
          ? { ...p, scenarios: [...p.scenarios, scenario], updatedAt: Date.now() }
          : p
      )
    );
  }, []);

  const updateScenario = useCallback(
    (projectId: string, scenarioId: string, updates: Partial<Scenario>) => {
      setProjects((prev) =>
        prev.map((p) =>
          p.id === projectId
            ? {
                ...p,
                scenarios: p.scenarios.map((s) => (s.id === scenarioId ? { ...s, ...updates } : s)),
                updatedAt: Date.now()
              }
            : p
        )
      );
    },
    []
  );

  const deleteScenario = useCallback((projectId: string, scenarioId: string) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === projectId
          ? {
              ...p,
              scenarios: p.scenarios.filter((s) => s.id !== scenarioId),
              updatedAt: Date.now()
            }
          : p
      )
    );
  }, []);

  // Comparison operations
  const createNewComparison = useCallback(
    (projectId: string, name: string, description: string, runIds: string[]) => {
      const newComparison = createExperimentComparison(projectId, name, description, runIds);
      setComparisons((prev) => [...prev, newComparison]);
      return newComparison;
    },
    []
  );

  const updateComparison = useCallback((id: string, updates: Partial<ExperimentComparison>) => {
    setComparisons((prev) => prev.map((c) => (c.id === id ? { ...c, ...updates } : c)));
  }, []);

  const deleteComparison = useCallback((id: string) => {
    setComparisons((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const getComparisonById = useCallback(
    (id: string) => comparisons.find((c) => c.id === id),
    [comparisons]
  );

  const getComparisonsByProject = useCallback(
    (projectId: string) => comparisons.filter((c) => c.projectId === projectId),
    [comparisons]
  );

  // Decision operations
  const createNewDecision = useCallback(
    (
      projectId: string,
      problem: string,
      alternatives: string[],
      decision: string,
      kpi: string,
      affectedRunIds: string[],
      witness: string = 'local-user',
      category?: string
    ) => {
      const newDecision = createDecisionEntry(
        projectId,
        problem,
        alternatives,
        decision,
        kpi,
        affectedRunIds,
        witness,
        category
      );
      setDecisions((prev) => [...prev, newDecision]);
      return newDecision;
    },
    []
  );

  const updateDecision = useCallback((id: string, updates: Partial<DecisionEntry>) => {
    setDecisions((prev) => prev.map((d) => (d.id === id ? { ...d, ...updates } : d)));
  }, []);

  const deleteDecision = useCallback((id: string) => {
    setDecisions((prev) => prev.filter((d) => d.id !== id));
  }, []);

  const getDecisionById = useCallback((id: string) => decisions.find((d) => d.id === id), [decisions]);

  const getDecisionsByProject = useCallback(
    (projectId: string) => decisions.filter((d) => d.projectId === projectId),
    [decisions]
  );

  // Utility
  const getProjectById = useCallback((id: string) => projects.find((p) => p.id === id), [projects]);

  const getRunById = useCallback((id: string) => runs.find((r) => r.id === id), [runs]);

  const getRunsByProject = useCallback(
    (projectId: string) => runs.filter((r) => r.projectId === projectId),
    [runs]
  );

  const value: ProjectContextValue = {
    projects,
    runs,
    comparisons,
    decisions,
    activeProjectId,
    activeRunId,
    activeProject,
    activeRun,
    projectRuns,
    projectComparisons,
    projectDecisions,
    createNewProject,
    updateProject,
    deleteProject,
    setActiveProject,
    createNewRun,
    updateRun,
    deleteRun,
    setActiveRun,
    getRunExecutionStatus,
    addScenarioToProject,
    updateScenario,
    deleteScenario,
    createNewComparison,
    updateComparison,
    deleteComparison,
    getComparisonById,
    getComparisonsByProject,
    createNewDecision,
    updateDecision,
    deleteDecision,
    getDecisionById,
    getDecisionsByProject,
    getProjectById,
    getRunById,
    getRunsByProject
  };

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

/**
 * Hook to access ProjectContext
 */
export function useProjects() {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error('useProjects must be used within a ProjectProvider');
  }
  return context;
}

/**
 * Helper hook to create a scenario easily
 */
export function useCreateScenario() {
  return useCallback((name: string, prompt: string, expectedResponse?: string) => {
    return createScenario(name, prompt, expectedResponse);
  }, []);
}

/**
 * Helper hook to create a decision ledger easily
 */
export function useCreateDecisionLedger() {
  return useCallback(
    (
      rationale: string,
      witness?: string,
      expiry?: string | null,
      rollback?: 'keep' | 'delete-after-expiry' | 'archive'
    ) => {
      return createDecisionLedger(rationale, witness, expiry, rollback);
    },
    []
  );
}
