/**
 * Export Utilities for Experiment Data
 * Exports projects, runs, comparisons, and decisions to JSON/CSV
 * Part of Σ-SIG Experiment Explorer
 */

import type { Project, Run } from '../types/project';
import type {
  ExperimentComparison,
  DecisionEntry,
  ProjectExport,
  RunCSVRow,
  ComparisonCSVRow,
  DecisionCSVRow
} from '../types/experiment';
import { EXPORT_SCHEMA_VERSION } from '../types/experiment';

/**
 * Current package version (should match package.json)
 */
const PACKAGE_VERSION = '3.3.0';

/**
 * Create a full project export with all related data
 */
export function createProjectExport(
  projects: Project[],
  runs: Run[],
  comparisons: ExperimentComparison[],
  decisions: DecisionEntry[]
): ProjectExport {
  return {
    schemaVersion: EXPORT_SCHEMA_VERSION,
    metadata: {
      exportedAt: Date.now(),
      exportedBy: 'local-user',
      source: 'neuro-lingua-domestica',
      version: PACKAGE_VERSION
    },
    projects,
    runs,
    comparisons,
    decisions
  };
}

/**
 * Convert a Run to a flattened CSV row
 */
export function runToCSVRow(run: Run, projectName: string): RunCSVRow {
  const config = run.config;
  const results = run.results;

  return {
    // Identifiers
    runId: run.id,
    runName: run.name,
    projectId: run.projectId,
    projectName,
    status: run.status,
    createdAt: new Date(run.createdAt).toISOString(),
    completedAt: run.completedAt ? new Date(run.completedAt).toISOString() : '',
    // Config
    architecture: config.architecture,
    hiddenSize: config.hiddenSize,
    epochs: config.epochs,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    momentum: config.momentum,
    dropout: config.dropout,
    contextSize: config.contextSize,
    useGPU: config.useGPU,
    // Results
    finalLoss: results?.finalLoss ?? 0,
    finalAccuracy: results?.finalAccuracy ?? 0,
    finalPerplexity: results?.finalPerplexity ?? 0,
    // Corpus
    corpusLength: run.corpus.length,
    corpusChecksum: run.corpusChecksum,
    // Decision
    decisionRationale: run.decisionLedger.rationale,
    decisionWitness: run.decisionLedger.witness
  };
}

/**
 * Convert an ExperimentComparison to a CSV row
 */
export function comparisonToCSVRow(
  comparison: ExperimentComparison,
  projectName: string
): ComparisonCSVRow {
  return {
    comparisonId: comparison.id,
    comparisonName: comparison.name,
    projectName,
    runCount: comparison.runIds.length,
    runIds: comparison.runIds.join(', '),
    createdAt: new Date(comparison.createdAt).toISOString(),
    notes: comparison.notes ?? ''
  };
}

/**
 * Convert a DecisionEntry to a CSV row
 */
export function decisionToCSVRow(decision: DecisionEntry, projectName: string): DecisionCSVRow {
  return {
    decisionId: decision.id,
    projectName,
    problem: decision.problem,
    decision: decision.decision,
    kpi: decision.kpi,
    alternatives: decision.alternatives.join('; '),
    affectedRuns: decision.affectedRunIds.join(', '),
    category: decision.category ?? '',
    witness: decision.witness,
    createdAt: new Date(decision.createdAt).toISOString()
  };
}

/**
 * Convert array of objects to CSV string
 */
export function arrayToCSV<T extends Record<string, string | number | boolean>>(data: T[]): string {
  if (data.length === 0) {
    return '';
  }

  // Get headers from first object
  const headers = Object.keys(data[0]) as (keyof T)[];
  const headerRow = headers.join(',');

  // Convert each row
  const rows = data.map((row) => {
    return headers
      .map((header) => {
        const value = row[header];

        // Handle different value types
        if (value === null || value === undefined) {
          return '';
        }

        // Convert to string and escape quotes
        const stringValue = String(value);

        // Wrap in quotes if contains comma, newline, or quote
        if (stringValue.includes(',') || stringValue.includes('\n') || stringValue.includes('"')) {
          return `"${stringValue.replace(/"/g, '""')}"`;
        }

        return stringValue;
      })
      .join(',');
  });

  return [headerRow, ...rows].join('\n');
}

/**
 * Export runs to CSV format
 */
export function exportRunsToCSV(runs: Run[], projects: Project[]): string {
  const projectMap = new Map(projects.map((p) => [p.id, p.name]));

  const rows = runs.map((run) => runToCSVRow(run, projectMap.get(run.projectId) ?? 'Unknown'));

  return arrayToCSV(rows);
}

/**
 * Export comparisons to CSV format
 */
export function exportComparisonsToCSV(
  comparisons: ExperimentComparison[],
  projects: Project[]
): string {
  const projectMap = new Map(projects.map((p) => [p.id, p.name]));

  const rows = comparisons.map((comparison) =>
    comparisonToCSVRow(comparison, projectMap.get(comparison.projectId) ?? 'Unknown')
  );

  return arrayToCSV(rows);
}

/**
 * Export decisions to CSV format
 */
export function exportDecisionsToCSV(decisions: DecisionEntry[], projects: Project[]): string {
  const projectMap = new Map(projects.map((p) => [p.id, p.name]));

  const rows = decisions.map((decision) =>
    decisionToCSVRow(decision, projectMap.get(decision.projectId) ?? 'Unknown')
  );

  return arrayToCSV(rows);
}

/**
 * Download data as a file
 */
export function downloadFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Download JSON export
 */
export function downloadJSONExport(exportData: ProjectExport, filename?: string): void {
  const json = JSON.stringify(exportData, null, 2);
  const defaultFilename = `neuro-lingua-export-${Date.now()}.json`;
  downloadFile(filename ?? defaultFilename, json, 'application/json');
}

/**
 * Download CSV export
 */
export function downloadCSVExport(csv: string, filename: string): void {
  downloadFile(filename, csv, 'text/csv;charset=utf-8;');
}

/**
 * Export all data for a specific project
 */
export function exportProjectData(
  project: Project,
  runs: Run[],
  comparisons: ExperimentComparison[],
  decisions: DecisionEntry[]
): ProjectExport {
  // Filter data for this project only
  const projectRuns = runs.filter((r) => r.projectId === project.id);
  const projectComparisons = comparisons.filter((c) => c.projectId === project.id);
  const projectDecisions = decisions.filter((d) => d.projectId === project.id);

  return createProjectExport([project], projectRuns, projectComparisons, projectDecisions);
}

/**
 * Import and validate a project export
 */
export function importProjectExport(jsonString: string): ProjectExport {
  try {
    const data = JSON.parse(jsonString) as ProjectExport;

    // Validate schema version
    if (!data.schemaVersion) {
      throw new Error('Missing schema version');
    }

    // Version compatibility check (simple major version check)
    const [importMajor] = data.schemaVersion.split('.');
    const [currentMajor] = '1.0.0'.split('.');

    if (importMajor !== currentMajor) {
      throw new Error(
        `Incompatible schema version: ${data.schemaVersion}. ` +
          `Expected major version ${currentMajor}.x.x`
      );
    }

    // Validate required fields
    if (!data.metadata || !data.projects || !data.runs) {
      throw new Error('Invalid export format: missing required fields');
    }

    return data;
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error('Invalid JSON format');
    }
    throw error;
  }
}

/**
 * Generate a summary report of exported data
 */
export function generateExportSummary(exportData: ProjectExport): string {
  const { projects, runs, comparisons = [], decisions = [] } = exportData;

  const completedRuns = runs.filter((r) => r.status === 'completed').length;
  const totalDecisions = decisions.length;
  const totalComparisons = comparisons.length;

  let summary = '# Export Summary\n\n';
  summary += `**Exported:** ${new Date(exportData.metadata.exportedAt).toLocaleString()}\n`;
  summary += `**Version:** ${exportData.metadata.version}\n`;
  summary += `**Schema:** ${exportData.schemaVersion}\n\n`;
  summary += '## Contents\n\n';
  summary += `- **Projects:** ${projects.length}\n`;
  summary += `- **Runs:** ${runs.length} (${completedRuns} completed)\n`;
  summary += `- **Comparisons:** ${totalComparisons}\n`;
  summary += `- **Decisions:** ${totalDecisions}\n\n`;

  // Per-project breakdown
  summary += '## Project Breakdown\n\n';
  for (const project of projects) {
    const projectRuns = runs.filter((r) => r.projectId === project.id);
    const projectComparisons = comparisons.filter((c) => c.projectId === project.id);
    const projectDecisions = decisions.filter((d) => d.projectId === project.id);

    summary += `### ${project.name}\n`;
    summary += `- Runs: ${projectRuns.length}\n`;
    summary += `- Comparisons: ${projectComparisons.length}\n`;
    summary += `- Decisions: ${projectDecisions.length}\n\n`;
  }

  return summary;
}
