/**
 * ExportPanel - Data export UI for JSON and CSV formats
 * Allows exporting projects, runs, comparisons, and decisions
 */

import React, { useState, useMemo } from 'react';
import { useProjects } from '../contexts/ProjectContext';
import {
  exportProjectToJSON,
  downloadProjectJSON,
  exportRunsToCSV,
  exportComparisonsToCSV,
  exportDecisionsToCSV,
  downloadCSV
} from '../lib/experimentComparison';

interface ExportPanelProps {
  /** Callback when panel is closed */
  onClose?: () => void;
  /** Direction for RTL support */
  direction?: 'ltr' | 'rtl';
}

type ExportFormat = 'json' | 'csv';
type ExportScope = 'all' | 'project';
type ExportType = 'complete' | 'runs' | 'comparisons' | 'decisions';

/**
 * Main export panel component
 */
export function ExportPanel({ onClose, direction = 'ltr' }: ExportPanelProps) {
  const { projects, runs, comparisons, decisions } = useProjects();

  // Export configuration state
  const [exportFormat, setExportFormat] = useState<ExportFormat>('json');
  const [exportScope, setExportScope] = useState<ExportScope>('all');
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [exportType, setExportType] = useState<ExportType>('complete');
  const [exporting, setExporting] = useState(false);
  const [lastExport, setLastExport] = useState<string | null>(null);

  // Filter data based on scope
  const filteredData = useMemo(() => {
    if (exportScope === 'all') {
      return { projects, runs, comparisons, decisions };
    } else if (selectedProjectId) {
      return {
        projects: projects.filter((p) => p.id === selectedProjectId),
        runs: runs.filter((r) => r.projectId === selectedProjectId),
        comparisons: comparisons.filter((c) => c.projectId === selectedProjectId),
        decisions: decisions.filter((d) => d.projectId === selectedProjectId)
      };
    }
    return { projects: [], runs: [], comparisons: [], decisions: [] };
  }, [exportScope, selectedProjectId, projects, runs, comparisons, decisions]);

  // Statistics
  const stats = useMemo(() => {
    return {
      projects: filteredData.projects.length,
      runs: filteredData.runs.length,
      comparisons: filteredData.comparisons.length,
      decisions: filteredData.decisions.length,
      completedRuns: filteredData.runs.filter((r) => r.status === 'completed').length
    };
  }, [filteredData]);

  /**
   * Handle export action
   */
  const handleExport = () => {
    if (exportScope === 'project' && !selectedProjectId) {
      alert('Please select a project to export');
      return;
    }

    setExporting(true);

    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);

      if (exportFormat === 'json') {
        // JSON export
        const exportData = exportProjectToJSON(
          filteredData.projects,
          filteredData.runs,
          filteredData.comparisons,
          filteredData.decisions
        );

        const filename =
          exportScope === 'all'
            ? `neuro-lingua-full-export-${timestamp}.json`
            : `neuro-lingua-${filteredData.projects[0]?.name || 'project'}-${timestamp}.json`;

        downloadProjectJSON(exportData, filename);
        setLastExport(`JSON: ${filename}`);
      } else {
        // CSV export
        const projectsMap = new Map(filteredData.projects.map((p) => [p.id, p]));

        if (exportType === 'complete') {
          // Export all as separate CSV files
          const runsCSV = exportRunsToCSV(filteredData.runs, projectsMap);
          downloadCSV(runsCSV, `neuro-lingua-runs-${timestamp}.csv`);

          if (filteredData.comparisons.length > 0) {
            const comparisonsCSV = exportComparisonsToCSV(filteredData.comparisons, projectsMap);
            downloadCSV(comparisonsCSV, `neuro-lingua-comparisons-${timestamp}.csv`);
          }

          if (filteredData.decisions.length > 0) {
            const decisionsCSV = exportDecisionsToCSV(filteredData.decisions, projectsMap);
            downloadCSV(decisionsCSV, `neuro-lingua-decisions-${timestamp}.csv`);
          }

          setLastExport(
            `CSV: Exported ${filteredData.runs.length} runs, ${filteredData.comparisons.length} comparisons, ${filteredData.decisions.length} decisions`
          );
        } else if (exportType === 'runs') {
          const runsCSV = exportRunsToCSV(filteredData.runs, projectsMap);
          downloadCSV(runsCSV, `neuro-lingua-runs-${timestamp}.csv`);
          setLastExport(`CSV: ${filteredData.runs.length} runs`);
        } else if (exportType === 'comparisons') {
          const comparisonsCSV = exportComparisonsToCSV(filteredData.comparisons, projectsMap);
          downloadCSV(comparisonsCSV, `neuro-lingua-comparisons-${timestamp}.csv`);
          setLastExport(`CSV: ${filteredData.comparisons.length} comparisons`);
        } else if (exportType === 'decisions') {
          const decisionsCSV = exportDecisionsToCSV(filteredData.decisions, projectsMap);
          downloadCSV(decisionsCSV, `neuro-lingua-decisions-${timestamp}.csv`);
          setLastExport(`CSV: ${filteredData.decisions.length} decisions`);
        }
      }

      setTimeout(() => setExporting(false), 500);
    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please check the console for details.');
      setExporting(false);
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1002,
        padding: 20,
        direction
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape' || e.key === 'Enter') {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        style={{
          background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          borderRadius: 16,
          padding: 24,
          maxWidth: 700,
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '2px solid #475569'
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 20
          }}
        >
          <h2 style={{ margin: 0, color: '#a78bfa', fontSize: '1.5rem' }}>📦 Data Export</h2>
          <button
            onClick={onClose}
            style={{
              background: '#374151',
              border: '1px solid #4b5563',
              borderRadius: 8,
              color: '#e5e7eb',
              padding: '8px 16px',
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            ✕ Close
          </button>
        </div>

        {/* Export Format */}
        <div
          style={{
            background: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, color: '#a78bfa', marginBottom: 12 }}>
            1. Select Export Format
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              onClick={() => setExportFormat('json')}
              style={{
                flex: 1,
                padding: 12,
                background: exportFormat === 'json' ? '#7c3aed' : 'rgba(30, 41, 59, 0.5)',
                border: exportFormat === 'json' ? '2px solid #a78bfa' : '1px solid #475569',
                borderRadius: 8,
                color: '#e2e8f0',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              📄 JSON
              <div style={{ fontSize: 11, fontWeight: 400, marginTop: 4 }}>
                Complete data with full structure
              </div>
            </button>
            <button
              onClick={() => setExportFormat('csv')}
              style={{
                flex: 1,
                padding: 12,
                background: exportFormat === 'csv' ? '#7c3aed' : 'rgba(30, 41, 59, 0.5)',
                border: exportFormat === 'csv' ? '2px solid #a78bfa' : '1px solid #475569',
                borderRadius: 8,
                color: '#e2e8f0',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              📊 CSV
              <div style={{ fontSize: 11, fontWeight: 400, marginTop: 4 }}>
                Tabular data for analysis
              </div>
            </button>
          </div>
        </div>

        {/* Export Scope */}
        <div
          style={{
            background: 'rgba(34, 197, 94, 0.08)',
            border: '1px solid rgba(34, 197, 94, 0.25)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, color: '#22c55e', marginBottom: 12 }}>
            2. Select Scope
          </div>
          <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
            <button
              onClick={() => setExportScope('all')}
              style={{
                flex: 1,
                padding: 10,
                background: exportScope === 'all' ? '#059669' : 'rgba(30, 41, 59, 0.5)',
                border: exportScope === 'all' ? '2px solid #10b981' : '1px solid #475569',
                borderRadius: 8,
                color: '#e2e8f0',
                fontWeight: 600,
                cursor: 'pointer',
                fontSize: 13
              }}
            >
              All Projects
            </button>
            <button
              onClick={() => setExportScope('project')}
              style={{
                flex: 1,
                padding: 10,
                background: exportScope === 'project' ? '#059669' : 'rgba(30, 41, 59, 0.5)',
                border: exportScope === 'project' ? '2px solid #10b981' : '1px solid #475569',
                borderRadius: 8,
                color: '#e2e8f0',
                fontWeight: 600,
                cursor: 'pointer',
                fontSize: 13
              }}
            >
              Single Project
            </button>
          </div>

          {exportScope === 'project' && (
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              style={{
                width: '100%',
                padding: 10,
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 8,
                color: '#e2e8f0',
                fontSize: 13
              }}
            >
              <option value="">-- Select Project --</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name} ({runs.filter((r) => r.projectId === project.id).length} runs)
                </option>
              ))}
            </select>
          )}
        </div>

        {/* CSV-specific: Export Type */}
        {exportFormat === 'csv' && (
          <div
            style={{
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
              borderRadius: 12,
              padding: 16,
              marginBottom: 16
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f59e0b', marginBottom: 12 }}>
              3. Select Data Type
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <button
                onClick={() => setExportType('complete')}
                style={{
                  padding: 8,
                  background: exportType === 'complete' ? '#d97706' : 'rgba(30, 41, 59, 0.5)',
                  border: exportType === 'complete' ? '2px solid #f59e0b' : '1px solid #475569',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Complete (All)
              </button>
              <button
                onClick={() => setExportType('runs')}
                style={{
                  padding: 8,
                  background: exportType === 'runs' ? '#d97706' : 'rgba(30, 41, 59, 0.5)',
                  border: exportType === 'runs' ? '2px solid #f59e0b' : '1px solid #475569',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Runs Only
              </button>
              <button
                onClick={() => setExportType('comparisons')}
                style={{
                  padding: 8,
                  background: exportType === 'comparisons' ? '#d97706' : 'rgba(30, 41, 59, 0.5)',
                  border: exportType === 'comparisons' ? '2px solid #f59e0b' : '1px solid #475569',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Comparisons Only
              </button>
              <button
                onClick={() => setExportType('decisions')}
                style={{
                  padding: 8,
                  background: exportType === 'decisions' ? '#d97706' : 'rgba(30, 41, 59, 0.5)',
                  border: exportType === 'decisions' ? '2px solid #f59e0b' : '1px solid #475569',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Decisions Only
              </button>
            </div>
          </div>
        )}

        {/* Statistics */}
        <div
          style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid #334155',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8', marginBottom: 12 }}>
            📊 Export Preview
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#a78bfa' }}>
                {stats.projects}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Projects</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981' }}>{stats.runs}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Runs ({stats.completedRuns} completed)
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#3b82f6' }}>
                {stats.comparisons}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Comparisons</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#f59e0b' }}>
                {stats.decisions}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Decisions</div>
            </div>
          </div>
        </div>

        {/* Export Button */}
        <button
          onClick={handleExport}
          disabled={exporting || (exportScope === 'project' && !selectedProjectId)}
          style={{
            width: '100%',
            padding: 16,
            background:
              exporting || (exportScope === 'project' && !selectedProjectId)
                ? '#374151'
                : 'linear-gradient(90deg, #7c3aed, #059669)',
            border: 'none',
            borderRadius: 10,
            color: 'white',
            fontWeight: 700,
            fontSize: 16,
            cursor:
              exporting || (exportScope === 'project' && !selectedProjectId)
                ? 'not-allowed'
                : 'pointer',
            opacity: exporting || (exportScope === 'project' && !selectedProjectId) ? 0.5 : 1
          }}
        >
          {exporting ? '⏳ Exporting...' : `⬇️ Export ${exportFormat.toUpperCase()}`}
        </button>

        {/* Last Export Info */}
        {lastExport && (
          <div
            style={{
              marginTop: 12,
              padding: 12,
              background: 'rgba(16, 185, 129, 0.08)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: 8,
              fontSize: 12,
              color: '#10b981',
              textAlign: 'center'
            }}
          >
            ✅ Last export: {lastExport}
          </div>
        )}
      </div>
    </div>
  );
}
