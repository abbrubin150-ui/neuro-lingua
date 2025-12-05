/**
 * DecisionEntry2Panel - Decision Ledger 2.0 UI
 * Structured decision tracking with problem-alternatives-decision-KPI format
 */

import React, { useState, useMemo } from 'react';
import { useProjects } from '../contexts/ProjectContext';
import type { DecisionEntry } from '../types/experiment';

interface DecisionEntry2PanelProps {
  /** Optional project ID to filter decisions */
  projectId?: string;
  /** Callback when panel is closed */
  onClose?: () => void;
  /** Direction for RTL support */
  direction?: 'ltr' | 'rtl';
}

/**
 * Render a single decision card
 */
function DecisionCard({
  decision,
  projectName,
  onDelete
}: {
  decision: DecisionEntry;
  projectName: string;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const categoryColor = {
    compression: '#f59e0b',
    optimizer: '#3b82f6',
    architecture: '#a78bfa',
    hyperparameter: '#10b981',
    default: '#6b7280'
  }[decision.category || 'default'];

  const handleToggle = () => setExpanded(!expanded);

  return (
    <div
      role="button"
      tabIndex={0}
      style={{
        background: 'rgba(30, 41, 59, 0.6)',
        border: `1px solid ${categoryColor}40`,
        borderLeft: `4px solid ${categoryColor}`,
        borderRadius: 10,
        padding: 16,
        marginBottom: 12,
        cursor: 'pointer'
      }}
      onClick={handleToggle}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleToggle();
        }
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 8
        }}
      >
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 18 }}>{expanded ? '📖' : '📄'}</span>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>
              {decision.problem}
            </div>
            {decision.category && (
              <span
                style={{
                  fontSize: 10,
                  padding: '2px 8px',
                  background: categoryColor,
                  borderRadius: 12,
                  color: 'white',
                  fontWeight: 600
                }}
              >
                {decision.category.toUpperCase()}
              </span>
            )}
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>
            {projectName} • {decision.witness} • {new Date(decision.createdAt).toLocaleDateString()}
          </div>
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            if (window.confirm('Are you sure you want to delete this decision?')) {
              onDelete();
            }
          }}
          style={{
            background: '#dc2626',
            border: 'none',
            borderRadius: 6,
            color: 'white',
            padding: '4px 10px',
            cursor: 'pointer',
            fontSize: 11,
            fontWeight: 600
          }}
        >
          Delete
        </button>
      </div>

      {/* Collapsed View */}
      {!expanded && (
        <div
          style={{
            fontSize: 13,
            color: '#cbd5e1',
            padding: '8px 12px',
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: 6
          }}
        >
          ✅ <strong>Decision:</strong> {decision.decision}
        </div>
      )}

      {/* Expanded View */}
      {expanded && (
        <div style={{ marginTop: 12 }}>
          {/* Alternatives */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#f59e0b', marginBottom: 6 }}>
              🔄 Alternatives Considered:
            </div>
            <ul
              style={{
                margin: 0,
                paddingLeft: 20,
                fontSize: 12,
                color: '#94a3b8',
                listStyleType: 'circle'
              }}
            >
              {decision.alternatives.map((alt, i) => (
                <li key={i}>{alt}</li>
              ))}
            </ul>
          </div>

          {/* Decision */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#10b981', marginBottom: 6 }}>
              ✅ Chosen Decision:
            </div>
            <div
              style={{
                fontSize: 13,
                color: '#cbd5e1',
                padding: '8px 12px',
                background: 'rgba(16, 185, 129, 0.08)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: 6
              }}
            >
              {decision.decision}
            </div>
          </div>

          {/* KPI */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#3b82f6', marginBottom: 6 }}>
              📊 Key Performance Indicator:
            </div>
            <div
              style={{
                fontSize: 13,
                color: '#cbd5e1',
                padding: '8px 12px',
                background: 'rgba(59, 130, 246, 0.08)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: 6
              }}
            >
              {decision.kpi}
            </div>
          </div>

          {/* Affected Runs */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#a78bfa', marginBottom: 6 }}>
              🔗 Affected Runs:
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              {decision.affectedRunIds.length} run(s):{' '}
              {decision.affectedRunIds.slice(0, 3).join(', ')}
              {decision.affectedRunIds.length > 3 &&
                ` + ${decision.affectedRunIds.length - 3} more`}
            </div>
          </div>

          {/* Notes */}
          {decision.notes && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#64748b', marginBottom: 6 }}>
                📝 Notes:
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: '#94a3b8',
                  padding: '8px 12px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: 6,
                  fontStyle: 'italic'
                }}
              >
                {decision.notes}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Main decision panel component
 */
export function DecisionEntry2Panel({
  projectId,
  onClose,
  direction = 'ltr'
}: DecisionEntry2PanelProps) {
  const {
    projects,
    runs,
    decisions,
    createNewDecision,
    deleteDecision,
    getDecisionsByProject,
    getRunsByProject
  } = useProjects();

  // Filter state
  const [selectedProjectId, setSelectedProjectId] = useState<string>(projectId || '');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Form state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formProjectId, setFormProjectId] = useState<string>(projectId || '');
  const [formProblem, setFormProblem] = useState('');
  const [formAlternatives, setFormAlternatives] = useState<string[]>(['', '']);
  const [formDecision, setFormDecision] = useState('');
  const [formKPI, setFormKPI] = useState('');
  const [formCategory, setFormCategory] = useState('');
  const [formAffectedRunIds, setFormAffectedRunIds] = useState<string[]>([]);
  const [formWitness, setFormWitness] = useState('local-user');
  const [formNotes, setFormNotes] = useState('');

  // Filtered decisions
  const filteredDecisions = useMemo(() => {
    let result = selectedProjectId ? getDecisionsByProject(selectedProjectId) : decisions;

    if (categoryFilter !== 'all') {
      result = result.filter((d) => d.category === categoryFilter);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (d) =>
          d.problem.toLowerCase().includes(query) ||
          d.decision.toLowerCase().includes(query) ||
          d.kpi.toLowerCase().includes(query) ||
          d.alternatives.some((alt) => alt.toLowerCase().includes(query))
      );
    }

    return result.sort((a, b) => b.createdAt - a.createdAt);
  }, [decisions, selectedProjectId, categoryFilter, searchQuery, getDecisionsByProject]);

  // Available runs for selection
  const _availableRuns = useMemo(() => {
    if (formProjectId) {
      return getRunsByProject(formProjectId);
    }
    return runs;
  }, [formProjectId, runs, getRunsByProject]);

  // Categories for filtering
  const categories = useMemo(() => {
    const uniqueCategories = new Set(
      decisions.map((d) => d.category).filter((c): c is string => c !== undefined)
    );
    return ['all', ...Array.from(uniqueCategories)];
  }, [decisions]);

  // Handle create decision
  const handleCreateDecision = () => {
    if (!formProjectId || !formProblem || !formDecision || !formKPI) {
      alert('Please fill in all required fields');
      return;
    }

    const validAlternatives = formAlternatives.filter((alt) => alt.trim() !== '');
    if (validAlternatives.length === 0) {
      alert('Please provide at least one alternative');
      return;
    }

    createNewDecision(
      formProjectId,
      formProblem,
      validAlternatives,
      formDecision,
      formKPI,
      formAffectedRunIds,
      formWitness,
      formCategory || undefined
    );

    // Reset form
    setShowCreateForm(false);
    setFormProblem('');
    setFormAlternatives(['', '']);
    setFormDecision('');
    setFormKPI('');
    setFormCategory('');
    setFormAffectedRunIds([]);
    setFormNotes('');
  };

  // Add alternative field
  const addAlternative = () => {
    setFormAlternatives([...formAlternatives, '']);
  };

  // Remove alternative field
  const removeAlternative = (index: number) => {
    setFormAlternatives(formAlternatives.filter((_, i) => i !== index));
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
        zIndex: 1003,
        padding: 20,
        direction,
        overflow: 'auto'
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
          maxWidth: 900,
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
          <h2 style={{ margin: 0, color: '#a78bfa', fontSize: '1.5rem' }}>
            📋 Decision Ledger 2.0
          </h2>
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

        {/* Create Button */}
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          style={{
            width: '100%',
            padding: 12,
            background: showCreateForm ? '#374151' : 'linear-gradient(90deg, #7c3aed, #059669)',
            border: 'none',
            borderRadius: 10,
            color: 'white',
            fontWeight: 700,
            cursor: 'pointer',
            marginBottom: 16
          }}
        >
          {showCreateForm ? '✕ Cancel' : '+ Create New Decision'}
        </button>

        {/* Create Form */}
        {showCreateForm && (
          <div
            style={{
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              borderRadius: 12,
              padding: 16,
              marginBottom: 16
            }}
          >
            <h3 style={{ margin: '0 0 16px 0', color: '#a78bfa', fontSize: '1.1rem' }}>
              New Decision Entry
            </h3>

            {/* Project Selection */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-project"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Project *
              </label>
              <select
                id="decision-project"
                value={formProjectId}
                onChange={(e) => setFormProjectId(e.target.value)}
                style={{
                  width: '100%',
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              >
                <option value="">-- Select Project --</option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Problem Statement */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-problem"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Problem Statement *
              </label>
              <input
                id="decision-problem"
                type="text"
                value={formProblem}
                onChange={(e) => setFormProblem(e.target.value)}
                placeholder="What problem needs to be solved?"
                style={{
                  width: '100%',
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              />
            </div>

            {/* Alternatives */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-alternative-0"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Alternatives Considered *
              </label>
              {formAlternatives.map((alt, i) => (
                <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                  <input
                    id={`decision-alternative-${i}`}
                    type="text"
                    value={alt}
                    onChange={(e) => {
                      const newAlts = [...formAlternatives];
                      newAlts[i] = e.target.value;
                      setFormAlternatives(newAlts);
                    }}
                    placeholder={`Alternative ${i + 1}`}
                    style={{
                      flex: 1,
                      padding: 8,
                      background: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: 6,
                      color: '#e2e8f0',
                      fontSize: 13
                    }}
                  />
                  {formAlternatives.length > 1 && (
                    <button
                      onClick={() => removeAlternative(i)}
                      style={{
                        padding: '0 10px',
                        background: '#dc2626',
                        border: 'none',
                        borderRadius: 6,
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: 12
                      }}
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
              <button
                onClick={addAlternative}
                style={{
                  padding: '6px 12px',
                  background: '#059669',
                  border: 'none',
                  borderRadius: 6,
                  color: 'white',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                + Add Alternative
              </button>
            </div>

            {/* Decision */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-chosen"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Chosen Decision *
              </label>
              <textarea
                id="decision-chosen"
                value={formDecision}
                onChange={(e) => setFormDecision(e.target.value)}
                placeholder="What was decided?"
                style={{
                  width: '100%',
                  minHeight: 60,
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13,
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
            </div>

            {/* KPI */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-kpi"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Key Performance Indicator *
              </label>
              <input
                id="decision-kpi"
                type="text"
                value={formKPI}
                onChange={(e) => setFormKPI(e.target.value)}
                placeholder="How will success be measured?"
                style={{
                  width: '100%',
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              />
            </div>

            {/* Category */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-category"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Category (optional)
              </label>
              <select
                id="decision-category"
                value={formCategory}
                onChange={(e) => setFormCategory(e.target.value)}
                style={{
                  width: '100%',
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              >
                <option value="">-- Select Category --</option>
                <option value="architecture">Architecture</option>
                <option value="optimizer">Optimizer</option>
                <option value="hyperparameter">Hyperparameter</option>
                <option value="compression">Compression</option>
              </select>
            </div>

            {/* Witness */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-witness"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Witness (Decision Maker)
              </label>
              <input
                id="decision-witness"
                type="text"
                value={formWitness}
                onChange={(e) => setFormWitness(e.target.value)}
                style={{
                  width: '100%',
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              />
            </div>

            {/* Notes */}
            <div style={{ marginBottom: 12 }}>
              <label
                htmlFor="decision-notes"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Additional Notes
              </label>
              <textarea
                id="decision-notes"
                value={formNotes}
                onChange={(e) => setFormNotes(e.target.value)}
                placeholder="Any additional context..."
                style={{
                  width: '100%',
                  minHeight: 50,
                  padding: 8,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 13,
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
            </div>

            {/* Create Button */}
            <button
              onClick={handleCreateDecision}
              style={{
                width: '100%',
                padding: 12,
                background: 'linear-gradient(90deg, #7c3aed, #059669)',
                border: 'none',
                borderRadius: 8,
                color: 'white',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              ✓ Create Decision
            </button>
          </div>
        )}

        {/* Filters */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 2fr',
            gap: 12,
            marginBottom: 16
          }}
        >
          {/* Project Filter */}
          {!projectId && (
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              style={{
                padding: 8,
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                color: '#e2e8f0',
                fontSize: 12
              }}
            >
              <option value="">All Projects</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          )}

          {/* Category Filter */}
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            style={{
              padding: 8,
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              color: '#e2e8f0',
              fontSize: 12
            }}
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat === 'all' ? 'All Categories' : cat}
              </option>
            ))}
          </select>

          {/* Search */}
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search decisions..."
            style={{
              padding: 8,
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              color: '#e2e8f0',
              fontSize: 12,
              gridColumn: projectId ? 'span 2' : 'auto'
            }}
          />
        </div>

        {/* Decisions List */}
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8', marginBottom: 12 }}>
            {filteredDecisions.length} decision{filteredDecisions.length !== 1 ? 's' : ''} found
          </div>

          {filteredDecisions.length === 0 && (
            <div
              style={{
                textAlign: 'center',
                padding: 40,
                color: '#64748b',
                fontSize: 14
              }}
            >
              No decisions found. Create your first decision to get started!
            </div>
          )}

          {filteredDecisions.map((decision) => {
            const project = projects.find((p) => p.id === decision.projectId);
            return (
              <DecisionCard
                key={decision.id}
                decision={decision}
                projectName={project?.name || 'Unknown'}
                onDelete={() => deleteDecision(decision.id)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
