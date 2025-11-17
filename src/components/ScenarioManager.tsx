/**
 * ScenarioManager - UI for managing test scenarios within a project
 */

import React, { useState } from 'react';
import { useProjects, useCreateScenario } from '../contexts/ProjectContext';

interface ScenarioManagerProps {
  direction?: 'ltr' | 'rtl';
}

export function ScenarioManager({ direction = 'ltr' }: ScenarioManagerProps) {
  const { activeProject, addScenarioToProject, deleteScenario } = useProjects();
  const createScenario = useCreateScenario();

  const [isAdding, setIsAdding] = useState(false);
  const [newName, setNewName] = useState('');
  const [newPrompt, setNewPrompt] = useState('');
  const [newExpected, setNewExpected] = useState('');

  if (!activeProject) {
    return (
      <div
        style={{
          background: 'rgba(30,41,59,0.9)',
          border: '1px solid #334155',
          borderRadius: 16,
          padding: 20,
          marginBottom: 20,
          direction
        }}
      >
        <h3 style={{ color: '#94a3b8', margin: '0 0 12px 0' }}>🎯 Scenario Suite</h3>
        <div style={{ fontSize: 13, color: '#64748b', padding: 20, textAlign: 'center' }}>
          Please select or create a project first to manage scenarios.
        </div>
      </div>
    );
  }

  const handleAddScenario = () => {
    if (!newName.trim() || !newPrompt.trim()) return;

    const scenario = createScenario(newName, newPrompt, newExpected || undefined);
    addScenarioToProject(activeProject.id, scenario);

    setNewName('');
    setNewPrompt('');
    setNewExpected('');
    setIsAdding(false);
  };

  const handleDeleteScenario = (scenarioId: string) => {
    if (window.confirm('Delete this scenario?')) {
      deleteScenario(activeProject.id, scenarioId);
    }
  };

  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        marginBottom: 20,
        direction
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16
        }}
      >
        <h3 style={{ color: '#a78bfa', margin: 0 }}>
          🎯 Scenario Suite ({activeProject.scenarios.length})
        </h3>
        <button
          onClick={() => setIsAdding(!isAdding)}
          style={{
            padding: '8px 16px',
            background: isAdding ? '#374151' : '#10b981',
            border: 'none',
            borderRadius: 8,
            color: 'white',
            fontWeight: 600,
            cursor: 'pointer',
            fontSize: 12
          }}
        >
          {isAdding ? 'Cancel' : '+ Add Scenario'}
        </button>
      </div>

      {/* Add Scenario Form */}
      {isAdding && (
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ display: 'grid', gap: 12 }}>
            <div>
              <label
                htmlFor="scenario-name"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Scenario Name
              </label>
              <input
                id="scenario-name"
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g., Greeting response"
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: '#e2e8f0',
                  fontSize: 13
                }}
              />
            </div>

            <div>
              <label
                htmlFor="scenario-prompt"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Prompt
              </label>
              <textarea
                id="scenario-prompt"
                value={newPrompt}
                onChange={(e) => setNewPrompt(e.target.value)}
                placeholder="The input text to test..."
                style={{
                  width: '100%',
                  minHeight: 60,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: '#e2e8f0',
                  fontSize: 13,
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
            </div>

            <div>
              <label
                htmlFor="scenario-expected"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Expected Response (Optional)
              </label>
              <textarea
                id="scenario-expected"
                value={newExpected}
                onChange={(e) => setNewExpected(e.target.value)}
                placeholder="What should the model ideally respond?"
                style={{
                  width: '100%',
                  minHeight: 60,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: '#e2e8f0',
                  fontSize: 13,
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
            </div>

            <button
              onClick={handleAddScenario}
              disabled={!newName.trim() || !newPrompt.trim()}
              style={{
                padding: '10px 16px',
                background: newName.trim() && newPrompt.trim() ? '#10b981' : '#374151',
                border: 'none',
                borderRadius: 8,
                color: 'white',
                fontWeight: 600,
                cursor: newName.trim() && newPrompt.trim() ? 'pointer' : 'not-allowed'
              }}
            >
              ✓ Add Scenario
            </button>
          </div>
        </div>
      )}

      {/* Scenarios List */}
      {activeProject.scenarios.length === 0 && !isAdding && (
        <div
          style={{
            padding: 20,
            textAlign: 'center',
            color: '#64748b',
            fontSize: 13,
            background: 'rgba(0,0,0,0.2)',
            borderRadius: 8
          }}
        >
          No scenarios yet. Add scenarios to automatically test your model during training.
        </div>
      )}

      {activeProject.scenarios.map((scenario) => (
        <div
          key={scenario.id}
          style={{
            background: 'rgba(0,0,0,0.2)',
            border: '1px solid #475569',
            borderRadius: 8,
            padding: 12,
            marginBottom: 8
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: 8
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 14, marginBottom: 4 }}>
                {scenario.name}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
                <strong>Prompt:</strong> {scenario.prompt}
              </div>
              {scenario.expectedResponse && (
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
                  <strong>Expected:</strong> {scenario.expectedResponse}
                </div>
              )}
              {scenario.lastScore !== undefined && (
                <div style={{ fontSize: 11, color: '#64748b' }}>
                  Last score: {(scenario.lastScore * 100).toFixed(1)}%
                  {scenario.lastRunAt && (
                    <> • Run {new Date(scenario.lastRunAt).toLocaleString()}</>
                  )}
                </div>
              )}
            </div>

            <button
              onClick={() => handleDeleteScenario(scenario.id)}
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
        </div>
      ))}
    </div>
  );
}
