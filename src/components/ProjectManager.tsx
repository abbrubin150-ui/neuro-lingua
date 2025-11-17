/**
 * ProjectManager - Main UI for managing Projects and Runs
 */

import React, { useState } from 'react';
import { useProjects, useCreateDecisionLedger } from '../contexts/ProjectContext';
import { DecisionLedgerEditor } from './DecisionLedgerEditor';
import type { DecisionLedger } from '../types/project';

interface ProjectManagerProps {
  direction?: 'ltr' | 'rtl';
  onClose?: () => void;
}

export function ProjectManager({ direction = 'ltr', onClose }: ProjectManagerProps) {
  const {
    projects,
    activeProject,
    activeProjectId,
    projectRuns,
    createNewProject,
    setActiveProject,
    deleteProject,
    updateProject
  } = useProjects();

  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [newProjectLanguage, setNewProjectLanguage] = useState<'en' | 'he' | 'mixed'>('en');

  const handleCreateProject = () => {
    if (!newProjectName.trim()) return;
    createNewProject(newProjectName, newProjectDescription, newProjectLanguage);
    setNewProjectName('');
    setNewProjectDescription('');
    setIsCreatingProject(false);
  };

  const handleDeleteProject = (id: string) => {
    if (window.confirm('Are you sure you want to delete this project and all its runs?')) {
      deleteProject(id);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: 20,
        direction
      }}
      onClick={onClose}
    >
      <div
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
        onClick={(e) => e.stopPropagation()}
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
            📁 Project Manager
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

        {/* Create New Project Button */}
        {!isCreatingProject && (
          <button
            onClick={() => setIsCreatingProject(true)}
            style={{
              width: '100%',
              padding: '12px 20px',
              background: 'linear-gradient(90deg, #7c3aed, #059669)',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 700,
              cursor: 'pointer',
              marginBottom: 20
            }}
          >
            + Create New Project
          </button>
        )}

        {/* Create Project Form */}
        {isCreatingProject && (
          <div
            style={{
              background: 'rgba(99, 102, 241, 0.1)',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              borderRadius: 12,
              padding: 16,
              marginBottom: 20
            }}
          >
            <h3 style={{ margin: '0 0 12px 0', color: '#a78bfa', fontSize: '1.1rem' }}>
              New Project
            </h3>
            <div style={{ display: 'grid', gap: 12 }}>
              <div>
                <label
                  htmlFor="project-name"
                  style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
                >
                  Project Name
                </label>
                <input
                  id="project-name"
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  placeholder="e.g., Hebrew Poetry Analysis"
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: '#e2e8f0',
                    fontSize: 14
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="project-description"
                  style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
                >
                  Description
                </label>
                <textarea
                  id="project-description"
                  value={newProjectDescription}
                  onChange={(e) => setNewProjectDescription(e.target.value)}
                  placeholder="Brief description of the project goals..."
                  style={{
                    width: '100%',
                    minHeight: 60,
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: '#e2e8f0',
                    fontSize: 14,
                    fontFamily: 'inherit',
                    resize: 'vertical'
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="project-language"
                  style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
                >
                  Primary Language
                </label>
                <select
                  id="project-language"
                  value={newProjectLanguage}
                  onChange={(e) =>
                    setNewProjectLanguage(e.target.value as 'en' | 'he' | 'mixed')
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: '#e2e8f0',
                    fontSize: 14
                  }}
                >
                  <option value="en">English</option>
                  <option value="he">Hebrew (עברית)</option>
                  <option value="mixed">Mixed / Multilingual</option>
                </select>
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={handleCreateProject}
                  disabled={!newProjectName.trim()}
                  style={{
                    flex: 1,
                    padding: '10px 16px',
                    background: newProjectName.trim() ? '#10b981' : '#374151',
                    border: 'none',
                    borderRadius: 8,
                    color: 'white',
                    fontWeight: 600,
                    cursor: newProjectName.trim() ? 'pointer' : 'not-allowed'
                  }}
                >
                  ✓ Create
                </button>
                <button
                  onClick={() => {
                    setIsCreatingProject(false);
                    setNewProjectName('');
                    setNewProjectDescription('');
                  }}
                  style={{
                    flex: 1,
                    padding: '10px 16px',
                    background: '#374151',
                    border: '1px solid #4b5563',
                    borderRadius: 8,
                    color: '#e5e7eb',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Projects List */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ margin: '0 0 12px 0', color: '#94a3b8', fontSize: '1rem' }}>
            Your Projects ({projects.length})
          </h3>

          {projects.length === 0 && (
            <div
              style={{
                padding: 40,
                textAlign: 'center',
                color: '#64748b',
                fontSize: 14,
                background: 'rgba(0,0,0,0.2)',
                borderRadius: 8
              }}
            >
              No projects yet. Create your first project to get started!
            </div>
          )}

          {projects.map((project) => {
            const isActive = project.id === activeProjectId;
            const runs = projectRuns.filter((r) => r.projectId === project.id);

            return (
              <div
                key={project.id}
                style={{
                  background: isActive ? 'rgba(139, 92, 246, 0.15)' : 'rgba(30, 41, 59, 0.5)',
                  border: isActive ? '2px solid #a78bfa' : '1px solid #475569',
                  borderRadius: 12,
                  padding: 16,
                  marginBottom: 12,
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onClick={() => setActiveProject(isActive ? null : project.id)}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ fontSize: 18 }}>
                        {isActive ? '📂' : '📁'}
                      </span>
                      <h4 style={{ margin: 0, color: '#e2e8f0', fontSize: '1.1rem' }}>
                        {project.name}
                      </h4>
                      <span
                        style={{
                          fontSize: 11,
                          padding: '2px 8px',
                          background: '#334155',
                          borderRadius: 4,
                          color: '#94a3b8'
                        }}
                      >
                        {project.language.toUpperCase()}
                      </span>
                      <span
                        style={{
                          fontSize: 11,
                          padding: '2px 8px',
                          background: '#334155',
                          borderRadius: 4,
                          color: '#94a3b8'
                        }}
                      >
                        {project.defaultArchitecture}
                      </span>
                    </div>
                    <p style={{ margin: '8px 0', color: '#94a3b8', fontSize: 13 }}>
                      {project.description || 'No description'}
                    </p>
                    <div style={{ fontSize: 12, color: '#64748b' }}>
                      {runs.length} run{runs.length !== 1 ? 's' : ''} •{' '}
                      {project.scenarios.length} scenario{project.scenarios.length !== 1 ? 's' : ''}{' '}
                      • Created {new Date(project.createdAt).toLocaleDateString()}
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteProject(project.id);
                    }}
                    style={{
                      background: '#dc2626',
                      border: 'none',
                      borderRadius: 6,
                      color: 'white',
                      padding: '6px 12px',
                      cursor: 'pointer',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    Delete
                  </button>
                </div>

                {/* Show runs if active */}
                {isActive && runs.length > 0 && (
                  <div
                    style={{
                      marginTop: 12,
                      paddingTop: 12,
                      borderTop: '1px solid #475569'
                    }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#a78bfa', marginBottom: 8 }}>
                      Runs in this project:
                    </div>
                    {runs.map((run) => (
                      <div
                        key={run.id}
                        style={{
                          background: 'rgba(0,0,0,0.2)',
                          padding: 8,
                          borderRadius: 6,
                          marginBottom: 6,
                          fontSize: 12
                        }}
                      >
                        <div style={{ fontWeight: 600, color: '#e2e8f0' }}>
                          {run.name}
                        </div>
                        <div style={{ color: '#94a3b8', fontSize: 11 }}>
                          {run.config.architecture} • {run.status} •{' '}
                          {new Date(run.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Active Project Details */}
        {activeProject && (
          <div
            style={{
              background: 'rgba(139, 92, 246, 0.1)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: 12,
              padding: 16
            }}
          >
            <h3 style={{ margin: '0 0 12px 0', color: '#a78bfa', fontSize: '1.1rem' }}>
              Active Project: {activeProject.name}
            </h3>
            <div style={{ fontSize: 13, color: '#94a3b8' }}>
              <p style={{ margin: '0 0 8px 0' }}>{activeProject.description}</p>
              <div>
                <strong>Language:</strong> {activeProject.language} •{' '}
                <strong>Architecture:</strong> {activeProject.defaultArchitecture} •{' '}
                <strong>Corpus Type:</strong> {activeProject.corpusType}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
