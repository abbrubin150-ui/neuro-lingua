/**
 * BrainPanel - UI for Brain History and Autonomous Mode
 *
 * Displays:
 * - Brain vitals (creativity, stability, mood)
 * - Feeding interface
 * - Diary of events
 * - Autonomous mode controls
 * - Suggestions from autonomous mode
 */

import React, { useState, useCallback } from 'react';
import { useBrain, useBrainFeed } from '../contexts/BrainContext';
import { calculateHeavinessScore } from '../lib/BrainEngine';

interface BrainPanelProps {
  onActionSuggestion?: (action: 'FEED' | 'TRAIN') => void;
}

export function BrainPanel({ onActionSuggestion }: BrainPanelProps = {}) {
  const {
    brain,
    suggestions,
    statusMessage,
    setAutonomyEnabled,
    dismissSuggestion,
    actOnSuggestion,
    resetBrain,
    setBrainLabel,
    exportBrainHistory,
    isAutonomous
  } = useBrain();

  const dispatchFeed = useBrainFeed();

  const [feedText, setFeedText] = useState('');
  const [feedSummary, setFeedSummary] = useState('');
  const [editingLabel, setEditingLabel] = useState(false);
  const [labelInput, setLabelInput] = useState(brain.label);
  const [expandedSection, setExpandedSection] = useState<'vitals' | 'feed' | 'diary' | null>(
    'vitals'
  );
  const [diaryFilter, setDiaryFilter] = useState('');
  const [diaryTypeFilter, setDiaryTypeFilter] = useState<string>('ALL');

  // ========================================================================
  // Handlers
  // ========================================================================

  const handleFeed = useCallback(() => {
    if (!feedText.trim()) return;

    const words = feedText.trim().split(/\s+/);
    const uniqueWords = new Set(words);
    const newWordsCount = uniqueWords.size;
    const heavinessScore = calculateHeavinessScore(feedText);
    const summary = feedSummary.trim() || `${newWordsCount} words`;

    dispatchFeed(newWordsCount, heavinessScore, summary);

    setFeedText('');
    setFeedSummary('');
  }, [feedText, feedSummary, dispatchFeed]);

  const handleSaveLabel = useCallback(() => {
    if (labelInput.trim()) {
      setBrainLabel(labelInput.trim());
    }
    setEditingLabel(false);
  }, [labelInput, setBrainLabel]);

  const toggleSection = (section: 'vitals' | 'feed' | 'diary') => {
    setExpandedSection((prev) => (prev === section ? null : section));
  };

  const handleExportHistory = useCallback(() => {
    const jsonData = exportBrainHistory();
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `brain-history-${brain.id}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [exportBrainHistory, brain.id]);

  const handleSuggestionAction = useCallback((suggestionId: string, action: 'FEED' | 'TRAIN') => {
    actOnSuggestion(suggestionId);
    if (onActionSuggestion) {
      onActionSuggestion(action);
    }
  }, [actOnSuggestion, onActionSuggestion]);

  // ========================================================================
  // Diary filtering
  // ========================================================================

  const filteredDiary = React.useMemo(() => {
    let entries = [...brain.diary].reverse();

    // Filter by type
    if (diaryTypeFilter !== 'ALL') {
      entries = entries.filter((entry) => entry.type === diaryTypeFilter);
    }

    // Filter by search text
    if (diaryFilter.trim()) {
      const searchLower = diaryFilter.toLowerCase();
      entries = entries.filter((entry) =>
        entry.message.toLowerCase().includes(searchLower) ||
        entry.type.toLowerCase().includes(searchLower)
      );
    }

    return entries;
  }, [brain.diary, diaryFilter, diaryTypeFilter]);

  // ========================================================================
  // Mood visualization
  // ========================================================================

  const getMoodColor = (mood: string): string => {
    switch (mood) {
      case 'FOCUSED':
        return '#10b981'; // green
      case 'DREAMY':
        return '#a78bfa'; // purple
      case 'AGITATED':
        return '#ef4444'; // red
      case 'BURNT_OUT':
        return '#f59e0b'; // orange
      default:
        return '#6366f1'; // blue (CALM)
    }
  };

  const getMoodEmoji = (mood: string): string => {
    switch (mood) {
      case 'FOCUSED':
        return '🎯';
      case 'DREAMY':
        return '💭';
      case 'AGITATED':
        return '😰';
      case 'BURNT_OUT':
        return '🔥';
      default:
        return '😌'; // CALM
    }
  };

  // ========================================================================
  // Render
  // ========================================================================

  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 12,
        overflow: 'hidden'
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '16px 20px',
          borderBottom: '1px solid #334155',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(15,23,42,0.5)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 24 }}>🧠</span>
          {editingLabel ? (
            <input
              type="text"
              value={labelInput}
              onChange={(e) => setLabelInput(e.target.value)}
              onBlur={handleSaveLabel}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveLabel()}
              style={{
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: '4px 8px',
                color: '#e2e8f0',
                fontSize: 16,
                fontWeight: 600
              }}
            />
          ) : (
            <button
              onClick={() => setEditingLabel(true)}
              title="Click to edit label"
              style={{
                color: '#a78bfa',
                margin: 0,
                cursor: 'pointer',
                background: 'transparent',
                border: 'none',
                padding: 0,
                fontSize: 16,
                fontWeight: 600
              }}
            >
              {brain.label}
            </button>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>Autonomous:</span>
            <input
              type="checkbox"
              checked={isAutonomous}
              onChange={(e) => setAutonomyEnabled(e.target.checked)}
              style={{ cursor: 'pointer' }}
            />
          </label>
          <button
            onClick={handleExportHistory}
            title="Export brain history to JSON file"
            style={{
              padding: '6px 12px',
              background: '#10b981',
              border: 'none',
              borderRadius: 6,
              color: 'white',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            📥 Export
          </button>
          <button
            onClick={resetBrain}
            title="Reset brain to initial state"
            style={{
              padding: '6px 12px',
              background: '#ef4444',
              border: 'none',
              borderRadius: 6,
              color: 'white',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Suggestions (if any) */}
      {suggestions.length > 0 && (
        <div
          style={{
            padding: '12px 20px',
            background: 'rgba(99,102,241,0.1)',
            borderBottom: '1px solid #334155'
          }}
        >
          {suggestions.map((suggestion) => (
            <div
              key={suggestion.id}
              style={{
                padding: 12,
                background: 'rgba(99,102,241,0.2)',
                border: '1px solid #6366f1',
                borderRadius: 8,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 8,
                gap: 12
              }}
            >
              <span style={{ color: '#e2e8f0', fontSize: 14, flex: 1 }}>
                💡 {suggestion.message}
              </span>
              <div style={{ display: 'flex', gap: 6 }}>
                {suggestion.action !== 'NONE' && (
                  <button
                    onClick={() => handleSuggestionAction(suggestion.id, suggestion.action as 'FEED' | 'TRAIN')}
                    style={{
                      padding: '6px 12px',
                      background: '#6366f1',
                      border: 'none',
                      borderRadius: 6,
                      color: 'white',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                    title={`Click to ${suggestion.action.toLowerCase()}`}
                  >
                    {suggestion.action === 'FEED' ? '🍎 Feed' : '🎓 Train'}
                  </button>
                )}
                <button
                  onClick={() => dismissSuggestion(suggestion.id)}
                  style={{
                    padding: '6px 12px',
                    background: 'transparent',
                    border: '1px solid #6366f1',
                    borderRadius: 6,
                    color: '#6366f1',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div style={{ padding: 20 }}>
        {/* Status Message */}
        <div
          style={{
            padding: 12,
            background: 'rgba(15,23,42,0.5)',
            border: `1px solid ${getMoodColor(brain.mood)}`,
            borderRadius: 8,
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            gap: 12
          }}
        >
          <span style={{ fontSize: 28 }}>{getMoodEmoji(brain.mood)}</span>
          <div>
            <div style={{ color: getMoodColor(brain.mood), fontWeight: 600, fontSize: 14 }}>
              Mood: {brain.mood}
            </div>
            <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>{statusMessage}</div>
          </div>
        </div>

        {/* Vitals Section */}
        <div style={{ marginBottom: 16 }}>
          <button
            onClick={() => toggleSection('vitals')}
            style={{
              width: '100%',
              padding: '12px 16px',
              background: 'rgba(15,23,42,0.5)',
              border: '1px solid #334155',
              borderRadius: 8,
              color: '#a78bfa',
              fontWeight: 600,
              fontSize: 14,
              cursor: 'pointer',
              textAlign: 'left',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>📊 Brain Vitals</span>
            <span>{expandedSection === 'vitals' ? '▼' : '▶'}</span>
          </button>

          {expandedSection === 'vitals' && (
            <div
              style={{
                padding: 16,
                background: 'rgba(15,23,42,0.3)',
                borderRadius: 8,
                marginTop: 8
              }}
            >
              {/* Creativity Bar */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>Creativity</span>
                  <span style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>
                    {brain.creativity.toFixed(0)}%
                  </span>
                </div>
                <div
                  style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${brain.creativity}%`,
                      background: 'linear-gradient(90deg, #a78bfa 0%, #6366f1 100%)',
                      transition: 'width 0.3s ease'
                    }}
                  />
                </div>
              </div>

              {/* Stability Bar */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>Stability</span>
                  <span style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>
                    {brain.stability.toFixed(0)}%
                  </span>
                </div>
                <div
                  style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${brain.stability}%`,
                      background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                      transition: 'width 0.3s ease'
                    }}
                  />
                </div>
              </div>

              {/* Stats Grid */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: 12,
                  marginTop: 16
                }}
              >
                <div style={{ padding: 12, background: '#1e293b', borderRadius: 8 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11 }}>Training Steps</div>
                  <div style={{ color: '#e2e8f0', fontSize: 18, fontWeight: 600 }}>
                    {brain.totalTrainSteps.toLocaleString()}
                  </div>
                </div>
                <div style={{ padding: 12, background: '#1e293b', borderRadius: 8 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11 }}>Tokens Seen</div>
                  <div style={{ color: '#e2e8f0', fontSize: 18, fontWeight: 600 }}>
                    {brain.totalTokensSeen.toLocaleString()}
                  </div>
                </div>
                <div style={{ padding: 12, background: '#1e293b', borderRadius: 8 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11 }}>Vocabulary</div>
                  <div style={{ color: '#e2e8f0', fontSize: 18, fontWeight: 600 }}>
                    {brain.vocabSize.toLocaleString()}
                  </div>
                </div>
                <div style={{ padding: 12, background: '#1e293b', borderRadius: 8 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11 }}>Last Updated</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>
                    {new Date(brain.updatedAt).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Feed Section */}
        <div style={{ marginBottom: 16 }}>
          <button
            onClick={() => toggleSection('feed')}
            style={{
              width: '100%',
              padding: '12px 16px',
              background: 'rgba(15,23,42,0.5)',
              border: '1px solid #334155',
              borderRadius: 8,
              color: '#a78bfa',
              fontWeight: 600,
              fontSize: 14,
              cursor: 'pointer',
              textAlign: 'left',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🍎 Feed Brain</span>
            <span>{expandedSection === 'feed' ? '▼' : '▶'}</span>
          </button>

          {expandedSection === 'feed' && (
            <div
              style={{
                padding: 16,
                background: 'rgba(15,23,42,0.3)',
                borderRadius: 8,
                marginTop: 8
              }}
            >
              <textarea
                value={feedText}
                onChange={(e) => setFeedText(e.target.value)}
                placeholder="Enter new text to feed the brain..."
                style={{
                  width: '100%',
                  minHeight: 100,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  padding: 12,
                  color: '#e2e8f0',
                  fontSize: 13,
                  resize: 'vertical',
                  marginBottom: 8
                }}
              />
              <input
                type="text"
                value={feedSummary}
                onChange={(e) => setFeedSummary(e.target.value)}
                placeholder="Summary (e.g., 'poetry', 'legal text')..."
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  padding: '8px 12px',
                  color: '#e2e8f0',
                  fontSize: 13,
                  marginBottom: 8
                }}
              />
              <button
                onClick={handleFeed}
                disabled={!feedText.trim()}
                style={{
                  width: '100%',
                  padding: '10px 16px',
                  background: feedText.trim() ? '#6366f1' : '#475569',
                  border: 'none',
                  borderRadius: 8,
                  color: 'white',
                  fontWeight: 600,
                  fontSize: 14,
                  cursor: feedText.trim() ? 'pointer' : 'not-allowed'
                }}
              >
                Feed Brain
              </button>
              {brain.lastFeedSummary && (
                <div style={{ marginTop: 8, fontSize: 12, color: '#94a3b8' }}>
                  Last fed: {brain.lastFeedSummary}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Diary Section */}
        <div>
          <button
            onClick={() => toggleSection('diary')}
            style={{
              width: '100%',
              padding: '12px 16px',
              background: 'rgba(15,23,42,0.5)',
              border: '1px solid #334155',
              borderRadius: 8,
              color: '#a78bfa',
              fontWeight: 600,
              fontSize: 14,
              cursor: 'pointer',
              textAlign: 'left',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>📔 Diary ({brain.diary.length})</span>
            <span>{expandedSection === 'diary' ? '▼' : '▶'}</span>
          </button>

          {expandedSection === 'diary' && (
            <div
              style={{
                padding: 16,
                background: 'rgba(15,23,42,0.3)',
                borderRadius: 8,
                marginTop: 8
              }}
            >
              {/* Filter Controls */}
              <div style={{ marginBottom: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <input
                  type="text"
                  value={diaryFilter}
                  onChange={(e) => setDiaryFilter(e.target.value)}
                  placeholder="Search diary..."
                  style={{
                    flex: 1,
                    minWidth: 200,
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: '6px 10px',
                    color: '#e2e8f0',
                    fontSize: 12
                  }}
                />
                <select
                  value={diaryTypeFilter}
                  onChange={(e) => setDiaryTypeFilter(e.target.value)}
                  style={{
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: '6px 10px',
                    color: '#e2e8f0',
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                >
                  <option value="ALL">All Types</option>
                  <option value="TRAIN">Train</option>
                  <option value="GEN">Generation</option>
                  <option value="FEED">Feed</option>
                  <option value="MOOD_SHIFT">Mood Shift</option>
                  <option value="SUGGESTION">Suggestion</option>
                </select>
                {(diaryFilter || diaryTypeFilter !== 'ALL') && (
                  <button
                    onClick={() => {
                      setDiaryFilter('');
                      setDiaryTypeFilter('ALL');
                    }}
                    style={{
                      padding: '6px 10px',
                      background: '#475569',
                      border: 'none',
                      borderRadius: 6,
                      color: '#e2e8f0',
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Clear
                  </button>
                )}
              </div>

              {/* Diary Entries */}
              <div style={{ maxHeight: 350, overflowY: 'auto' }}>
                {filteredDiary.length === 0 ? (
                  <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>
                    {brain.diary.length === 0
                      ? 'No diary entries yet. Start training or feeding to create a history!'
                      : 'No entries match your filters.'}
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {filteredDiary.map((entry, index) => (
                      <div
                        key={index}
                        style={{
                          padding: 10,
                          background: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: 6
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: 4
                          }}
                        >
                          <span style={{ fontSize: 11, color: '#6366f1', fontWeight: 600 }}>
                            {entry.type}
                          </span>
                          <span style={{ fontSize: 11, color: '#94a3b8' }}>
                            {new Date(entry.timestamp).toLocaleString()}
                          </span>
                        </div>
                        <div style={{ fontSize: 13, color: '#e2e8f0' }}>{entry.message}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Autonomous Mode Info */}
        {isAutonomous && (
          <div
            style={{
              marginTop: 16,
              padding: 12,
              background: 'rgba(99,102,241,0.1)',
              border: '1px solid #6366f1',
              borderRadius: 8,
              fontSize: 12,
              color: '#94a3b8'
            }}
          >
            <strong style={{ color: '#6366f1' }}>Autonomous Mode Active</strong>
            <div style={{ marginTop: 4 }}>
              The brain will:
              <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                <li>Age naturally over time (idle decay)</li>
                <li>Suggest actions when needed</li>
                <li>Track all events in the diary</li>
              </ul>
              <em>Note: No heavy operations run automatically - only suggestions.</em>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
