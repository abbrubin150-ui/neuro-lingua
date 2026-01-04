/**
 * Triadic Operator Panel
 *
 * Visualization component for the triadic operator table.
 * Displays all domains and their triadic relationships across different
 * conceptual frameworks.
 */

import React, { useState } from 'react';
import { TRIADIC_TABLE, getTriadicCell } from '../data/triadicTable';
import { triadicOperator, triadicVectorToString } from '../lib/triadicOperator';
import type { TriadicCell, TriadicVector } from '../types/triadic';

interface TriadicOperatorPanelProps {
  language?: 'en' | 'he';
}

interface CellDisplayProps {
  cell: TriadicCell;
  domainId: string;
  columnIndex: number;
  onClick: () => void;
}

/**
 * Individual cell display component
 */
const CellDisplay: React.FC<CellDisplayProps> = ({ cell, onClick }) => {
  // Compute the triadic vector for this cell (assuming all true for visualization)
  const vector = triadicOperator(true, true, true);

  return (
    <div
      onClick={onClick}
      className="triadic-cell"
      style={{
        padding: '8px',
        margin: '2px',
        border: '1px solid #ddd',
        borderRadius: '4px',
        cursor: 'pointer',
        fontSize: '12px',
        textAlign: 'center',
        backgroundColor: vector.strong ? '#e8f5e9' : vector.tension ? '#fff3e0' : '#f5f5f5',
      }}
      title={`${cell.a} → ${cell.b} → ${cell.c}\n${triadicVectorToString(vector)}`}
    >
      <div style={{ fontSize: '16px', marginBottom: '4px' }}>
        {cell.emojiA} → {cell.emojiB} → {cell.emojiC}
      </div>
      <div style={{ fontSize: '9px', color: '#666' }}>
        {cell.a} → {cell.b} → {cell.c}
      </div>
    </div>
  );
};

/**
 * Detail panel for a selected cell
 */
interface CellDetailProps {
  cell: TriadicCell;
  vector: TriadicVector;
  onClose: () => void;
}

const CellDetail: React.FC<CellDetailProps> = ({ cell, vector, onClose }) => {
  return (
    <div
      style={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: 'white',
        padding: '24px',
        borderRadius: '8px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        zIndex: 1000,
        minWidth: '400px',
      }}
    >
      <h3 style={{ marginTop: 0 }}>Triadic Cell Details</h3>

      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '24px', marginBottom: '8px' }}>
          {cell.emojiA} → {cell.emojiB} → {cell.emojiC}
        </div>
        <div style={{ color: '#666' }}>
          {cell.a} → {cell.b} → {cell.c}
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <h4>State Vector: {triadicVectorToString(vector)}</h4>
        <ul style={{ listStyle: 'none', padding: 0 }}>
          <li style={{ color: vector.weak ? '#4caf50' : '#999' }}>
            {vector.weak ? '✓' : '✗'} Weak Link (W): At least one connection through B
          </li>
          <li style={{ color: vector.strong ? '#4caf50' : '#999' }}>
            {vector.strong ? '✓' : '✗'} Strong Link (S): Complete triad with both links
          </li>
          <li style={{ color: vector.tension ? '#ff9800' : '#999' }}>
            {vector.tension ? '✓' : '✗'} Tension (T): Exactly one strong link
          </li>
          <li style={{ color: vector.null ? '#f44336' : '#999' }}>
            {vector.null ? '✓' : '✗'} Null (N): No connection at all
          </li>
        </ul>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <h4>Interpretation</h4>
        {vector.strong && (
          <p>
            <strong>Complete Triad:</strong> The mediator <em>{cell.b}</em> ({cell.emojiB})
            successfully connects <em>{cell.a}</em> ({cell.emojiA}) to <em>{cell.c}</em> (
            {cell.emojiC}), forming a stable three-way relationship.
          </p>
        )}
        {vector.tension && !vector.strong && (
          <p>
            <strong>Partial Connection:</strong> An asymmetric situation where the mediator{' '}
            <em>{cell.b}</em> ({cell.emojiB}) connects to only one side, creating systemic
            tension.
          </p>
        )}
        {vector.weak && !vector.strong && !vector.tension && (
          <p>
            <strong>Weak Connection:</strong> Some pathway exists through <em>{cell.b}</em> (
            {cell.emojiB}), but it doesn't create stable organization.
          </p>
        )}
        {vector.null && (
          <p>
            <strong>No Connection:</strong> The three elements do not form any meaningful
            relationship.
          </p>
        )}
      </div>

      <button
        onClick={onClose}
        style={{
          padding: '8px 16px',
          backgroundColor: '#2196f3',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
        }}
      >
        Close
      </button>
    </div>
  );
};

/**
 * Main triadic operator panel component
 */
export const TriadicOperatorPanel: React.FC<TriadicOperatorPanelProps> = ({
  language = 'en',
}) => {
  const [selectedCell, setSelectedCell] = useState<{
    cell: TriadicCell;
    domainId: string;
    columnIndex: number;
  } | null>(null);
  const [selectedDomainId, setSelectedDomainId] = useState<string | null>(null);

  const handleCellClick = (cell: TriadicCell, domainId: string, columnIndex: number) => {
    setSelectedCell({ cell, domainId, columnIndex });
  };

  const handleCloseDetail = () => {
    setSelectedCell(null);
  };

  const displayedDomains = selectedDomainId
    ? TRIADIC_TABLE.domains.filter((d) => d.id === selectedDomainId)
    : TRIADIC_TABLE.domains;

  return (
    <div style={{ padding: '16px', fontFamily: 'system-ui, sans-serif' }}>
      <h2>𝕋 Triadic Operator Table</h2>

      <p style={{ marginBottom: '16px', color: '#666' }}>
        A unified framework for analyzing three-way relationships using NAND-based logic. Each
        cell represents a triadic relationship 𝕋(A,B,C) = ⟨W,S,T,N⟩.
      </p>

      {/* Domain filter */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ marginRight: '8px' }}>Filter by Domain:</label>
        <select
          value={selectedDomainId || ''}
          onChange={(e) => setSelectedDomainId(e.target.value || null)}
          style={{ padding: '4px 8px' }}
        >
          <option value="">All Domains</option>
          {TRIADIC_TABLE.domains.map((domain) => (
            <option key={domain.id} value={domain.id}>
              {domain.emoji} {language === 'he' ? domain.nameHe : domain.nameEn}
            </option>
          ))}
        </select>
      </div>

      {/* Legend */}
      <div style={{ marginBottom: '16px', fontSize: '12px' }}>
        <strong>Color Legend:</strong>{' '}
        <span style={{ backgroundColor: '#e8f5e9', padding: '2px 8px', marginLeft: '8px' }}>
          Strong (S)
        </span>
        <span style={{ backgroundColor: '#fff3e0', padding: '2px 8px', marginLeft: '8px' }}>
          Tension (T)
        </span>
        <span style={{ backgroundColor: '#f5f5f5', padding: '2px 8px', marginLeft: '8px' }}>
          Other
        </span>
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        {displayedDomains.map((domain) => (
          <div key={domain.id} style={{ marginBottom: '24px' }}>
            <h3 style={{ margin: '8px 0' }}>
              {domain.emoji} {language === 'he' ? domain.nameHe : domain.nameEn}
            </h3>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${TRIADIC_TABLE.columns.length}, 1fr)`,
                gap: '4px',
              }}
            >
              {domain.cells.map((cell, idx) => (
                <CellDisplay
                  key={idx}
                  cell={cell}
                  domainId={domain.id}
                  columnIndex={idx}
                  onClick={() => handleCellClick(cell, domain.id, idx)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detail modal */}
      {selectedCell && (
        <>
          {/* Backdrop */}
          <div
            onClick={handleCloseDetail}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.5)',
              zIndex: 999,
            }}
          />

          {/* Detail panel */}
          <CellDetail
            cell={selectedCell.cell}
            vector={triadicOperator(true, true, true)}
            onClose={handleCloseDetail}
          />
        </>
      )}

      {/* Footer */}
      <div style={{ marginTop: '32px', padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
        <h4>About the Triadic Operator</h4>
        <p style={{ fontSize: '14px', color: '#666', margin: '8px 0' }}>
          The triadic operator 𝕋 computes a 4-bit vector ⟨W,S,T,N⟩ using only NAND gates:
        </p>
        <ul style={{ fontSize: '13px', color: '#666' }}>
          <li><strong>W (Weak):</strong> At least one weak link through B</li>
          <li><strong>S (Strong):</strong> Both links are strong (complete triad)</li>
          <li><strong>T (Tension):</strong> Exactly one strong link (XOR)</li>
          <li><strong>N (Null):</strong> No weak link at all</li>
        </ul>
        <p style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
          See <code>/docs/theory/triadic-operator.md</code> for complete documentation.
        </p>
      </div>
    </div>
  );
};

export default TriadicOperatorPanel;
