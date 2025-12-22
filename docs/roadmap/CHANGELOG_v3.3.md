# Neuro-Lingua DOMESTICA v3.3 - Project & Run Management

## Overview

Major architectural expansion implementing **Projects**, **Runs**, and **Σ-SIG Decision Ledger** governance framework. This update transforms Neuro-Lingua from single-training mode to full experiment tracking and management.

> Note: The current runtime/version string in the UI remains **v3.2.4**; the governance/export fields documented here are already emitted by the live app.

---

## 🎯 Core Features Added

### 1. **Project & Run Architecture** (Axis A - Logical Structure)

#### Projects

- **Container for multiple training runs** with:
  - Name, description, and primary language (EN/HE/Mixed)
  - Default architecture selection
  - Corpus type classification (plain-text / dialogue-embedded)
  - Test scenarios suite
  - Tags for organization

#### Runs

- **Frozen training execution** with complete snapshot:
  - All hyperparameters (learning rate, epochs, optimizer, etc.)
  - Architecture configuration (feedforward/advanced/transformer)
  - Tokenizer settings
  - Corpus text with SHA256-like checksum
  - Decision Ledger (governance)
  - Training results and history
  - Model weights (JSON serialized)

### 2. **Decision Ledger System** (Axis C - Governance)

Implements **Σ-SIG / EXACT1** compliance framework:

#### Fields

- **Rationale**: Why this training run is necessary
- **Witness**: Who authorized the training (e.g., "local-user")
- **Expiry**: Optional expiration date (ISO 8601)
- **Rollback**: Action after expiry (keep/delete-after-expiry/archive)

#### Execution Status

- **EXECUTE**: All checks passed, training permitted ✅
- **HOLD**: Run expired or paused ⏸️
- **ESCALATE**: Missing rationale/witness, review required 🚨

### 3. **Scenario Suite** (Axis B - Testing)

- Define test scenarios per project
- Each scenario includes:
  - Name and prompt
  - Expected response (optional)
  - Last score and run timestamp
- Scenarios can be run automatically during training (future enhancement)

### 4. **Trace Export** (Axis C - Audit Trail)

Enhanced model export format with full traceability:

```json
{
  "modelWeights": "...",
  "config": {...},
  "tokenizer": {...},
  "projectMeta": {
    "projectId": "...",
    "projectName": "...",
    "runId": "...",
    "runName": "..."
  },
  "decisionLedger": {
    "rationale": "...",
    "witness": "local-user",
    "expiry": "2026-01-01T00:00:00.000Z",
    "rollback": "delete-after-expiry",
    "createdAt": 1234567890
  },
  "trainingTrace": {
    "epochs": 25,
    "finalLoss": 0.42,
    "finalAccuracy": 0.87,
    "finalPerplexity": 1.52,
    "trainLoss": [...],
    "trainAccuracy": [...],
    "timestamps": [...],
    "sha256_corpus": "a1b2c3d4",
    "scenariosScores": {...}
  },
  "exportedAt": 1234567890,
  "exportedBy": "local-user",
  "version": "3.3.0"
}
```

---

## 📁 New Files & Components

### Type Definitions

- `src/types/project.ts`: Core types for Project, Run, DecisionLedger, Scenario
  - Helper functions: `createProject`, `createRun`, `createScenario`, `createDecisionLedger`
  - Validation: `computeExecutionStatus`, `generateCorpusChecksum`

### Context Management

- `src/contexts/ProjectContext.tsx`: React Context for Projects/Runs
  - CRUD operations for projects
  - CRUD operations for runs
  - Scenario management
  - localStorage persistence
  - Hooks: `useProjects`, `useCreateScenario`, `useCreateDecisionLedger`

### UI Components

- `src/components/ProjectManager.tsx`: Main project management interface
  - Create/select/delete projects
  - View runs within projects
  - Visual hierarchy

- `src/components/DecisionLedgerEditor.tsx`: Governance editor
  - Edit rationale, witness, expiry, rollback
  - Live execution status indicator (EXECUTE/HOLD/ESCALATE)
  - Color-coded status badges

- `src/components/ScenarioManager.tsx`: Test scenario management
  - Add/edit/delete scenarios
  - View scenario results
  - Integrated with project context

### Utilities

- `src/lib/traceExport.ts`: Enhanced export functionality
  - `createTraceExport`: Generate full audit trail
  - `generateTraceFilename`: Consistent naming
  - `validateTraceExport`: Import validation

---

## 🔧 Modified Files

### App.tsx

- Wrapped application in `<ProjectProvider>`
- Added "📁 Projects" button to header
- Added `showProjectManager` state
- Modal overlay for ProjectManager

### components/index.ts

- Exported new components: `ProjectManager`, `DecisionLedgerEditor`, `ScenarioManager`

---

## 🎨 UI/UX Enhancements

- **Projects button** in top-right header (purple gradient)
- **Modal project manager** with:
  - Project creation form
  - Project list with active state highlighting
  - Run listing within active project
  - Delete confirmation dialogs

- **Color-coded status indicators**:
  - Green (#10b981): EXECUTE
  - Amber (#f59e0b): HOLD
  - Red (#ef4444): ESCALATE

- **Responsive layouts** with RTL support for Hebrew

---

## 📊 Data Model

### Storage Keys (localStorage)

- `neuro-lingua-projects-v1`: Array of projects
- `neuro-lingua-runs-v1`: Array of runs
- `neuro-lingua-active-project-v1`: Currently selected project ID
- `neuro-lingua-active-run-v1`: Currently selected run ID

### Relationships

```
Project (1) ──── (N) Run
  │
  └──── (N) Scenario

Run ──── (1) DecisionLedger
  │
  └──── (1) TrainingConfig
  │
  └──── (1) Results
         └──── (N) ScenarioResult
```

---

## 🚀 Future Enhancements (Not Implemented)

### Deferred from original plan:

1. **Privacy-Guard**: PII scanner before training (skipped for personal use)
2. **Automatic Scenario Execution**: Run scenarios every N epochs
3. **Enhanced Tokenization**: Semantic token detection and "Propose semantic splits"
4. **Experiment Comparison**: Side-by-side run comparison with charts
5. **Dialogue-Embedded Corpus**: Special handling for [HUMAN]/[MODEL] tags
6. **Stop-Rules**: Early stopping based on scenario performance
7. **Tiny-MoE Architecture**: Mixture of experts model

---

## 🧪 Testing

### Build Status

✅ TypeScript compilation successful
✅ Vite build successful
✅ No runtime errors in development mode

### Manual Testing Checklist

- [ ] Create new project
- [ ] View project list
- [ ] Select active project
- [ ] Add scenario to project
- [ ] Delete scenario
- [ ] Create run (future: integrate with training)
- [ ] Edit Decision Ledger
- [ ] Verify execution status changes
- [ ] Export model with trace
- [ ] Delete project with confirmation

---

## 📝 Migration Notes

### Backward Compatibility

- Existing localStorage models remain untouched
- Projects/Runs system is **opt-in**
- Original single-training workflow still works

### Breaking Changes

None. All changes are additive.

---

## 🔒 Governance Philosophy (Σ-SIG)

The Decision Ledger implements:

- **Σ** (Sigma): Signature/witness - who is accountable
- **SIG**: Signal - rationale for action
- **EXACT1**: Expiry-based accountability with clear rollback policy

This ensures:

1. **Traceability**: Every training run has documented purpose
2. **Accountability**: Clear ownership via witness field
3. **Temporal bounds**: Expiry dates prevent indefinite model drift
4. **Ethical AI**: Explicit rationale forces consideration of purpose

---

## 📦 Version Compatibility

- **React**: 18.2.0
- **TypeScript**: 5.2.2
- **Vite**: 4.4.5
- **Node**: 20+ recommended

---

## 👥 Contributors

- Implementation based on user specification for Σ-SIG/EXACT1 framework
- Designed for bilingual (EN/HE) research and personal AI development

---

## 📄 License

Same as main project (private/personal use)

---

**Next Release (v3.4)**: Scenario auto-execution, enhanced tokenization, experiment comparison UI
