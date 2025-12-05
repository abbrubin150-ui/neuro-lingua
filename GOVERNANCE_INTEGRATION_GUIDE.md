# Governance System Integration Guide

## Overview

This guide explains how to integrate the autonomous governance system into Neuro-Lingua's training workflow.

## Architecture Components

The governance system consists of:

1. **GovernanceEngine** (`src/lib/GovernanceEngine.ts`) - Core calibration logic
2. **Governance Types** (`src/types/governance.ts`) - Type definitions
3. **ProjectContext Integration** (`src/contexts/ProjectContext.tsx`) - State management
4. **GovernanceBoard** (`src/components/GovernanceBoard.tsx`) - UI component

## Integration Steps

### Step 1: Add GovernanceBoard to Your App

Add the GovernanceBoard component to your main App component or training interface:

```tsx
import { GovernanceBoard } from './components/GovernanceBoard';

function App() {
  return (
    <div>
      {/* Existing components */}
      <GovernanceBoard visible={true} />
      {/* Training panel, etc. */}
    </div>
  );
}
```

### Step 2: Record Metrics During Training

In your training loop (typically in `TrainingPanel.tsx` or where the model trains), record metrics after each epoch:

```tsx
import { useProjects } from '../contexts/ProjectContext';

function TrainingPanel() {
  const { recordTrainingMetrics, checkGovernance } = useProjects();

  // Generate unique session ID for this training session
  const sessionId = `train_${Date.now()}`;

  const trainModel = async () => {
    for (let epoch = 0; epoch < epochs; epoch++) {
      // ... existing training code ...

      // Record metrics after each epoch
      recordTrainingMetrics(
        sessionId,
        epoch,
        currentLoss,        // Training loss
        currentAccuracy,    // Training accuracy
        perplexity,         // Perplexity
        validationLoss,     // Optional: validation loss
        validationAccuracy  // Optional: validation accuracy
      );

      // ... continue training ...
    }
  };
}
```

### Step 3: Check for Governance After Training Sessions

After completing a training session, check if governance should activate:

```tsx
const trainModel = async () => {
  // ... training loop ...

  // After training completes
  const calibrationActions = checkGovernance(
    currentLearningRate,
    currentDropout,
    sessionId
  );

  // Apply calibration actions if any
  if (calibrationActions.length > 0) {
    calibrationActions.forEach(action => {
      if (action.parameter === 'learningRate') {
        setLearningRate(action.newValue);
        console.log(`🔧 Learning rate calibrated: ${action.previousValue.toFixed(4)} → ${action.newValue.toFixed(4)}`);
        console.log(`   Reason: ${action.reason}`);
      } else if (action.parameter === 'dropout') {
        setDropout(action.newValue);
        console.log(`🔧 Dropout calibrated: ${action.previousValue.toFixed(4)} → ${action.newValue.toFixed(4)}`);
        console.log(`   Reason: ${action.reason}`);
      }
    });
  }
};
```

### Step 4: Complete Integration Example

Here's a complete example of how to integrate governance into a training function:

```tsx
import React, { useState } from 'react';
import { useProjects } from '../contexts/ProjectContext';
import { GovernanceBoard } from './GovernanceBoard';

export function TrainingPanel() {
  const { recordTrainingMetrics, checkGovernance, getActiveAlerts } = useProjects();

  const [learningRate, setLearningRate] = useState(0.08);
  const [dropout, setDropout] = useState(0.1);
  const [isTraining, setIsTraining] = useState(false);

  const handleTrain = async () => {
    setIsTraining(true);

    // Generate unique session ID
    const sessionId = `train_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

    try {
      // Initialize model with current parameters
      const model = new YourModel({
        learningRate,
        dropout,
        // ... other params
      });

      // Training loop
      for (let epoch = 0; epoch < epochs; epoch++) {
        // Train one epoch
        const { loss, accuracy, perplexity } = await model.trainEpoch(corpus);

        // Record metrics for governance
        recordTrainingMetrics(
          sessionId,
          epoch,
          loss,
          accuracy,
          perplexity
        );

        // Update UI
        console.log(`Epoch ${epoch + 1}/${epochs}: Loss=${loss.toFixed(4)}, Accuracy=${accuracy.toFixed(2)}%`);
      }

      // After training session completes, check governance
      const calibrationActions = checkGovernance(learningRate, dropout, sessionId);

      // Apply calibration actions
      if (calibrationActions.length > 0) {
        console.log('\n🛡️ Autonomous Governance Activated');

        calibrationActions.forEach(action => {
          console.log(`\n🔧 ${action.parameter} calibrated:`);
          console.log(`   ${action.previousValue.toFixed(6)} → ${action.newValue.toFixed(6)}`);
          console.log(`   Reason: ${action.reason}`);

          // Apply the changes
          if (action.parameter === 'learningRate') {
            setLearningRate(action.newValue);
          } else if (action.parameter === 'dropout') {
            setDropout(action.newValue);
          }
        });

        // Show alerts to user
        const alerts = getActiveAlerts();
        alerts.forEach(alert => {
          console.log(`\n${alert.severity === 'critical' ? '🚨' : '⚠️'} ${alert.message}`);
        });
      }

    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div>
      <h2>Training Panel</h2>

      {/* Governance Board */}
      <GovernanceBoard visible={true} />

      {/* Training Controls */}
      <div>
        <label>
          Learning Rate: {learningRate.toFixed(6)}
          <input
            type="range"
            min="0.001"
            max="1"
            step="0.001"
            value={learningRate}
            onChange={e => setLearningRate(Number(e.target.value))}
            disabled={isTraining}
          />
        </label>

        <label>
          Dropout: {dropout.toFixed(2)}
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.01"
            value={dropout}
            onChange={e => setDropout(Number(e.target.value))}
            disabled={isTraining}
          />
        </label>

        <button onClick={handleTrain} disabled={isTraining}>
          {isTraining ? 'Training...' : 'Train Model'}
        </button>
      </div>
    </div>
  );
}
```

## Governance Configuration

You can customize the governance behavior by updating the governor configuration:

```tsx
import { useProjects } from '../contexts/ProjectContext';

function GovernanceSettings() {
  const { updateGovernorConfig } = useProjects();

  const handleConfigUpdate = () => {
    updateGovernorConfig({
      enabled: true,
      checkInterval: 3,              // Check every 3 sessions instead of 2
      activationProbability: 0.7,    // 70% activation chance
      improvementThreshold: 2.0,     // Require 2% improvement
      learningRate: {
        min: 1e-5,
        max: 0.5,
        decreaseFactor: 0.7,         // Decrease by 30%
        increaseFactor: 1.2          // Increase by 20%
      },
      dropout: {
        min: 0.0,
        max: 0.4,
        increaseStep: 0.1,           // Increase by 10%
        decreaseStep: 0.05           // Decrease by 5%
      },
      overfittingThreshold: 15.0,    // 15% train/val gap
      underfittingThreshold: 2.5,
      plateauWindow: 3               // Look back 3 sessions
    });
  };

  return (
    <button onClick={handleConfigUpdate}>
      Update Governance Config
    </button>
  );
}
```

## How the Governance System Works

### 1. **Metric Recording**
Every time you call `recordTrainingMetrics()`, the system stores:
- Training loss
- Training accuracy
- Perplexity
- Optional validation metrics
- Session ID and epoch number

### 2. **Activation Logic**
When you call `checkGovernance()`, the system:
1. Checks if enough sessions have passed (default: 2)
2. Applies probabilistic activation (default: 50% chance)
3. If both conditions met, proceeds to analysis

### 3. **Pattern Detection**
The GovernanceEngine analyzes recent metrics to detect:

- **Plateau**: No improvement for N sessions
- **Overfitting**: Train/val gap > threshold
- **Underfitting**: Both losses > threshold
- **Divergence**: Loss increasing
- **Oscillation**: High variance in loss

### 4. **Calibration Priority**
Actions are applied in priority order (only one per session):

1. **Divergence** → Reduce learning rate
2. **Oscillation** → Reduce learning rate
3. **Overfitting** → Increase dropout
4. **Plateau** → Reduce learning rate
5. **Underfitting** (early sessions) → Reduce dropout

### 5. **Safety Bounds**
All parameter changes respect configured min/max limits:
- Learning rate: 1e-6 to 1.0
- Dropout: 0.0 to 0.5

### 6. **Σ-SIG Compliance**
All decisions are logged in the governance ledger with:
- **Consistency**: Complete state analysis before action
- **Controlled Change**: Single parameter, small adjustments
- **Documented Decision**: Full audit trail with rationale

## Benefits

1. **Automatic Parameter Tuning**: No manual intervention needed
2. **Faster Convergence**: Detects and responds to training issues quickly
3. **Full Traceability**: Every decision logged with rationale
4. **User Awareness**: Visual alerts and calibration history
5. **Safety**: Bounded changes prevent catastrophic parameter values

## Advanced Usage

### Resetting Governance

To reset the governance state (clears all history):

```tsx
const { resetGovernance } = useProjects();

// Clear all governance history
resetGovernance();
```

### Acknowledging Alerts

```tsx
const { getActiveAlerts, acknowledgeAlert } = useProjects();

const alerts = getActiveAlerts();
alerts.forEach(alert => {
  console.log(alert.message);
  acknowledgeAlert(alert.id); // Mark as acknowledged
});
```

### Viewing Ledger

```tsx
const { getGovernanceLedger } = useProjects();

const ledger = getGovernanceLedger();
ledger.forEach(entry => {
  console.log(`[${entry.type}] ${entry.description}`);
  console.log(`  Session: ${entry.sessionId}`);
  console.log(`  Time: ${new Date(entry.timestamp).toLocaleString()}`);
});
```

## Testing the Integration

1. **Start with default parameters**
2. **Train for 2-3 sessions** (click Train button 2-3 times)
3. **Check GovernanceBoard** for alerts and calibrations
4. **Observe parameter changes** applied automatically
5. **Review ledger** to see decision history

## Troubleshooting

**Governance not activating?**
- Check that `enabled: true` in config
- Ensure you're calling `checkGovernance()` after training
- Verify `checkInterval` (default: 2 sessions)

**Too many calibrations?**
- Increase `checkInterval` (e.g., 3 or 4)
- Decrease `activationProbability` (e.g., 0.3)
- Increase `improvementThreshold`

**No alerts showing?**
- Check that metrics are being recorded properly
- Ensure training is actually running
- Verify alert severities aren't being filtered

## Next Steps

1. Add the GovernanceBoard to your App.tsx
2. Integrate metric recording in your training loop
3. Add governance checks after training sessions
4. Test with 2-3 training runs
5. Review alerts and calibration history
6. Adjust configuration as needed

For more details, see:
- `src/lib/GovernanceEngine.ts` - Core implementation
- `src/types/governance.ts` - Type definitions
- `src/components/GovernanceBoard.tsx` - UI component
