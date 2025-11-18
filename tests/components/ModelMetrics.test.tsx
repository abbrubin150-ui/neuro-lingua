import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ModelMetrics } from '../../src/components/ModelMetrics';
import type { ModelMetaStore } from '../../src/types/modelMeta';

vi.mock('../../src/lib/utils', async () => {
  const actual = await vi.importActual<typeof import('../../src/lib/utils')>('../../src/lib/utils');
  return {
    ...actual,
    createTrainingHistoryCsv: vi.fn(() => new Blob(['epoch,loss,accuracy'], { type: 'text/csv' })),
    downloadBlob: vi.fn()
  };
});

import { createTrainingHistoryCsv, downloadBlob } from '../../src/lib/utils';

describe('ModelMetrics', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders statistics, active architecture, and comparison badges', () => {
    const props = createProps({
      trainingHistory: [
        { loss: 0.8, accuracy: 0.5, timestamp: Date.now() },
        { loss: 0.6, accuracy: 0.6, timestamp: Date.now() + 1_000 }
      ],
      modelComparisons: {
        feedforward: {
          architecture: 'feedforward',
          timestamp: Date.now(),
          vocab: 128,
          perplexity: 1.4
        },
        transformer: {
          architecture: 'transformer',
          timestamp: Date.now(),
          vocab: 128,
          perplexity: 1.2
        }
      }
    });

    render(<ModelMetrics {...props} />);

    expect(screen.getByText(/Loss \(Avg\)/)).toBeInTheDocument();
    expect(screen.getByText(/Active architecture/)).toHaveTextContent('Standard (ProNeural)');
    expect(screen.getByText(/BEST PPL/)).toBeInTheDocument();
  });

  it('informs the user when exporting without history', () => {
    const props = createProps({ trainingHistory: [] });
    render(<ModelMetrics {...props} />);

    fireEvent.click(screen.getByRole('button', { name: /Export history CSV/i }));

    expect(props.onMessage).toHaveBeenCalledWith(
      'ℹ️ Train the model to generate history before exporting CSV.'
    );
    expect(vi.mocked(createTrainingHistoryCsv)).not.toHaveBeenCalled();
    expect(vi.mocked(downloadBlob)).not.toHaveBeenCalled();
  });

  it('downloads a CSV when history exists', () => {
    const props = createProps({
      trainingHistory: [{ loss: 0.2, accuracy: 0.9, timestamp: Date.now() }]
    });
    render(<ModelMetrics {...props} />);

    fireEvent.click(screen.getByRole('button', { name: /Export history CSV/i }));

    expect(vi.mocked(createTrainingHistoryCsv)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(downloadBlob)).toHaveBeenCalledTimes(1);
  });
});

function createProps(overrides: Partial<Parameters<typeof ModelMetrics>[0]>) {
  const stats = { loss: 0.42, acc: 0.91, ppl: 1.2, lossEMA: 0.4, tokensPerSec: 120 };
  const info = { V: 256, P: 123_456 };
  const modelComparisons: ModelMetaStore = overrides.modelComparisons ?? {
    feedforward: {
      architecture: 'feedforward',
      timestamp: Date.now(),
      vocab: 256,
      perplexity: 1.5
    }
  };
  const onMessage = overrides.onMessage ?? vi.fn();

  return {
    stats,
    info,
    activeArchitecture: 'feedforward' as const,
    activeModelMeta: { architecture: 'feedforward', timestamp: Date.now(), vocab: 256 },
    modelComparisons,
    trainingHistory: overrides.trainingHistory ?? [],
    gpuMetrics: {
      available: true,
      enabled: true,
      totalOperations: 1,
      totalTimeMs: 1,
      averageTimeMs: 1,
      utilizationPercent: 50
    },
    edgeLearningDiagnostics: {
      fisherInformation: 0.2,
      entropy: 0.5,
      estimatorCovariance: 0.3,
      cramerRaoBound: 0.01,
      efficiency: 0.8,
      variance: 0.2,
      timestamp: Date.now(),
      status: 'success'
    },
    onMessage
  } satisfies Parameters<typeof ModelMetrics>[0];
}
