/**
 * Edge Learning Diagnostics
 *
 * Utilities for computing On-the-Edge learning metrics based on
 * information-theoretic principles. These diagnostics assess whether
 * a trained model operates at the "edge of efficiency" - a regime where
 * statistical efficiency is near-optimal and generalization is favored.
 *
 * @see docs/on-the-edge-formalism.md for theoretical background
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface EdgeDiagnostics {
  fisherInformationRange: [number, number];
  entropyRange: [number, number];
  averageEfficiency: number;
  inFlatRegion: boolean;
  edgeBandPercentage: number;
}

export interface EdgeLearningOptions {
  thetaMin?: number;
  thetaMax?: number;
  nMin?: number;
  nMax?: number;
  nGrid?: number;
  mGrid?: number;
}

/**
 * Computes edge learning diagnostics by invoking the Python implementation.
 *
 * This function runs the OnTheEdgeLearning algorithm which assesses whether
 * a model operates at the theoretical "edge of efficiency" where:
 * - Fisher information is near-optimal
 * - Entropy is maximized (flat loss landscape)
 * - Statistical efficiency ≈ 1
 *
 * @param options Configuration for the edge learning analysis grid
 * @returns Edge learning diagnostic metrics
 */
export async function computeEdgeDiagnostics(
  options: EdgeLearningOptions = {}
): Promise<EdgeDiagnostics> {
  const {
    thetaMin = -3.0,
    thetaMax = 3.0,
    nMin = 10,
    nMax = 1000,
    nGrid = 50,
    mGrid = 30
  } = options;

  const pythonScript = `
import sys
sys.path.insert(0, '.')
from symmetry_coupling.on_the_edge_learning import OnTheEdgeLearning
import json

# Create and run the model
model = OnTheEdgeLearning(
    theta_min=${thetaMin},
    theta_max=${thetaMax},
    n_min=${nMin},
    n_max=${nMax},
    n_grid=${nGrid},
    m_grid=${mGrid}
)
model.run()

# Compute diagnostics
avg_efficiency = sum(sum(row) for row in model.E_p) / (model.n_grid * model.m_grid)
edge_count = sum(
    1 for i in range(model.n_grid)
    for j in range(model.m_grid)
    if 1.0 - model.delta <= model.E_max[i][j] <= 1.0 + model.delta
)
edge_percent = 100 * edge_count / (model.n_grid * model.m_grid)
is_flat = model.check_flat_region()

# Output JSON
result = {
    "fisherInformationRange": [min(model.I1), max(model.I1)],
    "entropyRange": [min(model.H), max(model.H)],
    "averageEfficiency": avg_efficiency,
    "inFlatRegion": is_flat,
    "edgeBandPercentage": edge_percent
}
print(json.dumps(result))
`;

  try {
    const { stdout, stderr } = await execAsync(`python3 -c '${pythonScript}'`, {
      cwd: process.cwd()
    });

    if (stderr) {
      console.warn('Edge diagnostics warning:', stderr);
    }

    const result = JSON.parse(stdout.trim());
    return {
      fisherInformationRange: result.fisherInformationRange as [number, number],
      entropyRange: result.entropyRange as [number, number],
      averageEfficiency: result.averageEfficiency,
      inFlatRegion: result.inFlatRegion,
      edgeBandPercentage: result.edgeBandPercentage
    };
  } catch (error) {
    console.error('Failed to compute edge diagnostics:', error);
    throw new Error(
      `Edge diagnostics computation failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Interprets edge learning diagnostics to provide actionable insights.
 *
 * @param diagnostics The computed edge diagnostics
 * @returns Human-readable interpretation
 */
export function interpretEdgeDiagnostics(diagnostics: EdgeDiagnostics): string {
  const { averageEfficiency, inFlatRegion, edgeBandPercentage } = diagnostics;

  const parts: string[] = [];

  // Efficiency interpretation
  if (Math.abs(averageEfficiency - 1.0) < 0.05) {
    parts.push('✓ Model operates near optimal statistical efficiency (≈1.0)');
  } else if (averageEfficiency < 0.9) {
    parts.push('⚠ Model is statistically inefficient (<<1.0) - may underfit');
  } else if (averageEfficiency > 1.1) {
    parts.push('⚠ Model exceeds efficiency bound (>>1.0) - may overfit');
  }

  // Flat region interpretation
  if (inFlatRegion) {
    parts.push('✓ Model is in flat region (edge-of-chaos) - good for generalization');
  } else {
    parts.push('○ Model is not in flat region - may benefit from regularization');
  }

  // Edge band interpretation
  if (edgeBandPercentage > 80) {
    parts.push(
      `✓ ${edgeBandPercentage.toFixed(0)}% of parameter space in edge band - well-calibrated`
    );
  } else if (edgeBandPercentage > 50) {
    parts.push(`○ ${edgeBandPercentage.toFixed(0)}% of parameter space in edge band - acceptable`);
  } else {
    parts.push(
      `⚠ ${edgeBandPercentage.toFixed(0)}% of parameter space in edge band - poorly calibrated`
    );
  }

  return parts.join('\n');
}

/**
 * Checks if edge learning diagnostics are available in the current environment.
 *
 * @returns true if Python3 and required modules are available
 */
export async function isEdgeDiagnosticsAvailable(): Promise<boolean> {
  try {
    await execAsync('python3 --version');
    return true;
  } catch {
    return false;
  }
}
