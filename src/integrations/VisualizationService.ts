/**
 * VisualizationService - Plotly Integration for Advanced Charts
 *
 * Provides chart generation and export for EmbeddingVisualizationPanel
 * All operations are local without actual Plotly API calls
 */

import type {
  PlotlyTrace,
  PlotlyLayout,
  PlotlyConfig,
  PlotlyChart,
  PlotlyMarker,
  PlotlyAxis
} from './types';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_PLOTLY_CONFIG: PlotlyConfig = {
  enabled: true,
  responsive: true,
  displayModeBar: 'hover',
  displaylogo: false,
  modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'neuro-lingua-chart',
    height: 800,
    width: 1200,
    scale: 2
  }
};

const DEFAULT_LAYOUT: PlotlyLayout = {
  showlegend: true,
  legend: { x: 1, y: 1, orientation: 'v' },
  margin: { l: 50, r: 50, t: 50, b: 50 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'monospace', size: 12, color: '#e0e0e0' },
  hovermode: 'closest'
};

// ============================================================================
// Color Palettes
// ============================================================================

export const COLOR_PALETTES = {
  default: ['#6366f1', '#22d3ee', '#f472b6', '#a78bfa', '#34d399', '#fbbf24', '#fb7185'],
  viridis: ['#440154', '#482878', '#3e4a89', '#31688e', '#26838f', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
  plasma: ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
  coolwarm: ['#3b4cc0', '#688aef', '#99b7f5', '#c8d4eb', '#edd1c2', '#f7a889', '#e26952', '#b40426'],
  spectral: ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
};

// ============================================================================
// VisualizationService Class
// ============================================================================

export class VisualizationService {
  private config: PlotlyConfig;
  private charts: Map<string, PlotlyChart> = new Map();
  private listeners: Set<(charts: Map<string, PlotlyChart>) => void> = new Set();

  constructor(config?: Partial<PlotlyConfig>) {
    this.config = { ...DEFAULT_PLOTLY_CONFIG, ...config };
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  updateConfig(config: Partial<PlotlyConfig>): void {
    this.config = { ...this.config, ...config };
  }

  getConfig(): PlotlyConfig {
    return { ...this.config };
  }

  // ==========================================================================
  // Chart Creation - Embeddings
  // ==========================================================================

  /**
   * Create 2D scatter plot for embeddings (t-SNE/UMAP)
   */
  createEmbeddingScatter2D(
    points: { x: number; y: number; label?: string; group?: string }[],
    options: {
      title?: string;
      colorByGroup?: boolean;
      showLabels?: boolean;
      markerSize?: number;
    } = {}
  ): PlotlyChart {
    const { title = 'Embedding Visualization', colorByGroup = true, showLabels = false, markerSize = 8 } = options;

    // Group points if needed
    const groups = new Map<string, typeof points>();
    for (const point of points) {
      const group = point.group || 'default';
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group)!.push(point);
    }

    const traces: PlotlyTrace[] = [];
    const colors = COLOR_PALETTES.default;
    let colorIndex = 0;

    for (const [group, groupPoints] of groups) {
      const trace: PlotlyTrace = {
        type: 'scatter',
        mode: showLabels ? 'markers+text' : 'markers',
        name: group,
        x: groupPoints.map(p => p.x),
        y: groupPoints.map(p => p.y),
        text: groupPoints.map(p => p.label || ''),
        marker: {
          color: colorByGroup ? colors[colorIndex % colors.length] : undefined,
          size: markerSize,
          opacity: 0.8,
          line: { color: '#ffffff', width: 1 }
        }
      };

      if (!colorByGroup) {
        trace.marker!.color = groupPoints.map((_, i) => i);
        trace.marker!.colorscale = 'Viridis';
        trace.marker!.showscale = true;
      }

      traces.push(trace);
      colorIndex++;
    }

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: 'Dimension 1', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'Dimension 2', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces,
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  /**
   * Create 3D scatter plot for embeddings
   */
  createEmbeddingScatter3D(
    points: { x: number; y: number; z: number; label?: string; group?: string }[],
    options: {
      title?: string;
      colorByGroup?: boolean;
      markerSize?: number;
    } = {}
  ): PlotlyChart {
    const { title = '3D Embedding Visualization', colorByGroup = true, markerSize = 5 } = options;

    const groups = new Map<string, typeof points>();
    for (const point of points) {
      const group = point.group || 'default';
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group)!.push(point);
    }

    const traces: PlotlyTrace[] = [];
    const colors = COLOR_PALETTES.default;
    let colorIndex = 0;

    for (const [group, groupPoints] of groups) {
      const trace: PlotlyTrace = {
        type: 'scatter3d',
        mode: 'markers',
        name: group,
        x: groupPoints.map(p => p.x),
        y: groupPoints.map(p => p.y),
        z: groupPoints.map(p => p.z),
        text: groupPoints.map(p => p.label || ''),
        marker: {
          color: colorByGroup ? colors[colorIndex % colors.length] : groupPoints.map((_, i) => i),
          size: markerSize,
          opacity: 0.8,
          colorscale: colorByGroup ? undefined : 'Viridis'
        }
      };
      traces.push(trace);
      colorIndex++;
    }

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      scene: {
        xaxis: { title: 'Dim 1', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'Dim 2', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
        zaxis: { title: 'Dim 3', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }
      }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces,
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  // ==========================================================================
  // Chart Creation - Training Metrics
  // ==========================================================================

  /**
   * Create line chart for training metrics over epochs
   */
  createTrainingMetricsChart(
    data: {
      epochs: number[];
      loss?: number[];
      accuracy?: number[];
      perplexity?: number[];
      valLoss?: number[];
      valAccuracy?: number[];
    },
    options: { title?: string; showValidation?: boolean } = {}
  ): PlotlyChart {
    const { title = 'Training Metrics', showValidation = true } = options;

    const traces: PlotlyTrace[] = [];
    const colors = COLOR_PALETTES.default;

    if (data.loss) {
      traces.push({
        type: 'line',
        mode: 'lines+markers',
        name: 'Train Loss',
        x: data.epochs,
        y: data.loss,
        line: { color: colors[0], width: 2 },
        marker: { size: 6 }
      });
    }

    if (data.accuracy) {
      traces.push({
        type: 'line',
        mode: 'lines+markers',
        name: 'Train Accuracy',
        x: data.epochs,
        y: data.accuracy,
        line: { color: colors[1], width: 2 },
        marker: { size: 6 }
      });
    }

    if (showValidation && data.valLoss) {
      traces.push({
        type: 'line',
        mode: 'lines+markers',
        name: 'Val Loss',
        x: data.epochs,
        y: data.valLoss,
        line: { color: colors[0], width: 2, dash: 'dash' },
        marker: { size: 6, symbol: 'diamond' }
      });
    }

    if (showValidation && data.valAccuracy) {
      traces.push({
        type: 'line',
        mode: 'lines+markers',
        name: 'Val Accuracy',
        x: data.epochs,
        y: data.valAccuracy,
        line: { color: colors[1], width: 2, dash: 'dash' },
        marker: { size: 6, symbol: 'diamond' }
      });
    }

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: 'Epoch', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'Value', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces,
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  /**
   * Create heatmap for attention weights or confusion matrix
   */
  createHeatmap(
    data: number[][],
    options: {
      title?: string;
      xLabels?: string[];
      yLabels?: string[];
      colorscale?: string;
    } = {}
  ): PlotlyChart {
    const {
      title = 'Heatmap',
      xLabels,
      yLabels,
      colorscale = 'Viridis'
    } = options;

    const trace: PlotlyTrace = {
      type: 'heatmap',
      z: data,
      x: xLabels,
      y: yLabels,
      colorscale,
      showscale: true
    };

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: 'X', showgrid: false },
      yaxis: { title: 'Y', showgrid: false }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces: [trace],
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  // ==========================================================================
  // Chart Creation - Brain & Governance
  // ==========================================================================

  /**
   * Create gauge-style chart for brain health metrics
   */
  createBrainHealthChart(
    creativity: number,
    stability: number,
    options: { title?: string } = {}
  ): PlotlyChart {
    const { title = 'Brain Health' } = options;

    const traces: PlotlyTrace[] = [
      {
        type: 'bar',
        name: 'Creativity',
        x: ['Creativity'],
        y: [creativity],
        marker: { color: '#6366f1' }
      },
      {
        type: 'bar',
        name: 'Stability',
        x: ['Stability'],
        y: [stability],
        marker: { color: '#22d3ee' }
      }
    ];

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: '' },
      yaxis: { title: 'Score', range: [0, 100], showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces,
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  /**
   * Create timeline chart for governance events
   */
  createGovernanceTimeline(
    events: { timestamp: number; type: string; severity: string; message: string }[],
    options: { title?: string } = {}
  ): PlotlyChart {
    const { title = 'Governance Events Timeline' } = options;

    const severityColors: Record<string, string> = {
      info: '#22d3ee',
      warning: '#fbbf24',
      critical: '#ef4444'
    };

    const severityY: Record<string, number> = {
      info: 1,
      warning: 2,
      critical: 3
    };

    const traces: PlotlyTrace[] = [{
      type: 'scatter',
      mode: 'markers+text',
      x: events.map(e => new Date(e.timestamp)),
      y: events.map(e => severityY[e.severity] || 1),
      text: events.map(e => e.type),
      marker: {
        color: events.map(e => severityColors[e.severity] || '#6366f1'),
        size: 12,
        symbol: 'circle'
      }
    }];

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: 'Time', type: 'date', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: {
        title: 'Severity',
        ticktext: ['Info', 'Warning', 'Critical'],
        tickvals: [1, 2, 3],
        range: [0.5, 3.5],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)'
      }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces,
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  // ==========================================================================
  // Chart Creation - Distribution
  // ==========================================================================

  /**
   * Create histogram for distribution visualization
   */
  createHistogram(
    values: number[],
    options: {
      title?: string;
      bins?: number;
      xLabel?: string;
      color?: string;
    } = {}
  ): PlotlyChart {
    const { title = 'Distribution', bins = 30, xLabel = 'Value', color = '#6366f1' } = options;

    const trace: PlotlyTrace = {
      type: 'histogram',
      x: values,
      marker: { color, line: { color: '#ffffff', width: 1 } }
    };

    const layout: PlotlyLayout = {
      ...DEFAULT_LAYOUT,
      title: { text: title, font: { size: 16 } },
      xaxis: { title: xLabel, showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'Count', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }
    };

    const chart: PlotlyChart = {
      id: this.generateChartId(),
      traces: [trace],
      layout,
      config: this.config
    };

    this.charts.set(chart.id, chart);
    this.notifyListeners();
    return chart;
  }

  // ==========================================================================
  // Chart Management
  // ==========================================================================

  getChart(chartId: string): PlotlyChart | undefined {
    return this.charts.get(chartId);
  }

  getAllCharts(): PlotlyChart[] {
    return Array.from(this.charts.values());
  }

  updateChart(chartId: string, updates: Partial<PlotlyChart>): boolean {
    const chart = this.charts.get(chartId);
    if (!chart) return false;

    if (updates.traces) chart.traces = updates.traces;
    if (updates.layout) chart.layout = { ...chart.layout, ...updates.layout };
    if (updates.config) chart.config = { ...chart.config, ...updates.config };

    this.charts.set(chartId, chart);
    this.notifyListeners();
    return true;
  }

  deleteChart(chartId: string): boolean {
    const deleted = this.charts.delete(chartId);
    if (deleted) this.notifyListeners();
    return deleted;
  }

  clearCharts(): void {
    this.charts.clear();
    this.notifyListeners();
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  /**
   * Export chart data as JSON
   */
  exportChartJSON(chartId: string): string | null {
    const chart = this.charts.get(chartId);
    if (!chart) return null;

    return JSON.stringify(chart, null, 2);
  }

  /**
   * Export chart for Plotly.js rendering
   */
  exportForPlotly(chartId: string): { data: PlotlyTrace[]; layout: PlotlyLayout; config: Partial<PlotlyConfig> } | null {
    const chart = this.charts.get(chartId);
    if (!chart) return null;

    return {
      data: chart.traces,
      layout: chart.layout,
      config: chart.config || this.config
    };
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  subscribe(listener: (charts: Map<string, PlotlyChart>) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private generateChartId(): string {
    return `chart_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.charts);
    }
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  destroy(): void {
    this.charts.clear();
    this.listeners.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let visualizationInstance: VisualizationService | null = null;

export function getVisualizationService(): VisualizationService {
  if (!visualizationInstance) {
    visualizationInstance = new VisualizationService();
  }
  return visualizationInstance;
}

export function resetVisualizationService(): void {
  if (visualizationInstance) {
    visualizationInstance.destroy();
    visualizationInstance = null;
  }
}
