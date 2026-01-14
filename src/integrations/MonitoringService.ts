/**
 * MonitoringService - Prometheus & Datadog Integration (Mock)
 *
 * Provides metrics collection and export for BrainTelemetryPanel
 * All operations are simulated locally without actual API calls
 */

import type {
  MetricPoint,
  PrometheusConfig,
  DatadogConfig,
  MonitoringState
} from './types';
import type { BrainStats } from '../lib/BrainEngine';
import type { MetricSnapshot, BoardAlert } from '../types/governance';

// ============================================================================
// Default Configurations
// ============================================================================

const DEFAULT_PROMETHEUS_CONFIG: PrometheusConfig = {
  enabled: false,
  jobName: 'neuro-lingua',
  instanceId: `instance-${Date.now()}`,
  pushInterval: 15000,
  prefix: 'nl_'
};

const DEFAULT_DATADOG_CONFIG: DatadogConfig = {
  enabled: false,
  site: 'datadoghq.com',
  tags: ['env:development', 'app:neuro-lingua'],
  prefix: 'neuro_lingua.'
};

// ============================================================================
// MonitoringService Class
// ============================================================================

export class MonitoringService {
  private prometheusConfig: PrometheusConfig;
  private datadogConfig: DatadogConfig;
  private state: MonitoringState;
  private pushInterval: ReturnType<typeof setInterval> | null = null;
  private listeners: Set<(state: MonitoringState) => void> = new Set();

  constructor(
    prometheusConfig?: Partial<PrometheusConfig>,
    datadogConfig?: Partial<DatadogConfig>
  ) {
    this.prometheusConfig = { ...DEFAULT_PROMETHEUS_CONFIG, ...prometheusConfig };
    this.datadogConfig = { ...DEFAULT_DATADOG_CONFIG, ...datadogConfig };
    this.state = {
      metrics: [],
      lastPush: 0,
      pushCount: 0,
      errors: []
    };
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  updatePrometheusConfig(config: Partial<PrometheusConfig>): void {
    this.prometheusConfig = { ...this.prometheusConfig, ...config };
    if (config.enabled !== undefined) {
      if (config.enabled) {
        this.startPushInterval();
      } else {
        this.stopPushInterval();
      }
    }
  }

  updateDatadogConfig(config: Partial<DatadogConfig>): void {
    this.datadogConfig = { ...this.datadogConfig, ...config };
  }

  getPrometheusConfig(): PrometheusConfig {
    return { ...this.prometheusConfig };
  }

  getDatadogConfig(): DatadogConfig {
    return { ...this.datadogConfig };
  }

  // ==========================================================================
  // Metric Collection
  // ==========================================================================

  /**
   * Record a single metric point
   */
  recordMetric(
    name: string,
    value: number,
    labels: Record<string, string> = {},
    type: MetricPoint['type'] = 'gauge'
  ): void {
    const metric: MetricPoint = {
      name: this.formatMetricName(name),
      value,
      labels: {
        ...labels,
        job: this.prometheusConfig.jobName,
        instance: this.prometheusConfig.instanceId
      },
      timestamp: Date.now(),
      type
    };

    this.state.metrics.push(metric);

    // Keep only last 1000 metrics
    if (this.state.metrics.length > 1000) {
      this.state.metrics = this.state.metrics.slice(-1000);
    }

    this.notifyListeners();
  }

  /**
   * Record training metrics
   */
  recordTrainingMetrics(snapshot: MetricSnapshot): void {
    const labels = {
      session_id: snapshot.sessionId,
      epoch: String(snapshot.epoch)
    };

    this.recordMetric('training_loss', snapshot.trainLoss, labels, 'gauge');
    this.recordMetric('training_accuracy', snapshot.trainAccuracy, labels, 'gauge');
    this.recordMetric('training_perplexity', snapshot.perplexity, labels, 'gauge');

    if (snapshot.valLoss !== undefined) {
      this.recordMetric('validation_loss', snapshot.valLoss, labels, 'gauge');
    }
    if (snapshot.valAccuracy !== undefined) {
      this.recordMetric('validation_accuracy', snapshot.valAccuracy, labels, 'gauge');
    }

    this.recordMetric('training_epochs_total', snapshot.epoch, labels, 'counter');
  }

  /**
   * Record brain state metrics
   */
  recordBrainMetrics(brain: BrainStats): void {
    const labels = {
      brain_id: brain.id,
      mood: brain.mood
    };

    this.recordMetric('brain_creativity', brain.creativity, labels, 'gauge');
    this.recordMetric('brain_stability', brain.stability, labels, 'gauge');
    this.recordMetric('brain_train_steps_total', brain.totalTrainSteps, labels, 'counter');
    this.recordMetric('brain_tokens_seen_total', brain.totalTokensSeen, labels, 'counter');
    this.recordMetric('brain_last_loss', brain.lastLoss, labels, 'gauge');

    // Calculate health score
    const healthScore = (brain.creativity + brain.stability) / 2;
    this.recordMetric('brain_health_score', healthScore, labels, 'gauge');

    // Mood as numeric value
    const moodValues: Record<BrainStats['mood'], number> = {
      CALM: 0,
      FOCUSED: 1,
      AGITATED: 2,
      DREAMY: 3,
      BURNT_OUT: 4
    };
    this.recordMetric('brain_mood_value', moodValues[brain.mood], labels, 'gauge');
  }

  /**
   * Record governance alert as metric
   */
  recordAlertMetric(alert: BoardAlert): void {
    const labels = {
      alert_type: alert.type,
      severity: alert.severity,
      alert_id: alert.id
    };

    this.recordMetric('governance_alert', 1, labels, 'counter');

    if (alert.value !== undefined) {
      this.recordMetric(`alert_${alert.type}_value`, alert.value, labels, 'gauge');
    }
  }

  /**
   * Record custom histogram
   */
  recordHistogram(name: string, values: number[], labels: Record<string, string> = {}): void {
    if (values.length === 0) return;

    const sorted = [...values].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);
    const count = sorted.length;

    this.recordMetric(`${name}_sum`, sum, labels, 'histogram');
    this.recordMetric(`${name}_count`, count, labels, 'histogram');
    this.recordMetric(`${name}_min`, sorted[0], labels, 'histogram');
    this.recordMetric(`${name}_max`, sorted[count - 1], labels, 'histogram');
    this.recordMetric(`${name}_avg`, sum / count, labels, 'histogram');

    // Percentiles
    const p50 = sorted[Math.floor(count * 0.5)];
    const p90 = sorted[Math.floor(count * 0.9)];
    const p99 = sorted[Math.floor(count * 0.99)];

    this.recordMetric(`${name}_p50`, p50, { ...labels, quantile: '0.5' }, 'histogram');
    this.recordMetric(`${name}_p90`, p90, { ...labels, quantile: '0.9' }, 'histogram');
    this.recordMetric(`${name}_p99`, p99, { ...labels, quantile: '0.99' }, 'histogram');
  }

  // ==========================================================================
  // Export Formats
  // ==========================================================================

  /**
   * Export metrics in Prometheus text format
   */
  exportPrometheusFormat(): string {
    const lines: string[] = [];
    const grouped = this.groupMetricsByName();

    for (const [name, metrics] of grouped) {
      // Add TYPE and HELP comments
      const type = metrics[0]?.type || 'gauge';
      lines.push(`# HELP ${name} Auto-generated metric from Neuro-Lingua`);
      lines.push(`# TYPE ${name} ${type}`);

      // Add metric lines
      for (const metric of metrics) {
        const labelStr = Object.entries(metric.labels)
          .map(([k, v]) => `${k}="${v}"`)
          .join(',');
        lines.push(`${name}{${labelStr}} ${metric.value} ${metric.timestamp}`);
      }

      lines.push('');
    }

    return lines.join('\n');
  }

  /**
   * Export metrics in Datadog JSON format
   */
  exportDatadogFormat(): object {
    const series = this.state.metrics.map(metric => ({
      metric: this.datadogConfig.prefix + metric.name,
      type: metric.type === 'counter' ? 'count' : 'gauge',
      points: [[Math.floor(metric.timestamp / 1000), metric.value]],
      tags: [
        ...this.datadogConfig.tags,
        ...Object.entries(metric.labels).map(([k, v]) => `${k}:${v}`)
      ]
    }));

    return { series };
  }

  /**
   * Export metrics as JSON
   */
  exportJSON(): string {
    return JSON.stringify({
      timestamp: Date.now(),
      prometheus: this.prometheusConfig,
      datadog: this.datadogConfig,
      metrics: this.state.metrics,
      state: {
        lastPush: this.state.lastPush,
        pushCount: this.state.pushCount,
        errors: this.state.errors
      }
    }, null, 2);
  }

  // ==========================================================================
  // Mock Push Operations
  // ==========================================================================

  /**
   * Simulate pushing metrics to Prometheus PushGateway
   */
  async pushToPrometheus(): Promise<boolean> {
    if (!this.prometheusConfig.enabled) {
      return false;
    }

    // Simulate network delay
    await this.simulateDelay(100, 500);

    // Simulate occasional failures (5% chance)
    if (Math.random() < 0.05) {
      const error = `[MOCK] Prometheus push failed: Connection timeout`;
      this.state.errors.push(error);
      this.notifyListeners();
      return false;
    }

    this.state.lastPush = Date.now();
    this.state.pushCount++;
    this.notifyListeners();

    console.log(`[MonitoringService] Pushed ${this.state.metrics.length} metrics to Prometheus (mock)`);
    return true;
  }

  /**
   * Simulate pushing metrics to Datadog
   */
  async pushToDatadog(): Promise<boolean> {
    if (!this.datadogConfig.enabled) {
      return false;
    }

    // Simulate network delay
    await this.simulateDelay(150, 600);

    // Simulate occasional failures (3% chance)
    if (Math.random() < 0.03) {
      const error = `[MOCK] Datadog push failed: API rate limit exceeded`;
      this.state.errors.push(error);
      this.notifyListeners();
      return false;
    }

    this.state.lastPush = Date.now();
    this.state.pushCount++;
    this.notifyListeners();

    console.log(`[MonitoringService] Pushed ${this.state.metrics.length} metrics to Datadog (mock)`);
    return true;
  }

  /**
   * Push to all enabled services
   */
  async pushAll(): Promise<{ prometheus: boolean; datadog: boolean }> {
    const [prometheus, datadog] = await Promise.all([
      this.pushToPrometheus(),
      this.pushToDatadog()
    ]);

    return { prometheus, datadog };
  }

  // ==========================================================================
  // Automatic Push Interval
  // ==========================================================================

  startPushInterval(): void {
    if (this.pushInterval) {
      return;
    }

    this.pushInterval = setInterval(() => {
      this.pushAll();
    }, this.prometheusConfig.pushInterval);

    console.log(`[MonitoringService] Started auto-push every ${this.prometheusConfig.pushInterval}ms`);
  }

  stopPushInterval(): void {
    if (this.pushInterval) {
      clearInterval(this.pushInterval);
      this.pushInterval = null;
      console.log('[MonitoringService] Stopped auto-push');
    }
  }

  // ==========================================================================
  // State & Queries
  // ==========================================================================

  getState(): MonitoringState {
    return { ...this.state };
  }

  getMetrics(filter?: { name?: string; since?: number }): MetricPoint[] {
    let metrics = [...this.state.metrics];

    if (filter?.name) {
      metrics = metrics.filter(m => m.name.includes(filter.name!));
    }

    if (filter?.since) {
      metrics = metrics.filter(m => m.timestamp >= filter.since!);
    }

    return metrics;
  }

  getLatestMetric(name: string): MetricPoint | undefined {
    const formattedName = this.formatMetricName(name);
    return [...this.state.metrics]
      .reverse()
      .find(m => m.name === formattedName);
  }

  getMetricHistory(name: string, limit: number = 100): MetricPoint[] {
    const formattedName = this.formatMetricName(name);
    return this.state.metrics
      .filter(m => m.name === formattedName)
      .slice(-limit);
  }

  clearMetrics(): void {
    this.state.metrics = [];
    this.state.errors = [];
    this.notifyListeners();
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  subscribe(listener: (state: MonitoringState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private formatMetricName(name: string): string {
    // Add prefix and sanitize name
    const sanitized = name
      .toLowerCase()
      .replace(/[^a-z0-9_]/g, '_')
      .replace(/__+/g, '_');
    return this.prometheusConfig.prefix + sanitized;
  }

  private groupMetricsByName(): Map<string, MetricPoint[]> {
    const grouped = new Map<string, MetricPoint[]>();

    for (const metric of this.state.metrics) {
      const existing = grouped.get(metric.name) || [];
      existing.push(metric);
      grouped.set(metric.name, existing);
    }

    return grouped;
  }

  private notifyListeners(): void {
    const state = this.getState();
    for (const listener of this.listeners) {
      listener(state);
    }
  }

  private simulateDelay(min: number, max: number): Promise<void> {
    const delay = min + Math.random() * (max - min);
    return new Promise(resolve => setTimeout(resolve, delay));
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  destroy(): void {
    this.stopPushInterval();
    this.listeners.clear();
    this.state.metrics = [];
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let monitoringInstance: MonitoringService | null = null;

export function getMonitoringService(): MonitoringService {
  if (!monitoringInstance) {
    monitoringInstance = new MonitoringService();
  }
  return monitoringInstance;
}

export function resetMonitoringService(): void {
  if (monitoringInstance) {
    monitoringInstance.destroy();
    monitoringInstance = null;
  }
}
