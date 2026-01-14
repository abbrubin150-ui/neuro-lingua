/**
 * WebhookService - Slack & Discord Integration (Mock)
 *
 * Provides webhook notifications for BrainPanel alerts
 * All operations are simulated locally without actual webhook calls
 */

import type {
  WebhookConfig,
  WebhookPayload,
  WebhookDelivery,
  WebhookEventType,
  SlackMessage,
  SlackBlock,
  SlackAttachment,
  DiscordMessage,
  DiscordEmbed
} from './types';
import type { BoardAlert } from '../types/governance';
import type { BrainStats } from '../lib/BrainEngine';
import { StorageManager } from '../lib/storage';

// ============================================================================
// Default Configurations
// ============================================================================

const STORAGE_KEY = 'neuro-lingua-webhooks-v1';

const DEFAULT_WEBHOOK_CONFIG: Omit<WebhookConfig, 'id' | 'name'> = {
  enabled: false,
  type: 'slack',
  events: ['alert_critical', 'training_completed', 'brain_burnout'],
  retryCount: 3,
  retryDelay: 5000
};

// ============================================================================
// Event Severity Mapping
// ============================================================================

const EVENT_SEVERITY: Record<WebhookEventType, 'info' | 'warning' | 'critical'> = {
  training_started: 'info',
  training_completed: 'info',
  training_failed: 'critical',
  training_progress: 'info',
  alert_critical: 'critical',
  alert_warning: 'warning',
  alert_info: 'info',
  brain_burnout: 'critical',
  brain_recovery: 'info',
  brain_mood_change: 'info',
  model_saved: 'info',
  model_loaded: 'info',
  governance_action: 'warning',
  calibration_applied: 'info'
};

const SEVERITY_COLORS = {
  info: '#22d3ee',
  warning: '#fbbf24',
  critical: '#ef4444'
};

const DISCORD_SEVERITY_COLORS = {
  info: 0x22d3ee,
  warning: 0xfbbf24,
  critical: 0xef4444
};

// ============================================================================
// WebhookService Class
// ============================================================================

export class WebhookService {
  private configs: Map<string, WebhookConfig> = new Map();
  private deliveries: WebhookDelivery[] = [];
  private listeners: Set<(deliveries: WebhookDelivery[]) => void> = new Set();
  private retryTimeouts: Map<string, ReturnType<typeof setTimeout>> = new Map();

  constructor() {
    this.loadState();
  }

  // ==========================================================================
  // Webhook Configuration
  // ==========================================================================

  /**
   * Add a new webhook configuration
   */
  addWebhook(name: string, config: Partial<WebhookConfig>): WebhookConfig {
    const id = `webhook_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const webhook: WebhookConfig = {
      ...DEFAULT_WEBHOOK_CONFIG,
      ...config,
      id,
      name
    };

    this.configs.set(id, webhook);
    this.saveState();
    return webhook;
  }

  /**
   * Update webhook configuration
   */
  updateWebhook(id: string, updates: Partial<WebhookConfig>): boolean {
    const webhook = this.configs.get(id);
    if (!webhook) return false;

    Object.assign(webhook, updates);
    this.configs.set(id, webhook);
    this.saveState();
    return true;
  }

  /**
   * Delete webhook configuration
   */
  deleteWebhook(id: string): boolean {
    const deleted = this.configs.delete(id);
    if (deleted) this.saveState();
    return deleted;
  }

  /**
   * Get webhook by ID
   */
  getWebhook(id: string): WebhookConfig | undefined {
    return this.configs.get(id);
  }

  /**
   * Get all webhooks
   */
  getAllWebhooks(): WebhookConfig[] {
    return Array.from(this.configs.values());
  }

  /**
   * Toggle webhook enabled state
   */
  toggleWebhook(id: string): boolean {
    const webhook = this.configs.get(id);
    if (!webhook) return false;

    webhook.enabled = !webhook.enabled;
    this.saveState();
    return true;
  }

  // ==========================================================================
  // Event Dispatch
  // ==========================================================================

  /**
   * Dispatch an event to all configured webhooks
   */
  async dispatch(
    event: WebhookEventType,
    data: Record<string, unknown>,
    options?: { projectId?: string; runId?: string; brainId?: string }
  ): Promise<WebhookDelivery[]> {
    const payload: WebhookPayload = {
      event,
      timestamp: Date.now(),
      projectId: options?.projectId,
      runId: options?.runId,
      brainId: options?.brainId,
      data,
      message: this.generateMessage(event, data),
      severity: EVENT_SEVERITY[event]
    };

    const deliveries: WebhookDelivery[] = [];

    for (const webhook of this.configs.values()) {
      if (!webhook.enabled) continue;
      if (!webhook.events.includes(event)) continue;

      const delivery = await this.sendToWebhook(webhook, payload);
      deliveries.push(delivery);
    }

    return deliveries;
  }

  /**
   * Dispatch governance alert
   */
  async dispatchAlert(alert: BoardAlert, projectId?: string): Promise<WebhookDelivery[]> {
    const eventType: WebhookEventType =
      alert.severity === 'critical' ? 'alert_critical' :
      alert.severity === 'warning' ? 'alert_warning' : 'alert_info';

    return this.dispatch(eventType, {
      alertId: alert.id,
      type: alert.type,
      message: alert.message,
      metric: alert.metric,
      value: alert.value
    }, { projectId });
  }

  /**
   * Dispatch brain state change
   */
  async dispatchBrainState(brain: BrainStats, event: 'burnout' | 'recovery' | 'mood_change'): Promise<WebhookDelivery[]> {
    const eventType: WebhookEventType =
      event === 'burnout' ? 'brain_burnout' :
      event === 'recovery' ? 'brain_recovery' : 'brain_mood_change';

    return this.dispatch(eventType, {
      brainId: brain.id,
      mood: brain.mood,
      creativity: brain.creativity,
      stability: brain.stability,
      label: brain.label
    }, { brainId: brain.id });
  }

  /**
   * Dispatch training event
   */
  async dispatchTrainingEvent(
    event: 'started' | 'completed' | 'failed' | 'progress',
    runId: string,
    data: Record<string, unknown>
  ): Promise<WebhookDelivery[]> {
    const eventType: WebhookEventType = `training_${event}` as WebhookEventType;
    return this.dispatch(eventType, data, { runId });
  }

  // ==========================================================================
  // Webhook Delivery
  // ==========================================================================

  private async sendToWebhook(webhook: WebhookConfig, payload: WebhookPayload): Promise<WebhookDelivery> {
    const delivery: WebhookDelivery = {
      id: `delivery_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      webhookId: webhook.id,
      event: payload.event,
      status: 'pending',
      payload,
      timestamp: Date.now(),
      attempts: 0
    };

    this.deliveries.push(delivery);
    this.notifyListeners();

    // Simulate sending
    await this.attemptDelivery(webhook, delivery);

    return delivery;
  }

  private async attemptDelivery(webhook: WebhookConfig, delivery: WebhookDelivery): Promise<void> {
    delivery.attempts++;
    delivery.lastAttempt = Date.now();
    delivery.status = 'pending';
    this.notifyListeners();

    // Simulate network delay
    await this.simulateDelay(200, 800);

    // Simulate success/failure (90% success rate)
    const success = Math.random() > 0.1;

    if (success) {
      delivery.status = 'sent';
      console.log(`[WebhookService] Delivered ${delivery.event} to ${webhook.name} (mock)`);
    } else {
      if (delivery.attempts < webhook.retryCount) {
        delivery.status = 'retrying';
        delivery.error = 'Connection timeout (simulated)';

        // Schedule retry
        const timeout = setTimeout(() => {
          this.attemptDelivery(webhook, delivery);
          this.retryTimeouts.delete(delivery.id);
        }, webhook.retryDelay);

        this.retryTimeouts.set(delivery.id, timeout);
      } else {
        delivery.status = 'failed';
        delivery.error = `Failed after ${webhook.retryCount} attempts (simulated)`;
      }
    }

    this.saveState();
    this.notifyListeners();
  }

  // ==========================================================================
  // Message Formatting
  // ==========================================================================

  private generateMessage(event: WebhookEventType, data: Record<string, unknown>): string {
    const messages: Record<WebhookEventType, (data: Record<string, unknown>) => string> = {
      training_started: (d) => `Training started for run ${d.runId || 'unknown'}`,
      training_completed: (d) => `Training completed! Final loss: ${d.loss || 'N/A'}, Accuracy: ${d.accuracy || 'N/A'}`,
      training_failed: (d) => `Training failed: ${d.error || 'Unknown error'}`,
      training_progress: (d) => `Training progress: Epoch ${d.epoch}/${d.totalEpochs}, Loss: ${d.loss}`,
      alert_critical: (d) => `CRITICAL ALERT: ${d.message || d.type}`,
      alert_warning: (d) => `Warning: ${d.message || d.type}`,
      alert_info: (d) => `Info: ${d.message || d.type}`,
      brain_burnout: (d) => `Brain "${d.label || d.brainId}" is burnt out! Creativity: ${d.creativity}, Stability: ${d.stability}`,
      brain_recovery: (d) => `Brain "${d.label || d.brainId}" has recovered. Mood: ${d.mood}`,
      brain_mood_change: (d) => `Brain mood changed to ${d.mood}`,
      model_saved: (d) => `Model saved: ${d.name || d.modelId}`,
      model_loaded: (d) => `Model loaded: ${d.name || d.modelId}`,
      governance_action: (d) => `Governance action: ${d.action || d.type}`,
      calibration_applied: (d) => `Calibration applied: ${d.parameter} changed from ${d.previousValue} to ${d.newValue}`
    };

    return messages[event]?.(data) || `Event: ${event}`;
  }

  /**
   * Format payload for Slack
   */
  formatSlackMessage(payload: WebhookPayload): SlackMessage {
    const severity = payload.severity || 'info';
    const color = SEVERITY_COLORS[severity];

    const blocks: SlackBlock[] = [
      {
        type: 'header',
        text: { type: 'plain_text', text: `Neuro-Lingua: ${payload.event.replace(/_/g, ' ').toUpperCase()}` }
      },
      {
        type: 'section',
        text: { type: 'mrkdwn', text: payload.message }
      }
    ];

    // Add data fields
    const fields = Object.entries(payload.data)
      .filter(([_, v]) => v !== undefined && v !== null)
      .slice(0, 10)
      .map(([k, v]) => ({
        type: 'mrkdwn' as const,
        text: `*${k}:* ${String(v)}`
      }));

    if (fields.length > 0) {
      blocks.push({
        type: 'section',
        fields
      });
    }

    blocks.push({
      type: 'context',
      elements: [{ type: 'mrkdwn', text: `_${new Date(payload.timestamp).toISOString()}_` }]
    });

    const attachments: SlackAttachment[] = [{
      color,
      footer: 'Neuro-Lingua Integration',
      ts: Math.floor(payload.timestamp / 1000)
    }];

    return { blocks, attachments };
  }

  /**
   * Format payload for Discord
   */
  formatDiscordMessage(payload: WebhookPayload): DiscordMessage {
    const severity = payload.severity || 'info';
    const color = DISCORD_SEVERITY_COLORS[severity];

    const fields = Object.entries(payload.data)
      .filter(([_, v]) => v !== undefined && v !== null)
      .slice(0, 25)
      .map(([k, v]) => ({
        name: k,
        value: String(v).slice(0, 1024),
        inline: String(v).length < 50
      }));

    const embed: DiscordEmbed = {
      title: `Neuro-Lingua: ${payload.event.replace(/_/g, ' ')}`,
      description: payload.message,
      color,
      fields,
      footer: { text: 'Neuro-Lingua Integration' },
      timestamp: new Date(payload.timestamp).toISOString()
    };

    return {
      embeds: [embed],
      username: 'Neuro-Lingua'
    };
  }

  // ==========================================================================
  // Delivery Management
  // ==========================================================================

  /**
   * Get all deliveries
   */
  getDeliveries(filter?: {
    webhookId?: string;
    event?: WebhookEventType;
    status?: WebhookDelivery['status'];
    since?: number;
  }): WebhookDelivery[] {
    let deliveries = [...this.deliveries];

    if (filter?.webhookId) {
      deliveries = deliveries.filter(d => d.webhookId === filter.webhookId);
    }
    if (filter?.event) {
      deliveries = deliveries.filter(d => d.event === filter.event);
    }
    if (filter?.status) {
      deliveries = deliveries.filter(d => d.status === filter.status);
    }
    if (filter?.since) {
      deliveries = deliveries.filter(d => d.timestamp >= filter.since!);
    }

    return deliveries.sort((a, b) => b.timestamp - a.timestamp);
  }

  /**
   * Get delivery statistics
   */
  getStats(): {
    total: number;
    pending: number;
    sent: number;
    failed: number;
    retrying: number;
  } {
    return {
      total: this.deliveries.length,
      pending: this.deliveries.filter(d => d.status === 'pending').length,
      sent: this.deliveries.filter(d => d.status === 'sent').length,
      failed: this.deliveries.filter(d => d.status === 'failed').length,
      retrying: this.deliveries.filter(d => d.status === 'retrying').length
    };
  }

  /**
   * Retry a failed delivery
   */
  async retryDelivery(deliveryId: string): Promise<boolean> {
    const delivery = this.deliveries.find(d => d.id === deliveryId);
    if (!delivery || delivery.status !== 'failed') return false;

    const webhook = this.configs.get(delivery.webhookId);
    if (!webhook) return false;

    delivery.attempts = 0;
    await this.attemptDelivery(webhook, delivery);
    return true;
  }

  /**
   * Clear delivery history
   */
  clearDeliveries(filter?: { status?: WebhookDelivery['status']; before?: number }): number {
    const initialCount = this.deliveries.length;

    if (filter?.status) {
      this.deliveries = this.deliveries.filter(d => d.status !== filter.status);
    } else if (filter?.before) {
      this.deliveries = this.deliveries.filter(d => d.timestamp >= filter.before!);
    } else {
      this.deliveries = [];
    }

    const cleared = initialCount - this.deliveries.length;
    this.saveState();
    this.notifyListeners();
    return cleared;
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  subscribe(listener: (deliveries: WebhookDelivery[]) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // ==========================================================================
  // Persistence
  // ==========================================================================

  private loadState(): void {
    const saved = StorageManager.get<{
      configs: [string, WebhookConfig][];
      deliveries: WebhookDelivery[];
    }>(STORAGE_KEY, { configs: [], deliveries: [] });

    this.configs = new Map(saved.configs);
    this.deliveries = saved.deliveries.slice(-100); // Keep last 100 deliveries
  }

  private saveState(): void {
    StorageManager.set(STORAGE_KEY, {
      configs: Array.from(this.configs.entries()),
      deliveries: this.deliveries.slice(-100)
    });
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.deliveries);
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
    // Cancel pending retries
    for (const timeout of this.retryTimeouts.values()) {
      clearTimeout(timeout);
    }
    this.retryTimeouts.clear();
    this.listeners.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let webhookInstance: WebhookService | null = null;

export function getWebhookService(): WebhookService {
  if (!webhookInstance) {
    webhookInstance = new WebhookService();
  }
  return webhookInstance;
}

export function resetWebhookService(): void {
  if (webhookInstance) {
    webhookInstance.destroy();
    webhookInstance = null;
  }
}
