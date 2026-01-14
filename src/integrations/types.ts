/**
 * Integration Types for Neuro-Lingua
 * External services and real-time communication
 */

import type { BrainStats } from '../lib/BrainEngine';
import type { BoardAlert } from '../types/governance';

// ============================================================================
// Monitoring Integration Types (Prometheus/Datadog)
// ============================================================================

export interface MetricPoint {
  name: string;
  value: number;
  labels: Record<string, string>;
  timestamp: number;
  type: 'gauge' | 'counter' | 'histogram';
}

export interface PrometheusConfig {
  enabled: boolean;
  jobName: string;
  instanceId: string;
  pushInterval: number; // ms
  prefix: string;
}

export interface DatadogConfig {
  enabled: boolean;
  site: 'datadoghq.com' | 'datadoghq.eu';
  tags: string[];
  prefix: string;
}

export interface MonitoringState {
  metrics: MetricPoint[];
  lastPush: number;
  pushCount: number;
  errors: string[];
}

// ============================================================================
// Storage Integration Types (HuggingFace/ModelZoo)
// ============================================================================

export interface ModelMetadata {
  id: string;
  name: string;
  description: string;
  author: string;
  tags: string[];
  createdAt: number;
  updatedAt: number;
  downloads: number;
  likes: number;
  version: string;
  framework: string;
  license: string;
}

export interface ModelUploadOptions {
  name: string;
  description: string;
  tags: string[];
  isPublic: boolean;
  readme?: string;
}

export interface ModelVersion {
  version: string;
  createdAt: number;
  size: number;
  commitHash: string;
  message?: string;
}

export interface HuggingFaceConfig {
  enabled: boolean;
  organization?: string;
  defaultPrivate: boolean;
  autoSave: boolean;
}

export interface ModelZooConfig {
  enabled: boolean;
  storageKey: string;
  maxModels: number;
}

export interface StorageState {
  models: ModelMetadata[];
  uploading: string[];
  downloading: string[];
  lastSync: number;
}

// ============================================================================
// Visualization Integration Types (Plotly)
// ============================================================================

export interface PlotlyTrace {
  x?: number[];
  y?: number[];
  z?: number[];
  type: 'scatter' | 'scatter3d' | 'line' | 'bar' | 'heatmap' | 'surface' | 'histogram';
  mode?: 'lines' | 'markers' | 'lines+markers' | 'text';
  name?: string;
  text?: string[];
  marker?: PlotlyMarker;
  line?: PlotlyLine;
  colorscale?: string;
  showscale?: boolean;
}

export interface PlotlyMarker {
  color?: string | number[];
  size?: number | number[];
  symbol?: string;
  colorscale?: string;
  showscale?: boolean;
  opacity?: number;
  line?: { color: string; width: number };
}

export interface PlotlyLine {
  color?: string;
  width?: number;
  dash?: 'solid' | 'dot' | 'dash' | 'longdash' | 'dashdot';
}

export interface PlotlyLayout {
  title?: string | { text: string; font?: { size: number } };
  xaxis?: PlotlyAxis;
  yaxis?: PlotlyAxis;
  zaxis?: PlotlyAxis;
  showlegend?: boolean;
  legend?: { x: number; y: number; orientation?: 'h' | 'v' };
  width?: number;
  height?: number;
  margin?: { l: number; r: number; t: number; b: number };
  paper_bgcolor?: string;
  plot_bgcolor?: string;
  font?: { family: string; size: number; color: string };
  scene?: { xaxis: PlotlyAxis; yaxis: PlotlyAxis; zaxis: PlotlyAxis };
  hovermode?: 'closest' | 'x' | 'y' | false;
  annotations?: PlotlyAnnotation[];
}

export interface PlotlyAxis {
  title?: string | { text: string };
  range?: [number, number];
  type?: 'linear' | 'log' | 'date' | 'category';
  showgrid?: boolean;
  gridcolor?: string;
  zeroline?: boolean;
  showticklabels?: boolean;
  tickformat?: string;
}

export interface PlotlyAnnotation {
  x: number;
  y: number;
  text: string;
  showarrow?: boolean;
  arrowhead?: number;
  font?: { size: number; color: string };
}

export interface PlotlyConfig {
  enabled: boolean;
  responsive: boolean;
  displayModeBar: boolean | 'hover';
  displaylogo: boolean;
  modeBarButtonsToRemove?: string[];
  toImageButtonOptions?: {
    format: 'png' | 'svg' | 'jpeg' | 'webp';
    filename: string;
    height: number;
    width: number;
    scale: number;
  };
}

export interface PlotlyChart {
  id: string;
  traces: PlotlyTrace[];
  layout: PlotlyLayout;
  config?: Partial<PlotlyConfig>;
}

// ============================================================================
// Webhook Integration Types (Slack/Discord)
// ============================================================================

export type WebhookEventType =
  | 'training_started'
  | 'training_completed'
  | 'training_failed'
  | 'training_progress'
  | 'alert_critical'
  | 'alert_warning'
  | 'alert_info'
  | 'brain_burnout'
  | 'brain_recovery'
  | 'brain_mood_change'
  | 'model_saved'
  | 'model_loaded'
  | 'governance_action'
  | 'calibration_applied';

export interface WebhookConfig {
  id: string;
  enabled: boolean;
  name: string;
  type: 'slack' | 'discord' | 'generic';
  events: WebhookEventType[];
  retryCount: number;
  retryDelay: number; // ms
}

export interface WebhookPayload {
  event: WebhookEventType;
  timestamp: number;
  projectId?: string;
  runId?: string;
  brainId?: string;
  data: Record<string, unknown>;
  message: string;
  severity?: 'info' | 'warning' | 'critical';
}

export interface WebhookDelivery {
  id: string;
  webhookId: string;
  event: WebhookEventType;
  status: 'pending' | 'sent' | 'failed' | 'retrying';
  payload: WebhookPayload;
  timestamp: number;
  attempts: number;
  lastAttempt?: number;
  error?: string;
}

export interface SlackMessage {
  text?: string;
  blocks?: SlackBlock[];
  attachments?: SlackAttachment[];
  channel?: string;
  username?: string;
  icon_emoji?: string;
}

export interface SlackBlock {
  type: 'section' | 'divider' | 'header' | 'context' | 'actions';
  text?: { type: 'plain_text' | 'mrkdwn'; text: string };
  fields?: { type: 'plain_text' | 'mrkdwn'; text: string }[];
  accessory?: unknown;
  elements?: unknown[];
}

export interface SlackAttachment {
  color?: string;
  title?: string;
  text?: string;
  fields?: { title: string; value: string; short?: boolean }[];
  footer?: string;
  ts?: number;
}

export interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: { name: string; value: string; inline?: boolean }[];
  footer?: { text: string; icon_url?: string };
  timestamp?: string;
  thumbnail?: { url: string };
  author?: { name: string; icon_url?: string };
}

export interface DiscordMessage {
  content?: string;
  embeds?: DiscordEmbed[];
  username?: string;
  avatar_url?: string;
}

// ============================================================================
// WebSocket Real-time Types
// ============================================================================

export type WebSocketMessageType =
  | 'training_progress'
  | 'metrics_update'
  | 'alert_fired'
  | 'brain_state_changed'
  | 'project_updated'
  | 'run_updated'
  | 'user_presence'
  | 'cursor_position'
  | 'sync_request'
  | 'sync_response'
  | 'ping'
  | 'pong';

export interface WebSocketMessage<T = unknown> {
  id: string;
  type: WebSocketMessageType;
  channel: string;
  payload: T;
  timestamp: number;
  senderId: string;
  senderName?: string;
}

export interface WebSocketChannel {
  id: string;
  type: 'project' | 'run' | 'global' | 'brain';
  entityId?: string;
  subscribers: Set<string>;
  lastActivity: number;
  messageCount: number;
}

export interface WebSocketUser {
  id: string;
  name: string;
  color: string;
  joinedAt: number;
  lastSeen: number;
  currentChannel?: string;
  cursorPosition?: { x: number; y: number };
}

export interface WebSocketConfig {
  enabled: boolean;
  heartbeatInterval: number; // ms
  reconnectDelay: number; // ms
  maxReconnectAttempts: number;
  messageBufferSize: number;
}

export interface WebSocketState {
  connected: boolean;
  connectionId: string;
  userId: string;
  userName: string;
  lastHeartbeat: number;
  reconnectAttempts: number;
  subscribedChannels: string[];
  users: Map<string, WebSocketUser>;
  messageBuffer: WebSocketMessage[];
}

// ============================================================================
// Training Progress Types (for WebSocket)
// ============================================================================

export interface TrainingProgressPayload {
  runId: string;
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  loss: number;
  accuracy: number;
  perplexity: number;
  learningRate: number;
  throughput: number; // tokens/sec
  eta: number; // estimated time remaining (ms)
}

export interface MetricsUpdatePayload {
  runId: string;
  metrics: {
    loss: number;
    accuracy: number;
    perplexity: number;
    gradientNorm?: number;
    memoryUsage?: number;
  };
  timestamp: number;
}

export interface BrainStatePayload {
  brainId: string;
  mood: BrainStats['mood'];
  creativity: number;
  stability: number;
  healthScore: number;
  lastEvent?: string;
}

export interface UserPresencePayload {
  action: 'join' | 'leave' | 'update';
  user: WebSocketUser;
}

export interface CursorPositionPayload {
  userId: string;
  position: { x: number; y: number };
  viewport?: { width: number; height: number };
}

// ============================================================================
// Integration Hub State
// ============================================================================

export interface IntegrationHubState {
  monitoring: {
    prometheus: PrometheusConfig;
    datadog: DatadogConfig;
    state: MonitoringState;
  };
  storage: {
    huggingface: HuggingFaceConfig;
    modelzoo: ModelZooConfig;
    state: StorageState;
  };
  visualization: {
    plotly: PlotlyConfig;
    charts: Map<string, PlotlyChart>;
  };
  webhooks: {
    configs: WebhookConfig[];
    deliveries: WebhookDelivery[];
    pendingCount: number;
  };
  websocket: {
    config: WebSocketConfig;
    state: WebSocketState;
    channels: Map<string, WebSocketChannel>;
  };
}
