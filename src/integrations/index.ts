/**
 * Neuro-Lingua Integration Layer
 *
 * External integrations and real-time communication:
 * - Monitoring: Prometheus, Datadog → BrainTelemetryPanel
 * - Storage: HuggingFace Hub, ModelZoo → ModelSnapshot
 * - Visualization: Plotly API → EmbeddingVisualizationPanel
 * - Webhooks: Slack/Discord → BrainPanel alerts
 * - Real-time: WebSocket simulation → Cross-tab sync, collaboration
 */

// Types
export * from './types';

// Services
export {
  MonitoringService,
  getMonitoringService,
  resetMonitoringService
} from './MonitoringService';

export {
  StorageService,
  getStorageService,
  resetStorageService
} from './StorageService';

export {
  VisualizationService,
  getVisualizationService,
  resetVisualizationService,
  COLOR_PALETTES
} from './VisualizationService';

export {
  WebhookService,
  getWebhookService,
  resetWebhookService
} from './WebhookService';

export {
  WebSocketService,
  getWebSocketService,
  resetWebSocketService
} from './WebSocketService';

// React Context & Hooks
export {
  IntegrationProvider,
  useIntegrations,
  useMonitoring,
  useModelStorage,
  useVisualization,
  useWebhooks,
  useRealtime,
  useBrainTelemetry
} from './IntegrationContext';
