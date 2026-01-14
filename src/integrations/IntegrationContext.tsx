/**
 * IntegrationContext - React Context for External Integrations
 *
 * Provides unified access to:
 * - Monitoring (Prometheus, Datadog)
 * - Storage (HuggingFace, ModelZoo)
 * - Visualization (Plotly)
 * - Webhooks (Slack, Discord)
 * - Real-time (WebSocket)
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef
} from 'react';

import {
  MonitoringService,
  getMonitoringService
} from './MonitoringService';
import {
  StorageService,
  getStorageService
} from './StorageService';
import {
  VisualizationService,
  getVisualizationService
} from './VisualizationService';
import {
  WebhookService,
  getWebhookService
} from './WebhookService';
import {
  WebSocketService,
  getWebSocketService
} from './WebSocketService';

import type {
  MonitoringState,
  StorageState,
  PlotlyChart,
  WebhookDelivery,
  WebSocketMessage,
  WebSocketUser,
  PrometheusConfig,
  DatadogConfig,
  HuggingFaceConfig,
  ModelZooConfig,
  PlotlyConfig,
  WebhookConfig,
  WebSocketConfig
} from './types';
import type { BrainStats } from '../lib/BrainEngine';
import type { MetricSnapshot, BoardAlert } from '../types/governance';

// ============================================================================
// Context Types
// ============================================================================

interface IntegrationContextValue {
  // Services
  monitoring: MonitoringService;
  storage: StorageService;
  visualization: VisualizationService;
  webhooks: WebhookService;
  websocket: WebSocketService;

  // State
  monitoringState: MonitoringState;
  storageState: StorageState;
  charts: PlotlyChart[];
  webhookDeliveries: WebhookDelivery[];
  onlineUsers: WebSocketUser[];
  isWebSocketConnected: boolean;

  // Quick Actions
  recordMetrics: (snapshot: MetricSnapshot) => void;
  recordBrainMetrics: (brain: BrainStats) => void;
  dispatchAlert: (alert: BoardAlert, projectId?: string) => Promise<void>;
  dispatchBrainEvent: (brain: BrainStats, event: 'burnout' | 'recovery' | 'mood_change') => Promise<void>;
  sendTrainingProgress: (runId: string, progress: {
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
  }) => void;

  // Configuration
  updateMonitoringConfig: (prometheus?: Partial<PrometheusConfig>, datadog?: Partial<DatadogConfig>) => void;
  updateStorageConfig: (huggingface?: Partial<HuggingFaceConfig>, modelzoo?: Partial<ModelZooConfig>) => void;
  updateVisualizationConfig: (plotly: Partial<PlotlyConfig>) => void;
  addWebhook: (name: string, config: Partial<WebhookConfig>) => WebhookConfig;
}

// ============================================================================
// Context
// ============================================================================

const IntegrationContext = createContext<IntegrationContextValue | null>(null);

// ============================================================================
// Provider
// ============================================================================

interface IntegrationProviderProps {
  children: React.ReactNode;
  userName?: string;
}

export function IntegrationProvider({ children, userName }: IntegrationProviderProps) {
  // Service refs
  const monitoringRef = useRef(getMonitoringService());
  const storageRef = useRef(getStorageService());
  const visualizationRef = useRef(getVisualizationService());
  const webhooksRef = useRef(getWebhookService());
  const websocketRef = useRef(getWebSocketService(userName));

  // State
  const [monitoringState, setMonitoringState] = useState<MonitoringState>(() =>
    monitoringRef.current.getState()
  );
  const [storageState, setStorageState] = useState<StorageState>(() =>
    storageRef.current.getState()
  );
  const [charts, setCharts] = useState<PlotlyChart[]>([]);
  const [webhookDeliveries, setWebhookDeliveries] = useState<WebhookDelivery[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<WebSocketUser[]>([]);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);

  // ==========================================================================
  // Subscribe to Service Updates
  // ==========================================================================

  useEffect(() => {
    const unsubMonitoring = monitoringRef.current.subscribe(setMonitoringState);
    const unsubStorage = storageRef.current.subscribe(setStorageState);
    const unsubVisualization = visualizationRef.current.subscribe((chartsMap) => {
      setCharts(Array.from(chartsMap.values()));
    });
    const unsubWebhooks = webhooksRef.current.subscribe(setWebhookDeliveries);

    // WebSocket updates
    const updateWebSocketState = () => {
      setOnlineUsers(websocketRef.current.getOnlineUsers());
      setIsWebSocketConnected(websocketRef.current.isConnected());
    };

    const unsubWebSocket = websocketRef.current.onType('user_presence', updateWebSocketState);
    updateWebSocketState();

    return () => {
      unsubMonitoring();
      unsubStorage();
      unsubVisualization();
      unsubWebhooks();
      unsubWebSocket();
    };
  }, []);

  // ==========================================================================
  // Quick Actions
  // ==========================================================================

  const recordMetrics = useCallback((snapshot: MetricSnapshot) => {
    monitoringRef.current.recordTrainingMetrics(snapshot);
  }, []);

  const recordBrainMetrics = useCallback((brain: BrainStats) => {
    monitoringRef.current.recordBrainMetrics(brain);
  }, []);

  const dispatchAlert = useCallback(async (alert: BoardAlert, projectId?: string) => {
    await webhooksRef.current.dispatchAlert(alert, projectId);
  }, []);

  const dispatchBrainEvent = useCallback(async (
    brain: BrainStats,
    event: 'burnout' | 'recovery' | 'mood_change'
  ) => {
    await webhooksRef.current.dispatchBrainState(brain, event);
    websocketRef.current.sendBrainState(brain);
  }, []);

  const sendTrainingProgress = useCallback((
    runId: string,
    progress: {
      epoch: number;
      totalEpochs: number;
      loss: number;
      accuracy: number;
    }
  ) => {
    websocketRef.current.sendTrainingProgress(runId, {
      ...progress,
      step: 0,
      totalSteps: 0,
      perplexity: Math.exp(progress.loss),
      learningRate: 0,
      throughput: 0,
      eta: 0
    });
  }, []);

  // ==========================================================================
  // Configuration Updates
  // ==========================================================================

  const updateMonitoringConfig = useCallback((
    prometheus?: Partial<PrometheusConfig>,
    datadog?: Partial<DatadogConfig>
  ) => {
    if (prometheus) {
      monitoringRef.current.updatePrometheusConfig(prometheus);
    }
    if (datadog) {
      monitoringRef.current.updateDatadogConfig(datadog);
    }
  }, []);

  const updateStorageConfig = useCallback((
    huggingface?: Partial<HuggingFaceConfig>,
    modelzoo?: Partial<ModelZooConfig>
  ) => {
    if (huggingface) {
      storageRef.current.updateHuggingFaceConfig(huggingface);
    }
    if (modelzoo) {
      storageRef.current.updateModelZooConfig(modelzoo);
    }
  }, []);

  const updateVisualizationConfig = useCallback((plotly: Partial<PlotlyConfig>) => {
    visualizationRef.current.updateConfig(plotly);
  }, []);

  const addWebhook = useCallback((name: string, config: Partial<WebhookConfig>) => {
    return webhooksRef.current.addWebhook(name, config);
  }, []);

  // ==========================================================================
  // Context Value
  // ==========================================================================

  const value = useMemo<IntegrationContextValue>(() => ({
    // Services
    monitoring: monitoringRef.current,
    storage: storageRef.current,
    visualization: visualizationRef.current,
    webhooks: webhooksRef.current,
    websocket: websocketRef.current,

    // State
    monitoringState,
    storageState,
    charts,
    webhookDeliveries,
    onlineUsers,
    isWebSocketConnected,

    // Quick Actions
    recordMetrics,
    recordBrainMetrics,
    dispatchAlert,
    dispatchBrainEvent,
    sendTrainingProgress,

    // Configuration
    updateMonitoringConfig,
    updateStorageConfig,
    updateVisualizationConfig,
    addWebhook
  }), [
    monitoringState,
    storageState,
    charts,
    webhookDeliveries,
    onlineUsers,
    isWebSocketConnected,
    recordMetrics,
    recordBrainMetrics,
    dispatchAlert,
    dispatchBrainEvent,
    sendTrainingProgress,
    updateMonitoringConfig,
    updateStorageConfig,
    updateVisualizationConfig,
    addWebhook
  ]);

  return (
    <IntegrationContext.Provider value={value}>
      {children}
    </IntegrationContext.Provider>
  );
}

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Main hook to access all integrations
 */
export function useIntegrations() {
  const context = useContext(IntegrationContext);
  if (!context) {
    throw new Error('useIntegrations must be used within an IntegrationProvider');
  }
  return context;
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for monitoring (Prometheus/Datadog)
 */
export function useMonitoring() {
  const { monitoring, monitoringState, updateMonitoringConfig, recordMetrics, recordBrainMetrics } = useIntegrations();

  return {
    service: monitoring,
    state: monitoringState,
    recordMetrics,
    recordBrainMetrics,
    updateConfig: updateMonitoringConfig,
    exportPrometheus: () => monitoring.exportPrometheusFormat(),
    exportDatadog: () => monitoring.exportDatadogFormat(),
    getMetricHistory: (name: string, limit?: number) => monitoring.getMetricHistory(name, limit)
  };
}

/**
 * Hook for model storage (HuggingFace/ModelZoo)
 */
export function useModelStorage() {
  const { storage, storageState, updateStorageConfig } = useIntegrations();

  const uploadModel = useCallback(async (
    modelData: unknown,
    options: { name: string; description: string; tags: string[]; isPublic: boolean },
    destination: 'huggingface' | 'modelzoo' = 'modelzoo',
    onProgress?: (progress: number) => void
  ) => {
    if (destination === 'huggingface') {
      return storage.uploadToHuggingFace(modelData, options, onProgress);
    }
    return storage.saveToModelZoo(modelData, options, onProgress);
  }, [storage]);

  const downloadModel = useCallback(async (
    modelId: string,
    source: 'huggingface' | 'modelzoo' = 'modelzoo',
    onProgress?: (progress: number) => void
  ) => {
    if (source === 'huggingface') {
      return storage.downloadFromHuggingFace(modelId, onProgress);
    }
    return storage.loadFromModelZoo(modelId, onProgress);
  }, [storage]);

  return {
    service: storage,
    state: storageState,
    models: storageState.models,
    isUploading: storage.isUploading(),
    isDownloading: storage.isDownloading(),
    uploadModel,
    downloadModel,
    deleteModel: storage.deleteModel.bind(storage),
    updateConfig: updateStorageConfig,
    getVersions: storage.getModelVersions.bind(storage)
  };
}

/**
 * Hook for Plotly visualization
 */
export function useVisualization() {
  const { visualization, charts, updateVisualizationConfig } = useIntegrations();

  return {
    service: visualization,
    charts,
    createEmbeddingScatter2D: visualization.createEmbeddingScatter2D.bind(visualization),
    createEmbeddingScatter3D: visualization.createEmbeddingScatter3D.bind(visualization),
    createTrainingMetricsChart: visualization.createTrainingMetricsChart.bind(visualization),
    createHeatmap: visualization.createHeatmap.bind(visualization),
    createBrainHealthChart: visualization.createBrainHealthChart.bind(visualization),
    createGovernanceTimeline: visualization.createGovernanceTimeline.bind(visualization),
    createHistogram: visualization.createHistogram.bind(visualization),
    getChart: visualization.getChart.bind(visualization),
    updateChart: visualization.updateChart.bind(visualization),
    deleteChart: visualization.deleteChart.bind(visualization),
    exportForPlotly: visualization.exportForPlotly.bind(visualization),
    updateConfig: updateVisualizationConfig
  };
}

/**
 * Hook for webhooks (Slack/Discord)
 */
export function useWebhooks() {
  const { webhooks, webhookDeliveries, addWebhook, dispatchAlert, dispatchBrainEvent } = useIntegrations();

  return {
    service: webhooks,
    deliveries: webhookDeliveries,
    configs: webhooks.getAllWebhooks(),
    stats: webhooks.getStats(),
    addWebhook,
    updateWebhook: webhooks.updateWebhook.bind(webhooks),
    deleteWebhook: webhooks.deleteWebhook.bind(webhooks),
    toggleWebhook: webhooks.toggleWebhook.bind(webhooks),
    dispatchAlert,
    dispatchBrainEvent,
    dispatch: webhooks.dispatch.bind(webhooks),
    retryDelivery: webhooks.retryDelivery.bind(webhooks),
    clearDeliveries: webhooks.clearDeliveries.bind(webhooks),
    formatSlackMessage: webhooks.formatSlackMessage.bind(webhooks),
    formatDiscordMessage: webhooks.formatDiscordMessage.bind(webhooks)
  };
}

/**
 * Hook for WebSocket real-time features
 */
export function useRealtime() {
  const { websocket, onlineUsers, isWebSocketConnected, sendTrainingProgress } = useIntegrations();

  const [messages, setMessages] = useState<WebSocketMessage[]>([]);

  // Subscribe to messages
  useEffect(() => {
    const unsubscribe = websocket.onAny((message) => {
      setMessages(prev => [...prev.slice(-99), message]);
    });
    return unsubscribe;
  }, [websocket]);

  return {
    service: websocket,
    connected: isWebSocketConnected,
    users: onlineUsers,
    messages,
    currentUser: websocket.getCurrentUser(),
    subscribe: websocket.subscribe.bind(websocket),
    unsubscribe: websocket.unsubscribe.bind(websocket),
    send: websocket.send.bind(websocket),
    sendTrainingProgress,
    sendBrainState: websocket.sendBrainState.bind(websocket),
    sendCursorPosition: websocket.sendCursorPosition.bind(websocket),
    requestSync: websocket.requestSync.bind(websocket),
    respondSync: websocket.respondSync.bind(websocket),
    on: websocket.on.bind(websocket),
    onType: websocket.onType.bind(websocket),
    setUserName: websocket.setUserName.bind(websocket),
    getStats: websocket.getStats.bind(websocket)
  };
}

/**
 * Hook for brain telemetry with integrations
 */
export function useBrainTelemetry(brain: BrainStats | null) {
  const { recordBrainMetrics, dispatchBrainEvent } = useIntegrations();
  const prevMoodRef = useRef<BrainStats['mood'] | null>(null);

  // Record metrics when brain changes
  useEffect(() => {
    if (!brain) return;

    recordBrainMetrics(brain);

    // Check for mood changes
    if (prevMoodRef.current && prevMoodRef.current !== brain.mood) {
      if (brain.mood === 'BURNT_OUT') {
        dispatchBrainEvent(brain, 'burnout');
      } else if (prevMoodRef.current === 'BURNT_OUT') {
        dispatchBrainEvent(brain, 'recovery');
      } else {
        dispatchBrainEvent(brain, 'mood_change');
      }
    }

    prevMoodRef.current = brain.mood;
  }, [brain, recordBrainMetrics, dispatchBrainEvent]);

  return {
    recordMetrics: () => brain && recordBrainMetrics(brain),
    dispatchBurnout: () => brain && dispatchBrainEvent(brain, 'burnout'),
    dispatchRecovery: () => brain && dispatchBrainEvent(brain, 'recovery'),
    dispatchMoodChange: () => brain && dispatchBrainEvent(brain, 'mood_change')
  };
}
