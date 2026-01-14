/**
 * WebSocketService - Real-time Communication Simulation
 *
 * Provides WebSocket-like functionality for:
 * - Training sync between tabs
 * - Team collaboration
 * - Real-time metrics updates
 *
 * Uses BroadcastChannel API for cross-tab communication
 * All operations are local without actual WebSocket server
 */

import type {
  WebSocketMessage,
  WebSocketMessageType,
  WebSocketChannel,
  WebSocketUser,
  WebSocketConfig,
  WebSocketState,
  TrainingProgressPayload,
  MetricsUpdatePayload,
  BrainStatePayload,
  UserPresencePayload,
  CursorPositionPayload
} from './types';
import type { BrainStats } from '../lib/BrainEngine';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: WebSocketConfig = {
  enabled: true,
  heartbeatInterval: 30000,
  reconnectDelay: 3000,
  maxReconnectAttempts: 5,
  messageBufferSize: 100
};

const USER_COLORS = [
  '#6366f1', '#22d3ee', '#f472b6', '#a78bfa', '#34d399',
  '#fbbf24', '#fb7185', '#818cf8', '#2dd4bf', '#f97316'
];

// ============================================================================
// WebSocketService Class
// ============================================================================

export class WebSocketService {
  private config: WebSocketConfig;
  private state: WebSocketState;
  private channels: Map<string, WebSocketChannel> = new Map();
  private broadcastChannel: BroadcastChannel | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private listeners: Map<string, Set<(message: WebSocketMessage) => void>> = new Map();
  private globalListeners: Set<(message: WebSocketMessage) => void> = new Set();

  constructor(config?: Partial<WebSocketConfig>, userName?: string) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = {
      connected: false,
      connectionId: this.generateId('conn'),
      userId: this.generateId('user'),
      userName: userName || `User-${Math.random().toString(36).substr(2, 4)}`,
      lastHeartbeat: 0,
      reconnectAttempts: 0,
      subscribedChannels: [],
      users: new Map(),
      messageBuffer: []
    };

    if (this.config.enabled) {
      this.connect();
    }
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Initialize connection (using BroadcastChannel)
   */
  connect(): void {
    if (this.state.connected) return;

    try {
      // Use BroadcastChannel for cross-tab communication
      this.broadcastChannel = new BroadcastChannel('neuro-lingua-realtime');
      this.broadcastChannel.onmessage = (event) => {
        this.handleIncomingMessage(event.data);
      };

      this.state.connected = true;
      this.state.reconnectAttempts = 0;
      this.state.lastHeartbeat = Date.now();

      // Start heartbeat
      this.startHeartbeat();

      // Add self to users
      const selfUser: WebSocketUser = {
        id: this.state.userId,
        name: this.state.userName,
        color: this.getRandomColor(),
        joinedAt: Date.now(),
        lastSeen: Date.now()
      };
      this.state.users.set(this.state.userId, selfUser);

      // Broadcast presence
      this.broadcastPresence('join');

      console.log(`[WebSocketService] Connected with ID: ${this.state.connectionId}`);
    } catch (error) {
      console.warn('[WebSocketService] BroadcastChannel not available, falling back to local-only mode');
      this.state.connected = true; // Still mark as connected for local operations
    }
  }

  /**
   * Disconnect from the service
   */
  disconnect(): void {
    if (!this.state.connected) return;

    this.broadcastPresence('leave');
    this.stopHeartbeat();

    if (this.broadcastChannel) {
      this.broadcastChannel.close();
      this.broadcastChannel = null;
    }

    this.state.connected = false;
    console.log('[WebSocketService] Disconnected');
  }

  /**
   * Reconnect to the service
   */
  reconnect(): void {
    this.disconnect();
    setTimeout(() => {
      if (this.state.reconnectAttempts < this.config.maxReconnectAttempts) {
        this.state.reconnectAttempts++;
        this.connect();
      }
    }, this.config.reconnectDelay);
  }

  // ==========================================================================
  // Channel Management
  // ==========================================================================

  /**
   * Subscribe to a channel
   */
  subscribe(channelId: string, type: WebSocketChannel['type'] = 'global', entityId?: string): boolean {
    if (this.state.subscribedChannels.includes(channelId)) return true;

    // Create or get channel
    let channel = this.channels.get(channelId);
    if (!channel) {
      channel = {
        id: channelId,
        type,
        entityId,
        subscribers: new Set(),
        lastActivity: Date.now(),
        messageCount: 0
      };
      this.channels.set(channelId, channel);
    }

    channel.subscribers.add(this.state.userId);
    this.state.subscribedChannels.push(channelId);

    // Notify others of subscription
    this.send('user_presence', channelId, {
      action: 'join',
      user: this.state.users.get(this.state.userId)!
    } as UserPresencePayload);

    console.log(`[WebSocketService] Subscribed to channel: ${channelId}`);
    return true;
  }

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(channelId: string): boolean {
    const channel = this.channels.get(channelId);
    if (!channel) return false;

    channel.subscribers.delete(this.state.userId);
    this.state.subscribedChannels = this.state.subscribedChannels.filter(c => c !== channelId);

    // Notify others
    this.send('user_presence', channelId, {
      action: 'leave',
      user: this.state.users.get(this.state.userId)!
    } as UserPresencePayload);

    // Clean up empty channels
    if (channel.subscribers.size === 0) {
      this.channels.delete(channelId);
    }

    console.log(`[WebSocketService] Unsubscribed from channel: ${channelId}`);
    return true;
  }

  /**
   * Get channel by ID
   */
  getChannel(channelId: string): WebSocketChannel | undefined {
    return this.channels.get(channelId);
  }

  /**
   * Get all subscribed channels
   */
  getSubscribedChannels(): string[] {
    return [...this.state.subscribedChannels];
  }

  // ==========================================================================
  // Message Sending
  // ==========================================================================

  /**
   * Send a message to a channel
   */
  send<T>(type: WebSocketMessageType, channel: string, payload: T): void {
    if (!this.state.connected) {
      console.warn('[WebSocketService] Not connected, message not sent');
      return;
    }

    const message: WebSocketMessage<T> = {
      id: this.generateId('msg'),
      type,
      channel,
      payload,
      timestamp: Date.now(),
      senderId: this.state.userId,
      senderName: this.state.userName
    };

    // Add to local buffer
    this.addToBuffer(message);

    // Broadcast to other tabs
    if (this.broadcastChannel) {
      this.broadcastChannel.postMessage(message);
    }

    // Notify local listeners
    this.notifyListeners(channel, message);

    // Update channel activity
    const ch = this.channels.get(channel);
    if (ch) {
      ch.lastActivity = Date.now();
      ch.messageCount++;
    }
  }

  /**
   * Send training progress update
   */
  sendTrainingProgress(runId: string, progress: Omit<TrainingProgressPayload, 'runId'>): void {
    this.send('training_progress', `run:${runId}`, {
      runId,
      ...progress
    } as TrainingProgressPayload);
  }

  /**
   * Send metrics update
   */
  sendMetricsUpdate(runId: string, metrics: MetricsUpdatePayload['metrics']): void {
    this.send('metrics_update', `run:${runId}`, {
      runId,
      metrics,
      timestamp: Date.now()
    } as MetricsUpdatePayload);
  }

  /**
   * Send brain state update
   */
  sendBrainState(brain: BrainStats): void {
    const healthScore = (brain.creativity + brain.stability) / 2;
    this.send('brain_state_changed', `brain:${brain.id}`, {
      brainId: brain.id,
      mood: brain.mood,
      creativity: brain.creativity,
      stability: brain.stability,
      healthScore
    } as BrainStatePayload);
  }

  /**
   * Send cursor position (for collaboration)
   */
  sendCursorPosition(channelId: string, position: { x: number; y: number }): void {
    this.send('cursor_position', channelId, {
      userId: this.state.userId,
      position
    } as CursorPositionPayload);
  }

  /**
   * Request sync from other tabs
   */
  requestSync(channelId: string): void {
    this.send('sync_request', channelId, {
      requesterId: this.state.userId,
      timestamp: Date.now()
    });
  }

  /**
   * Respond to sync request
   */
  respondSync(channelId: string, data: unknown): void {
    this.send('sync_response', channelId, {
      responderId: this.state.userId,
      timestamp: Date.now(),
      data
    });
  }

  // ==========================================================================
  // Message Listening
  // ==========================================================================

  /**
   * Listen to messages on a specific channel
   */
  on(channel: string, listener: (message: WebSocketMessage) => void): () => void {
    if (!this.listeners.has(channel)) {
      this.listeners.set(channel, new Set());
    }
    this.listeners.get(channel)!.add(listener);

    return () => {
      this.listeners.get(channel)?.delete(listener);
    };
  }

  /**
   * Listen to all messages (global listener)
   */
  onAny(listener: (message: WebSocketMessage) => void): () => void {
    this.globalListeners.add(listener);
    return () => this.globalListeners.delete(listener);
  }

  /**
   * Listen to specific message type
   */
  onType(type: WebSocketMessageType, listener: (message: WebSocketMessage) => void): () => void {
    const wrapper = (message: WebSocketMessage) => {
      if (message.type === type) {
        listener(message);
      }
    };
    this.globalListeners.add(wrapper);
    return () => this.globalListeners.delete(wrapper);
  }

  // ==========================================================================
  // User & Presence Management
  // ==========================================================================

  /**
   * Get online users
   */
  getOnlineUsers(): WebSocketUser[] {
    return Array.from(this.state.users.values());
  }

  /**
   * Get user by ID
   */
  getUser(userId: string): WebSocketUser | undefined {
    return this.state.users.get(userId);
  }

  /**
   * Get current user
   */
  getCurrentUser(): WebSocketUser | undefined {
    return this.state.users.get(this.state.userId);
  }

  /**
   * Update current user name
   */
  setUserName(name: string): void {
    this.state.userName = name;
    const user = this.state.users.get(this.state.userId);
    if (user) {
      user.name = name;
      this.broadcastPresence('update');
    }
  }

  private broadcastPresence(action: 'join' | 'leave' | 'update'): void {
    const user = this.state.users.get(this.state.userId);
    if (!user) return;

    user.lastSeen = Date.now();

    // Broadcast to all subscribed channels
    for (const channel of this.state.subscribedChannels) {
      this.send('user_presence', channel, {
        action,
        user: { ...user }
      } as UserPresencePayload);
    }
  }

  // ==========================================================================
  // State & Queries
  // ==========================================================================

  /**
   * Get current state
   */
  getState(): WebSocketState {
    return {
      ...this.state,
      subscribedChannels: [...this.state.subscribedChannels],
      users: new Map(this.state.users),
      messageBuffer: [...this.state.messageBuffer]
    };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.state.connected;
  }

  /**
   * Get message buffer
   */
  getMessageBuffer(channel?: string, limit?: number): WebSocketMessage[] {
    let messages = [...this.state.messageBuffer];

    if (channel) {
      messages = messages.filter(m => m.channel === channel);
    }

    if (limit) {
      messages = messages.slice(-limit);
    }

    return messages;
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    connected: boolean;
    channels: number;
    users: number;
    messages: number;
    uptime: number;
  } {
    const firstMessage = this.state.messageBuffer[0];
    return {
      connected: this.state.connected,
      channels: this.channels.size,
      users: this.state.users.size,
      messages: this.state.messageBuffer.length,
      uptime: firstMessage ? Date.now() - firstMessage.timestamp : 0
    };
  }

  // ==========================================================================
  // Internal Handlers
  // ==========================================================================

  private handleIncomingMessage(message: WebSocketMessage): void {
    // Ignore own messages
    if (message.senderId === this.state.userId) return;

    // Add to buffer
    this.addToBuffer(message);

    // Handle special message types
    switch (message.type) {
      case 'user_presence':
        this.handlePresenceMessage(message);
        break;
      case 'ping':
        this.send('pong', message.channel, { timestamp: Date.now() });
        break;
      case 'sync_request':
        // Emit event for handlers to respond
        break;
    }

    // Notify listeners
    this.notifyListeners(message.channel, message);
  }

  private handlePresenceMessage(message: WebSocketMessage<UserPresencePayload>): void {
    const payload = message.payload as UserPresencePayload;

    switch (payload.action) {
      case 'join':
      case 'update':
        this.state.users.set(payload.user.id, payload.user);
        break;
      case 'leave':
        this.state.users.delete(payload.user.id);
        break;
    }
  }

  private addToBuffer(message: WebSocketMessage): void {
    this.state.messageBuffer.push(message);

    // Keep buffer within limits
    if (this.state.messageBuffer.length > this.config.messageBufferSize) {
      this.state.messageBuffer = this.state.messageBuffer.slice(-this.config.messageBufferSize);
    }
  }

  private notifyListeners(channel: string, message: WebSocketMessage): void {
    // Channel-specific listeners
    const channelListeners = this.listeners.get(channel);
    if (channelListeners) {
      for (const listener of channelListeners) {
        listener(message);
      }
    }

    // Global listeners
    for (const listener of this.globalListeners) {
      listener(message);
    }
  }

  // ==========================================================================
  // Heartbeat
  // ==========================================================================

  private startHeartbeat(): void {
    if (this.heartbeatInterval) return;

    this.heartbeatInterval = setInterval(() => {
      this.state.lastHeartbeat = Date.now();

      // Update self in users
      const self = this.state.users.get(this.state.userId);
      if (self) {
        self.lastSeen = Date.now();
      }

      // Clean up stale users (not seen in 2 minutes)
      const staleThreshold = Date.now() - 120000;
      for (const [userId, user] of this.state.users) {
        if (userId !== this.state.userId && user.lastSeen < staleThreshold) {
          this.state.users.delete(userId);
        }
      }

      // Send ping to keep connection alive
      for (const channel of this.state.subscribedChannels) {
        this.send('ping', channel, { timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getRandomColor(): string {
    return USER_COLORS[Math.floor(Math.random() * USER_COLORS.length)];
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  destroy(): void {
    this.disconnect();
    this.listeners.clear();
    this.globalListeners.clear();
    this.channels.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let websocketInstance: WebSocketService | null = null;

export function getWebSocketService(userName?: string): WebSocketService {
  if (!websocketInstance) {
    websocketInstance = new WebSocketService(undefined, userName);
  }
  return websocketInstance;
}

export function resetWebSocketService(): void {
  if (websocketInstance) {
    websocketInstance.destroy();
    websocketInstance = null;
  }
}
