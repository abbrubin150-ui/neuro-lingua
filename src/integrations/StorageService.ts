/**
 * StorageService - HuggingFace Hub & ModelZoo Integration (Mock)
 *
 * Provides model storage, versioning, and sharing for ModelSnapshot
 * All operations are simulated locally without actual API calls
 */

import type {
  ModelMetadata,
  ModelUploadOptions,
  ModelVersion,
  HuggingFaceConfig,
  ModelZooConfig,
  StorageState
} from './types';
import { StorageManager } from '../lib/storage';

// ============================================================================
// Default Configurations
// ============================================================================

const DEFAULT_HUGGINGFACE_CONFIG: HuggingFaceConfig = {
  enabled: false,
  organization: undefined,
  defaultPrivate: true,
  autoSave: false
};

const DEFAULT_MODELZOO_CONFIG: ModelZooConfig = {
  enabled: true,
  storageKey: 'neuro-lingua-modelzoo-v1',
  maxModels: 50
};

const STORAGE_KEY = 'neuro-lingua-storage-state-v1';

// ============================================================================
// StorageService Class
// ============================================================================

export class StorageService {
  private huggingfaceConfig: HuggingFaceConfig;
  private modelzooConfig: ModelZooConfig;
  private state: StorageState;
  private listeners: Set<(state: StorageState) => void> = new Set();

  constructor(
    huggingfaceConfig?: Partial<HuggingFaceConfig>,
    modelzooConfig?: Partial<ModelZooConfig>
  ) {
    this.huggingfaceConfig = { ...DEFAULT_HUGGINGFACE_CONFIG, ...huggingfaceConfig };
    this.modelzooConfig = { ...DEFAULT_MODELZOO_CONFIG, ...modelzooConfig };

    // Load persisted state
    const saved = StorageManager.get<StorageState>(STORAGE_KEY, {
      models: [],
      uploading: [],
      downloading: [],
      lastSync: 0
    });
    this.state = saved;
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  updateHuggingFaceConfig(config: Partial<HuggingFaceConfig>): void {
    this.huggingfaceConfig = { ...this.huggingfaceConfig, ...config };
  }

  updateModelZooConfig(config: Partial<ModelZooConfig>): void {
    this.modelzooConfig = { ...this.modelzooConfig, ...config };
  }

  getHuggingFaceConfig(): HuggingFaceConfig {
    return { ...this.huggingfaceConfig };
  }

  getModelZooConfig(): ModelZooConfig {
    return { ...this.modelzooConfig };
  }

  // ==========================================================================
  // Model Upload (Mock)
  // ==========================================================================

  /**
   * Upload model to HuggingFace Hub (mock)
   */
  async uploadToHuggingFace(
    modelData: unknown,
    options: ModelUploadOptions,
    onProgress?: (progress: number) => void
  ): Promise<ModelMetadata> {
    if (!this.huggingfaceConfig.enabled) {
      throw new Error('HuggingFace integration is disabled');
    }

    const modelId = this.generateModelId('hf');
    this.state.uploading.push(modelId);
    this.notifyListeners();

    try {
      // Simulate upload with progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await this.simulateDelay(100, 300);
        onProgress?.(progress);
      }

      const metadata: ModelMetadata = {
        id: modelId,
        name: options.name,
        description: options.description,
        author: this.huggingfaceConfig.organization || 'anonymous',
        tags: options.tags,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        downloads: 0,
        likes: 0,
        version: '1.0.0',
        framework: 'neuro-lingua',
        license: 'MIT'
      };

      // Store model data locally
      this.saveModelLocally(modelId, modelData, metadata);

      this.state.models.push(metadata);
      this.state.uploading = this.state.uploading.filter(id => id !== modelId);
      this.state.lastSync = Date.now();
      this.persistState();
      this.notifyListeners();

      console.log(`[StorageService] Uploaded model to HuggingFace: ${modelId} (mock)`);
      return metadata;
    } catch (error) {
      this.state.uploading = this.state.uploading.filter(id => id !== modelId);
      this.notifyListeners();
      throw error;
    }
  }

  /**
   * Save model to local ModelZoo
   */
  async saveToModelZoo(
    modelData: unknown,
    options: ModelUploadOptions,
    onProgress?: (progress: number) => void
  ): Promise<ModelMetadata> {
    if (!this.modelzooConfig.enabled) {
      throw new Error('ModelZoo is disabled');
    }

    const modelId = this.generateModelId('zoo');
    this.state.uploading.push(modelId);
    this.notifyListeners();

    try {
      // Simulate save progress
      for (let progress = 0; progress <= 100; progress += 20) {
        await this.simulateDelay(50, 150);
        onProgress?.(progress);
      }

      const metadata: ModelMetadata = {
        id: modelId,
        name: options.name,
        description: options.description,
        author: 'local-user',
        tags: options.tags,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        downloads: 0,
        likes: 0,
        version: '1.0.0',
        framework: 'neuro-lingua',
        license: 'private'
      };

      // Store model data
      this.saveModelLocally(modelId, modelData, metadata);

      // Enforce max models limit
      if (this.state.models.length >= this.modelzooConfig.maxModels) {
        const oldest = this.state.models[0];
        this.deleteModelLocally(oldest.id);
        this.state.models.shift();
      }

      this.state.models.push(metadata);
      this.state.uploading = this.state.uploading.filter(id => id !== modelId);
      this.state.lastSync = Date.now();
      this.persistState();
      this.notifyListeners();

      console.log(`[StorageService] Saved model to ModelZoo: ${modelId}`);
      return metadata;
    } catch (error) {
      this.state.uploading = this.state.uploading.filter(id => id !== modelId);
      this.notifyListeners();
      throw error;
    }
  }

  // ==========================================================================
  // Model Download (Mock)
  // ==========================================================================

  /**
   * Download model from HuggingFace Hub (mock)
   */
  async downloadFromHuggingFace(
    modelId: string,
    onProgress?: (progress: number) => void
  ): Promise<{ modelData: unknown; metadata: ModelMetadata }> {
    if (!this.huggingfaceConfig.enabled) {
      throw new Error('HuggingFace integration is disabled');
    }

    this.state.downloading.push(modelId);
    this.notifyListeners();

    try {
      // Simulate download with progress
      for (let progress = 0; progress <= 100; progress += 5) {
        await this.simulateDelay(50, 200);
        onProgress?.(progress);
      }

      // Check if model exists locally (mock remote lookup)
      const localData = this.loadModelLocally(modelId);
      if (localData) {
        this.state.downloading = this.state.downloading.filter(id => id !== modelId);
        this.notifyListeners();
        return localData;
      }

      // Generate mock model if not found
      const mockMetadata: ModelMetadata = {
        id: modelId,
        name: `Model ${modelId}`,
        description: 'Downloaded from HuggingFace Hub (mock)',
        author: 'community',
        tags: ['downloaded', 'mock'],
        createdAt: Date.now() - 86400000, // 1 day ago
        updatedAt: Date.now(),
        downloads: Math.floor(Math.random() * 1000),
        likes: Math.floor(Math.random() * 100),
        version: '1.0.0',
        framework: 'neuro-lingua',
        license: 'MIT'
      };

      const mockModelData = this.generateMockModelData();

      // Update download count
      const existing = this.state.models.find(m => m.id === modelId);
      if (existing) {
        existing.downloads++;
      } else {
        this.state.models.push(mockMetadata);
      }

      this.state.downloading = this.state.downloading.filter(id => id !== modelId);
      this.persistState();
      this.notifyListeners();

      console.log(`[StorageService] Downloaded model from HuggingFace: ${modelId} (mock)`);
      return { modelData: mockModelData, metadata: mockMetadata };
    } catch (error) {
      this.state.downloading = this.state.downloading.filter(id => id !== modelId);
      this.notifyListeners();
      throw error;
    }
  }

  /**
   * Load model from local ModelZoo
   */
  async loadFromModelZoo(
    modelId: string,
    onProgress?: (progress: number) => void
  ): Promise<{ modelData: unknown; metadata: ModelMetadata } | null> {
    this.state.downloading.push(modelId);
    this.notifyListeners();

    try {
      // Simulate load progress
      for (let progress = 0; progress <= 100; progress += 25) {
        await this.simulateDelay(20, 80);
        onProgress?.(progress);
      }

      const result = this.loadModelLocally(modelId);

      this.state.downloading = this.state.downloading.filter(id => id !== modelId);
      this.notifyListeners();

      if (result) {
        console.log(`[StorageService] Loaded model from ModelZoo: ${modelId}`);
      }

      return result;
    } catch (error) {
      this.state.downloading = this.state.downloading.filter(id => id !== modelId);
      this.notifyListeners();
      throw error;
    }
  }

  // ==========================================================================
  // Model Management
  // ==========================================================================

  /**
   * List all models
   */
  listModels(filter?: { tags?: string[]; author?: string; search?: string }): ModelMetadata[] {
    let models = [...this.state.models];

    if (filter?.tags && filter.tags.length > 0) {
      models = models.filter(m =>
        filter.tags!.some(tag => m.tags.includes(tag))
      );
    }

    if (filter?.author) {
      models = models.filter(m => m.author === filter.author);
    }

    if (filter?.search) {
      const search = filter.search.toLowerCase();
      models = models.filter(m =>
        m.name.toLowerCase().includes(search) ||
        m.description.toLowerCase().includes(search)
      );
    }

    return models.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  /**
   * Get model by ID
   */
  getModel(modelId: string): ModelMetadata | undefined {
    return this.state.models.find(m => m.id === modelId);
  }

  /**
   * Update model metadata
   */
  updateModelMetadata(modelId: string, updates: Partial<ModelUploadOptions>): boolean {
    const model = this.state.models.find(m => m.id === modelId);
    if (!model) return false;

    if (updates.name) model.name = updates.name;
    if (updates.description) model.description = updates.description;
    if (updates.tags) model.tags = updates.tags;
    model.updatedAt = Date.now();

    this.persistState();
    this.notifyListeners();
    return true;
  }

  /**
   * Delete model
   */
  deleteModel(modelId: string): boolean {
    const index = this.state.models.findIndex(m => m.id === modelId);
    if (index === -1) return false;

    this.state.models.splice(index, 1);
    this.deleteModelLocally(modelId);
    this.persistState();
    this.notifyListeners();

    console.log(`[StorageService] Deleted model: ${modelId}`);
    return true;
  }

  /**
   * Like a model
   */
  likeModel(modelId: string): boolean {
    const model = this.state.models.find(m => m.id === modelId);
    if (!model) return false;

    model.likes++;
    model.updatedAt = Date.now();
    this.persistState();
    this.notifyListeners();
    return true;
  }

  // ==========================================================================
  // Version Management
  // ==========================================================================

  /**
   * Get model versions (mock)
   */
  getModelVersions(modelId: string): ModelVersion[] {
    const model = this.state.models.find(m => m.id === modelId);
    if (!model) return [];

    // Generate mock versions
    const versions: ModelVersion[] = [
      {
        version: model.version,
        createdAt: model.createdAt,
        size: Math.floor(Math.random() * 100000000), // Up to 100MB
        commitHash: this.generateCommitHash(),
        message: 'Initial version'
      }
    ];

    // Add some historical versions
    const numVersions = Math.floor(Math.random() * 5) + 1;
    for (let i = 1; i < numVersions; i++) {
      versions.unshift({
        version: `0.${numVersions - i}.0`,
        createdAt: model.createdAt - i * 86400000,
        size: Math.floor(Math.random() * 100000000),
        commitHash: this.generateCommitHash(),
        message: `Version ${numVersions - i}`
      });
    }

    return versions;
  }

  /**
   * Create new version of model
   */
  async createModelVersion(
    modelId: string,
    modelData: unknown,
    message?: string,
    onProgress?: (progress: number) => void
  ): Promise<ModelVersion | null> {
    const model = this.state.models.find(m => m.id === modelId);
    if (!model) return null;

    // Simulate upload
    for (let progress = 0; progress <= 100; progress += 15) {
      await this.simulateDelay(50, 150);
      onProgress?.(progress);
    }

    // Increment version
    const [major, minor, patch] = model.version.split('.').map(Number);
    model.version = `${major}.${minor}.${patch + 1}`;
    model.updatedAt = Date.now();

    // Save model data
    this.saveModelLocally(modelId, modelData, model);
    this.persistState();
    this.notifyListeners();

    const newVersion: ModelVersion = {
      version: model.version,
      createdAt: Date.now(),
      size: JSON.stringify(modelData).length,
      commitHash: this.generateCommitHash(),
      message: message || `Version ${model.version}`
    };

    console.log(`[StorageService] Created new version ${model.version} for model: ${modelId}`);
    return newVersion;
  }

  // ==========================================================================
  // State & Queries
  // ==========================================================================

  getState(): StorageState {
    return {
      ...this.state,
      uploading: [...this.state.uploading],
      downloading: [...this.state.downloading]
    };
  }

  isUploading(modelId?: string): boolean {
    if (modelId) {
      return this.state.uploading.includes(modelId);
    }
    return this.state.uploading.length > 0;
  }

  isDownloading(modelId?: string): boolean {
    if (modelId) {
      return this.state.downloading.includes(modelId);
    }
    return this.state.downloading.length > 0;
  }

  getStorageStats(): {
    totalModels: number;
    totalSize: number;
    oldestModel: number;
    newestModel: number;
  } {
    const models = this.state.models;
    return {
      totalModels: models.length,
      totalSize: models.reduce((sum, _m) => sum + Math.floor(Math.random() * 10000000), 0),
      oldestModel: models.length > 0 ? Math.min(...models.map(m => m.createdAt)) : 0,
      newestModel: models.length > 0 ? Math.max(...models.map(m => m.createdAt)) : 0
    };
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  subscribe(listener: (state: StorageState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // ==========================================================================
  // Local Storage Helpers
  // ==========================================================================

  private saveModelLocally(modelId: string, modelData: unknown, metadata: ModelMetadata): void {
    const key = `neuro-lingua-model-${modelId}`;
    StorageManager.set(key, { modelData, metadata });
  }

  private loadModelLocally(modelId: string): { modelData: unknown; metadata: ModelMetadata } | null {
    const key = `neuro-lingua-model-${modelId}`;
    return StorageManager.get(key, null);
  }

  private deleteModelLocally(modelId: string): void {
    const key = `neuro-lingua-model-${modelId}`;
    StorageManager.remove(key);
  }

  private persistState(): void {
    StorageManager.set(STORAGE_KEY, this.state);
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private generateModelId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateCommitHash(): string {
    return Array.from({ length: 40 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }

  private generateMockModelData(): object {
    return {
      architecture: 'feedforward',
      weights: {
        embedding: Array.from({ length: 100 }, () => Math.random() - 0.5),
        hidden: Array.from({ length: 256 }, () => Math.random() - 0.5),
        output: Array.from({ length: 100 }, () => Math.random() - 0.5)
      },
      config: {
        vocabSize: 1000,
        hiddenSize: 256,
        contextSize: 32
      }
    };
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
    this.listeners.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let storageInstance: StorageService | null = null;

export function getStorageService(): StorageService {
  if (!storageInstance) {
    storageInstance = new StorageService();
  }
  return storageInstance;
}

export function resetStorageService(): void {
  if (storageInstance) {
    storageInstance.destroy();
    storageInstance = null;
  }
}
