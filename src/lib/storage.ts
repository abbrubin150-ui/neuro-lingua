/**
 * Centralized localStorage abstraction layer
 * Provides type-safe storage operations with consistent error handling
 */

export class StorageManager {
  /**
   * Safely get an item from localStorage
   * @param key - Storage key
   * @param defaultValue - Default value if key doesn't exist or parsing fails
   * @returns Parsed value or default
   */
  static get<T>(key: string, defaultValue: T): T {
    try {
      const raw = localStorage.getItem(key);
      if (raw === null) return defaultValue;
      return JSON.parse(raw) as T;
    } catch (err) {
      console.warn(`Failed to retrieve ${key} from localStorage`, err);
      return defaultValue;
    }
  }

  /**
   * Safely set an item in localStorage
   * @param key - Storage key
   * @param value - Value to store (will be JSON stringified)
   * @returns true if successful, false otherwise
   */
  static set<T>(key: string, value: T): boolean {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (err) {
      console.warn(`Failed to persist ${key} to localStorage`, err);
      return false;
    }
  }

  /**
   * Safely remove an item from localStorage
   * @param key - Storage key to remove
   * @returns true if successful, false otherwise
   */
  static remove(key: string): boolean {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (err) {
      console.warn(`Failed to remove ${key} from localStorage`, err);
      return false;
    }
  }

  /**
   * Check if a key exists in localStorage
   * @param key - Storage key to check
   * @returns true if key exists, false otherwise
   */
  static has(key: string): boolean {
    try {
      return localStorage.getItem(key) !== null;
    } catch (err) {
      console.warn(`Failed to check ${key} in localStorage`, err);
      return false;
    }
  }

  /**
   * Clear multiple keys from localStorage
   * @param keys - Array of storage keys to remove
   * @returns Number of successfully removed keys
   */
  static removeMultiple(keys: string[]): number {
    let removed = 0;
    for (const key of keys) {
      if (this.remove(key)) removed++;
    }
    return removed;
  }
}
