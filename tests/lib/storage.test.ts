import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StorageManager } from '../../src/lib/storage';

describe('StorageManager', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  describe('get', () => {
    it('returns stored value when key exists', () => {
      const testData = { foo: 'bar', count: 42 };
      localStorage.setItem('test-key', JSON.stringify(testData));

      const result = StorageManager.get('test-key', {});
      expect(result).toEqual(testData);
    });

    it('returns default value when key does not exist', () => {
      const defaultValue = { default: true };
      const result = StorageManager.get('non-existent-key', defaultValue);
      expect(result).toEqual(defaultValue);
    });

    it('returns default value when JSON parsing fails', () => {
      localStorage.setItem('corrupt-key', 'invalid-json{{{');
      const defaultValue = { fallback: true };

      const result = StorageManager.get('corrupt-key', defaultValue);
      expect(result).toEqual(defaultValue);
    });

    it('handles null value correctly', () => {
      const result = StorageManager.get('null-key', null);
      expect(result).toBeNull();
    });

    it('preserves type safety for different data types', () => {
      // String
      localStorage.setItem('string-key', JSON.stringify('hello'));
      expect(StorageManager.get('string-key', '')).toBe('hello');

      // Number
      localStorage.setItem('number-key', JSON.stringify(123));
      expect(StorageManager.get('number-key', 0)).toBe(123);

      // Boolean
      localStorage.setItem('bool-key', JSON.stringify(true));
      expect(StorageManager.get('bool-key', false)).toBe(true);

      // Array
      localStorage.setItem('array-key', JSON.stringify([1, 2, 3]));
      expect(StorageManager.get('array-key', [])).toEqual([1, 2, 3]);

      // Object
      localStorage.setItem('obj-key', JSON.stringify({ a: 1, b: 2 }));
      expect(StorageManager.get('obj-key', {})).toEqual({ a: 1, b: 2 });
    });

    it('warns when retrieval fails', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Force an error by corrupting localStorage
      localStorage.setItem('bad-key', 'not valid json}}}');

      StorageManager.get('bad-key', {});

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to retrieve bad-key'),
        expect.anything()
      );

      consoleSpy.mockRestore();
    });
  });

  describe('set', () => {
    it('stores value successfully and returns true', () => {
      const testData = { test: 'data', value: 123 };
      const result = StorageManager.set('test-key', testData);

      expect(result).toBe(true);
      expect(localStorage.getItem('test-key')).toBe(JSON.stringify(testData));
    });

    it('stores different data types correctly', () => {
      StorageManager.set('string', 'hello');
      StorageManager.set('number', 42);
      StorageManager.set('boolean', true);
      StorageManager.set('array', [1, 2, 3]);
      StorageManager.set('object', { key: 'value' });
      StorageManager.set('null', null);

      expect(StorageManager.get('string', '')).toBe('hello');
      expect(StorageManager.get('number', 0)).toBe(42);
      expect(StorageManager.get('boolean', false)).toBe(true);
      expect(StorageManager.get('array', [])).toEqual([1, 2, 3]);
      expect(StorageManager.get('object', {})).toEqual({ key: 'value' });
      expect(StorageManager.get('null', 'default')).toBeNull();
    });

    it('overwrites existing values', () => {
      StorageManager.set('key', 'original');
      expect(StorageManager.get('key', '')).toBe('original');

      StorageManager.set('key', 'updated');
      expect(StorageManager.get('key', '')).toBe('updated');
    });

    it('handles quota exceeded errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Mock localStorage.setItem to throw quota exceeded error
      const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        const error = new Error('QuotaExceededError');
        error.name = 'QuotaExceededError';
        throw error;
      });

      const result = StorageManager.set('test-key', 'large data');

      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to persist test-key'),
        expect.anything()
      );

      // Restore
      setItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });

    it('returns false and warns on other errors', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('Random error');
      });

      const result = StorageManager.set('test-key', { data: 'test' });

      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalled();

      setItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });
  });

  describe('remove', () => {
    it('removes existing key and returns true', () => {
      localStorage.setItem('test-key', 'test-value');
      expect(localStorage.getItem('test-key')).toBeTruthy();

      const result = StorageManager.remove('test-key');

      expect(result).toBe(true);
      expect(localStorage.getItem('test-key')).toBeNull();
    });

    it('returns true even when key does not exist', () => {
      const result = StorageManager.remove('non-existent-key');
      expect(result).toBe(true);
    });

    it('handles errors gracefully and returns false', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const removeItemSpy = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
        throw new Error('Remove error');
      });

      const result = StorageManager.remove('test-key');

      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to remove test-key'),
        expect.anything()
      );

      removeItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });
  });

  describe('has', () => {
    it('returns true when key exists', () => {
      localStorage.setItem('test-key', 'value');
      expect(StorageManager.has('test-key')).toBe(true);
    });

    it('returns false when key does not exist', () => {
      expect(StorageManager.has('non-existent-key')).toBe(false);
    });

    it('returns true even for empty string values', () => {
      localStorage.setItem('empty-key', '');
      expect(StorageManager.has('empty-key')).toBe(true);
    });

    it('handles errors gracefully and returns false', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const getItemSpy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('Access error');
      });

      const result = StorageManager.has('test-key');

      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalled();

      getItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });
  });

  describe('removeMultiple', () => {
    it('removes all specified keys', () => {
      localStorage.setItem('key1', 'value1');
      localStorage.setItem('key2', 'value2');
      localStorage.setItem('key3', 'value3');

      const removed = StorageManager.removeMultiple(['key1', 'key2', 'key3']);

      expect(removed).toBe(3);
      expect(localStorage.getItem('key1')).toBeNull();
      expect(localStorage.getItem('key2')).toBeNull();
      expect(localStorage.getItem('key3')).toBeNull();
    });

    it('counts only successfully removed keys', () => {
      localStorage.setItem('key1', 'value1');
      // key2 doesn't exist
      localStorage.setItem('key3', 'value3');

      const removed = StorageManager.removeMultiple(['key1', 'key2', 'key3']);

      // All should be removed successfully (even if key2 didn't exist)
      expect(removed).toBe(3);
    });

    it('handles empty array', () => {
      const removed = StorageManager.removeMultiple([]);
      expect(removed).toBe(0);
    });

    it('continues removing even if one key fails', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      localStorage.setItem('key1', 'value1');
      localStorage.setItem('key2', 'value2');
      localStorage.setItem('key3', 'value3');

      let callCount = 0;
      const removeItemSpy = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(function (this: Storage, key: string) {
        callCount++;
        if (key === 'key2') {
          throw new Error('Remove error');
        }
        // Actually remove the key from internal storage
        const storage = this as any;
        delete storage[key];
      });

      const removed = StorageManager.removeMultiple(['key1', 'key2', 'key3']);

      expect(removed).toBe(2); // key1 and key3 removed, key2 failed
      expect(localStorage.getItem('key2')).toBe('value2'); // Still exists

      removeItemSpy.mockRestore();
      consoleSpy.mockRestore();
    });
  });

  describe('Integration scenarios', () => {
    it('handles complete lifecycle: set → get → remove → verify', () => {
      const data = { name: 'test', count: 10 };

      // Set
      expect(StorageManager.set('lifecycle-key', data)).toBe(true);

      // Get
      expect(StorageManager.get('lifecycle-key', {})).toEqual(data);
      expect(StorageManager.has('lifecycle-key')).toBe(true);

      // Remove
      expect(StorageManager.remove('lifecycle-key')).toBe(true);

      // Verify
      expect(StorageManager.has('lifecycle-key')).toBe(false);
      expect(StorageManager.get('lifecycle-key', null)).toBeNull();
    });

    it('handles concurrent updates to same key', () => {
      StorageManager.set('counter', 1);
      StorageManager.set('counter', 2);
      StorageManager.set('counter', 3);

      expect(StorageManager.get('counter', 0)).toBe(3);
    });

    it('isolates different keys', () => {
      StorageManager.set('key-a', 'value-a');
      StorageManager.set('key-b', 'value-b');

      expect(StorageManager.get('key-a', '')).toBe('value-a');
      expect(StorageManager.get('key-b', '')).toBe('value-b');

      StorageManager.remove('key-a');

      expect(StorageManager.has('key-a')).toBe(false);
      expect(StorageManager.has('key-b')).toBe(true);
    });

    it('handles large objects efficiently', () => {
      const largeObject = {
        data: new Array(1000).fill(0).map((_, i) => ({
          id: i,
          value: `item-${i}`,
          nested: { deep: { value: i * 2 } }
        }))
      };

      const success = StorageManager.set('large-object', largeObject);
      expect(success).toBe(true);

      const retrieved = StorageManager.get('large-object', {});
      expect(retrieved).toEqual(largeObject);
      expect(retrieved.data.length).toBe(1000);
    });

    it('preserves complex nested structures', () => {
      const complex = {
        string: 'text',
        number: 42,
        boolean: true,
        null: null,
        array: [1, 2, 3],
        nested: {
          level1: {
            level2: {
              level3: 'deep value'
            }
          }
        },
        mixed: [
          { id: 1, name: 'first' },
          { id: 2, name: 'second' }
        ]
      };

      StorageManager.set('complex', complex);
      const retrieved = StorageManager.get('complex', {});

      expect(retrieved).toEqual(complex);
      expect(retrieved.nested.level1.level2.level3).toBe('deep value');
      expect(retrieved.mixed[1].name).toBe('second');
    });
  });

  describe('Edge cases', () => {
    it('handles undefined values by storing as JSON', () => {
      StorageManager.set('undefined-key', undefined);
      const result = StorageManager.get('undefined-key', 'default');

      // JSON.stringify(undefined) returns undefined (not a string)
      // so localStorage won't actually store it, and get returns default
      expect(result).toBe('default');
    });

    it('handles special characters in keys', () => {
      const specialKeys = [
        'key-with-dashes',
        'key_with_underscores',
        'key.with.dots',
        'key:with:colons',
        'key/with/slashes'
      ];

      specialKeys.forEach(key => {
        StorageManager.set(key, `value-${key}`);
        expect(StorageManager.get(key, '')).toBe(`value-${key}`);
      });
    });

    it('handles empty string as key', () => {
      StorageManager.set('', 'empty-key-value');
      expect(StorageManager.get('', '')).toBe('empty-key-value');
      expect(StorageManager.has('')).toBe(true);
      StorageManager.remove('');
      expect(StorageManager.has('')).toBe(false);
    });

    it('handles circular reference objects gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const circular: any = { name: 'circular' };
      circular.self = circular;

      const result = StorageManager.set('circular', circular);

      // Should fail because JSON.stringify can't handle circular refs
      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });
});
