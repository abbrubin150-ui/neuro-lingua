// @vitest-environment node
/**
 * Guard: Neuro-Lingua produces and runs models without any external AI model, API or service.
 *
 * Scans the application, CLI and desktop code (and the dependency list) for
 * - client libraries of hosted AI services or downloaders of pretrained models,
 * - hostnames of hosted AI APIs,
 * - network calls (fetch / XMLHttpRequest / WebSocket / EventSource / sendBeacon / importScripts).
 * The only allowed network call is the offline service worker's same-origin cache handler.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = path.resolve(__dirname, '../..');
const SCANNED_DIRECTORIES = ['src', 'scripts', 'electron', 'public'];
const SCANNED_FILES = ['index.html', 'vite.config.ts'];
const EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.mjs', '.cjs', '.html', '.json', '.webmanifest']);

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) return entry === 'node_modules' ? [] : walk(full);
    return EXTENSIONS.has(path.extname(entry)) ? [full] : [];
  });
}

const files = [
  ...SCANNED_DIRECTORIES.flatMap((d) => walk(path.join(ROOT, d))),
  ...SCANNED_FILES.map((f) => path.join(ROOT, f))
].map((file) => ({ file: path.relative(ROOT, file), text: readFileSync(file, 'utf8') }));

/** Lines of the scanned files that match a pattern, as "file:line: text". */
function matches(pattern: RegExp, allow: (file: string) => boolean = () => false): string[] {
  const found: string[] = [];
  for (const { file, text } of files) {
    if (allow(file)) continue;
    text.split('\n').forEach((line, i) => {
      if (pattern.test(line)) found.push(`${file}:${i + 1}: ${line.trim()}`);
    });
  }
  return found;
}

const AI_PACKAGES = [
  'openai',
  '@anthropic-ai/sdk',
  '@google/generative-ai',
  '@google/genai',
  '@google-cloud/aiplatform',
  '@google-cloud/vertexai',
  'cohere-ai',
  '@huggingface/inference',
  '@huggingface/hub',
  '@xenova/transformers',
  '@huggingface/transformers',
  '@mistralai/mistralai',
  'groq-sdk',
  'replicate',
  'ollama',
  'langchain',
  '@langchain/core',
  'llamaindex',
  'ai',
  '@ai-sdk/openai',
  '@tensorflow-models/universal-sentence-encoder',
  '@aws-sdk/client-bedrock-runtime',
  '@azure/openai'
];

const AI_HOSTS = [
  'api.openai.com',
  'openai.azure.com',
  'api.anthropic.com',
  'generativelanguage.googleapis.com',
  'aiplatform.googleapis.com',
  'api.cohere.ai',
  'api.cohere.com',
  'api-inference.huggingface.co',
  'router.huggingface.co',
  'huggingface.co/api',
  'api.mistral.ai',
  'api.together.xyz',
  'api.groq.com',
  'api.replicate.com',
  'openrouter.ai',
  'api.deepseek.com',
  'api.perplexity.ai',
  'api.fireworks.ai',
  'bedrock-runtime'
];

/** An import / require / dynamic import of a package (or one of its sub-paths). */
const importPattern = (name: string) =>
  new RegExp(
    `(from\\s+|require\\(\\s*|import\\(\\s*)['"]${name.replace(/[/.]/g, '\\$&')}(/[^'"]*)?['"]`
  );
const HOST_PATTERN = new RegExp(AI_HOSTS.map((h) => h.replace(/[.]/g, '\\.')).join('|'), 'i');
const NETWORK_PATTERN =
  /\bfetch\s*\(|new\s+WebSocket\s*\(|new\s+EventSource\s*\(|XMLHttpRequest|sendBeacon\s*\(|importScripts\s*\(/;

describe('no external AI services', () => {
  it('the checks detect what they guard against', () => {
    expect(importPattern('openai').test("import OpenAI from 'openai';")).toBe(true);
    expect(importPattern('@anthropic-ai/sdk').test("require('@anthropic-ai/sdk/client')")).toBe(true);
    expect(importPattern('ai').test("const m = await import('ai');")).toBe(true);
    expect(importPattern('ai').test("import { x } from './ai';")).toBe(false);
    expect(HOST_PATTERN.test("const url = 'https://api.openai.com/v1/chat/completions';")).toBe(
      true
    );
    expect(NETWORK_PATTERN.test('const r = await fetch(endpoint);')).toBe(true);
    expect(NETWORK_PATTERN.test("const ws = new WebSocket('wss://example.com');")).toBe(true);
    expect(NETWORK_PATTERN.test('const fetched = cache.get(key);')).toBe(false);
  });

  it('scans the application, CLI and desktop sources', () => {
    expect(files.length).toBeGreaterThan(100);
    expect(files.some((f) => f.file === path.join('src', 'App.tsx'))).toBe(true);
    expect(files.some((f) => f.file.startsWith('scripts'))).toBe(true);
  });

  it('depends on no AI service client or pretrained-model downloader', () => {
    const pkg = JSON.parse(readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
    const dependencies = Object.keys({
      ...pkg.dependencies,
      ...pkg.devDependencies,
      ...pkg.optionalDependencies,
      ...pkg.peerDependencies
    });
    expect(dependencies.filter((d) => AI_PACKAGES.includes(d))).toEqual([]);

    expect(AI_PACKAGES.flatMap((name) => matches(importPattern(name)))).toEqual([]);
  });

  it('contains no hosted AI API endpoint', () => {
    expect(matches(HOST_PATTERN)).toEqual([]);
  });

  it('makes no network requests (only the offline worker serves same-origin files from its cache)', () => {
    const calls = matches(
      NETWORK_PATTERN,
      (file) => file === path.join('src', 'pwa', 'serviceWorker.ts')
    );
    expect(calls).toEqual([]);
  });

  it('the local inference server listens on the loopback interface by default', () => {
    const cli = readFileSync(path.join(ROOT, 'scripts', 'lm.ts'), 'utf8');
    expect(cli).toMatch(/127\.0\.0\.1/);
    expect(cli).not.toMatch(/['"]0\.0\.0\.0['"]\s*[,;)]/);
  });
});
