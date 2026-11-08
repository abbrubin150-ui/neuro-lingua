/**
 * Local inference server for a model produced by the model factory.
 *
 * Endpoints (OpenAI-compatible shapes so that local tools can talk to it):
 *   GET  /health                → { status: "ok" }
 *   GET  /v1/models             → the loaded model
 *   POST /v1/completions        { prompt, max_tokens?, temperature?, top_p?, top_k?, stop?, seed? }
 *   POST /v1/chat/completions   { messages: [{ role, content }], ...same options }
 *
 * The server only runs the local model: it never forwards requests anywhere. It listens on
 * 127.0.0.1 by default and sends CORS headers only when explicitly enabled.
 */

import http from 'node:http';
import { ProNeuralLM } from '../../lib/ProNeuralLM';

export interface InferenceServerOptions {
  modelName: string;
  /** Allow browser pages from any origin to call the server (default false). */
  cors?: boolean;
  /** Upper bound for max_tokens (default 512). */
  maxTokensLimit?: number;
}

interface CompletionRequest {
  prompt?: string;
  messages?: { role: string; content: string }[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string | string[];
}

function cutAtStop(text: string, stop?: string | string[]): { text: string; stopped: boolean } {
  const stops = (Array.isArray(stop) ? stop : stop ? [stop] : []).filter((s) => s.length > 0);
  let cut = text.length;
  for (const s of stops) {
    const i = text.indexOf(s);
    if (i >= 0 && i < cut) cut = i;
  }
  return { text: text.slice(0, cut), stopped: cut < text.length };
}

/** Run one completion with the local model. */
export async function complete(
  model: ProNeuralLM,
  request: CompletionRequest,
  maxTokensLimit = 512
): Promise<{
  text: string;
  finishReason: 'stop' | 'length';
  promptTokens: number;
  completionTokens: number;
}> {
  const prompt =
    request.prompt ??
    (request.messages ?? [])
      .filter((m) => m.role !== 'system')
      .map((m) => m.content)
      .join('\n');
  const maxTokens = Math.max(1, Math.min(maxTokensLimit, Math.floor(request.max_tokens ?? 64)));
  const temperature = Math.max(0.05, request.temperature ?? 0.8);
  const generated = await model.generate(
    prompt,
    maxTokens,
    temperature,
    Math.max(0, Math.floor(request.top_k ?? 0)),
    Math.min(1, Math.max(0, request.top_p ?? 0.9))
  );
  const { text, stopped } = cutAtStop(generated, request.stop);
  const config = model.getTokenizerConfig();
  const completionTokens = ProNeuralLM.tokenizeText(text, config).length;
  return {
    text,
    finishReason: stopped || completionTokens < maxTokens ? 'stop' : 'length',
    promptTokens: ProNeuralLM.tokenizeText(prompt, config).length,
    completionTokens
  };
}

export function createInferenceServer(
  model: ProNeuralLM,
  options: InferenceServerOptions
): http.Server {
  const limit = options.maxTokensLimit ?? 512;
  return http.createServer(async (req, res) => {
    const send = (status: number, body: unknown) => {
      const headers: Record<string, string> = { 'Content-Type': 'application/json; charset=utf-8' };
      if (options.cors) {
        headers['Access-Control-Allow-Origin'] = '*';
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization';
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS';
      }
      res.writeHead(status, headers);
      res.end(JSON.stringify(body));
    };
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      if (req.method === 'OPTIONS') return send(204, {});
      if (req.method === 'GET' && url.pathname === '/health') return send(200, { status: 'ok' });
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        return send(200, {
          object: 'list',
          data: [{ id: options.modelName, object: 'model', owned_by: 'local' }]
        });
      }
      const isChat = url.pathname === '/v1/chat/completions';
      if (req.method === 'POST' && (url.pathname === '/v1/completions' || isChat)) {
        let raw = '';
        for await (const chunk of req) {
          raw += chunk;
          if (raw.length > 1_000_000) return send(413, { error: { message: 'Request too large' } });
        }
        const body = (raw ? JSON.parse(raw) : {}) as CompletionRequest;
        if (isChat ? !Array.isArray(body.messages) : typeof body.prompt !== 'string') {
          return send(400, {
            error: { message: isChat ? '"messages" must be an array' : '"prompt" must be a string' }
          });
        }
        const result = await complete(model, body, limit);
        const created = Math.floor(Date.now() / 1000);
        const usage = {
          prompt_tokens: result.promptTokens,
          completion_tokens: result.completionTokens,
          total_tokens: result.promptTokens + result.completionTokens
        };
        return send(
          200,
          isChat
            ? {
                id: `chatcmpl-${created}`,
                object: 'chat.completion',
                created,
                model: options.modelName,
                choices: [
                  {
                    index: 0,
                    message: { role: 'assistant', content: result.text },
                    finish_reason: result.finishReason
                  }
                ],
                usage
              }
            : {
                id: `cmpl-${created}`,
                object: 'text_completion',
                created,
                model: options.modelName,
                choices: [{ index: 0, text: result.text, finish_reason: result.finishReason }],
                usage
              }
        );
      }
      return send(404, { error: { message: `No route for ${req.method} ${url.pathname}` } });
    } catch (error) {
      return send(400, {
        error: { message: error instanceof Error ? error.message : String(error) }
      });
    }
  });
}
