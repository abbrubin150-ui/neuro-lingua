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
import type { ChatTemplate } from '../../lib/modelPackage';
import { qaPrompt } from '../../data/qa';
import { answerStopToken } from '../answer';

export interface InferenceServerOptions {
  modelName: string;
  /** Prompt layout of a model trained on question–answer pairs (from its model card). */
  chat?: ChatTemplate;
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
  /** Reseed the sampler for a reproducible completion. */
  seed?: number;
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

/**
 * Run one completion with the local model. For a chat request to a Q&A model (`chat` given), the
 * last user message is asked with the model's question / answer labels and the answer ends where
 * the model starts the next question.
 */
export async function complete(
  model: ProNeuralLM,
  request: CompletionRequest,
  maxTokensLimit = 512,
  chat?: ChatTemplate
): Promise<{
  text: string;
  finishReason: 'stop' | 'length';
  promptTokens: number;
  completionTokens: number;
}> {
  const lastUser = [...(request.messages ?? [])].reverse().find((m) => m.role === 'user');
  const prompt =
    request.prompt ??
    (chat && lastUser
      ? qaPrompt(lastUser.content, chat)
      : (request.messages ?? [])
          .filter((m) => m.role !== 'system')
          .map((m) => m.content)
          .join('\n'));
  const stops = [
    ...(Array.isArray(request.stop) ? request.stop : request.stop ? [request.stop] : []),
    ...(chat && request.prompt === undefined ? [chat.questionLabel] : [])
  ];
  const maxTokens = Math.max(1, Math.min(maxTokensLimit, Math.floor(request.max_tokens ?? 64)));
  const temperature = Math.max(0.05, request.temperature ?? 0.8);
  if (typeof request.seed === 'number' && Number.isFinite(request.seed)) {
    model.reseed(Math.floor(request.seed));
  }
  const asked = chat && request.prompt === undefined;
  const generated = await model.generateDetailed(
    prompt,
    maxTokens,
    temperature,
    Math.max(0, Math.floor(request.top_k ?? 0)),
    Math.min(1, Math.max(0, request.top_p ?? 0.9)),
    0,
    0,
    0,
    0,
    0.1,
    // A Q&A model ends its answer with the first token of the next question.
    asked ? [answerStopToken(model, chat)] : []
  );
  const cut = cutAtStop(generated.text, stops);
  const stopped = cut.stopped;
  // A chat answer is returned without the space that followed the answer label.
  const text = asked ? cut.text.trim() : cut.text;
  const config = model.getTokenizerConfig();
  return {
    text,
    // 'length' only when the token limit (not the end-of-sequence token or a stop string) ended it.
    finishReason: stopped || generated.finishReason === 'eos' ? 'stop' : 'length',
    promptTokens: ProNeuralLM.tokenizeText(prompt, config).length,
    // Tokens actually sampled; re-counted only for the part kept before a stop string.
    completionTokens: stopped ? ProNeuralLM.tokenizeText(text, config).length : generated.tokens
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
        const result = await complete(model, body, limit, options.chat);
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
