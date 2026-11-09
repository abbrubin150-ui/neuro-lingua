/**
 * Answering questions with a model trained on question–answer pairs (src/data/qa.ts): the prompt
 * uses the model's labels and generation stops at the token the model learned to emit when an
 * answer is complete (the first token of the next question label).
 */

import { ProNeuralLM } from '../lib/ProNeuralLM';
import { qaPrompt, qaStopToken, type QaTemplate } from '../data/qa';

export interface AnswerOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
}

/** The end-of-answer token of a Q&A model. */
export function answerStopToken(model: ProNeuralLM, template: QaTemplate): string {
  const config = model.getTokenizerConfig();
  return qaStopToken((text) => ProNeuralLM.tokenizeWithOffsets(text, config), template);
}

export async function answerQuestion(
  model: ProNeuralLM,
  question: string,
  template: QaTemplate,
  options: AnswerOptions = {}
): Promise<{ text: string; tokens: number; finishReason: 'eos' | 'length' }> {
  const result = await model.generateDetailed(
    qaPrompt(question, template),
    options.maxTokens ?? 60,
    options.temperature ?? 0.7,
    options.topK ?? 0,
    options.topP ?? 0.9,
    0,
    0,
    0,
    0,
    0.1,
    [answerStopToken(model, template)]
  );
  // A model that spells the label out anyway: cut at the label text.
  const cut = result.text.indexOf(template.questionLabel);
  return { ...result, text: (cut >= 0 ? result.text.slice(0, cut) : result.text).trim() };
}
