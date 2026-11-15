/**
 * Held-out validation for training in the app: a deterministic split of the pasted text and the
 * early-stopping bookkeeping. Pure functions (no model classes), shared by the UI and the tests.
 *
 * The CLI factory has its own split (src/data/corpusPrep.ts: by paragraph hash over documents);
 * pasted text often has few paragraphs, so this one falls back to sentences and lines.
 */

export interface ValidationSplit {
  train: string;
  validation: string;
  /** What was held out: every `every`-th paragraph or sentence. */
  unit: 'paragraph' | 'sentence';
  every: number;
  units: number;
}

export interface ValidationSplitOptions {
  /** Share of the units held out (default 0.1 → every 10th unit). */
  fraction?: number;
  /** Fewer units than this: no split (default 20). */
  minUnits?: number;
  /** A validation part shorter than this is too noisy to steer training (default 200). */
  minValidationChars?: number;
}

const PARAGRAPH_BREAK = /\n\s*\n/;
// End of a sentence (Latin and Hebrew punctuation) followed by whitespace, or a line break.
const SENTENCE_BREAK = /(?<=[.!?…׃:;])\s+|\s*\n+\s*/u;

function nonEmpty(parts: string[]): string[] {
  return parts.map((p) => p.trim()).filter((p) => p.length > 0);
}

/**
 * Hold out every k-th paragraph (or sentence / line when there are few paragraphs) of `text`.
 * Interleaving keeps the two parts alike in style and topic, unlike cutting off the end.
 * Returns null when the text is too small to spare a meaningful validation part; training then
 * uses all of it, as before.
 */
export function splitForValidation(
  text: string,
  options: ValidationSplitOptions = {}
): ValidationSplit | null {
  const fraction = Math.min(0.5, Math.max(0.01, options.fraction ?? 0.1));
  const minUnits = Math.max(2, options.minUnits ?? 20);
  const minValidationChars = options.minValidationChars ?? 200;
  const every = Math.max(2, Math.round(1 / fraction));

  const paragraphs = nonEmpty(text.split(PARAGRAPH_BREAK));
  const useParagraphs = paragraphs.length >= minUnits;
  const units = useParagraphs ? paragraphs : nonEmpty(text.split(SENTENCE_BREAK));
  if (units.length < minUnits) return null;

  const separator = useParagraphs ? '\n\n' : ' ';
  const train: string[] = [];
  const validation: string[] = [];
  units.forEach((unit, index) => (index % every === every - 1 ? validation : train).push(unit));

  const validationText = validation.join(separator);
  if (validationText.length < minValidationChars) return null;
  return {
    train: train.join(separator),
    validation: validationText,
    unit: useParagraphs ? 'paragraph' : 'sentence',
    every,
    units: units.length
  };
}

/** Bits per character of a held-out text: tokenizer-independent, lower is better. */
export function bitsPerCharacter(loss: number, tokens: number, characters: number): number {
  return tokens > 0 && characters > 0 ? (loss * tokens) / characters / Math.LN2 : NaN;
}

/** Tracks the best validation loss over the epochs and says when to stop. */
export class EarlyStopping {
  private bestLoss = Infinity;
  private bestEpoch = 0;
  private lastEpoch = 0;

  /** @param patience epochs without improvement before stopping; 0 never stops */
  constructor(private readonly patience: number) {}

  /** Record the validation loss of `epoch` (1-based). Returns true when it is the best so far. */
  record(epoch: number, loss: number): boolean {
    this.lastEpoch = epoch;
    if (Number.isFinite(loss) && loss < this.bestLoss) {
      this.bestLoss = loss;
      this.bestEpoch = epoch;
      return true;
    }
    return false;
  }

  shouldStop(): boolean {
    return (
      this.patience > 0 && this.bestEpoch > 0 && this.lastEpoch - this.bestEpoch >= this.patience
    );
  }

  get best(): { epoch: number; loss: number } | null {
    return this.bestEpoch > 0 ? { epoch: this.bestEpoch, loss: this.bestLoss } : null;
  }
}
