/**
 * BPE Tokenizer Implementation
 * Browser-native Byte Pair Encoding with training, versioning, and metrics
 */

import {
  BPEConfig,
  BPEArtifact,
  BPEMergeRule,
  VocabEntry,
  TokenizerMetrics,
  BPEProgressCallback,
  EncodingResult,
  DecodingOptions,
  DEFAULT_BPE_CONFIG
} from '../types/tokenizer';

async function sha256(message: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

function quickHash(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export class BPETokenizer {
  private config: BPEConfig;
  private merges: BPEMergeRule[] = [];
  private vocab: Map<string, number> = new Map();
  private vocabToId: Map<string, number> = new Map();
  private idToVocab: Map<number, string> = new Map();
  private mergeRank: Map<string, number> = new Map();
  private trained: boolean = false;
  private artifactHash: string = '';
  private trainingStats = { totalTokens: 0, uniqueWords: 0, corpusHash: '' };

  constructor(config: Partial<BPEConfig> = {}) {
    this.config = { ...DEFAULT_BPE_CONFIG, ...config };
    this.initializeSpecialTokens();
  }

  private initializeSpecialTokens(): void {
    let id = 0;
    for (const token of this.config.specialTokens) {
      this.vocabToId.set(token, id);
      this.idToVocab.set(id, token);
      this.vocab.set(token, 0);
      id++;
    }
  }

  async train(corpus: string, onProgress?: BPEProgressCallback): Promise<BPEArtifact> {
    const startTime = performance.now();
    this.trainingStats.corpusHash = await sha256(corpus);

    const wordFreqs = this.preTokenize(corpus);
    this.trainingStats.uniqueWords = wordFreqs.size;
    this.trainingStats.totalTokens = Array.from(wordFreqs.values()).reduce((a, b) => a + b, 0);

    const wordSplits = this.initializeWordSplits(wordFreqs);

    const baseVocabSize = this.vocabToId.size;
    const targetMerges = Math.max(
      0,
      this.config.vocabSize - baseVocabSize - this.getUniqueChars(wordSplits).size
    );

    let mergeCount = 0;

    while (mergeCount < targetMerges) {
      const pairFreqs = this.countPairs(wordSplits, wordFreqs);
      if (pairFreqs.size === 0) break;

      const bestPair = this.findBestPair(pairFreqs);
      if (!bestPair || pairFreqs.get(bestPair)! < this.config.minFrequency) break;

      const [left, right] = bestPair.split(' ');
      const merged = left + right;

      const rule: BPEMergeRule = { left, right, merged, priority: mergeCount };
      this.merges.push(rule);
      this.mergeRank.set(bestPair, mergeCount);

      const newId = this.vocabToId.size;
      this.vocabToId.set(merged, newId);
      this.idToVocab.set(newId, merged);
      this.vocab.set(merged, pairFreqs.get(bestPair)!);

      this.applyMerge(wordSplits, left, right, merged);
      mergeCount++;

      if (onProgress && mergeCount % 50 === 0) {
        onProgress({
          currentMerges: mergeCount,
          targetMerges,
          currentVocabSize: this.vocabToId.size,
          targetVocabSize: this.config.vocabSize,
          lastMerge: rule,
          elapsedMs: performance.now() - startTime
        });
      }
    }

    for (const char of this.getUniqueChars(wordSplits)) {
      if (!this.vocabToId.has(char)) {
        const id = this.vocabToId.size;
        this.vocabToId.set(char, id);
        this.idToVocab.set(id, char);
        this.vocab.set(char, 0);
      }
    }

    this.trained = true;
    return this.buildArtifact();
  }

  private preTokenize(corpus: string): Map<string, number> {
    const wordFreqs = new Map<string, number>();
    const words = corpus
      .toLowerCase()
      .split(/(\s+)/)
      .filter((w) => w.trim().length > 0);
    for (const word of words) {
      const markedWord = '_' + word;
      wordFreqs.set(markedWord, (wordFreqs.get(markedWord) || 0) + 1);
    }
    return wordFreqs;
  }

  private initializeWordSplits(wordFreqs: Map<string, number>): Map<string, string[]> {
    const splits = new Map<string, string[]>();
    for (const word of wordFreqs.keys()) {
      splits.set(word, [...word]);
    }
    return splits;
  }

  private getUniqueChars(wordSplits: Map<string, string[]>): Set<string> {
    const chars = new Set<string>();
    for (const split of wordSplits.values()) {
      for (const char of split) chars.add(char);
    }
    return chars;
  }

  private countPairs(
    wordSplits: Map<string, string[]>,
    wordFreqs: Map<string, number>
  ): Map<string, number> {
    const pairFreqs = new Map<string, number>();
    for (const [word, split] of wordSplits) {
      const freq = wordFreqs.get(word) || 0;
      for (let i = 0; i < split.length - 1; i++) {
        const pair = `${split[i]} ${split[i + 1]}`;
        pairFreqs.set(pair, (pairFreqs.get(pair) || 0) + freq);
      }
    }
    return pairFreqs;
  }

  private findBestPair(pairFreqs: Map<string, number>): string | null {
    let bestPair: string | null = null;
    let bestFreq = 0;
    for (const [pair, freq] of pairFreqs) {
      if (freq > bestFreq) {
        bestFreq = freq;
        bestPair = pair;
      }
    }
    return bestPair;
  }

  private applyMerge(
    wordSplits: Map<string, string[]>,
    left: string,
    right: string,
    merged: string
  ): void {
    for (const [word, split] of wordSplits) {
      const newSplit: string[] = [];
      let i = 0;
      while (i < split.length) {
        if (i < split.length - 1 && split[i] === left && split[i + 1] === right) {
          newSplit.push(merged);
          i += 2;
        } else {
          newSplit.push(split[i]);
          i++;
        }
      }
      wordSplits.set(word, newSplit);
    }
  }

  private async buildArtifact(): Promise<BPEArtifact> {
    const vocabEntries: VocabEntry[] = [];
    for (const [token, freq] of this.vocab) {
      vocabEntries.push({
        token,
        frequency: freq,
        isMerge: this.merges.some((m) => m.merged === token)
      });
    }

    const hashInput = JSON.stringify({
      merges: this.merges,
      vocab: Array.from(this.vocabToId.entries())
    });
    this.artifactHash = await sha256(hashInput);

    return {
      version: '1.0.0',
      hash: this.artifactHash,
      createdAt: new Date().toISOString(),
      config: this.config,
      merges: this.merges,
      vocab: vocabEntries,
      vocabToId: Object.fromEntries(this.vocabToId),
      idToVocab: Object.fromEntries(this.idToVocab),
      trainingCorpusStats: this.trainingStats
    };
  }

  static fromArtifact(artifact: BPEArtifact): BPETokenizer {
    const tokenizer = new BPETokenizer(artifact.config);
    tokenizer.merges = artifact.merges;
    tokenizer.vocabToId = new Map(Object.entries(artifact.vocabToId));
    tokenizer.idToVocab = new Map(
      Object.entries(artifact.idToVocab).map(([k, v]) => [parseInt(k), v])
    );
    for (const entry of artifact.vocab) {
      tokenizer.vocab.set(entry.token, entry.frequency);
    }
    for (const merge of artifact.merges) {
      tokenizer.mergeRank.set(`${merge.left} ${merge.right}`, merge.priority);
    }
    tokenizer.trained = true;
    tokenizer.artifactHash = artifact.hash;
    tokenizer.trainingStats = artifact.trainingCorpusStats;
    return tokenizer;
  }

  encode(text: string): EncodingResult {
    if (!this.trained) throw new Error('Tokenizer not trained. Call train() first.');
    const tokens: string[] = [];
    const ids: number[] = [];
    const offsets: Array<[number, number]> = [];
    const words = text.toLowerCase().split(/(\s+)/);
    let charOffset = 0;

    for (const word of words) {
      if (word.trim().length === 0) {
        charOffset += word.length;
        continue;
      }
      const markedWord = '_' + word;
      const wordTokens = this.tokenizeWord(markedWord);

      for (const token of wordTokens) {
        const cleanToken = token.startsWith('_') ? token.slice(1) : token;
        tokens.push(token);
        const id = this.vocabToId.get(token) ?? this.vocabToId.get(this.config.unknownToken)!;
        ids.push(id);
        const tokenLen = cleanToken.length;
        offsets.push([charOffset, charOffset + tokenLen]);
        charOffset += tokenLen;
      }
      charOffset += 1;
    }
    return { ids, tokens, offsets };
  }

  private tokenizeWord(word: string): string[] {
    let tokens = [...word];
    let changed = true;
    while (changed) {
      changed = false;
      let bestIdx = -1;
      let bestPriority = Infinity;
      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${tokens[i]} ${tokens[i + 1]}`;
        const priority = this.mergeRank.get(pair);
        if (priority !== undefined && priority < bestPriority) {
          bestIdx = i;
          bestPriority = priority;
        }
      }
      if (bestIdx >= 0) {
        const merged = tokens[bestIdx] + tokens[bestIdx + 1];
        tokens = [...tokens.slice(0, bestIdx), merged, ...tokens.slice(bestIdx + 2)];
        changed = true;
      }
    }
    return tokens;
  }

  decode(ids: number[], options: Partial<DecodingOptions> = {}): string {
    const opts: DecodingOptions = { skipSpecialTokens: true, cleanupSpaces: true, ...options };
    const tokens: string[] = [];
    for (const id of ids) {
      const token = this.idToVocab.get(id);
      if (!token) continue;
      if (opts.skipSpecialTokens && this.config.specialTokens.includes(token)) continue;
      tokens.push(token);
    }
    let text = tokens.join('').replace(/_/g, ' ');
    if (opts.cleanupSpaces) text = text.replace(/\s+/g, ' ').trim();
    return text;
  }

  get vocabSize(): number {
    return this.vocabToId.size;
  }
  get isTrained(): boolean {
    return this.trained;
  }
  get hash(): string {
    return this.artifactHash;
  }

  calculateMetrics(testCorpus: string): TokenizerMetrics {
    if (!this.trained) throw new Error('Tokenizer not trained.');
    const encoding = this.encode(testCorpus);
    const words = testCorpus
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 0);
    const totalChars = testCorpus.length;
    const unkId = this.vocabToId.get(this.config.unknownToken)!;
    const unkCount = encoding.ids.filter((id) => id === unkId).length;
    const usedTokens = new Set(encoding.ids);
    const tokenCounts = new Map<number, number>();
    for (const id of encoding.ids) {
      tokenCounts.set(id, (tokenCounts.get(id) || 0) + 1);
    }
    let entropy = 0;
    const totalTokens = encoding.ids.length;
    for (const count of tokenCounts.values()) {
      const p = count / totalTokens;
      entropy -= p * Math.log2(p);
    }
    return {
      coverage: 1 - unkCount / totalTokens,
      entropy,
      fertility: totalTokens / words.length,
      compressionRatio: totalChars / totalTokens,
      vocabUtilization: usedTokens.size / this.vocabSize,
      unknownRate: unkCount / totalTokens
    };
  }

  async toJSON(): Promise<string> {
    const artifact = await this.buildArtifact();
    return JSON.stringify(artifact, null, 2);
  }

  static fromJSON(json: string): BPETokenizer {
    const artifact: BPEArtifact = JSON.parse(json);
    return BPETokenizer.fromArtifact(artifact);
  }

  tokenToId(token: string): number | undefined {
    return this.vocabToId.get(token);
  }
  idToToken(id: number): string | undefined {
    return this.idToVocab.get(id);
  }

  get specialTokenIds(): Record<string, number> {
    return {
      pad: this.vocabToId.get(this.config.padToken)!,
      bos: this.vocabToId.get(this.config.bosToken)!,
      eos: this.vocabToId.get(this.config.eosToken)!,
      unk: this.vocabToId.get(this.config.unknownToken)!
    };
  }

  async verifyArtifact(artifact: BPEArtifact): Promise<boolean> {
    const hashInput = JSON.stringify({
      merges: artifact.merges,
      vocab: Object.entries(artifact.vocabToId)
    });
    const computedHash = await sha256(hashInput);
    return computedHash === artifact.hash;
  }
}

export { quickHash, sha256 };
