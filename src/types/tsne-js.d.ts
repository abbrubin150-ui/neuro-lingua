declare module 'tsne-js' {
  export interface TSNEInitOptions {
    dim?: number;
    perplexity?: number;
    earlyExaggeration?: number;
    learningRate?: number;
    nIter?: number;
    metric?: string;
  }

  export interface TSNEInitData {
    data: number[][];
    type: 'dense' | 'sparse';
  }

  export default class TSNE {
    constructor(options?: TSNEInitOptions);
    init(config: TSNEInitData): void;
    run(): void;
    getOutput(): number[][];
    getOutputScaled(): number[][];
  }
}
