import fs from 'node:fs/promises';
import path from 'node:path';

import {
  compareProjections,
  NormalisationMode,
  ProjectionResult
} from '../src/visualization/embeddings';

type VisualizeArgs = {
  model: string;
  outputDir: string;
  normalise?: NormalisationMode;
};

type ModelExport = {
  vocab: string[];
  embedding: number[][];
};

async function readModel(modelPath: string): Promise<ModelExport> {
  const payload = await fs.readFile(modelPath, 'utf8');
  const data = JSON.parse(payload);
  if (!Array.isArray(data.embedding)) {
    throw new Error(`Model at ${modelPath} does not contain an embedding matrix.`);
  }
  return { vocab: data.vocab, embedding: data.embedding };
}

function summarise(result: ProjectionResult, vocab: string[]) {
  return {
    ...result,
    vocabularySample: vocab.slice(0, 10)
  };
}

async function visualizeEmbedding({ model, outputDir, normalise = 'zscore' }: VisualizeArgs) {
  const { vocab, embedding } = await readModel(model);
  const projections = compareProjections(
    embedding,
    { normalise, iterations: 500 },
    { normalise, nNeighbors: 8 }
  );

  const baseName = path.basename(model, path.extname(model));
  const tsnePath = path.join(outputDir, `${baseName}-tsne.json`);
  const umapPath = path.join(outputDir, `${baseName}-umap.json`);

  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(tsnePath, JSON.stringify(summarise(projections.tsne, vocab), null, 2), 'utf8');
  await fs.writeFile(umapPath, JSON.stringify(summarise(projections.umap, vocab), null, 2), 'utf8');
  console.log(`Saved projections to ${tsnePath} and ${umapPath}`);
}

async function main() {
  const model = process.argv[2];
  if (!model) {
    console.error('Usage: ts-node scripts/visualize_embeddings.ts <model_path> [output_dir]');
    process.exit(1);
  }
  const outputDir = process.argv[3] ?? path.join('docs', 'visuals');
  await visualizeEmbedding({ model, outputDir });
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
