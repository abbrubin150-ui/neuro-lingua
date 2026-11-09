/**
 * Node-only helpers of the model factory: read local corpus files and model packages.
 * (Not imported by the browser app.)
 */

import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { parseDocuments, type CorpusDocument } from '../../data/corpusPrep';
import { qaDocument, type QaTemplate } from '../../data/qa';
import { isModelPackage, restoreModel, type ModelPackage } from '../../lib/modelPackage';
import { parseModelImport } from '../../lib/modelImport';
import type { ProNeuralLM } from '../../lib/ProNeuralLM';

export const CORPUS_EXTENSIONS = [
  '.txt',
  '.text',
  '.md',
  '.markdown',
  '.jsonl',
  '.ndjson',
  '.json',
  '.csv'
];

const SKIPPED_DIRECTORIES = new Set(['node_modules', '.git', 'dist', 'build']);

/** Recursively list supported corpus files under the given files / directories (sorted). */
export async function listCorpusFiles(inputs: string[]): Promise<string[]> {
  const files: string[] = [];
  const visit = async (p: string) => {
    const info = await stat(p);
    if (info.isDirectory()) {
      const name = path.basename(p);
      if (SKIPPED_DIRECTORIES.has(name) || (name.startsWith('.') && name !== '.')) return;
      const entries = await readdir(p);
      for (const entry of entries.sort()) await visit(path.join(p, entry));
    } else if (CORPUS_EXTENSIONS.includes(path.extname(p).toLowerCase())) {
      files.push(p);
    }
  };
  for (const input of inputs) await visit(path.resolve(input));
  return [...new Set(files)].sort();
}

/** Read and parse every corpus file into documents. */
/**
 * Read the corpus files under `inputs`. With `options.qa` every file is read as question–answer
 * pairs (one document of formatted pairs per file, see src/data/qa.ts).
 */
export async function readCorpusDocuments(
  inputs: string[],
  options: { qa?: QaTemplate } = {}
): Promise<{ documents: CorpusDocument[]; files: string[]; pairs: number }> {
  const files = await listCorpusFiles(inputs);
  const documents: CorpusDocument[] = [];
  let pairs = 0;
  for (const file of files) {
    const content = await readFile(file, 'utf8');
    if (options.qa) {
      const doc = qaDocument(file, content, options.qa);
      pairs += doc.pairs;
      if (doc.pairs > 0) documents.push({ id: doc.id, text: doc.text, format: doc.format });
    } else {
      documents.push(...parseDocuments(file, content));
    }
  }
  return { documents, files, pairs };
}

/** Load a model from a package file or from a plain `toJSON()` export. */
export async function loadModelFile(
  file: string
): Promise<{ model: ProNeuralLM; pkg?: ModelPackage }> {
  // Packages (checksum verified), plain `toJSON()` exports and trace exports from the app.
  const data: unknown = JSON.parse(await readFile(file, 'utf8'));
  const parsed = parseModelImport(data);
  const model = restoreModel(parsed.modelData, parsed.architecture);
  return isModelPackage(data) ? { model, pkg: data } : { model };
}
