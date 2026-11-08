/**
 * Node-only helpers of the model factory: read local corpus files and model packages.
 * (Not imported by the browser app.)
 */

import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { parseDocuments, type CorpusDocument } from '../../data/corpusPrep';
import { isModelPackage, loadModelFromPackage, type ModelPackage } from '../../lib/modelPackage';
import { ProNeuralLM } from '../../lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../lib/AdvancedNeuralLM';
import { TransformerLM } from '../../lib/TransformerLM';

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
export async function readCorpusDocuments(
  inputs: string[]
): Promise<{ documents: CorpusDocument[]; files: string[] }> {
  const files = await listCorpusFiles(inputs);
  const documents: CorpusDocument[] = [];
  for (const file of files) {
    const content = await readFile(file, 'utf8');
    documents.push(...parseDocuments(file, content));
  }
  return { documents, files };
}

/** Load a model from a package file or from a plain `toJSON()` export. */
export async function loadModelFile(
  file: string
): Promise<{ model: ProNeuralLM; pkg?: ModelPackage }> {
  const data = JSON.parse(await readFile(file, 'utf8'));
  if (isModelPackage(data)) return { model: loadModelFromPackage(data), pkg: data };
  if (data?.architecture === 'transformer') return { model: TransformerLM.fromJSON(data) };
  if (data?.architecture === 'advanced') return { model: AdvancedNeuralLM.fromJSON(data) };
  return { model: ProNeuralLM.fromJSON(data) };
}
