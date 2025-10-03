import fs from 'node:fs';

console.log('Training script placeholder. Implement training pipeline here.');
const corpus = fs.readFileSync(new URL('../data/corpus.txt', import.meta.url), 'utf8');
console.log(`Loaded corpus with ${corpus.split(/\s+/).filter(Boolean).length} tokens.`);
