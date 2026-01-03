/**
 * Complete Triadic Operator Table
 *
 * This file contains the complete table with the 𝕋 operator for each cell.
 * The table maps various domains (Digital Foundation, Boolean Logic, etc.)
 * against different conceptual frameworks.
 *
 * Each cell contains three concepts (A, B, C) with their emoji representations,
 * representing a triadic relationship that can be computed using the 𝕋 operator.
 */

import type {
  TriadicTable,
  TriadicDomain,
  TriadicColumnGroup,
  TriadicCell,
} from '../types/triadic';

/**
 * All column groups in the triadic table
 */
export const TRIADIC_COLUMNS: TriadicColumnGroup[] = [
  {
    id: 'col1',
    nameEn: 'Noise → Regulation → Control',
    nameHe: '🔊 רעש → ⚙️ ויסות → 🎛️ שליטה',
    index: 0,
  },
  {
    id: 'col2',
    nameEn: 'Variation → Selection → Retention',
    nameHe: '🔀 שונות → 🧬 ברירה → 🗂️ שימור',
    index: 1,
  },
  {
    id: 'col3',
    nameEn: 'Fluctuation → Order Parameter → Slaving',
    nameHe: '🌊 תנודה → 🎚️ פרמטר סדר → 🧲 שעבוד',
    index: 2,
  },
  {
    id: 'col4',
    nameEn: 'Disorder → Interaction → Organization',
    nameHe: '🌪️ אי-סדר → 🤝 אינטראקציה → 🏗️ ארגון',
    index: 3,
  },
  {
    id: 'col5',
    nameEn: 'Oscillation → Interference → Resonant Attractor',
    nameHe: '🔄 תנודה → 💥 התערבות → 🎯 אטרקטור תהודה',
    index: 4,
  },
  {
    id: 'col6',
    nameEn: 'Rhythm → Harmony → Emergence',
    nameHe: '🥁 קצב → 🎶 הרמוניה → ✨ הופעה',
    index: 5,
  },
  {
    id: 'col7',
    nameEn: 'Drive → Constraint → Mediation',
    nameHe: '🚀 דחף → 🧱 אילוץ → 🤝 תיווך',
    index: 6,
  },
  {
    id: 'col8',
    nameEn: 'Explore → Exploit → Policy',
    nameHe: '🧭 חקירה → 💎 ניצול → 📜 מדיניות',
    index: 7,
  },
  {
    id: 'col9',
    nameEn: 'Prediction → Error → Model Update',
    nameHe: '🔮 תחזית → ❌ שגיאה → 🔁 עדכון מודל',
    index: 8,
  },
  {
    id: 'col10',
    nameEn: 'Signal → Code → Interpretation',
    nameHe: '📡 אות → 💻 קוד → 🧠 פרשנות',
    index: 9,
  },
  {
    id: 'col11',
    nameEn: 'Coordination → Alignment → Mandate',
    nameHe: '👥 תיאום → 🧲 יישור → 🏛️ מנדט',
    index: 10,
  },
  {
    id: 'col12',
    nameEn: 'Contribution → Interoperability → Infrastructure',
    nameHe: '🧩 תרומה → 🔌 יכולת פעולה → 🏗️ תשתית',
    index: 11,
  },
  {
    id: 'col13',
    nameEn: 'Boundary → State → Transition',
    nameHe: '🚧 גבול → 📍 מצב → 🔀 מעבר',
    index: 12,
  },
];

/**
 * All domains (rows) in the triadic table
 */
export const TRIADIC_DOMAINS: TriadicDomain[] = [
  {
    id: 'digital',
    nameEn: 'Digital Foundation',
    nameHe: 'יסוד דיגיטלי',
    emoji: '💻',
    cells: [
      { a: 'noise', b: 'protocol', c: 'system', emojiA: '📊', emojiB: '📑', emojiC: '🎛️' },
      { a: 'bits', b: 'encoding', c: 'data', emojiA: '🔢', emojiB: '🧬', emojiC: '🗄️' },
      { a: 'packets', b: 'bandwidth', c: 'traffic', emojiA: '📦', emojiB: '📶', emojiC: '🚦' },
      { a: 'entropy', b: 'compression', c: 'storage', emojiA: '🌫️', emojiB: '🗜️', emojiC: '🧱' },
      { a: 'clock', b: 'sync', c: 'timing', emojiA: '⏱️', emojiB: '🔄', emojiC: '🎯' },
      { a: 'data', b: 'flow', c: 'architecture', emojiA: '🌊', emojiB: '🎼', emojiC: '🏙️' },
      { a: 'compute', b: 'resource', c: 'load', emojiA: '⚡', emojiB: '🧱', emojiC: '⚖️' },
      { a: 'topology', b: 'protocol', c: 'infra', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'traffic', b: 'error', c: 'routing', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'binary', b: 'opcode', c: 'execute', emojiA: '🔢', emojiB: '💻', emojiC: '▶️' },
      { a: 'nodes', b: 'consensus', c: 'chain', emojiA: '🧑‍🤝‍🧑', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'hardware', b: 'API', c: 'cloud', emojiA: '🧩', emojiB: '🔌', emojiC: '☁️' },
      { a: 'boundary', b: 'state', c: 'error', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'logic',
    nameEn: 'Boolean Logic',
    nameHe: 'לוגיקה בוליאנית',
    emoji: '🔢',
    cells: [
      { a: 'input', b: 'gate', c: 'output', emojiA: '🔊', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'logic', b: 'table', c: 'ops', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'proposition', b: 'axiom', c: 'theorem', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'ambiguity', b: 'deduction', c: 'proof', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'toggle', b: 'negation', c: 'fixed', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'switch', b: 'combo', c: 'seq', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'truth', b: 'constraint', c: 'SAT', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'expr', b: 'simplify', c: 'reduce', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'outcome', b: 'contradiction', c: 'axiom', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'input', b: 'function', c: 'truth', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'ops', b: 'precedence', c: 'eval', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'gate', b: 'circuit', c: 'logic', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'var', b: 'assign', c: 'compute', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'questions',
    nameEn: 'Fundamental Questions',
    nameHe: 'שאלות יסוד',
    emoji: '❓',
    cells: [
      { a: 'epistemic', b: 'method', c: 'knowledge', emojiA: '🌫️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'hypothesis', b: 'evidence', c: 'paradigm', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'drift', b: 'framing', c: 'explain', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'concept', b: 'dialogue', c: 'philosophy', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'inquiry', b: 'debate', c: 'consensus', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'curiosity', b: 'insight', c: 'discovery', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'search', b: 'rigor', c: 'validity', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'unknown', b: 'answer', c: 'research', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'implication', b: 'falsify', c: 'theory', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'phenomenon', b: 'question', c: 'meaning', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'thinkers', b: 'discourse', c: 'canon', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'idea', b: 'citation', c: 'knowledge', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'ignorance', b: 'understanding', c: 'paradigm', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'governance',
    nameEn: 'Governance & Organization',
    nameHe: 'ממשל וארגון',
    emoji: '🏛️',
    cells: [
      { a: 'conflict', b: 'rules', c: 'power', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'policy', b: 'vote', c: 'institution', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'authority', b: 'leadership', c: 'hierarchy', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'institution', b: 'negotiation', c: 'authority', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'faction', b: 'coalition', c: 'gov', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'decision', b: 'process', c: 'policy', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'power', b: 'law', c: 'compromise', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'structure', b: 'mechanism', c: 'governance', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'outcome', b: 'failure', c: 'reform', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'citizen', b: 'law', c: 'rights', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'actors', b: 'roles', c: 'constitution', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'resource', b: 'division', c: 'org', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'jurisdiction', b: 'sovereignty', c: 'regime', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'standards',
    nameEn: 'Standards & Regulation',
    nameHe: 'תקנים ורגולציה',
    emoji: '📏',
    cells: [
      { a: 'market', b: 'spec', c: 'compliance', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'format', b: 'adopt', c: 'standard', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'requirement', b: 'committee', c: 'version', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'practice', b: 'harmonize', c: 'regulation', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'proposal', b: 'lobby', c: 'ratify', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'update', b: 'align', c: 'ecosystem', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'compliance', b: 'penalty', c: 'enforce', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'option', b: 'consensus', c: 'standardize', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'impact', b: 'violation', c: 'revise', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'requirement', b: 'clause', c: 'legal', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'body', b: 'jurisdiction', c: 'mandate', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'stakeholder', b: 'conform', c: 'standards', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'scope', b: 'compliant', c: 'deregulate', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'execution',
    nameEn: 'Execution & Implementation',
    nameHe: 'ביצוע ויישום',
    emoji: '⚙️',
    cells: [
      { a: 'resource', b: 'schedule', c: 'process', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'approach', b: 'priority', c: 'workflow', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'task', b: 'bottleneck', c: 'dependency', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'plan', b: 'coordinate', c: 'project', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'cycle', b: 'feedback', c: 'execution', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'operation', b: 'sync', c: 'outcome', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'urgency', b: 'deadline', c: 'resource', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'method', b: 'tool', c: 'implement', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'delay', b: 'deviation', c: 'plan', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'instruction', b: 'procedure', c: 'result', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'team', b: 'roles', c: 'delivery', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'effort', b: 'integrate', c: 'ops', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'start', b: 'progress', c: 'complete', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'measurement',
    nameEn: 'Measurement & Control',
    nameHe: 'מדידה ובקרה',
    emoji: '📐',
    cells: [
      { a: 'sensor', b: 'filter', c: 'feedback', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'metric', b: 'calibrate', c: 'reference', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'variable', b: 'setpoint', c: 'actuator', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'system', b: 'correction', c: 'stable', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'deviation', b: 'damping', c: 'equilibrium', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'sampling', b: 'loop', c: 'control', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'performance', b: 'limit', c: 'regulate', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'range', b: 'tolerance', c: 'policy', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'behavior', b: 'error', c: 'parameter', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'observation', b: 'scale', c: 'value', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'component', b: 'hierarchy', c: 'command', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'instrument', b: 'protocol', c: 'control', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'threshold', b: 'regulated', c: 'failure', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'monitoring',
    nameEn: 'Monitoring & Response',
    nameHe: 'ניטור ותגובה',
    emoji: '🚨',
    cells: [
      { a: 'event', b: 'alert', c: 'incident', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'anomaly', b: 'threshold', c: 'pattern', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'metric', b: 'baseline', c: 'alert', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'chaos', b: 'triage', c: 'resolve', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'alert', b: 'escalate', c: 'response', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'pulse', b: 'dashboard', c: 'situation', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'vigilance', b: 'protocol', c: 'action', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'incident', b: 'signal', c: 'response', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'threat', b: 'false+', c: 'rule', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'log', b: 'signature', c: 'threat', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'team', b: 'chain', c: 'mandate', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'data', b: 'tool', c: 'monitoring', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'normal', b: 'alert', c: 'recover', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'learning',
    nameEn: 'Learning & Improvement',
    nameHe: 'למידה ושיפור',
    emoji: '📚',
    cells: [
      { a: 'data', b: 'preprocess', c: 'train', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'feature', b: 'validate', c: 'model', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'performance', b: 'hyperparam', c: 'gradient', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'error', b: 'optimize', c: 'converge', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'epoch', b: 'regularize', c: 'optimum', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'iteration', b: 'converge', c: 'capability', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'curiosity', b: 'loss', c: 'learn', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'hypothesis', b: 'data', c: 'train', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'failure', b: 'analyze', c: 'refine', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'example', b: 'represent', c: 'knowledge', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'agent', b: 'curriculum', c: 'mandate', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'experience', b: 'transfer', c: 'knowledge', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'prior', b: 'posterior', c: 'insight', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'interface',
    nameEn: 'Interface & Experience',
    nameHe: 'ממשק וחוויה',
    emoji: '🎨',
    cells: [
      { a: 'input', b: 'validate', c: 'interact', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'design', b: 'usability', c: 'pattern', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'focus', b: 'affordance', c: 'attention', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'clutter', b: 'layout', c: 'intuitive', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'animation', b: 'overlap', c: 'coherence', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'flow', b: 'aesthetic', c: 'engage', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'user', b: 'access', c: 'afford', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'interaction', b: 'convention', c: 'design', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'behavior', b: 'frustration', c: 'iterate', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'user', b: 'UI', c: 'experience', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'element', b: 'hierarchy', c: 'nav', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'component', b: 'framework', c: 'interface', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'edge', b: 'focus', c: 'modal', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'rights',
    nameEn: 'Human Rights',
    nameHe: 'זכויות אדם',
    emoji: '⚖️',
    cells: [
      { a: 'violation', b: 'report', c: 'justice', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'claim', b: 'adjudicate', c: 'precedent', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'demand', b: 'principle', c: 'enforce', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'oppression', b: 'advocacy', c: 'rights', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'protest', b: 'repress', c: 'protect', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'struggle', b: 'solidarity', c: 'dignity', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'dignity', b: 'prohibit', c: 'remedy', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'freedom', b: 'protect', c: 'rights', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'abuse', b: 'violate', c: 'norm', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'grievance', b: 'article', c: 'right', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'society', b: 'align', c: 'law', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'individual', b: 'treaty', c: 'rights', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'restrict', b: 'protected', c: 'free', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'geopolitics',
    nameEn: 'Geopolitics',
    nameHe: 'גיאופוליטיקה',
    emoji: '🌍',
    cells: [
      { a: 'intel', b: 'diplomacy', c: 'influence', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'strategy', b: 'alliance', c: 'bloc', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'power', b: 'hegemon', c: 'vassal', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'conflict', b: 'negotiate', c: 'order', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'crisis', b: 'deter', c: 'balance', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'escalate', b: 'détente', c: 'stability', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'ambition', b: 'treaty', c: 'sphere', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'option', b: 'leverage', c: 'foreign', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'move', b: 'miscalc', c: 'strategy', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'event', b: 'narrative', c: 'interest', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'state', b: 'bloc', c: 'security', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'nation', b: 'interop', c: 'geo', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'border', b: 'sovereign', c: 'shift', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'commons',
    nameEn: 'Digital Commons',
    nameHe: 'נחלת הכלל הדיגיטלית',
    emoji: '🌐',
    cells: [
      { a: 'access', b: 'moderate', c: 'community', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'content', b: 'curate', c: 'repo', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'usage', b: 'govern', c: 'contribute', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'fragment', b: 'collaborate', c: 'commons', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'fork', b: 'merge', c: 'project', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'contribute', b: 'synergy', c: 'collective', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'share', b: 'license', c: 'permit', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'resource', b: 'fork', c: 'commons', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'sustain', b: 'tragedy', c: 'govern', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'create', b: 'format', c: 'share', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'member', b: 'norm', c: 'commons', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'user', b: 'protocol', c: 'digital', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'exclude', b: 'include', c: 'commons', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
  {
    id: 'evolution',
    nameEn: 'Evolution & Adaptation',
    nameHe: 'אבולוציה והתאמה',
    emoji: '🧬',
    cells: [
      { a: 'genetic', b: 'select', c: 'adapt', emojiA: '🌪️', emojiB: '⚙️', emojiC: '🎛️' },
      { a: 'phenotype', b: 'env', c: 'trait', emojiA: '🔀', emojiB: '🧬', emojiC: '🗂️' },
      { a: 'population', b: 'fitness', c: 'dominance', emojiA: '🌊', emojiB: '🎚️', emojiC: '🧲' },
      { a: 'ecology', b: 'species', c: 'ecosystem', emojiA: '🌪️', emojiB: '🤝', emojiC: '🏗️' },
      { a: 'predator', b: 'compete', c: 'cycle', emojiA: '🔄', emojiB: '💥', emojiC: '🎯' },
      { a: 'metabolism', b: 'symbiosis', c: 'speciate', emojiA: '🥁', emojiB: '🎶', emojiC: '✨' },
      { a: 'survive', b: 'resource', c: 'niche', emojiA: '🚀', emojiB: '🧱', emojiC: '🤝' },
      { a: 'mutate', b: 'niche', c: 'evolve', emojiA: '🧭', emojiB: '💎', emojiC: '📜' },
      { a: 'adapt', b: 'select', c: 'genome', emojiA: '🔮', emojiB: '❌', emojiC: '🔁' },
      { a: 'cue', b: 'epigenetic', c: 'develop', emojiA: '📡', emojiB: '💻', emojiC: '🧠' },
      { a: 'species', b: 'coevolve', c: 'nature', emojiA: '👥', emojiB: '🧲', emojiC: '🏛️' },
      { a: 'trait', b: 'gene-flow', c: 'pool', emojiA: '🧩', emojiB: '🔌', emojiC: '🏗️' },
      { a: 'reproduce', b: 'speciate', c: 'radiate', emojiA: '🚧', emojiB: '📍', emojiC: '🔀' },
    ],
  },
];

/**
 * The complete triadic operator table
 */
export const TRIADIC_TABLE: TriadicTable = {
  columns: TRIADIC_COLUMNS,
  domains: TRIADIC_DOMAINS,
  metadata: {
    version: '1.0.0',
    description:
      'Complete triadic operator table mapping domains to conceptual frameworks using NAND-based logic',
  },
};

/**
 * Get a specific cell from the table
 */
export function getTriadicCell(domainId: string, columnIndex: number): TriadicCell | null {
  const domain = TRIADIC_DOMAINS.find((d) => d.id === domainId);
  if (!domain || columnIndex < 0 || columnIndex >= domain.cells.length) {
    return null;
  }
  return domain.cells[columnIndex];
}

/**
 * Get all cells for a specific domain
 */
export function getDomainCells(domainId: string): TriadicCell[] {
  const domain = TRIADIC_DOMAINS.find((d) => d.id === domainId);
  return domain?.cells ?? [];
}

/**
 * Get all cells for a specific column
 */
export function getColumnCells(columnIndex: number): TriadicCell[] {
  return TRIADIC_DOMAINS.map((domain) => domain.cells[columnIndex]).filter(Boolean);
}
