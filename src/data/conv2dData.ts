/**
 * 2D Convolution Matrix Data
 *
 * Complete data for 14 triads × 14 digital layers = 196 cells
 *
 * Y-axis: Triads (T1-T14)
 * X-axis: Digital Layers (N1-N14)
 */

import type { Triad, TriadId, DigitalLayer, LayerId, Conv2DCell, ConvDepth } from '../types/conv2d';

// ============================================================================
// Triad Definitions (T1-T14) - Y-axis
// ============================================================================

export const TRIADS: Record<TriadId, Triad> = {
  T1: {
    id: 'T1',
    nameHe: 'רעש → רגולציה → שליטה',
    nameEn: 'Noise → Regulation → Control',
    components: ['Noise', 'Regulation', 'Control'],
    rowIndex: 0
  },
  T2: {
    id: 'T2',
    nameHe: 'וריאציה → סלקציה → שימור',
    nameEn: 'Variation → Selection → Retention',
    components: ['Variation', 'Selection', 'Retention'],
    rowIndex: 1
  },
  T3: {
    id: 'T3',
    nameHe: 'תנודה → פרמטר סדר → שעבוד',
    nameEn: 'Fluctuation → Order Parameter → Slaving',
    components: ['Fluctuation', 'Order Parameter', 'Slaving'],
    rowIndex: 2
  },
  T4: {
    id: 'T4',
    nameHe: 'אי-סדר → אינטראקציה → ארגון',
    nameEn: 'Disorder → Interaction → Organization',
    components: ['Disorder', 'Interaction', 'Organization'],
    rowIndex: 3
  },
  T5: {
    id: 'T5',
    nameHe: 'תנודה → הפרעה → אטרקטור תהודה',
    nameEn: 'Oscillation → Interference → Resonant Attractor',
    components: ['Oscillation', 'Interference', 'Resonant Attractor'],
    rowIndex: 4
  },
  T6: {
    id: 'T6',
    nameHe: 'קצב → הרמוניה → צמיחה',
    nameEn: 'Rhythm → Harmony → Emergence',
    components: ['Rhythm', 'Harmony', 'Emergence'],
    rowIndex: 5
  },
  T7: {
    id: 'T7',
    nameHe: 'דחף → אילוץ → תיווך',
    nameEn: 'Drive → Constraint → Mediation',
    components: ['Drive', 'Constraint', 'Mediation'],
    rowIndex: 6
  },
  T8: {
    id: 'T8',
    nameHe: 'חקור → נצל → מדיניות',
    nameEn: 'Explore → Exploit → Policy',
    components: ['Explore', 'Exploit', 'Policy'],
    rowIndex: 7
  },
  T9: {
    id: 'T9',
    nameHe: 'חיזוי → שגיאה → עדכון מודל',
    nameEn: 'Prediction → Error → Model Update',
    components: ['Prediction', 'Error', 'Model Update'],
    rowIndex: 8
  },
  T10: {
    id: 'T10',
    nameHe: 'אות → קוד → פרשנות',
    nameEn: 'Signal → Code → Interpretation',
    components: ['Signal', 'Code', 'Interpretation'],
    rowIndex: 9
  },
  T11: {
    id: 'T11',
    nameHe: 'תיאום → יישור → מנדט',
    nameEn: 'Coordination → Alignment → Mandate',
    components: ['Coordination', 'Alignment', 'Mandate'],
    rowIndex: 10
  },
  T12: {
    id: 'T12',
    nameHe: 'תרומה → הדדיות → תשתית',
    nameEn: 'Contribution → Interoperability → Infrastructure',
    components: ['Contribution', 'Interoperability', 'Infrastructure'],
    rowIndex: 11
  },
  T13: {
    id: 'T13',
    nameHe: 'גבול → מצב → מעבר',
    nameEn: 'Boundary → State → Transition',
    components: ['Boundary', 'State', 'Transition'],
    rowIndex: 12
  },
  T14: {
    id: 'T14',
    nameHe: 'בראשית → התאמה → התעלות',
    nameEn: 'Genesis → Adaptation → Transcendence',
    components: ['Genesis', 'Adaptation', 'Transcendence'],
    rowIndex: 13
  }
};

// ============================================================================
// Digital Layer Definitions (N1-N14) - X-axis
// ============================================================================

export const LAYERS: Record<LayerId, DigitalLayer> = {
  N1: {
    id: 'N1',
    nameHe: 'תשתית דיגיטלית',
    nameEn: 'Digital Foundation',
    keywords: ['data', 'network', 'compute', 'storage', 'identity', 'security'],
    colIndex: 0
  },
  N2: {
    id: 'N2',
    nameHe: 'לוגיקה בוליאנית',
    nameEn: 'Boolean Logic',
    keywords: ['and', 'or', 'not', 'xor', 'nand', 'nor'],
    colIndex: 1
  },
  N3: {
    id: 'N3',
    nameHe: 'שאלות יסוד',
    nameEn: 'Fundamental Questions',
    keywords: ['why', 'how', 'how-much'],
    colIndex: 2
  },
  N4: {
    id: 'N4',
    nameHe: 'ממשל וארגון',
    nameEn: 'Governance & Organization',
    keywords: ['govern', 'align', 'arbitrate', 'publish'],
    colIndex: 3
  },
  N5: {
    id: 'N5',
    nameHe: 'תקנים ורגולציה',
    nameEn: 'Standards & Regulation',
    keywords: ['standardize', 'certify', 'accredit', 'legislate'],
    colIndex: 4
  },
  N6: {
    id: 'N6',
    nameHe: 'ביצוע ויישום',
    nameEn: 'Execution & Implementation',
    keywords: ['plan', 'implement', 'configure', 'deploy'],
    colIndex: 5
  },
  N7: {
    id: 'N7',
    nameHe: 'מדידה ובקרה',
    nameEn: 'Measurement & Control',
    keywords: ['measure', 'verify', 'validate', 'audit', 'log'],
    colIndex: 6
  },
  N8: {
    id: 'N8',
    nameHe: 'ניטור ותגובה',
    nameEn: 'Monitoring & Response',
    keywords: ['monitor', 'detect', 'alert', 'escalate', 'rollback'],
    colIndex: 7
  },
  N9: {
    id: 'N9',
    nameHe: 'למידה ושיפור',
    nameEn: 'Learning & Improvement',
    keywords: ['postmortem', 'root-cause', 'refactor', 'benchmark'],
    colIndex: 8
  },
  N10: {
    id: 'N10',
    nameHe: 'ממשק וחוויה',
    nameEn: 'Interface & Experience',
    keywords: ['personas', 'wireframe', 'usability', 'accessibility'],
    colIndex: 9
  },
  N11: {
    id: 'N11',
    nameHe: 'זכויות אדם',
    nameEn: 'Human Rights',
    keywords: ['dignity', 'autonomy', 'privacy', 'consent', 'safety'],
    colIndex: 10
  },
  N12: {
    id: 'N12',
    nameHe: 'גיאופוליטיקה',
    nameEn: 'Geopolitics',
    keywords: ['sovereignty', 'treaties', 'sanctions', 'cyber-norms'],
    colIndex: 11
  },
  N13: {
    id: 'N13',
    nameHe: 'נכסים דיגיטליים משותפים',
    nameEn: 'Digital Commons',
    keywords: ['open-source', 'open-data', 'open-standards'],
    colIndex: 12
  },
  N14: {
    id: 'N14',
    nameHe: 'אבולוציה והתאמה',
    nameEn: 'Evolution & Adaptation',
    keywords: ['genetic', 'phenotypic', 'speciation', 'fitness'],
    colIndex: 13
  }
};

// ============================================================================
// Convolution Matrix Content (196 cells)
// ============================================================================

/**
 * T1: Noise → Regulation → Control
 */
const T1_CELLS: string[] = [
  // N1-N7
  'Raw data noise → protocol regulation → system control',
  'Input noise → gate regulation → output control',
  'Epistemic noise → methodological regulation → knowledge control',
  'Conflicting interests → rule regulation → power control',
  'Market noise → specification regulation → compliance control',
  'Resource noise → scheduling regulation → process control',
  'Sensor noise → filtering regulation → feedback control',
  // N8-N14
  'Event noise → alerting regulation → incident control',
  'Data noise → preprocessing regulation → training control',
  'Input noise → validation regulation → interaction control',
  'Violation noise → reporting regulation → justice control',
  'Intelligence noise → diplomacy regulation → influence control',
  'Access noise → moderation regulation → community control',
  'Genetic noise → selection regulation → adaptation control'
];

/**
 * T2: Variation → Selection → Retention
 */
const T2_CELLS: string[] = [
  // N1-N7
  'Bit variation → encoding selection → data retention',
  'Logical variants → truth-table selection → stable operators',
  'Hypothesis variation → evidence selection → paradigm retention',
  'Policy variation → voting selection → institution retention',
  'Format variation → adoption selection → standard retention',
  'Approach variation → prioritization selection → workflow retention',
  'Metric variation → calibration selection → reference retention',
  // N8-N14
  'Anomaly variation → thresholding selection → pattern retention',
  'Feature variation → validation selection → model retention',
  'Design variation → usability selection → pattern retention',
  'Claim variation → adjudication selection → precedent retention',
  'Strategy variation → alliance selection → bloc retention',
  'Content variation → curation selection → repository retention',
  'Phenotypic variation → environmental selection → trait retention'
];

/**
 * T3: Fluctuation → Order Parameter → Slaving
 */
const T3_CELLS: string[] = [
  // N1-N7
  'Packet fluctuation → bandwidth parameter → traffic slaving',
  'Proposition fluctuation → axiom parameter → theorem slaving',
  'Question drift → framing parameter → explanatory dominance',
  'Authority fluctuation → leadership parameter → hierarchy slaving',
  'Requirement fluctuation → committee parameter → version slaving',
  'Task fluctuation → bottleneck parameter → dependency slaving',
  'Variable fluctuation → setpoint parameter → actuator slaving',
  // N8-N14
  'Metric fluctuation → baseline parameter → alert slaving',
  'Performance fluctuation → hyperparameter → gradient slaving',
  'Focus fluctuation → affordance parameter → attention slaving',
  'Demand fluctuation → principle parameter → enforcement slaving',
  'Power fluctuation → hegemon parameter → vassal slaving',
  'Usage fluctuation → governance parameter → contribution slaving',
  'Population fluctuation → fitness parameter → adaptive dominance'
];

/**
 * T4: Disorder → Interaction → Organization
 */
const T4_CELLS: string[] = [
  // N1-N7
  'Entropy disorder → compression interaction → structured storage',
  'Ambiguity disorder → deduction interaction → proof organization',
  'Conceptual disorder → dialogue interaction → philosophical organization',
  'Institutional disorder → negotiation interaction → organized authority',
  'Practice disorder → harmonization interaction → regulatory organization',
  'Plan disorder → coordination interaction → project organization',
  'System disorder → correction interaction → stable organization',
  // N8-N14
  'Chaos disorder → triage interaction → resolution organization',
  'Error disorder → optimization interaction → converged organization',
  'Clutter disorder → layout interaction → intuitive organization',
  'Oppression disorder → advocacy interaction → rights organization',
  'Conflict disorder → negotiation interaction → order organization',
  'Fragmentation disorder → collaboration interaction → commons organization',
  'Ecological disorder → species interaction → ecosystem organization'
];

/**
 * T5: Oscillation → Interference → Resonant Attractor
 */
const T5_CELLS: string[] = [
  // N1-N7
  'Clock oscillation → sync interference → stable timing attractor',
  'Toggle oscillation → negation interference → fixed-point attractor',
  'Inquiry oscillation → debate interference → consensus attractor',
  'Faction oscillation → coalition interference → stable government attractor',
  'Proposal oscillation → lobbying interference → ratified standard attractor',
  'Cycle oscillation → feedback interference → stable execution attractor',
  'Deviation oscillation → damping interference → equilibrium attractor',
  // N8-N14
  'Alert oscillation → escalation interference → stable response attractor',
  'Epoch oscillation → regularization interference → optimum attractor',
  'Animation oscillation → overlap interference → coherent attractor',
  'Protest oscillation → repression interference → protected attractor',
  'Crisis oscillation → deterrence interference → balance attractor',
  'Fork oscillation → merge interference → coherent project attractor',
  'Predator-prey oscillation → competition interference → stable-cycle attractor'
];

/**
 * T6: Rhythm → Harmony → Emergence
 */
const T6_CELLS: string[] = [
  // N1-N7
  'Data rhythm → flow harmony → scalable architecture emergence',
  'Switching rhythm → combinational harmony → sequential emergence',
  'Curiosity rhythm → insight harmony → discovery emergence',
  'Decision rhythm → process harmony → policy emergence',
  'Update rhythm → alignment harmony → ecosystem emergence',
  'Operational rhythm → synchronization harmony → delivered outcome',
  'Sampling rhythm → loop harmony → controlled emergence',
  // N8-N14
  'Pulse rhythm → dashboard harmony → situational emergence',
  'Iteration rhythm → convergence harmony → capability emergence',
  'Flow rhythm → aesthetic harmony → engaging emergence',
  'Struggle rhythm → solidarity harmony → dignity emergence',
  'Escalation rhythm → détente harmony → stability emergence',
  'Contribution rhythm → synergy harmony → collective emergence',
  'Metabolic rhythm → symbiotic harmony → speciation emergence'
];

/**
 * T7: Drive → Constraint → Mediation
 */
const T7_CELLS: string[] = [
  // N1-N7
  'Compute drive → resource constraint → load mediation',
  'Truth drive → constraint logic → satisfiability mediation',
  'Search drive → rigor constraint → validity mediation',
  'Power drive → legal constraint → compromise mediation',
  'Compliance drive → penalty constraint → enforcement mediation',
  'Urgency drive → deadline constraint → resource mediation',
  'Performance drive → limit constraint → regulation mediation',
  // N8-N14
  'Vigilance drive → protocol constraint → action mediation',
  'Curiosity drive → loss constraint → learning mediation',
  'User drive → accessibility constraint → affordance mediation',
  'Dignity drive → prohibition constraint → remedy mediation',
  'Ambition drive → treaty constraint → sphere mediation',
  'Sharing drive → license constraint → permission mediation',
  'Survival drive → resource constraint → niche mediation'
];

/**
 * T8: Explore → Exploit → Policy
 */
const T8_CELLS: string[] = [
  // N1-N7
  'Explore topologies → exploit protocols → infrastructure policy',
  'Explore expressions → exploit simplifications → reduction policy',
  'Explore unknowns → exploit answers → research policy',
  'Explore structures → exploit mechanisms → governance policy',
  'Explore options → exploit consensus → standardization policy',
  'Explore methods → exploit tools → implementation policy',
  'Explore ranges → exploit tolerances → control policy',
  // N8-N14
  'Explore incidents → exploit signals → response policy',
  'Explore hypotheses → exploit data → training policy',
  'Explore interactions → exploit conventions → design policy',
  'Explore freedoms → exploit protections → rights policy',
  'Explore options → exploit leverage → foreign policy',
  'Explore resources → exploit forks → commons policy',
  'Explore mutations → exploit niches → evolutionary policy'
];

/**
 * T9: Prediction → Error → Model Update
 */
const T9_CELLS: string[] = [
  // N1-N7
  'Predict traffic → error detection → routing update',
  'Predict outcome → contradiction error → axiom update',
  'Predict implications → falsification error → theory update',
  'Predict outcomes → failure error → reform update',
  'Predict impact → violation error → revision update',
  'Predict delays → deviation error → plan update',
  'Predict behavior → error signal → parameter update',
  // N8-N14
  'Predict threats → false positive error → rule update',
  'Prediction failure → error analysis → model refinement',
  'Predict behavior → frustration error → iteration update',
  'Predict abuse → violation error → norm update',
  'Predict moves → miscalculation error → strategy update',
  'Predict sustainability → tragedy error → governance update',
  'Adaptive prediction → selection error → genetic-model update'
];

/**
 * T10: Signal → Code → Interpretation
 */
const T10_CELLS: string[] = [
  // N1-N7
  'Binary signal → opcode → executed instruction',
  'Input signal → function code → truth interpretation',
  'Phenomenon signal → question code → meaning interpretation',
  'Citizen signal → law code → rights interpretation',
  'Requirement signal → clause code → legal interpretation',
  'Instruction signal → procedure code → result interpretation',
  'Observation signal → scale code → value interpretation',
  // N8-N14
  'Log signal → signature code → threat interpretation',
  'Example signal → representation code → knowledge interpretation',
  'User signal → UI code → interpreted experience',
  'Grievance signal → article code → entitlement interpretation',
  'Event signal → narrative code → interest interpretation',
  'Creation signal → format code → shared interpretation',
  'Environmental cue → epigenetic code → developmental interpretation'
];

/**
 * T11: Coordination → Alignment → Mandate
 */
const T11_CELLS: string[] = [
  // N1-N7
  'Node coordination → consensus alignment → blockchain mandate',
  'Operator coordination → precedence alignment → evaluation mandate',
  'Thinker coordination → discourse alignment → canon mandate',
  'Actor coordination → role alignment → constitutional mandate',
  'Body coordination → jurisdiction alignment → regulatory mandate',
  'Team coordination → role alignment → delivery mandate',
  'Component coordination → hierarchy alignment → command mandate',
  // N8-N14
  'Team coordination → chain alignment → incident mandate',
  'Agent coordination → curriculum alignment → learning mandate',
  'Element coordination → hierarchy alignment → navigation mandate',
  'Social coordination → rights alignment → legal mandate',
  'State coordination → bloc alignment → security mandate',
  'Member coordination → norm alignment → commons mandate',
  'Species coordination → coevolutionary alignment → natural-selection mandate'
];

/**
 * T12: Contribution → Interoperability → Infrastructure
 */
const T12_CELLS: string[] = [
  // N1-N7
  'Hardware contribution → API interoperability → cloud infrastructure',
  'Gate contribution → circuit interoperability → logic infrastructure',
  'Idea contribution → citation interoperability → knowledge infrastructure',
  'Resource contribution → division interoperability → organizational infrastructure',
  'Stakeholder contribution → conformance interoperability → standards infrastructure',
  'Effort contribution → integration interoperability → operational infrastructure',
  'Instrument contribution → protocol interoperability → control infrastructure',
  // N8-N14
  'Data contribution → tool interoperability → monitoring infrastructure',
  'Experience contribution → transfer interoperability → knowledge infrastructure',
  'Component contribution → framework interoperability → interface infrastructure',
  'Individual contribution → treaty interoperability → rights infrastructure',
  'National contribution → treaty interoperability → geopolitical infrastructure',
  'User contribution → protocol interoperability → digital infrastructure',
  'Trait contribution → gene-flow interoperability → gene-pool infrastructure'
];

/**
 * T13: Boundary → State → Transition
 */
const T13_CELLS: string[] = [
  // N1-N7
  'Boundary noise → valid state → error transition',
  'Variable boundary → assignment state → computation transition',
  'Ignorance boundary → understanding state → paradigm transition',
  'Jurisdiction boundary → sovereignty state → regime transition',
  'Scope boundary → compliance state → deregulation transition',
  'Start boundary → progress state → completion transition',
  'Threshold boundary → regulated state → failure transition',
  // N8-N14
  'Normal boundary → alert state → recovery transition',
  'Prior boundary → posterior state → insight transition',
  'Edge boundary → focus state → modal transition',
  'Restriction boundary → protected state → emancipation transition',
  'Border boundary → sovereignty state → geopolitical transition',
  'Exclusion boundary → inclusion state → commons transition',
  'Reproductive boundary → speciation state → adaptive-radiation transition'
];

/**
 * T14: Genesis → Adaptation → Transcendence
 */
const T14_CELLS: string[] = [
  // N1-N7
  'Bit genesis → protocol adaptation → network transcendence',
  'Gate genesis → circuit adaptation → computational transcendence',
  'Question genesis → method adaptation → wisdom transcendence',
  'Power genesis → institutional adaptation → democratic transcendence',
  'Standard genesis → compliance adaptation → interoperability transcendence',
  'Plan genesis → execution adaptation → outcome transcendence',
  'Metric genesis → calibration adaptation → precision transcendence',
  // N8-N14
  'Alert genesis → response adaptation → resilience transcendence',
  'Model genesis → learning adaptation → intelligence transcendence',
  'Interface genesis → usability adaptation → experience transcendence',
  'Rights genesis → legal adaptation → dignity transcendence',
  'Treaty genesis → diplomatic adaptation → peace transcendence',
  'Commons genesis → collaborative adaptation → collective transcendence',
  'Species genesis → evolutionary adaptation → consciousness transcendence'
];

/**
 * All cell contents organized by triad
 */
export const CELL_CONTENTS: Record<TriadId, string[]> = {
  T1: T1_CELLS,
  T2: T2_CELLS,
  T3: T3_CELLS,
  T4: T4_CELLS,
  T5: T5_CELLS,
  T6: T6_CELLS,
  T7: T7_CELLS,
  T8: T8_CELLS,
  T9: T9_CELLS,
  T10: T10_CELLS,
  T11: T11_CELLS,
  T12: T12_CELLS,
  T13: T13_CELLS,
  T14: T14_CELLS
};

// ============================================================================
// K Depth Matrix (14×14)
// K(T,N) ∈ {0, 1, 2}
// ============================================================================

/**
 * Depth matrix K(T,N) where:
 * - 2 = Core (full connection)
 * - 1 = Relevant (partial connection)
 * - 0 = Empty (no connection)
 *
 * Row = Triad (T1-T14)
 * Col = Layer (N1-N14)
 */
export const DEPTH_MATRIX: ConvDepth[][] = [
  // T1:  N1 N2 N3 N4 N5 N6 N7 N8 N9 N10 N11 N12 N13 N14
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T2:
  [2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2],
  // T3:
  [2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2],
  // T4:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T5:
  [2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2],
  // T6:
  [2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2],
  // T7:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T8:
  [2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2],
  // T9:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T10:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T11:
  [2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T12:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T13:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  // T14:
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
];

// ============================================================================
// Main Diagonal Meanings (T=N, Reflexivity)
// ============================================================================

export const MAIN_DIAGONAL_MEANINGS: {
  meaningHe: string;
  meaningEn: string;
}[] = [
  {
    meaningHe: 'תשתית שולטת בעצמה',
    meaningEn: 'Infrastructure controls itself'
  },
  {
    meaningHe: 'לוגיקה מפרשת את עצמה',
    meaningEn: 'Logic interprets itself'
  },
  {
    meaningHe: 'שאלות משנות שאלות',
    meaningEn: 'Questions change questions'
  },
  {
    meaningHe: 'ממשל מממשל',
    meaningEn: 'Governance governs'
  },
  {
    meaningHe: 'סטנדרט מסטנדרט',
    meaningEn: 'Standard standardizes'
  },
  {
    meaningHe: 'ביצוע מביא ביצוע',
    meaningEn: 'Execution brings execution'
  },
  {
    meaningHe: 'מדידה מולידה שליטה',
    meaningEn: 'Measurement generates control'
  },
  {
    meaningHe: 'תגובה מייצבת תגובה',
    meaningEn: 'Response stabilizes response'
  },
  {
    meaningHe: 'למידה מולידה יכולת',
    meaningEn: 'Learning generates capability'
  },
  {
    meaningHe: 'UX מולידה חוויה',
    meaningEn: 'UX generates experience'
  },
  {
    meaningHe: 'זכויות מולידות כבוד',
    meaningEn: 'Rights generate dignity'
  },
  {
    meaningHe: 'גיאופוליטיקה מייצבת',
    meaningEn: 'Geopolitics stabilizes'
  },
  {
    meaningHe: 'קומונס מוליד קולקטיב',
    meaningEn: 'Commons generates collective'
  },
  {
    meaningHe: 'אבולוציה מתעלה',
    meaningEn: 'Evolution transcends'
  }
];

// ============================================================================
// Anti-Diagonal Meanings (T+N=15, Critical Transitions)
// ============================================================================

export const ANTI_DIAGONAL_MEANINGS: {
  triadIndex: number;
  layerIndex: number;
  meaningHe: string;
  meaningEn: string;
}[] = [
  {
    triadIndex: 0,
    layerIndex: 13,
    meaningHe: 'תשתית↔אבולוציה',
    meaningEn: 'Infrastructure ↔ Evolution'
  },
  {
    triadIndex: 1,
    layerIndex: 12,
    meaningHe: 'לוגיקה↔קומונס',
    meaningEn: 'Logic ↔ Commons'
  },
  {
    triadIndex: 2,
    layerIndex: 11,
    meaningHe: 'שאלות↔גיאופוליטיקה',
    meaningEn: 'Questions ↔ Geopolitics'
  },
  {
    triadIndex: 3,
    layerIndex: 10,
    meaningHe: 'אי-סדר↔זכויות',
    meaningEn: 'Disorder ↔ Rights'
  },
  {
    triadIndex: 4,
    layerIndex: 9,
    meaningHe: 'תנודה↔ממשק',
    meaningEn: 'Oscillation ↔ Interface'
  },
  {
    triadIndex: 5,
    layerIndex: 8,
    meaningHe: 'קצב↔למידה',
    meaningEn: 'Rhythm ↔ Learning'
  },
  {
    triadIndex: 6,
    layerIndex: 7,
    meaningHe: 'מדידה↔תגובה',
    meaningEn: 'Measurement ↔ Response'
  },
  {
    triadIndex: 7,
    layerIndex: 6,
    meaningHe: 'חקירה↔בקרה',
    meaningEn: 'Exploration ↔ Control'
  },
  {
    triadIndex: 8,
    layerIndex: 5,
    meaningHe: 'חיזוי↔ביצוע',
    meaningEn: 'Prediction ↔ Execution'
  },
  {
    triadIndex: 9,
    layerIndex: 4,
    meaningHe: 'אות↔תקינה',
    meaningEn: 'Signal ↔ Standards'
  },
  {
    triadIndex: 10,
    layerIndex: 3,
    meaningHe: 'תיאום↔ממשל',
    meaningEn: 'Coordination ↔ Governance'
  },
  {
    triadIndex: 11,
    layerIndex: 2,
    meaningHe: 'תרומה↔שאלות',
    meaningEn: 'Contribution ↔ Questions'
  },
  {
    triadIndex: 12,
    layerIndex: 1,
    meaningHe: 'גבול↔לוגיקה',
    meaningEn: 'Boundary ↔ Logic'
  },
  {
    triadIndex: 13,
    layerIndex: 0,
    meaningHe: 'אבולוציה↔תשתית',
    meaningEn: 'Evolution ↔ Infrastructure'
  }
];

// ============================================================================
// Helper function to build all cells
// ============================================================================

export function buildAllCells(): Conv2DCell[] {
  const cells: Conv2DCell[] = [];

  for (let row = 0; row < 14; row++) {
    const triadId = `T${row + 1}` as TriadId;
    const contents = CELL_CONTENTS[triadId];

    for (let col = 0; col < 14; col++) {
      const layerId = `N${col + 1}` as LayerId;
      const cellId = `C${String(row + 1).padStart(2, '0')}${String(col + 1).padStart(2, '0')}`;

      cells.push({
        type: 'CONV_2D_CELL',
        cellId,
        triadId,
        layerId,
        rowIndex: row,
        colIndex: col,
        content: contents[col],
        depth: DEPTH_MATRIX[row][col]
      });
    }
  }

  return cells;
}
