/**
 * Everything the page says about the runs: their names and groups, the "Key decisions" table, and the "Training"
 * section (a large chart for the chosen run, plus a grid of small charts for every run).
 *
 * main.ts uses GROUPS to build the model picker, then calls renderDecisions(root), renderGrid(root, onPick) and
 * showRun(root, run). Every number comes from data/results.json, which make_app_data.py builds from results/*.json,
 * so each score on the page traces back to a run file. Run names are the keys of that file: a misspelt run name fails
 * the TypeScript check instead of showing a wrong number.
 */

import results from "./data/results.json";

export type RunName = keyof typeof results;

/** One run's settings (make_app_data.py, settings()). */
interface Settings {
  arch: string;
  pretrained: boolean;
  frozen: boolean;
  imageSize: number;
  batchSize: number;
  optimizer: string;
  momentum: number;
  lrNew: number;
  lrPretrained: number;
  schedule: string;
  warmupEpochs: number;
  epochs: number;
  augment: boolean;
  augmentStrength: string;
  dropout: number;
  weightDecay: number;
  labelSmoothing: number;
  trainOnVal: boolean;
  svmKernel: string | null;
  epochKept: number | null;
  minutes: number;
  peakGpuGb: number | null;
}

/** One run as results.json stores it. val is null when the run trained on the validation photos (the final runs). */
interface Run {
  val: number | null;
  test: number;
  perClass: number;
  settings: Settings;
  curve: { train: number[]; val: number[] | null } | null; // Top-1 per epoch; null for the SVM runs
}

const RUNS = results as Record<RunName, Run>;

// ---------- Number helpers ----------

/** 0.8972 -> "89.7%"; null -> "–" */
export const pct = (x: number | null): string => (x === null ? "–" : `${(x * 100).toFixed(1)}%`);

/** Gap between two scores in percentage points: (0.874, 0.846) -> "2.8" */
const points = (a: number, b: number): string => Math.abs((a - b) * 100).toFixed(1);

/** The seed-43 repeat of a run, if one exists. */
function twin(run: RunName): RunName | null {
  const name = `${run}-seed43`;
  return name in RUNS ? (name as RunName) : null;
}

/** Validation Top-1 the experiments decided on: the mean over seeds 42 and 43 when both exist, else seed 42 alone. */
function val(run: RunName): number {
  const own = RUNS[run].val ?? 0; // only the final runs have no validation score, and they are never passed here
  const repeat = twin(run);
  const other = repeat ? RUNS[repeat].val : null;
  return other === null ? own : (own + other) / 2;
}

/** Largest validation gap between two seeds of one recipe: the noise floor. A smaller gap between recipes is a tie. */
const NOISE = Math.max(
  ...(Object.keys(RUNS) as RunName[]).flatMap((run) => {
    const repeat = twin(run);
    const [a, b] = [RUNS[run].val, repeat ? RUNS[repeat].val : null];
    return a !== null && b !== null ? [Math.abs(a - b)] : [];
  }),
);

const test = (run: RunName): number => RUNS[run].test;

/** The final model on this site: ConvNeXt-Tiny retrained on all 4,829 training photos. */
export const FINAL_RUN = "8-final-convnext" satisfies RunName;
export const FINAL = RUNS[FINAL_RUN];
/** The other network of the final round: the same ResNet-50 recipe, also retrained on all photos. */
export const RUNNER_UP = RUNS["8-final-resnet50"];

// ---------- Run names and groups ----------

/** One run as the model picker and the training grid list it. kept: carried forward after its experiment. */
interface RunEntry {
  run: RunName;
  label: string;
  kept?: boolean;
}

/** One experiment's runs, in the order the picker and the grid show them. */
interface Group {
  title: string;
  runs: RunEntry[];
}

export const GROUPS: Group[] = [
  {
    title: "Final model",
    runs: [
      { run: "8-final-convnext", label: "ConvNeXt-Tiny, final model", kept: true },
      { run: "8-final-resnet50", label: "ResNet-50, runner-up" },
    ],
  },
  {
    title: "1. Baselines",
    runs: [
      { run: "1-random-start", label: "Random start" },
      { run: "1-frozen-new-layer-plain", label: "Frozen, new layer, plain" },
      { run: "1-frozen-new-layer", label: "Frozen, new layer" },
      { run: "1-frozen-svm-linear", label: "Frozen, linear SVM" },
      { run: "1-frozen-svm-rbf", label: "Frozen, RBF SVM" },
    ],
  },
  {
    title: "2. Starting recipe",
    runs: [{ run: "2-control", label: "Adam, fixed learning rate", kept: true }],
  },
  {
    title: "3. Learning rate schedule and optimiser",
    runs: [
      { run: "3-adam-cosine", label: "Adam, cosine" },
      { run: "3-adam-cosine-seed43", label: "Adam, cosine, seed 43" },
      { run: "3-sgd-lr1e-03", label: "SGD, cosine, rate 0.001" },
      { run: "3-sgd-lr1e-03-seed43", label: "SGD, cosine, rate 0.001, seed 43" },
      { run: "3-sgd-lr3e-03", label: "SGD, cosine, rate 0.003", kept: true },
      { run: "3-sgd-lr3e-03-seed43", label: "SGD, cosine, rate 0.003, seed 43" },
      { run: "3-sgd-lr1e-02", label: "SGD, cosine, rate 0.01" },
    ],
  },
  {
    title: "4. Batch size",
    runs: [
      { run: "4-batch24-lr3e-03", label: "Batch 24" },
      { run: "4-batch24-lr3e-03-seed43", label: "Batch 24, seed 43" },
      { run: "4-batch24-lr6e-03", label: "Batch 24, learning rate doubled" },
    ],
  },
  {
    title: "5. Fighting overfitting",
    runs: [
      { run: "5-label-smoothing", label: "Label smoothing" },
      { run: "5-weight-decay", label: "Weight decay" },
      { run: "5-dropout", label: "Dropout" },
      { run: "5-strong-augment", label: "Strong augmentation" },
      { run: "5-combined", label: "All four together", kept: true },
      { run: "5-combined-seed43", label: "All four together, seed 43" },
    ],
  },
  {
    title: "6. Networks",
    runs: [
      { run: "6-resnet18-lr1e-03", label: "ResNet-18, rate 0.001" },
      { run: "6-resnet18-lr3e-03", label: "ResNet-18, rate 0.003" },
      { run: "6-resnet18-lr1e-02", label: "ResNet-18, rate 0.01" },
      { run: "6-convnext-lr1e-03", label: "ConvNeXt-Tiny, rate 0.001" },
      { run: "6-convnext-lr3e-03", label: "ConvNeXt-Tiny, rate 0.003", kept: true },
      { run: "6-convnext-lr3e-03-seed43", label: "ConvNeXt-Tiny, rate 0.003, seed 43" },
      { run: "6-convnext-lr1e-02", label: "ConvNeXt-Tiny, rate 0.01" },
    ],
  },
];

/** A run's label from GROUPS, e.g. "Adam, cosine". */
export function label(run: RunName): string {
  return GROUPS.flatMap((g) => g.runs).find((r) => r.run === run)?.label ?? run;
}

/** True for an SVM run: its outputs are SVM scores, not probabilities. */
export const isSvm = (run: RunName): boolean => RUNS[run].settings.svmKernel !== null;

export const testScore = test;

// ---------- Key decisions ----------

/** One row of the "Key decisions" table: the experiment's question, its decision, and the evidence in numbers. */
interface Decision {
  number: string;
  question: string;
  decision: string;
  evidence: string;
}

// The validation range of the four frozen-network baselines in experiment 1.
const frozen = (["1-frozen-new-layer", "1-frozen-new-layer-plain", "1-frozen-svm-linear", "1-frozen-svm-rbf"] as const).map(val);
// ResNet-18's best validation score over its three learning rates (experiment 6, one seed each).
const resnet18Best = Math.max(...(["6-resnet18-lr1e-03", "6-resnet18-lr3e-03", "6-resnet18-lr1e-02"] as const).map(val));

const DECISIONS: Decision[] = [
  {
    number: "1",
    question: "Do ImageNet weights and fine-tuning matter?",
    decision: "Start from ImageNet weights and fine-tune every layer.",
    evidence:
      `Random start ${pct(val("1-random-start"))}; frozen layers ${pct(Math.min(...frozen))} to ` +
      `${pct(Math.max(...frozen))}; fine-tuned ${pct(val("5-combined"))}.`,
  },
  {
    number: "2",
    question: "Where does a standard fine-tuning recipe land?",
    decision: "Baseline: ResNet-50, 448 × 448 photos, Adam at a fixed learning rate.",
    evidence: `${pct(val("2-control"))}.`,
  },
  {
    number: "3",
    question: "Is the fixed learning rate holding it back?",
    decision: "Cosine learning rate schedule with warm-up; SGD with momentum.",
    evidence: `${pct(val("2-control"))} to ${pct(val("3-sgd-lr3e-03"))}. SGD and Adam tied; lower validation loss picked SGD.`,
  },
  {
    number: "4",
    question: "Does a batch of 24 beat 12?",
    decision: "Keep batch 12.",
    evidence: `${pct(val("4-batch24-lr3e-03"))} against ${pct(val("3-sgd-lr3e-03"))}: a tie. Batch 12 needs about half the GPU memory.`,
  },
  {
    number: "5",
    question: "Can it stop memorising the training photos?",
    decision: "Label smoothing, weight decay, dropout and strong augmentation, together.",
    evidence: `${pct(val("3-sgd-lr3e-03"))} to ${pct(val("5-combined"))}. Each tool alone stayed inside the noise.`,
  },
  {
    number: "6",
    question: "Does the network design matter?",
    decision: "ResNet-50 and ConvNeXt-Tiny tie and both go to the final round. ResNet-18 is dropped.",
    evidence: `ConvNeXt-Tiny ${pct(val("6-convnext-lr3e-03"))}, ResNet-50 ${pct(val("5-combined"))}, ResNet-18 ${pct(resnet18Best)} at best.`,
  },
  {
    number: "Final",
    question: "Which network is the final model?",
    decision: "Both retrained on all 4,829 photos. ConvNeXt-Tiny scored higher on test and is the final model.",
    evidence:
      `Test Top-1: ConvNeXt-Tiny ${pct(FINAL.test)}, ResNet-50 ${pct(RUNNER_UP.test)}. Validation had tied them, ` +
      `so the test set decided; part of the gap may be chance.`,
  },
];

/** The sentence above the "Key decisions" table: what the numbers are and when a gap counts. */
export const DECISIONS_INTRO =
  `Each experiment asked one question. Scores are validation Top-1 on 800 photos held back from training (the mean ` +
  `of two seeds where both exist). Two seeds of one recipe differed by up to ${points(NOISE, 0)} points, so smaller ` +
  `gaps count as ties. Validation chose every setting; only the last choice, between two tied networks, used the test set.`;

/** Build the "Key decisions" table inside `root`. */
export function renderDecisions(root: HTMLElement): void {
  root.innerHTML = `
    <table class="table decisions">
      <thead><tr><th scope="col">#</th><th scope="col">Question</th><th scope="col">Decision</th><th scope="col">Evidence</th></tr></thead>
      <tbody>${DECISIONS.map(
        (d) => `
        <tr${d.number === "Final" ? ' class="is-final"' : ""}>
          <td>${d.number}</td>
          <td>${d.question}</td>
          <td class="decision">${d.decision}</td>
          <td class="evidence">${d.evidence}</td>
        </tr>`,
      ).join("")}</tbody>
    </table>`;
}

// ---------- Charts ----------

/** A final run trained on the validation photos, so it has no validation line. Its chart borrows the validation
 * line of the same recipe's earlier run, which held those 800 photos back. */
const RECIPE_TWIN: Partial<Record<RunName, RunName>> = {
  "8-final-convnext": "6-convnext-lr3e-03",
  "8-final-resnet50": "5-combined",
};

/** Last value of a curve. */
const last = (values: number[]): number => values[values.length - 1] ?? 0;

/** The lines of one run's chart: solid training and validation, plus a dashed final-run line when it has a twin. */
function chartLines(run: RunName): { train: number[]; val: number[]; final: number[] | null } | null {
  const source = RECIPE_TWIN[run];
  const held = RUNS[source ?? run].curve;
  if (!held?.val) return null;
  return { train: held.train, val: held.val, final: source ? RUNS[run].curve!.train : null };
}

/**
 * A Top-1 chart as an inline SVG. The large chart (`detail`) adds axis labels, the kept-epoch dot, and a bracket
 * that measures the gap between training and validation at the last epoch.
 */
function chartSvg(run: RunName, detail: boolean): string {
  const lines = chartLines(run)!;
  const [W, H, left, right, top, bottom] = detail ? [560, 260, 40, 92, 10, 26] : [200, 90, 4, 4, 4, 4];
  const n = lines.train.length;
  const x = (epoch: number) => left + ((epoch - 1) / (n - 1)) * (W - left - right);
  const y = (v: number) => top + (1 - v) * (H - top - bottom);
  const path = (values: number[]) => values.map((v, i) => `${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const polylines =
    (lines.final ? `<polyline class="final" points="${path(lines.final)}" />` : "") +
    `<polyline class="train" points="${path(lines.train)}" /><polyline class="val" points="${path(lines.val)}" />`;
  if (!detail) return `<svg class="mini" viewBox="0 0 ${W} ${H}" aria-hidden="true">${polylines}</svg>`;

  // Grid lines every 25 points with labels on the left; epoch labels along the bottom.
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => `<line class="grid" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}" /><text class="tick" x="${left - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`)
    .join("");
  const ticks = [1, 10, 20, n].map((e) => `<text class="tick" x="${x(e)}" y="${H - 6}" text-anchor="middle">${e}</text>`).join("");
  // The bracket at the last epoch: from the training line down to the validation line, labelled with the gap.
  const [yTrain, yVal] = [y(last(lines.train)), y(last(lines.val))];
  const bx = x(n) + 10;
  const bracket =
    `<path class="bracket" d="M${bx - 4},${yTrain} H${bx} V${yVal} H${bx - 4}" />` +
    `<text class="bracket-label" x="${bx + 6}" y="${(yTrain + yVal) / 2 - 2}">gap ${points(last(lines.train), last(lines.val))}</text>` +
    `<text class="bracket-label" x="${bx + 6}" y="${(yTrain + yVal) / 2 + 12}">points</text>`;
  // The kept epoch: the one whose weights were scored, the best validation epoch. A final run keeps its last epoch
  // and has no validation line of its own, so it gets no dot.
  const kept = RECIPE_TWIN[run] ? null : RUNS[run].settings.epochKept;
  const dot = kept ? `<circle class="kept" cx="${x(kept)}" cy="${y(lines.val[kept - 1]!)}" r="4" />` : "";
  return `
    <svg class="curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="Top-1 per epoch for ${run}">
      ${grid}${ticks}${polylines}${bracket}${dot}
    </svg>`;
}

// ---------- Settings in a few words ----------

const ARCH_NAME: Record<string, string> = { resnet50: "ResNet-50", resnet18: "ResNet-18", convnext_tiny: "ConvNeXt-Tiny" };

/** A learning rate as plain decimals: 0.003 -> "0.003". */
const rate = (x: number): string => x.toFixed(6).replace(/0+$/, "");

/** The settings list of one run: [setting, value]. */
function settingsRows(run: RunName): [string, string][] {
  const s = RUNS[run].settings;
  const what = s.svmKernel
    ? `200 SVMs (${s.svmKernel} kernel) on frozen features`
    : s.frozen
      ? "Only a new last layer; the rest frozen"
      : "Every layer";
  const rows: [string, string][] = [
    ["Network", `${ARCH_NAME[s.arch] ?? s.arch}, ${s.pretrained ? "ImageNet weights" : "random weights"}`],
    ["What learns", what],
    ["Training photos", s.trainOnVal ? "4,829 (validation photos included)" : "4,029 (800 held back for validation)"],
  ];
  if (!s.svmKernel) {
    const tools = [
      s.augment ? (s.augmentStrength === "strong" ? "strong augmentation" : "augmentation") : "",
      s.labelSmoothing ? `label smoothing ${s.labelSmoothing}` : "",
      s.weightDecay ? `weight decay ${s.weightDecay}` : "",
      s.dropout ? `dropout ${s.dropout}` : "",
    ].filter(Boolean);
    rows.push(
      ["Optimiser", s.optimizer === "sgd" ? `SGD, momentum ${s.momentum}` : "Adam"],
      ["Learning rate", s.pretrained && !s.frozen ? `${rate(s.lrNew)} new layer, ${rate(s.lrPretrained)} copied layers` : rate(s.lrNew)],
      ["Schedule", s.schedule === "cosine" ? `cosine, ${s.warmupEpochs} warm-up epoch` : "fixed"],
      ["Batch", `${s.batchSize} photos`],
      ["Against overfitting", tools.length ? tools.join(", ") : "none"],
    );
  }
  rows.push(["Training time", `${Math.round(s.minutes)} minutes on a laptop RTX 4050`]);
  return rows;
}

// ---------- Training section ----------

/** How to read the large chart, under it. */
const HOW_TO_READ = `
  <ul class="how-to-read">
    <li>Each point is Top-1 accuracy after one epoch, one pass over every training photo.</li>
    <li><span class="key key-train"></span>Grey: the training photos the network learns from. <span class="key key-val"></span>Blue: the 800 validation photos it never learns from.</li>
    <li>The gap between them is overfitting: the network knows its own training photos better than new ones. A smaller gap is better.</li>
    <li>The dot marks the kept epoch: the weights with the best validation score are the ones scored on test.</li>
  </ul>`;

/** The large chart, scores and settings of one run, inside `root`. */
export function showRun(root: HTMLElement, run: RunName, onTry: (run: RunName) => void): void {
  const r = RUNS[run];
  const source = RECIPE_TWIN[run];
  const chart = chartLines(run)
    ? chartSvg(run, true) +
      (source
        ? `<p class="chart-note"><span class="key key-final"></span>Dashed: this run, trained on all 4,829 photos. Its validation photos became training photos, so it has no validation line; the solid lines come from <code>${source}</code>, the same recipe with 800 photos held back.</p>`
        : "") +
      HOW_TO_READ
    : `<p class="chart-empty">No chart: an SVM is solved in one go, not trained epoch by epoch.</p>`;
  const scores: [string, string][] = [
    ["Validation Top-1", r.val === null ? "– (trained on those photos)" : pct(r.val)],
    ["Test Top-1", pct(r.test)],
    ["Test average per-class", pct(r.perClass)],
  ];
  root.innerHTML = `
    <header class="explorer-head">
      <div>
        <h3>${label(run)}</h3>
        <code class="run-key">${run}</code>
      </div>
      <button type="button" class="button button-quiet" id="try-run">Classify with this model</button>
    </header>
    <div class="explorer-grid">
      <div>${chart}</div>
      <div>
        <dl class="facts">${scores.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
        <dl class="facts facts-settings">${settingsRows(run).map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
      </div>
    </div>`;
  root.querySelector("#try-run")!.addEventListener("click", () => onTry(run));
}

/** The grid of small charts, one tile per run, grouped by experiment. Selecting a tile calls onPick. */
export function renderGrid(root: HTMLElement, onPick: (run: RunName) => void): void {
  root.innerHTML = GROUPS.map(
    (group) => `
    <section class="grid-group">
      <h3>${group.title}</h3>
      <ul class="tiles">${group.runs
        .map(
          ({ run, label: text, kept }) => `
        <li><button type="button" class="tile" data-run="${run}" aria-pressed="false">
          ${chartLines(run) ? chartSvg(run, false) : '<span class="mini mini-empty">SVM: no epochs</span>'}
          <span class="tile-label">${text}${kept ? ' <span class="kept-tag">kept</span>' : ""}</span>
          <span class="tile-scores">val ${pct(RUNS[run].val)} · test ${pct(RUNS[run].test)}</span>
        </button></li>`,
        )
        .join("")}</ul>
    </section>`,
  ).join("");
  root.addEventListener("click", (event) => {
    const tile = (event.target as HTMLElement).closest<HTMLElement>(".tile");
    if (tile) onPick(tile.dataset["run"] as RunName);
  });
}

/** Mark the tile of the run on show in the large chart. */
export function markTile(root: HTMLElement, run: RunName): void {
  for (const tile of root.querySelectorAll<HTMLElement>(".tile")) {
    tile.setAttribute("aria-pressed", String(tile.dataset["run"] === run));
  }
}

/** The sentence above the training charts. */
export const TRAINING_INTRO =
  `Every run's learning curve, grouped by experiment. Select a small chart to see it large, with its scores and ` +
  `settings, or to classify a photo with it. Runs marked "kept" were carried forward to the next experiment.`;
