/**
 * The "The experiments" section: one table row per experiment (1 to 8), and a detail panel for the selected one.
 *
 * main.ts calls renderExperiments(root) once. Selecting an experiment row opens its story (question, what changed,
 * result) and the runs it compared. Selecting a run shows that run's settings in plain words and its learning curve.
 * Every number comes from data/results.json, which make_app_data.py builds from results/*.json, so each score on
 * the page traces back to a run file. Run names are the keys of that file: a misspelt run name fails the TypeScript
 * check instead of showing a wrong number.
 */

import results from "./data/results.json";

export type RunName = keyof typeof results;

/** One trained run's settings (make_app_data.py, settings()). */
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

/** One run as results.json stores it. val is null when the run trained on the validation photos (experiment 8). */
interface Run {
  val: number | null;
  test: number;
  perClass: number;
  settings: Settings | null; // null for a combination, which trains nothing
  members: RunName[] | null; // the runs a combination averages
  flip: boolean | null; // a combination that also scores each photo's mirror image
  curve: { train: number[]; val: number[] | null } | null; // Top-1 per epoch
}

const RUNS = results as Record<RunName, Run>;

/** How a run is shown in an experiment's "Runs compared" table. */
type Role = "kept" | "dropped" | "baseline" | "comparison" | "member";

const ROLE_TEXT: Record<Role, string> = {
  kept: "Kept",
  dropped: "Dropped",
  baseline: "Baseline",
  comparison: "For comparison",
  member: "Member",
};

interface RunRow {
  run: RunName;
  label: string;
  role: Role;
}

interface Experiment {
  number: number;
  title: string;
  row: RunName; // the run the main table shows: the one carried forward after this experiment
  change: string; // one line for the main table
  question: string;
  changed: string;
  result: string;
  runs: RunRow[];
}

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
function val(run: RunName): number | null {
  const own = RUNS[run].val;
  const repeat = twin(run);
  const other = repeat ? RUNS[repeat].val : null;
  return own === null || other === null ? own : (own + other) / 2;
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

const ARCH_NAME: Record<string, string> = { resnet50: "ResNet-50", resnet18: "ResNet-18", convnext_tiny: "ConvNeXt-Tiny" };

/** A run's network name, or both names for a combination. */
function networkName(run: RunName): string {
  const { settings, members } = RUNS[run];
  if (settings) return ARCH_NAME[settings.arch] ?? settings.arch;
  return (members ?? []).map(networkName).join(" + ");
}

// ---------- The eight experiments ----------

const frozenRuns: RunName[] = ["1-frozen-new-layer", "1-frozen-new-layer-plain", "1-frozen-svm-linear", "1-frozen-svm-rbf"];
const resnet18Best = Math.max(...(["6-resnet18-lr1e-03", "6-resnet18-lr3e-03", "6-resnet18-lr1e-02"] as const).map((r) => RUNS[r].val ?? 0));

const EXPERIMENTS: Experiment[] = [
  {
    number: 1,
    title: "Baselines",
    row: "1-frozen-new-layer",
    change: "The finished recipe with ImageNet weights or fine-tuning taken away",
    question: "How much do the ImageNet weights and fine-tuning add?",
    changed:
      "The finished recipe (experiment 5) with one advantage taken away at a time. Random start: every weight " +
      "begins as noise. Frozen: the copied ImageNet layers never change, and only a new last layer learns. Two " +
      "frozen runs swap that last layer for 200 SVMs.",
    result:
      `From random weights, ResNet-50 reaches only ${pct(test("1-random-start"))} on test: about 24 photos per ` +
      `species are far too few to learn from scratch. Frozen ImageNet features reach ` +
      `${pct(Math.min(...frozenRuns.map(test)))} to ${pct(Math.max(...frozenRuns.map(test)))}, whatever sits on top. ` +
      `Letting every layer learn lifts it to ${pct(test("5-combined"))}.`,
    runs: [
      { run: "1-random-start", label: "Random start", role: "baseline" },
      { run: "1-frozen-new-layer-plain", label: "Frozen, new layer, plain", role: "baseline" },
      { run: "1-frozen-new-layer", label: "Frozen, new layer", role: "baseline" },
      { run: "1-frozen-svm-linear", label: "Frozen, linear SVM", role: "baseline" },
      { run: "1-frozen-svm-rbf", label: "Frozen, RBF SVM", role: "baseline" },
      { run: "5-combined", label: "Fine-tuned (experiment 5)", role: "comparison" },
    ],
  },
  {
    number: 2,
    title: "Starting recipe",
    row: "2-control",
    change: "ResNet-50 fine-tuned with Adam at a fixed learning rate",
    question: "Where does a standard fine-tuning recipe land?",
    changed:
      "Nothing yet: this is the first full recipe. ResNet-50 starts from ImageNet weights, every layer learns, " +
      "photos are 448 × 448, and Adam keeps one fixed learning rate for 30 epochs.",
    result:
      `${pct(test("2-control"))} on test. Validation peaked at epoch ${RUNS["2-control"].settings?.epochKept}, then ` +
      `went up and down for the rest of the run: the fixed step size never let the weights settle.`,
    runs: [{ run: "2-control", label: "Adam, fixed learning rate", role: "kept" }],
  },
  {
    number: 3,
    title: "Learning rate schedule and optimiser",
    row: "3-sgd-lr3e-03",
    change: "Cosine learning rate schedule; SGD with momentum instead of Adam",
    question: "Was the fixed learning rate holding it back?",
    changed:
      "The learning rate now starts small for one warm-up epoch, then glides down a cosine curve to near zero. " +
      "Adam and SGD with momentum were tried at several learning rates. The best three were repeated with a second seed.",
    result:
      `The schedule alone lifted test Top-1 from ${pct(test("2-control"))} to ${pct(test("3-adam-cosine"))}. SGD and ` +
      `Adam then tied on validation (${pct(val("3-sgd-lr3e-03"))} and ${pct(val("3-adam-cosine"))}, two seeds each). ` +
      `The tie rule fixed before the runs, lower validation loss wins, picked SGD.`,
    runs: [
      { run: "2-control", label: "Adam, fixed rate (experiment 2)", role: "comparison" },
      { run: "3-adam-cosine", label: "Adam, cosine", role: "dropped" },
      { run: "3-sgd-lr1e-03", label: "SGD, cosine, rate 0.001", role: "dropped" },
      { run: "3-sgd-lr3e-03", label: "SGD, cosine, rate 0.003", role: "kept" },
      { run: "3-sgd-lr1e-02", label: "SGD, cosine, rate 0.01", role: "dropped" },
    ],
  },
  {
    number: 4,
    title: "Batch size",
    row: "3-sgd-lr3e-03",
    change: "Batch 24 tried; batch 12 stays",
    question: "Does a batch of 24 photos per step beat 12?",
    changed:
      "Batch 24 at the same learning rate, and at twice the learning rate, which keeps the step size per photo the same.",
    result:
      `Validation ${pct(val("4-batch24-lr3e-03"))} for batch 24 against ${pct(val("3-sgd-lr3e-03"))} for batch 12, ` +
      `two seeds each: no gain. Batch 12 stays, because it needs about half the graphics memory ` +
      `(${RUNS["3-sgd-lr3e-03"].settings?.peakGpuGb} GB against ${RUNS["4-batch24-lr3e-03"].settings?.peakGpuGb} GB).`,
    runs: [
      { run: "3-sgd-lr3e-03", label: "Batch 12 (experiment 3)", role: "kept" },
      { run: "4-batch24-lr3e-03", label: "Batch 24", role: "dropped" },
      { run: "4-batch24-lr6e-03", label: "Batch 24, learning rate doubled", role: "dropped" },
    ],
  },
  {
    number: 5,
    title: "Fighting overfitting",
    row: "5-combined",
    change: "Label smoothing, weight decay, dropout and strong augmentation, together",
    question: "With about 24 photos per species, can it stop memorising the training photos?",
    changed:
      "Four standard tools, first one at a time, then all together. Label smoothing: the answer key gives the " +
      "right species about 90% instead of 100%. Weight decay: every weight is pulled a little toward zero. Dropout: " +
      "30% of the last layer's inputs are switched off at random. Strong augmentation: wider random crops and small turns.",
    result:
      `Alone, each tool moved validation by about a point at most, inside the noise. Together they lifted the ` +
      `two-seed validation mean from ${pct(val("3-sgd-lr3e-03"))} to ${pct(val("5-combined"))}, and test Top-1 from ` +
      `${pct(test("3-sgd-lr3e-03"))} to ${pct(test("5-combined"))}.`,
    runs: [
      { run: "3-sgd-lr3e-03", label: "No tools (experiment 3)", role: "comparison" },
      { run: "5-label-smoothing", label: "Label smoothing", role: "dropped" },
      { run: "5-weight-decay", label: "Weight decay", role: "dropped" },
      { run: "5-dropout", label: "Dropout", role: "dropped" },
      { run: "5-strong-augment", label: "Strong augmentation", role: "dropped" },
      { run: "5-combined", label: "All four", role: "kept" },
    ],
  },
  {
    number: 6,
    title: "Networks",
    row: "5-combined",
    change: "Same recipe on ResNet-18 and ConvNeXt-Tiny; ResNet-50 stays",
    question: "Does the network design matter?",
    changed:
      "The experiment 5 recipe on two other CNNs: ResNet-18, smaller, and ConvNeXt-Tiny, a 2022 update of the " +
      "ResNet design. Each was tried at three learning rates.",
    result:
      `ResNet-18 trailed ResNet-50 by ${points(RUNS["5-combined"].val ?? 0, resnet18Best)} validation points at best. ` +
      `ConvNeXt-Tiny tied ResNet-50 (${pct(val("6-convnext-lr3e-03"))} against ${pct(val("5-combined"))}, two seeds ` +
      `each). By the rule fixed in advance, a tie keeps ResNet-50. ConvNeXt-Tiny went on to experiment 7 as its partner.`,
    runs: [
      { run: "5-combined", label: "ResNet-50 (experiment 5)", role: "kept" },
      { run: "6-resnet18-lr1e-03", label: "ResNet-18, rate 0.001", role: "dropped" },
      { run: "6-resnet18-lr3e-03", label: "ResNet-18, rate 0.003", role: "dropped" },
      { run: "6-resnet18-lr1e-02", label: "ResNet-18, rate 0.01", role: "dropped" },
      { run: "6-convnext-lr1e-03", label: "ConvNeXt-Tiny, rate 0.001", role: "dropped" },
      { run: "6-convnext-lr3e-03", label: "ConvNeXt-Tiny, rate 0.003", role: "kept" },
      { run: "6-convnext-lr1e-02", label: "ConvNeXt-Tiny, rate 0.01", role: "dropped" },
    ],
  },
  {
    number: 7,
    title: "Two networks together",
    row: "7-ensemble",
    change: "Average the probabilities of ResNet-50 and ConvNeXt-Tiny",
    question: "Do two networks beat one?",
    changed:
      "No new training. Each finished network gives 200 probabilities, and the two lists are averaged with equal " +
      "weight. A second idea also scored each photo as its mirror image and averaged that in.",
    result:
      `Averaging raised the two-seed validation mean from ${pct(val("5-combined"))} to ${pct(val("7-ensemble"))}, ` +
      `and both seeds went up. The mirror image added nothing: ResNet-50 with it scored ${pct(val("7-flip"))}, ` +
      `without it ${pct(val("5-combined"))}. Training already mirrors half the photos.`,
    runs: [
      { run: "5-combined", label: "ResNet-50 alone", role: "comparison" },
      { run: "6-convnext-lr3e-03", label: "ConvNeXt-Tiny alone", role: "comparison" },
      { run: "7-ensemble", label: "Both, averaged", role: "kept" },
      { run: "7-flip", label: "ResNet-50 + mirror image", role: "dropped" },
      { run: "7-ensemble-flip", label: "Both + mirror image", role: "dropped" },
    ],
  },
  {
    number: 8,
    title: "Final model",
    row: "8-final-ensemble",
    change: "Both networks retrained on all 4,829 photos, then averaged",
    question: "Every choice is made. Can the 800 validation photos help as training photos?",
    changed:
      "The validation photos joined training, and both networks trained once more with the settings already chosen. " +
      "The rule was fixed before training: this pair is the final model, whatever it scores. The test set scored it once.",
    result:
      `${pct(test("8-final-ensemble"))} Top-1 and ${pct(RUNS["8-final-ensemble"].perClass)} average per-class ` +
      `accuracy on test. ResNet-50 alone ${pct(test("8-final-resnet50"))}, ConvNeXt-Tiny alone ` +
      `${pct(test("8-final-convnext"))}. There is no validation score: the networks trained on those photos.`,
    runs: [
      { run: "8-final-resnet50", label: "ResNet-50", role: "member" },
      { run: "8-final-convnext", label: "ConvNeXt-Tiny", role: "member" },
      { run: "8-final-ensemble", label: "Both, averaged (this site)", role: "kept" },
    ],
  },
];

/** Final scores of the model on this site, for the "How it works" section. */
export const FINAL = RUNS["8-final-ensemble"];

export const INTRO =
  `Eight experiments, each asking one question. Validation photos: 800 of the 4,829 training photos, held back to ` +
  `choose settings. Test photos: the 1,204 official ones, scored for every run but never used to choose. The ` +
  `closest recipes ran twice with different random seeds; their validation score is the mean of both. Two seeds of one ` +
  `recipe differed by up to ${points(NOISE, 0)} points, so smaller gaps count as ties. Select an experiment to see ` +
  `what it tried.`;

// ---------- Settings in plain words ----------

/** A learning rate as plain decimals: 0.003 -> "0.003". */
const rate = (x: number): string => x.toFixed(6).replace(/0+$/, "");

/** The settings table of one run: [setting, value, what it means]. */
function settingsRows(run: RunName): [string, string, string][] {
  const r = RUNS[run];
  const s = r.settings;
  if (!s) {
    // A combination: nothing trains, it averages finished networks.
    const members = (r.members ?? []).map(networkName).join(" and ");
    return [
      ["Networks", members, "Each network gives a probability for all 200 species. The two lists are averaged with equal weight."],
      ["Mirror image", r.flip ? "Yes" : "No", r.flip ? "Each photo is also scored as its mirror image, and that is averaged in." : "Each photo is scored once, as it is."],
    ];
  }
  const arch = ARCH_NAME[s.arch] ?? s.arch;
  const archNote: Record<string, string> = {
    resnet50: "50 layers, with shortcut connections that let a deep network train.",
    resnet18: "18 layers, with shortcut connections. The smallest network here.",
    convnext_tiny: "A 2022 update of the ResNet design, with larger filters and newer layer types.",
  };
  const rows: [string, string, string][] = [
    [
      "Network",
      `${arch}, ${s.pretrained ? "ImageNet weights" : "random weights"}`,
      `${archNote[s.arch] ?? ""} ${s.pretrained ? "Its filters already learned shapes and textures from 1.28 million ImageNet photos." : "Every filter starts as noise and must learn from the bird photos alone."}`,
    ],
  ];
  if (s.svmKernel) {
    rows.push(["What learns", `200 SVMs, ${s.svmKernel} kernel`, "The frozen network turns each photo into a list of numbers. One SVM per species learns to tell it apart from the rest."]);
  } else if (s.frozen) {
    rows.push(["What learns", "Only a new last layer", "The copied layers stay exactly as ImageNet left them. A new 200-species layer learns on top."]);
  } else if (s.pretrained) {
    rows.push(["What learns", "Every layer", `The copied layers adapt to birds at a learning rate ${Math.round(s.lrNew / s.lrPretrained)} times smaller than the new layer's, so they keep most of what they knew.`]);
  } else {
    rows.push(["What learns", "Every layer", "There is nothing to keep: every weight learns from the start."]);
  }
  rows.push(
    ["Photo size", `${s.imageSize} × ${s.imageSize}`, "Twice the 224 pixels these networks were designed for, so small marks such as eye rings and wing bars keep their detail."],
    s.trainOnVal
      ? ["Training photos", "4,829, all of them", "Every choice was already made, so the validation photos joined training."]
      : ["Training photos", "4,029", "4 photos per species are held back as validation photos, to choose between settings."],
  );
  if (!s.svmKernel) {
    rows.push(
      s.optimizer === "sgd"
        ? ["Optimiser", `SGD, momentum ${s.momentum}`, "Each step moves the weights downhill on the error. Momentum keeps part of the last step, so the path is smoother."]
        : ["Optimiser", "Adam", "Gives every weight its own step size, from the history of its past steps."],
      [
        "Learning rate",
        s.pretrained && !s.frozen ? `new layer ${rate(s.lrNew)}, copied layers ${rate(s.lrPretrained)}` : rate(s.lrNew),
        "The size of each step. Too big and training jumps around; too small and it crawls.",
      ],
      s.schedule === "cosine"
        ? ["Schedule", `Cosine, ${s.warmupEpochs} warm-up epoch`, "One epoch at a small rate, then the rate glides down a cosine curve to near zero, so the last steps are small and the weights settle."]
        : ["Schedule", "Fixed", "The same step size for every epoch."],
      ["Batch", `${s.batchSize} photos per step`, "The error is averaged over this many photos before each step."],
      [
        "Epochs",
        `${s.epochs}, kept epoch ${s.epochKept}`,
        s.trainOnVal
          ? "No validation photos were left to pick an epoch, so the last one is kept."
          : "One epoch is one pass over every training photo. The weights from the epoch with the best validation score are kept.",
      ],
    );
    const tools: [string, string][] = [];
    if (s.augment) {
      tools.push([
        s.augmentStrength === "strong" ? "strong augmentation" : "augmentation",
        s.augmentStrength === "strong"
          ? "Random crops down to 35% of the photo, turns of up to 15°, mirror images and small colour changes, so no photo looks the same twice."
          : "Random crops, mirror images and small colour changes, so no photo looks the same twice.",
      ]);
    }
    if (s.labelSmoothing) tools.push([`label smoothing ${s.labelSmoothing}`, "The answer key gives the right species about 90%, not 100%, so the network is never pushed to be fully sure."]);
    if (s.weightDecay) tools.push([`weight decay ${s.weightDecay}`, "Every weight is pulled a little toward zero at each step."]);
    if (s.dropout) tools.push([`dropout ${s.dropout}`, `${Math.round(s.dropout * 100)}% of the last layer's inputs are switched off at random during training.`]);
    rows.push([
      "Against overfitting",
      tools.length ? tools.map(([name]) => name).join(", ") : "None",
      tools.length ? tools.map(([, note]) => note).join(" ") : "Nothing stops the network from memorising the training photos.",
    ]);
  }
  rows.push([
    "Training time",
    `${Math.round(s.minutes)} minutes${s.peakGpuGb ? `, ${s.peakGpuGb} GB graphics memory` : ""}`,
    "On a laptop RTX 4050 with 6 GB of graphics memory.",
  ]);
  return rows;
}

// ---------- Learning curve ----------

/** A run's training and validation Top-1 per epoch as an inline SVG, or a sentence when it has no curve. */
function curveHtml(run: RunName): string {
  const { curve, settings } = RUNS[run];
  if (!curve) {
    return settings?.svmKernel
      ? `<p class="curve-empty">No curve: an SVM is solved in one go, not trained epoch by epoch.</p>`
      : `<p class="curve-empty">No curve: nothing trains in this run. It averages finished networks. Select one of them above to see its curve.</p>`;
  }
  const [W, H, left, right, top, bottom] = [480, 220, 40, 10, 10, 26];
  const n = curve.train.length;
  const x = (epoch: number) => left + ((epoch - 1) / (n - 1)) * (W - left - right);
  const y = (v: number) => top + (1 - v) * (H - top - bottom);
  const line = (values: number[]) => values.map((v, i) => `${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => `<line class="grid" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}" /><text class="tick" x="${left - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`)
    .join("");
  const ticks = [1, 10, 20, n]
    .filter((e, i, all) => all.indexOf(e) === i && e <= n)
    .map((e) => `<text class="tick" x="${x(e)}" y="${H - 6}" text-anchor="middle">${e}</text>`)
    .join("");
  const kept = settings?.epochKept ?? null;
  const keptValue = kept === null ? null : (curve.val ?? curve.train)[kept - 1];
  const marker = keptValue === undefined || keptValue === null || kept === null ? "" : `<circle class="kept" cx="${x(kept)}" cy="${y(keptValue)}" r="4" />`;

  const last = (values: number[]) => values[values.length - 1] ?? 0;
  const summary =
    `Training Top-1 ends at ${pct(last(curve.train))}` +
    (curve.val ? `; validation ends at ${pct(last(curve.val))}.` : ". No validation line: the validation photos were training photos here.") +
    (kept !== null ? ` The dot marks the kept epoch, ${kept}.` : "");

  return `
    <svg class="curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="Learning curve of ${run}. ${summary}">
      ${grid}${ticks}
      <polyline class="train" points="${line(curve.train)}" />
      ${curve.val ? `<polyline class="val" points="${line(curve.val)}" />` : ""}
      ${marker}
    </svg>
    <p class="curve-legend"><span class="key key-val"></span>Validation <span class="key key-train"></span>Training</p>
    <p class="curve-note">Top-1 accuracy after each epoch. ${summary} A wide gap between the lines means the network remembers its training photos better than it handles new ones.</p>`;
}

// ---------- Rendering ----------

/** The right-hand panel for the selected run: its name and key, learning curve, then settings in plain words. */
function runPanel(run: RunName, label: string): string {
  return `
    <h4 class="detail-heading">${label} <span class="run-key">${run}</span></h4>
    ${curveHtml(run)}
    <h4 class="detail-heading">Settings</h4>
    <table class="settings-table">
      <tbody>${settingsRows(run)
        .map(([name, value, note]) => `<tr><th scope="row">${name}</th><td><span class="setting-value">${value}</span><span class="setting-note">${note}</span></td></tr>`)
        .join("")}</tbody>
    </table>`;
}

function detailHtml(experiment: Experiment, selected: RunName): string {
  return `
    <h3 class="detail-title"><span class="detail-number">${experiment.number}</span>${experiment.title}</h3>
    <div class="detail-grid">
      <div>
        <dl class="story-list">
          <div><dt>Question</dt><dd>${experiment.question}</dd></div>
          <div><dt>What changed</dt><dd>${experiment.changed}</dd></div>
          <div><dt>Result</dt><dd>${experiment.result}</dd></div>
        </dl>
        <h4 class="detail-heading">Runs compared</h4>
        <table class="runs-table">
          <thead><tr><th scope="col">Run</th><th scope="col" class="num">Validation</th><th scope="col" class="num">Test Top-1</th><th scope="col" class="num per-class">Per-class</th><th scope="col">Decision</th></tr></thead>
          <tbody>${experiment.runs
            .map(
              (row) => `
            <tr class="${row.role === "kept" ? "is-kept" : ""}" data-run="${row.run}" aria-selected="${row.run === selected}">
              <td><button type="button" class="row-button">${row.label}</button><span class="run-key">${row.run}</span></td>
              <td class="num">${pct(val(row.run))}${twin(row.run) ? '<span class="seeds">2 seeds</span>' : ""}</td>
              <td class="num">${pct(test(row.run))}</td>
              <td class="num per-class">${pct(RUNS[row.run].perClass)}</td>
              <td>${ROLE_TEXT[row.role]}</td>
            </tr>`,
            )
            .join("")}</tbody>
        </table>
        <p class="table-note">Select a run to see its settings and learning curve.</p>
      </div>
      <div class="detail-run" aria-live="polite">${runPanel(selected, experiment.runs.find((r) => r.run === selected)?.label ?? selected)}</div>
    </div>`;
}

/** Build the experiments table and its detail panel inside `root`. Experiment 8 (the model on this site) opens first. */
export function renderExperiments(root: HTMLElement): void {
  root.innerHTML = `
    <table class="exp-table">
      <thead><tr><th scope="col" class="exp-number">#</th><th scope="col">Experiment</th><th scope="col" class="exp-change">What changed</th><th scope="col" class="num">Validation</th><th scope="col" class="num">Test Top-1</th><th scope="col" class="num per-class">Per-class</th></tr></thead>
      <tbody>${EXPERIMENTS.map(
        (e) => `
        <tr data-experiment="${e.number}">
          <td class="exp-number">${e.number}</td>
          <td><button type="button" class="row-button">${e.title}</button></td>
          <td class="exp-change">${e.change}</td>
          <td class="num">${pct(val(e.row))}</td>
          <td class="num">${pct(test(e.row))}</td>
          <td class="num per-class">${pct(RUNS[e.row].perClass)}</td>
        </tr>`,
      ).join("")}</tbody>
    </table>
    <p class="table-note">Each row shows the run carried forward after that experiment. Experiment 8 has no validation score: its networks trained on the validation photos.</p>
    <article class="exp-detail" id="exp-detail"></article>`;

  const detail = root.querySelector<HTMLElement>("#exp-detail")!;
  let current = EXPERIMENTS[EXPERIMENTS.length - 1]!;

  const show = (experiment: Experiment, run: RunName = experiment.row) => {
    current = experiment;
    detail.innerHTML = detailHtml(experiment, run);
    for (const tr of root.querySelectorAll<HTMLElement>("tr[data-experiment]")) {
      tr.setAttribute("aria-selected", String(tr.dataset["experiment"] === String(experiment.number)));
    }
  };

  // One click handler for both tables: a whole row is the target, its button is there for the keyboard.
  root.addEventListener("click", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLElement>("tr[data-experiment], tr[data-run]");
    if (!row) return;
    if (row.dataset["experiment"]) {
      const experiment = EXPERIMENTS.find((e) => String(e.number) === row.dataset["experiment"]);
      if (experiment && experiment !== current) show(experiment);
    } else {
      show(current, row.dataset["run"] as RunName);
      detail.querySelector<HTMLButtonElement>(`tr[data-run="${row.dataset["run"]}"] button`)?.focus();
    }
  });

  show(current);
}
