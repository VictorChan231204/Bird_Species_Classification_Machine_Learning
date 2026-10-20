/**
 * The page's behaviour: show a photo, run the chosen model on it, and write its answer onto the card.
 *
 * Flow: on load the first example photo shows at once and the final model (8-final-convnext) starts downloading.
 * When it is ready, the photo is classified. Every later photo (an example click, a chosen file, a drop or a paste)
 * is classified straight away. Choosing another model in the picker downloads that model, then classifies the photo
 * on show again. The "Training" section shows the chosen run's chart; its small charts can switch the model too.
 */

import "@fontsource-variable/libre-franklin";
import "./style.css";

import { MODELS, classify, loadModel, toPixels, topK, type ModelName } from "./classifier";
import confidence from "./data/confidence.json";
import examples from "./data/examples.json";
import species from "./data/species.json";
import {
  DECISIONS_INTRO,
  FINAL,
  FINAL_RUN,
  GROUPS,
  RUNNER_UP,
  TRAINING_INTRO,
  isSvm,
  label,
  markTile,
  pct,
  renderDecisions,
  renderGrid,
  showRun,
  testScore,
  type RunName,
} from "./experiments";

/** Find a page element by id, typed; a missing id is a bug in index.html, so it stops the page loudly. */
function $<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`index.html has no #${id}`);
  return element as T;
}

const ui = {
  demo: $("demo"),
  model: $<HTMLSelectElement>("model"),
  photo: $<HTMLImageElement>("photo"),
  score: $("label-score"),
  source: $("label-source"),
  species: $("label-species"),
  confidence: $("label-confidence"),
  progress: $("label-progress"),
  progressFill: $("progress-fill"),
  progressText: $("progress-text"),
  top5: $<HTMLOListElement>("top5"),
  truth: $("label-truth"),
  note: $("label-note"),
  fileInput: $<HTMLInputElement>("file-input"),
  examples: $<HTMLUListElement>("examples"),
  dropVeil: $("drop-veil"),
  explorer: $("explorer"),
  grid: $("run-grid"),
};

/** The photo on show: an example (with its known species) or the visitor's own photo. */
type Hanging = { kind: "example"; index: number } | { kind: "upload"; file: File };

// Below this top probability the card says the final model's confidence is low (0.3, set in make_app_data.py).
const LOW_CONFIDENCE = confidence.threshold;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let model: ModelName = FINAL_RUN; // the model in the picker
let modelReady = false; // true once `model` is downloaded and open
let current: { hanging: Hanging; bitmap: ImageBitmap } | null = null; // the photo on show and its decoded pixels
// Three counters, so a slow step for an older photo or model never overwrites a newer one:
let photoId = 0; // bumped by each new photo while it decodes
let loadId = 0; // bumped by each model choice while it downloads
let runId = 0; // bumped by each classification; an answer from an older one is dropped

// ---------- Hanging a photo ----------

/** Show a photo and classify it if the model is ready. */
async function hang(hanging: Hanging): Promise<void> {
  const id = ++photoId;
  const url = hanging.kind === "example" ? `examples/${examples[hanging.index]!.file}` : URL.createObjectURL(hanging.file);
  let bitmap: ImageBitmap;
  try {
    const blob = hanging.kind === "example" ? await (await fetch(url)).blob() : hanging.file;
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" }); // respects phone photo rotation
  } catch {
    showNote("That file is not a photo this browser can read. Try a JPG or PNG.");
    return;
  }
  if (id !== photoId) return; // a newer photo arrived while this one decoded

  if (current?.hanging.kind === "upload") URL.revokeObjectURL(ui.photo.src); // free the previous upload
  current = { hanging, bitmap };
  ui.photo.src = url;
  ui.photo.hidden = false;
  ui.photo.style.setProperty("--ratio", String(bitmap.width / bitmap.height)); // style.css sizes the photo from its shape
  ui.photo.alt = hanging.kind === "example" ? `Example photo ${hanging.index + 1}` : "Your photo";
  markActiveExample(hanging.kind === "example" ? hanging.index : -1);
  // Examples get a credit line ("Example photo 3 of 8"); an upload has nothing to credit, so the line is hidden.
  ui.source.hidden = hanging.kind !== "example";
  if (hanging.kind === "example") ui.source.textContent = `Example photo ${hanging.index + 1} of ${examples.length}`;
  fadeIn();

  if (modelReady) await run(++runId);
}

/** A new photo fades up in place. Skipped for reduced motion. */
function fadeIn(): void {
  if (reducedMotion.matches) return;
  ui.photo.animate([{ opacity: 0.4 }, { opacity: 1 }], { duration: 500, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
}

// ---------- Running the model ----------

/** Classify the photo on show with the chosen model and fill the card. */
async function run(id: number): Promise<void> {
  if (!current) return;
  const { hanging, bitmap } = current;
  ui.species.textContent = "Looking";
  ui.confidence.textContent = "";
  ui.truth.hidden = true;
  ui.note.hidden = true;
  try {
    const outputs = await classify(model, toPixels(bitmap));
    if (id !== runId) return;
    showAnswer(outputs, hanging);
  } catch (error) {
    if (id !== runId) return;
    ui.species.textContent = "Something went wrong";
    showNote(`The model could not read this photo (${(error as Error).message}). Try another photo.`);
  }
}

/** The model's answer: species, probability (or SVM score), top 5, the true species for examples, and any caution. */
function showAnswer(outputs: Float32Array, hanging: Hanging): void {
  const svm = isSvm(model);
  const [[best, value]] = topK(outputs, 1) as [[number, number]];
  ui.species.textContent = species[best]!;
  ui.confidence.textContent = svm
    ? `SVM score ${value.toFixed(2)}, the highest of the 200. An SVM gives scores, not probabilities: the highest score wins.`
    : `${pct(value)} probability`;
  renderTop5(outputs, svm);

  if (hanging.kind === "example") {
    const truth = examples[hanging.index]!.label;
    ui.truth.hidden = false;
    ui.truth.textContent =
      truth === best ? `Correct: the test set says ${species[truth]}.` : `Wrong: the test set says ${species[truth]}.`;
    ui.truth.dataset["verdict"] = truth === best ? "correct" : "wrong";
  }
  // The confidence check was measured on the final model's test probabilities only, so only it gets the line.
  if (model === FINAL_RUN && value < LOW_CONFIDENCE) {
    ui.confidence.textContent +=
      `. Low confidence: on test photos under ${pct(LOW_CONFIDENCE)}, the first guess was right ` +
      `${pct(confidence.unsure.right)} of the time.`;
  }
}

/**
 * Draw the five highest outputs. Probabilities fill their bar directly. SVM scores can be negative, so their bars
 * show where each score sits between the lowest of the 200 scores (empty) and the highest (full).
 * Rows are kept by species, so when the order changes they slide from their old place to the new one (measure,
 * re-order, then animate the difference) instead of being redrawn.
 */
function renderTop5(outputs: Float32Array, svm: boolean): void {
  const before = new Map<string, number>();
  for (const row of ui.top5.children) before.set((row as HTMLElement).dataset["species"]!, row.getBoundingClientRect().top);

  const top = topK(outputs, 5);
  const highest = top[0]![1];
  const lowest = Math.min(...outputs);
  const existing = new Map([...ui.top5.children].map((row) => [(row as HTMLElement).dataset["species"]!, row as HTMLLIElement]));
  const rows = top.map(([index, value], rank) => {
    const key = String(index);
    const row = existing.get(key) ?? document.createElement("li");
    row.dataset["species"] = key;
    row.classList.toggle("is-top", rank === 0);
    const fill = svm ? (value - lowest) / Math.max(highest - lowest, 1e-6) : value;
    row.innerHTML = `
      <span class="top5-name">${species[index]}</span>
      <span class="top5-value">${svm ? value.toFixed(2) : pct(value)}</span>
      <span class="bar" aria-hidden="true"><span style="transform:scaleX(${fill.toFixed(4)})"></span></span>`;
    return row;
  });
  ui.top5.replaceChildren(...rows);

  if (reducedMotion.matches) return;
  for (const row of rows) {
    const old = before.get(row.dataset["species"]!);
    const shift = old === undefined ? null : old - row.getBoundingClientRect().top;
    const easing = "cubic-bezier(0.16, 1, 0.3, 1)";
    if (shift === null) row.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, easing });
    else if (shift !== 0) row.animate([{ transform: `translateY(${shift}px)` }, { transform: "none" }], { duration: 600, easing });
  }
}

function showNote(text: string): void {
  ui.note.hidden = false;
  ui.note.textContent = text;
}

// ---------- Choosing and loading a model ----------

/** Switch to a model: show its name and test score, download and open it, then classify the photo on show. */
async function useModel(name: ModelName): Promise<void> {
  const id = ++loadId;
  runId++; // drop any answer the previous model is still computing
  model = name;
  modelReady = false;
  ui.model.value = name;
  ui.score.textContent = `${name === FINAL_RUN ? "Final model" : label(name)} · ${pct(testScore(name))} test Top-1`;
  ui.species.textContent = "Loading the model";
  ui.confidence.textContent = "";
  ui.top5.replaceChildren();
  ui.truth.hidden = true;
  ui.note.hidden = true;
  ui.progress.hidden = false;
  ui.progressFill.style.transform = "scaleX(0)";
  ui.progressText.textContent = `Downloading ${name}, ${(MODELS[name].bytes / 1e6).toFixed(0)} MB`;
  try {
    await loadModel(name, (loaded, total) => {
      ui.progressFill.style.transform = `scaleX(${Math.min(loaded / total, 1).toFixed(4)})`;
      ui.progressText.textContent = `${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
    });
  } catch (error) {
    if (id !== loadId) return;
    ui.species.textContent = "The model did not load";
    ui.progress.hidden = true;
    showNote(`Check the connection and try again. (${(error as Error).message})`);
    return;
  }
  if (id !== loadId) return; // the visitor picked another model while this one loaded
  modelReady = true;
  ui.progress.hidden = true;
  await run(++runId);
}

/** Fill the model picker: one option group per experiment, the final model first and selected. */
function buildPicker(): void {
  ui.model.replaceChildren(
    ...GROUPS.map((group) => {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group.title;
      for (const { run, label: text } of group.runs) optgroup.append(new Option(`${text} (${pct(testScore(run))})`, run));
      return optgroup;
    }),
  );
  ui.model.value = FINAL_RUN;
  ui.model.addEventListener("change", () => {
    const name = ui.model.value as ModelName;
    void useModel(name);
    showInExplorer(name);
  });
}

/** Show a run in the Training section's large chart and mark its small chart. */
function showInExplorer(run: RunName): void {
  showRun(ui.explorer, run, (picked) => {
    void useModel(picked);
    ui.demo.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth" });
  });
  markTile(ui.grid, run);
}

// ---------- Examples, file picker, drop and paste ----------

function buildExamples(): void {
  ui.examples.replaceChildren(
    ...examples.map((example, index) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example";
      button.setAttribute("aria-label", `Example photo ${index + 1}`);
      button.innerHTML = `<img src="examples/${example.file}" alt="" loading="lazy" />`;
      button.addEventListener("click", () => void hang({ kind: "example", index }));
      li.append(button);
      return li;
    }),
  );
}

function markActiveExample(active: number): void {
  [...ui.examples.querySelectorAll("button")].forEach((button, index) => {
    button.setAttribute("aria-current", String(index === active));
  });
}

/** The first image file in a drop or paste, if any. */
function firstImage(files: FileList | undefined | null): File | null {
  return [...(files ?? [])].find((file) => file.type.startsWith("image/")) ?? null;
}

ui.fileInput.addEventListener("change", () => {
  const file = firstImage(ui.fileInput.files);
  if (file) void hang({ kind: "upload", file });
  ui.fileInput.value = ""; // allow choosing the same file again
});

// Drag and drop anywhere on the page. dragenter/dragleave fire for every child element, so count them.
let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  if (!event.dataTransfer?.types.includes("Files")) return;
  dragDepth++;
  ui.dropVeil.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) ui.dropVeil.hidden = true;
});
window.addEventListener("dragover", (event) => event.preventDefault()); // without this the browser opens the file
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  ui.dropVeil.hidden = true;
  const file = firstImage(event.dataTransfer?.files);
  if (file) {
    void hang({ kind: "upload", file });
    ui.demo.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth" });
  } else {
    showNote("That was not a photo. Drop a JPG or PNG.");
  }
});
window.addEventListener("paste", (event) => {
  const file = firstImage(event.clipboardData?.files);
  if (file) void hang({ kind: "upload", file });
});

// ---------- Start ----------

// The text below the diagram and the results table, all from data/results.json and data/confidence.json.
$("why-final").textContent =
  `On the validation photos it tied ResNet-50, the main network of the experiments. Both were retrained on all ` +
  `training photos, and on the test photos ConvNeXt-Tiny scored ${pct(FINAL.test)} Top-1 against ResNet-50's ` +
  `${pct(RUNNER_UP.test)}. Because the test set made this last choice, part of that gap may be chance.`;

// The results table: the final model first and marked, then the runner-up.
const metricRows: [string, { test: number; perClass: number }, boolean][] = [
  ["ConvNeXt-Tiny (final model)", FINAL, true],
  ["ResNet-50 (runner-up)", RUNNER_UP, false],
];
$("metrics").replaceChildren(
  ...metricRows.map(([name, scores, isFinal]) => {
    const row = document.createElement("tr");
    row.classList.toggle("is-final", isFinal);
    row.innerHTML = `<th scope="row">${name}</th><td class="num">${pct(scores.test)}</td><td class="num">${pct(scores.perClass)}</td>`;
    return row;
  }),
);
$("metrics-note").textContent =
  `Scored on the 1,204 test photos. Top-1: the share whose first guess is right. Average per-class: each ` +
  `species scored alone, then the 200 scores averaged. Every other run's scores are in the model picker and the ` +
  `Training section.`;
$("limit-confidence").textContent =
  `Its probabilities read low on purpose: label smoothing trained it never to be fully sure. On the ` +
  `${confidence.unsure.photos} test photos where its top probability is under ${pct(LOW_CONFIDENCE)}, the first ` +
  `guess was right ${pct(confidence.unsure.right)} of the time; above that, ${pct(confidence.sure.right)}.`;
$("decisions-intro").textContent = DECISIONS_INTRO;
renderDecisions($("decisions"));
$("training-intro").textContent = TRAINING_INTRO;
renderGrid(ui.grid, showInExplorer);

buildPicker();
buildExamples();
showInExplorer(FINAL_RUN);
void hang({ kind: "example", index: 0 });
void useModel(FINAL_RUN);
