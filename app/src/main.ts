/**
 * The page's behaviour: show a photo, run the final model (the Ensemble), and write its answer onto the card.
 *
 * Flow: on load the first example photo shows at once and the two networks start downloading. When they are ready,
 * the photo is classified. Every later photo (an example click, a chosen file, a drop or a paste) is classified
 * straight away. The card first shows ResNet-50's top 5, then the rows move into the Ensemble's order when
 * ConvNeXt-Tiny's vote lands. The "Member models" table shows each network's own vote and test score.
 */

import "@fontsource-variable/libre-franklin";
import "./style.css";

import { classify, loadNetworks, toPixels, topK, type Vote } from "./classifier";
import confidence from "./data/confidence.json";
import examples from "./data/examples.json";
import species from "./data/species.json";
import { FINAL, INTRO, MEMBERS, pct, renderExperiments } from "./experiments";

/** Find a page element by id, typed; a missing id is a bug in index.html, so it stops the page loudly. */
function $<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`index.html has no #${id}`);
  return element as T;
}

const ui = {
  demo: $("demo"),
  photo: $<HTMLImageElement>("photo"),
  source: $("label-source"),
  species: $("label-species"),
  confidence: $("label-confidence"),
  progress: $("label-progress"),
  progressFill: $("progress-fill"),
  progressText: $("progress-text"),
  top5: $<HTMLOListElement>("top5"),
  // The "Member models" table: each network's top species and its probability, then the Ensemble's.
  vote: {
    "ResNet-50": { species: $("vote-resnet"), value: $("vote-resnet-value") },
    "ConvNeXt-Tiny": { species: $("vote-convnext"), value: $("vote-convnext-value") },
    Ensemble: { species: $("vote-ensemble"), value: $("vote-ensemble-value") },
  },
  truth: $("label-truth"),
  note: $("label-note"),
  fileInput: $<HTMLInputElement>("file-input"),
  examples: $<HTMLUListElement>("examples"),
  dropVeil: $("drop-veil"),
};

/** The photo on show: an example (with its known species) or the visitor's own photo. */
type Hanging = { kind: "example"; index: number } | { kind: "upload"; file: File };

// Below this top probability the label says the model is unsure (0.3, set and measured in make_app_data.py).
const LOW_CONFIDENCE = confidence.threshold;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let networksReady = false;
let current: { hanging: Hanging; bitmap: ImageBitmap } | null = null; // the photo on show and its decoded pixels
let runId = 0; // each new photo bumps this, so an answer for an older photo is dropped

// ---------- Hanging a photo ----------

/** Show a photo and classify it if the networks are ready. */
async function hang(hanging: Hanging): Promise<void> {
  const id = ++runId;
  const url = hanging.kind === "example" ? `examples/${examples[hanging.index]!.file}` : URL.createObjectURL(hanging.file);
  let bitmap: ImageBitmap;
  try {
    const blob = hanging.kind === "example" ? await (await fetch(url)).blob() : hanging.file;
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" }); // respects phone photo rotation
  } catch {
    showNote("That file is not a photo this browser can read. Try a JPG or PNG.");
    return;
  }
  if (id !== runId) return; // a newer photo arrived while this one decoded

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

  if (networksReady) await run(id);
}

/** A new photo fades up in place. Skipped for reduced motion. */
function fadeIn(): void {
  if (reducedMotion.matches) return;
  ui.photo.animate([{ opacity: 0.4 }, { opacity: 1 }], { duration: 500, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
}

// ---------- Running the model ----------

/** Classify the photo on show and fill the card and the table: first vote, second vote, then the Ensemble. */
async function run(id: number): Promise<void> {
  if (!current) return;
  const { hanging, bitmap } = current;
  ui.species.textContent = "Looking";
  ui.confidence.textContent = "ResNet-50 is looking first, then ConvNeXt-Tiny.";
  setVote("ResNet-50", "Looking", ""); // the previous photo's votes go until the new ones land
  setVote("ConvNeXt-Tiny", "Waiting", "");
  setVote("Ensemble", "Waiting", "");
  ui.truth.hidden = true;
  ui.note.hidden = true;

  try {
    const average = await classify(toPixels(bitmap), (vote) => {
      if (id !== runId) return;
      showVote(vote);
      if (vote.network === "ResNet-50") {
        renderTop5(vote.probs); // the first vote fills the list; the Ensemble will re-rank it in place
        ui.confidence.textContent = "ResNet-50 has voted. ConvNeXt-Tiny is looking.";
        setVote("ConvNeXt-Tiny", "Looking", "");
      }
    });
    if (id !== runId) return;
    showAnswer(average, hanging);
  } catch (error) {
    if (id !== runId) return;
    ui.species.textContent = "Something went wrong";
    ui.confidence.textContent = "";
    showNote(`The networks could not read this photo (${(error as Error).message}). Try another photo.`);
  }
}

/** Write one row's answer in the "Member models" table: a species (or a status word) and its probability. */
function setVote(row: keyof typeof ui.vote, text: string, value: string): void {
  ui.vote[row].species.textContent = text;
  ui.vote[row].value.textContent = value;
}

/** One network's top species and its probability, in its row of the "Member models" table. */
function showVote(vote: Vote): void {
  const [[index, p]] = topK(vote.probs, 1) as [[number, number]];
  setVote(vote.network, species[index]!, pct(p));
}

/** The Ensemble's answer: species, probability, re-ranked top 5, the true species for examples, and any caution. */
function showAnswer(average: Float32Array, hanging: Hanging): void {
  const [[best, p]] = topK(average, 1) as [[number, number]];
  ui.species.textContent = species[best]!;
  ui.confidence.textContent = `${pct(p)} probability`;
  setVote("Ensemble", species[best]!, pct(p));
  renderTop5(average);

  if (hanging.kind === "example") {
    const truth = examples[hanging.index]!.label;
    ui.truth.hidden = false;
    ui.truth.textContent =
      truth === best ? `Correct: the test set says ${species[truth]}.` : `Wrong: the test set says ${species[truth]}.`;
    ui.truth.dataset["verdict"] = truth === best ? "correct" : "wrong";
  }
  if (p < LOW_CONFIDENCE) {
    showNote(
      `Unsure. On test photos this unsure, the first guess was right ${pct(confidence.unsure.right)} of the time.`,
    );
  }
}

/**
 * Draw the five most likely species. Rows are kept by species, so when the order changes they slide from their
 * old place to the new one (measure, re-order, then animate the difference) instead of being redrawn.
 */
function renderTop5(probs: Float32Array): void {
  const before = new Map<string, number>();
  for (const row of ui.top5.children) before.set((row as HTMLElement).dataset["species"]!, row.getBoundingClientRect().top);

  const existing = new Map([...ui.top5.children].map((row) => [(row as HTMLElement).dataset["species"]!, row as HTMLLIElement]));
  const rows = topK(probs, 5).map(([index, p], rank) => {
    const key = String(index);
    const row = existing.get(key) ?? document.createElement("li");
    row.dataset["species"] = key;
    row.classList.toggle("is-top", rank === 0);
    row.innerHTML = `
      <span class="top5-name">${species[index]}</span>
      <span class="top5-value">${pct(p)}</span>
      <span class="bar" aria-hidden="true"><span style="transform:scaleX(${p.toFixed(4)})"></span></span>`;
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

// ---------- Loading the networks ----------

async function start(): Promise<void> {
  try {
    await loadNetworks((loaded, total) => {
      ui.progressFill.style.transform = `scaleX(${(loaded / total).toFixed(4)})`;
      ui.progressText.textContent = `${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB, once per visit`;
    });
  } catch (error) {
    ui.species.textContent = "The networks did not load";
    ui.progress.hidden = true;
    showNote(`Check the connection and reload the page. (${(error as Error).message})`);
    return;
  }
  networksReady = true;
  ui.progress.hidden = true;
  if (current) await run(runId);
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

// Test scores, all from data/results.json: the Ensemble owns the headline number; each member network shows its own.
$("score-resnet").textContent = pct(MEMBERS["ResNet-50"].test);
$("score-convnext").textContent = pct(MEMBERS["ConvNeXt-Tiny"].test);
$("score-ensemble").textContent = pct(FINAL.test);
$("label-score").textContent = `${pct(FINAL.test)} test Top-1`;
$("members-note").textContent =
  `${pct(FINAL.test)} is the Ensemble's Top-1 on the 1,204 test photos. ` +
  `Neither network reaches it alone; their probabilities averaged do.`;

// The model card's test metrics: one row per model, the Ensemble first and in bold.
const metricRows: [string, { test: number; perClass: number }, boolean][] = [
  ["Ensemble (final model)", FINAL, true],
  ["ResNet-50 alone", MEMBERS["ResNet-50"], false],
  ["ConvNeXt-Tiny alone", MEMBERS["ConvNeXt-Tiny"], false],
];
$("metrics").replaceChildren(
  ...metricRows.map(([name, run, isFinal]) => {
    const row = document.createElement("tr");
    row.classList.toggle("is-ensemble", isFinal);
    row.innerHTML = `<th scope="row">${name}</th><td class="num">${pct(run.test)}</td><td class="num">${pct(run.perClass)}</td>`;
    return row;
  }),
);
$("metrics-note").textContent =
  `Top-1: the share of the 1,204 test photos whose first guess is right. Average per-class: each species scored ` +
  `alone, then the 200 scores averaged. When the Ensemble's top probability is ${pct(confidence.threshold)} or more ` +
  `(${confidence.sure.photos.toLocaleString("en")} test photos), its first guess is right ${pct(confidence.sure.right)} ` +
  `of the time; below that (${confidence.unsure.photos} photos), ${pct(confidence.unsure.right)}.`;
$("limit-confidence").textContent =
  `Its probabilities read low on purpose: label smoothing trained both networks never to be fully sure. ` +
  `Below ${pct(confidence.threshold)}, the card marks the answer as unsure.`;
$("experiments-intro").textContent = INTRO;
renderExperiments($("experiments"));
buildExamples();
void hang({ kind: "example", index: 0 });
void start();
