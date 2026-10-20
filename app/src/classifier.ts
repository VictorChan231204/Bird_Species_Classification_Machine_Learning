/**
 * Any exported run, running in the browser: the final model by default, or any run the visitor picks.
 *
 * main.ts calls, in order:
 *   loadModel(run, onProgress) - downloads that run's ONNX file (23 to 66 MB) and opens an onnxruntime session
 *   toPixels(image)            - turns a photo into the 448 x 448 input every run expects
 *   classify(run, pixels)      - runs the network once and returns its 200 outputs
 *
 * The ONNX files come from export_onnx.py --all; data/models.json lists them. A CNN file takes pixels 0-1, shape
 * (1, 3, 448, 448), and returns 200 probabilities: the standardising (W5 s80) and the softmax (W3 s43) happen inside
 * the file. An SVM file (1-frozen-svm-*) returns 200 SVM scores instead, which are not probabilities.
 */

// The WebGPU build of onnxruntime-web, which falls back to WebAssembly on the processor. Vite copies the engine
// files it references (*.wasm) into the build, so the site serves them itself, with no CDN.
import * as ort from "onnxruntime-web/webgpu";

import models from "./data/models.json";

export type ModelName = keyof typeof models;

/** One runnable model as models.json lists it (make_app_data.py, write_models). */
export interface ModelFile {
  file: string;
  bytes: number;
  kind: "cnn" | "svm"; // "cnn": 200 probabilities; "svm": 200 SVM scores
}

export const MODELS = models as Record<ModelName, ModelFile>;

export const IMAGE_SIZE = 448; // the crop the networks were trained and tested on
const RESIZE_SIDE = 512; // short side before the crop: 448 * 256 / 224, as in src/data.py (RESIZE_RATIO)

// Only the open model's session is kept: 30 sessions at once would need gigabytes of memory.
let open: { name: ModelName; session: ort.InferenceSession } | null = null;
// The last few downloaded files stay in memory (about 50 MB each), so switching back to one is instant.
const downloaded = new Map<ModelName, Uint8Array>();
const KEEP_DOWNLOADS = 3;

/**
 * Download one model with a running byte count, then open its session (closing the previous one).
 * The execution providers are tried in order: WebGPU (graphics card) first, else WebAssembly on the processor.
 */
export async function loadModel(name: ModelName, onProgress: (loaded: number, total: number) => void): Promise<void> {
  if (open?.name === name) return;
  const { file, bytes: expected } = MODELS[name];
  let bytes = downloaded.get(name);
  if (!bytes) {
    bytes = await download(file, (loaded) => onProgress(loaded, expected));
    downloaded.set(name, bytes);
    // A Map keeps insertion order, so the first key is the oldest download.
    if (downloaded.size > KEEP_DOWNLOADS) downloaded.delete(downloaded.keys().next().value!);
  }

  // "?ep=wasm" in the address forces the processor path, for testing on machines without a working graphics card.
  const executionProviders = new URLSearchParams(location.search).get("ep") === "wasm" ? ["wasm"] : ["webgpu", "wasm"];
  await open?.session.release(); // free the previous model's memory before opening the next
  open = null;
  const started = performance.now();
  open = { name, session: await ort.InferenceSession.create(bytes, { executionProviders }) };
  console.info(`${name} opened in ${Math.round(performance.now() - started)} ms (${executionProviders.join(", ")})`);
}

/** Fetch a file as bytes, calling onBytes with the running total as chunks arrive. */
async function download(url: string, onBytes: (bytes: number) => void): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url} answered ${response.status}`);
  const chunks: Uint8Array[] = [];
  let received = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(received);
  }
  // Join the chunks into one block, which is what onnxruntime reads.
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

/**
 * The test transform from src/data.py, in the browser: resize the short side to 512, keep the middle 448 square,
 * pixels 0-255 -> 0-1, and lay the numbers out as 3 colour planes (red, then green, then blue).
 * Done in one drawImage call: the crop is cut from the original photo and scaled straight to 448 x 448.
 */
export function toPixels(image: ImageBitmap): Float32Array {
  const short = Math.min(image.width, image.height);
  const crop = (short * IMAGE_SIZE) / RESIZE_SIDE; // the part of the original that becomes the 448 square
  const canvas = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("This browser cannot read the photo's pixels.");
  ctx.imageSmoothingQuality = "high"; // smooth downscaling, close to the bilinear resize used in training
  ctx.drawImage(image, (image.width - crop) / 2, (image.height - crop) / 2, crop, crop, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const rgba = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data; // 4 numbers per pixel: red, green, blue, alpha
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const pixels = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    pixels[i] = rgba[i * 4]! / 255; // red plane
    pixels[plane + i] = rgba[i * 4 + 1]! / 255; // green plane
    pixels[2 * plane + i] = rgba[i * 4 + 2]! / 255; // blue plane
  }
  return pixels;
}

/** Run the open model on one photo: 200 outputs, index = species number (probabilities, or SVM scores). */
export async function classify(name: ModelName, pixels: Float32Array): Promise<Float32Array> {
  if (open?.name !== name) throw new Error(`${name} is not loaded yet.`);
  const input = new ort.Tensor("float32", pixels, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const output = await open.session.run({ pixels: input });
  const values = output[open.session.outputNames[0]!]!.data as Float32Array; // shape (1, 200)
  return Float32Array.from(values);
}

/** The k highest outputs, highest first, as [species number, value] pairs. */
export function topK(values: Float32Array, k: number): [number, number][] {
  return [...values.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
}
