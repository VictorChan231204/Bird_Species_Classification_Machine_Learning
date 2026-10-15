/**
 * The final model, 8-final-ensemble, running in the browser.
 *
 * main.ts calls, in order:
 *   loadNetworks(onProgress) - downloads both ONNX files (104.7 MB) and opens an onnxruntime session for each
 *   toPixels(image)          - turns a photo into the 448 x 448 input the networks expect
 *   classify(pixels, onVote) - runs both networks, reports each vote as it lands, returns the averaged answer
 *
 * The ONNX files come from export_onnx.py. Each takes pixels 0-1, shape (1, 3, 448, 448), and returns 200
 * probabilities: the standardising (W5 s80) and the softmax (W3 s43) happen inside the file.
 */

// The WebGPU build of onnxruntime-web, which falls back to WebAssembly on the processor. Vite copies the engine
// files it references (*.wasm) into the build, so the site serves them itself, with no CDN.
import * as ort from "onnxruntime-web/webgpu";

export const IMAGE_SIZE = 448; // the crop the networks were trained and tested on
const RESIZE_SIDE = 512; // short side before the crop: 448 * 256 / 224, as in src/data.py (RESIZE_RATIO)

/** One member of the ensemble. `bytes` is the file size, used for the progress bar when the server sends none. */
interface NetworkFile {
  name: "ResNet-50" | "ConvNeXt-Tiny";
  url: string;
  bytes: number;
}

const NETWORKS: NetworkFile[] = [
  { name: "ResNet-50", url: "models/8-final-resnet50.onnx", bytes: 48_092_002 },
  { name: "ConvNeXt-Tiny", url: "models/8-final-convnext.onnx", bytes: 56_590_897 },
];

export type NetworkName = NetworkFile["name"];

/** One network's opinion of one photo: 200 probabilities, index = species number. */
export interface Vote {
  network: NetworkName;
  probs: Float32Array;
}

let sessions: Map<NetworkName, ort.InferenceSession> | null = null;

/**
 * Download both networks with a running byte count, then open a session for each.
 * The execution providers are tried in order: WebGPU (graphics card) first, else WebAssembly on the processor.
 */
export async function loadNetworks(onProgress: (loaded: number, total: number) => void): Promise<void> {
  const total = NETWORKS.reduce((sum, n) => sum + n.bytes, 0);
  const loaded = new Map<string, number>();
  const report = () => onProgress([...loaded.values()].reduce((a, b) => a + b, 0), total);

  // Both downloads run at once; each reports its own byte count into the shared total.
  const buffers = await Promise.all(
    NETWORKS.map((n) => download(n.url, (bytes) => { loaded.set(n.url, bytes); report(); })),
  );

  // "?ep=wasm" in the address forces the processor path, for testing on machines without a working graphics card.
  const executionProviders = new URLSearchParams(location.search).get("ep") === "wasm" ? ["wasm"] : ["webgpu", "wasm"];
  const opened = new Map<NetworkName, ort.InferenceSession>();
  for (const [i, n] of NETWORKS.entries()) {
    // One at a time: opening a session briefly needs the weights twice in memory.
    const started = performance.now();
    opened.set(n.name, await ort.InferenceSession.create(buffers[i]!, { executionProviders }));
    console.info(`${n.name} opened in ${Math.round(performance.now() - started)} ms (${executionProviders.join(", ")})`);
  }
  sessions = opened;
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

/**
 * Run both networks on one photo, one after the other, so the first vote can show while the second is computed.
 * Returns the ensemble: the equal-weight average of the two probability lists (src/ensemble.py, average_probs).
 */
export async function classify(pixels: Float32Array, onVote: (vote: Vote) => void): Promise<Float32Array> {
  if (!sessions) throw new Error("The networks are not loaded yet.");
  const input = new ort.Tensor("float32", pixels, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const votes: Vote[] = [];
  for (const [network, session] of sessions) {
    const output = await session.run({ pixels: input });
    const probs = output["probs"]!.data as Float32Array; // shape (1, 200): one probability per species
    const vote = { network, probs: Float32Array.from(probs) };
    votes.push(vote);
    onVote(vote);
  }
  const average = new Float32Array(200);
  for (let i = 0; i < average.length; i++) {
    average[i] = votes.reduce((sum, vote) => sum + vote.probs[i]!, 0) / votes.length; // equal weights
  }
  return average;
}

/** The k most likely species, highest first, as [species number, probability] pairs. */
export function topK(probs: Float32Array, k: number): [number, number][] {
  return [...probs.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
}
