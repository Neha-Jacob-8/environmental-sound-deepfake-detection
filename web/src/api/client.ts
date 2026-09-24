/**
 * Talks to the results API.
 *
 * The measured numbers are NOT compiled into this bundle. They are fetched at
 * runtime from results/tables/ via the backend, so retraining a model and
 * re-running evaluate/compare changes what the page shows on the next reload -
 * no rebuild, no redeploy, and no chance of the site quietly disagreeing with
 * the repository.
 */

export interface ModelInfo {
  id: string;
  label: string;
  level: string;
  input: string;
  params: number;
  normalized: boolean | null;
  valEer: number;
  seenEer: number;
  unseenEer: number;
  gap: number;
  color: string;
}

export interface GeneratorInfo {
  id: string;
  name: string;
  mode: 'text-to-audio' | 'audio-to-audio';
  seen: boolean;
  notes?: string | null;
}

export interface ConfidenceInterval {
  seen: [number, number];
  unseen: [number, number];
  gap: [number, number];
  nResamples: number;
  pGapPositive: number;
}

export interface DatasetStats {
  name: string;
  protocol: string;
  totalClips: number;
  trainClips: number;
  valClips: number;
  testClips: number;
  testSourceGroups: number;
  sampleRate: string;
  duration: string;
}

export interface Summary {
  ready: boolean;
  title?: string;
  researchQuestion?: string;
  bestModel?: string;
  bestSeenEer?: number;
  bestUnseenEer?: number;
  bestGap?: number;
  nModels?: number;
  reason?: string;
}

export interface Bundle {
  summary: Summary;
  models: ModelInfo[];
  perGenerator: Record<string, Record<string, number>>;
  curves: Record<string, { valEer: number[]; trainLoss: number[] }>;
  confidence: Record<string, ConfidenceInterval>;
  generators: GeneratorInfo[];
  dataset: DatasetStats | null;
}

export interface Prediction {
  pFake: number;
  verdict: string;
  axisPos: number;
  ldaPoint: [number, number] | null;
  input: { originalSeconds: number; action: string; sampleRate: number };
  caveats: string[];
}

// Same-origin by default, so a deployment that puts the API behind the same
// host needs no configuration. In dev, vite proxies /api to the backend.
const BASE = import.meta.env.VITE_API_BASE ?? '';

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {signal});
  if (!res.ok) {
    // The backend answers 503 with a human-readable reason when a table has
    // not been generated yet; surface that rather than a bare status code.
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const fetchBundle = (signal?: AbortSignal) =>
  get<Bundle>('/api/bundle', signal);

export const audioUrl = (filename: string) => `${BASE}/api/audio/${filename}`;

export async function predict(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/api/predict`, {method: 'POST', body: form});
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}
