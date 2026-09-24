/**
 * Static presentation constants and formatters.
 *
 * The measured numbers used to live here as hard-coded arrays. They now come
 * from the API at runtime - see src/api/client.ts and src/data/DataContext.tsx.
 * What remains is the palette and the formatters, which are design decisions
 * rather than results and have nothing to go stale against.
 */

export interface ModelInfo {
  id: string;
  label: string;
  level: string;
  input: string;
  params: number;
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
  notes?: string;
}

// Bootstrap over the 300 test source groups, 2000 resamples.

export const PALETTE = {
  seen: "#eb6834",
  unseen: "#1baf7a",
  series1: "#2a78d6",
  series2: "#eb6834",
  series3: "#1baf7a",
  series4: "#eda100",
  series5: "#e87ba4",
  light: {
    surface: "#fcfcfb",
    text: "#0b0b0b",
    secondary: "#52514e",
    grid: "#dedddb",
  },
  dark: {
    surface: "#1a1a19",
    text: "#ffffff",
    secondary: "#c3c2b7",
    grid: "#33322f",
  }
};

/** Format numeric values to exact 4 decimal places */
export function formatEer(val: number): string {
  return val.toFixed(4);
}

/** Format parameter count with commas */
export function formatParams(val: number): string {
  return val.toLocaleString('en-US');
}
