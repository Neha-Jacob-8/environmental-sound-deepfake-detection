/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
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

export const MODELS: ModelInfo[] = [
  { id: "logmel_cnn", label: "Log-Mel CNN", level: "1 (baseline)", input: "logmel", params: 240737, valEer: 0.0033, seenEer: 0.0242, unseenEer: 0.0833, gap: 0.0592, color: "#2a78d6" },
  { id: "cnn", label: "Waveform CNN", level: "1", input: "waveform", params: 19073, valEer: 0.1442, seenEer: 0.2133, unseenEer: 0.2700, gap: 0.0567, color: "#eb6834" },
  { id: "aasist", label: "AASIST", level: "2", input: "waveform", params: 269427, valEer: 0.0892, seenEer: 0.1133, unseenEer: 0.2333, gap: 0.1200, color: "#1baf7a" },
  { id: "beats_aasist", label: "BEATs + AASIST", level: "3", input: "waveform", params: 263937, valEer: 0.1667, seenEer: 0.2400, unseenEer: 0.3433, gap: 0.1033, color: "#eda100" },
  { id: "fusion", label: "Feature Fusion", level: "proposed", input: "waveform", params: 324227, valEer: 0.1325, seenEer: 0.2033, unseenEer: 0.2967, gap: 0.0933, color: "#e87ba4" },

];

// EER per generator. G01-G04 seen, G05-G07 unseen.
export const PER_GENERATOR: Record<string, Record<string, number>> = {
  logmel_cnn: { G01:0.0100, G02:0.0133, G03:0.0367, G04:0.0300, G05:0.0967, G06:0.0667, G07:0.0833 },
  cnn: { G01:0.1767, G02:0.1900, G03:0.1800, G04:0.2933, G05:0.1333, G06:0.1633, G07:0.5100 },
  aasist: { G01:0.1000, G02:0.1033, G03:0.1733, G04:0.1033, G05:0.2867, G06:0.2367, G07:0.1133 },
  beats_aasist: { G01:0.2200, G02:0.2767, G03:0.2467, G04:0.2267, G05:0.3100, G06:0.2700, G07:0.5100 },
  fusion: { G01:0.1767, G02:0.2233, G03:0.2100, G04:0.1933, G05:0.1767, G06:0.2033, G07:0.5133 },

};

// Validation EER per epoch (index 0 = epoch 1).
export const CURVES: Record<string, { valEer: number[]; trainLoss: number[] }> = {
  logmel_cnn: { valEer:[0.1933,0.0825,0.0617,0.0467,0.0292,0.0400,0.0167,0.0133,0.0117,0.0200,0.0133,0.0117,0.0167,0.0092,0.0075,0.0158,0.0100,0.0075,0.0100,0.0125,0.0100,0.0067,0.0033,0.0033,0.0058,0.0100,0.0100,0.0058,0.0050,0.0067,0.0067], trainLoss:[0.2432,0.1562,0.0968,0.0804,0.0638,0.0545,0.0479,0.0463,0.0384,0.0347,0.0325,0.0278,0.0275,0.0217,0.0249,0.0284,0.0185,0.0193,0.0205,0.0207,0.0181,0.0126,0.0111,0.0106,0.0098,0.0092,0.0099,0.0095,0.0066,0.0069,0.0070] },
  cnn: { valEer:[0.2838,0.2558,0.2479,0.2333,0.2233,0.2429,0.2133,0.2129,0.2133,0.2033,0.1996,0.1933,0.1858,0.1754,0.1892,0.1729,0.1700,0.1754,0.1663,0.1658,0.1596,0.1604,0.1550,0.1596,0.1529,0.1533,0.1442,0.1521,0.1467,0.1471], trainLoss:[0.5865,0.4889,0.4661,0.4527,0.4466,0.4366,0.4326,0.4238,0.4194,0.4176,0.4106,0.4056,0.4013,0.3974,0.3951,0.3857,0.3889,0.3848,0.3802,0.3752,0.3696,0.3677,0.3665,0.3601,0.3590,0.3559,0.3547,0.3490,0.3511,0.3456] },
  aasist: { valEer:[0.2096,0.1617,0.1638,0.2279,0.1562,0.1500,0.3217,0.1167,0.1638,0.2750,0.1267,0.1825,0.1387,0.1700,0.1096,0.1367,0.1054,0.1600,0.0979,0.2008,0.1871,0.0929,0.1062,0.1742,0.1425,0.1517,0.1071,0.0967,0.0892,0.1312], trainLoss:[0.4214,0.3266,0.3004,0.2768,0.2658,0.2541,0.2344,0.2134,0.2095,0.1952,0.1956,0.1773,0.1668,0.1587,0.1564,0.1512,0.1507,0.1451,0.1485,0.1364,0.1321,0.1266,0.1197,0.1136,0.1150,0.1051,0.1130,0.1062,0.1052,0.1009] },
  beats_aasist: { valEer:[0.2708,0.2263,0.2263,0.2362,0.2129,0.2108,0.2246,0.1967,0.2000,0.2171,0.1667,0.1729,0.2033,0.1904,0.1708], trainLoss:[0.4537,0.3610,0.3143,0.2904,0.2825,0.2662,0.2524,0.2349,0.2247,0.2105,0.2062,0.1930,0.1817,0.1684,0.1597] },
  fusion: { valEer:[0.1967,0.1733,0.1437,0.1425,0.1425,0.1529,0.1996,0.1592,0.1504,0.1529,0.1387,0.1371,0.1325,0.1329,0.1496], trainLoss:[0.2941,0.2114,0.1808,0.1793,0.1654,0.1577,0.1529,0.1460,0.1413,0.1249,0.1264,0.1262,0.1109,0.1124,0.1119] },

};

// Bootstrap over the 300 test source groups, 2000 resamples.
export const CONFIDENCE: Record<string, { seen: [number, number]; unseen: [number, number]; gap: [number, number] }> = {
  logmel_cnn: { seen:[0.0167,0.0333], unseen:[0.0633,0.1033], gap:[0.0411,0.0767] },
  aasist:     { seen:[0.0900,0.1400], unseen:[0.1944,0.2700], gap:[0.0900,0.1467] },
};

export const GENERATORS: GeneratorInfo[] = [
  { id: "G01", name: "AudioLDM",   mode: "text-to-audio",  seen: true },
  { id: "G02", name: "AudioLDM 2", mode: "text-to-audio",  seen: true },
  { id: "G03", name: "AudioGen",   mode: "text-to-audio",  seen: true },
  { id: "G04", name: "AudioLDM",   mode: "audio-to-audio", seen: true, notes: "Preserves source recording structure; closely resembles real audio." },
  { id: "G05", name: "AudioLCM",   mode: "text-to-audio",  seen: false },
  { id: "G06", name: "TangoFlux",  mode: "text-to-audio",  seen: false },
  { id: "G07", name: "AudioLDM 2", mode: "audio-to-audio", seen: false, notes: "Same architecture as G02 in a different conditioning mode (unseen mode rather than unseen architecture). Preserves source recording structure." },
];

export const DATASET_STATS = {
  name: "EnvSDD",
  totalClips: 9900,
  trainClips: 6000,
  valClips: 1500,
  testClips: 2400,
  sampleRate: "16 kHz mono",
  duration: "4.000 s",
  testSourceGroups: 300,
  resamples: 2000,
};

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
