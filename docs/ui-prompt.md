Build an interactive web app that presents a final-year research project on
audio deepfake detection — stating its question, answering it, and explaining
the answer using real measurements from the trained models. React + TypeScript,
single page, production quality.

Attach `ui-data.json` (760 real data points) alongside this prompt and import it.

## The project — use this exact framing, do not invent your own

**Title:** Robust Environmental Sound Deepfake Detection Against Unseen Audio
Generators

**Research question:** Can combining complementary acoustic representations
improve robustness of environmental sound deepfake detection against unseen
audio generators?

Built on the EnvSDD dataset (Interspeech 2025, CC BY 4.0), following the ESDD
2026 challenge protocol. Detectors are trained on generators **G01–G04** and
evaluated on **G05–G07**, which appear only at test time. The primary metric is
**EER** — equal error rate, lower is better — reported per generator rather than
pooled.

Working subset: 9,900 clips, each 16 kHz mono and exactly 4.000 s.
Train 6,000 / validation 1,500 / test 2,400.

### The generators

| ID | Generator | Conditioning | Status |
|----|-----------|--------------|--------|
| G01 | AudioLDM | text-to-audio | seen in training |
| G02 | AudioLDM 2 | text-to-audio | seen in training |
| G03 | AudioGen | text-to-audio | seen in training |
| G04 | AudioLDM | audio-to-audio | seen in training |
| G05 | AudioLCM | text-to-audio | **unseen architecture** |
| G06 | TangoFlux | text-to-audio | **unseen architecture** |
| G07 | AudioLDM 2 | audio-to-audio | **unseen conditioning only** |

A distinction the app should make, because it is easy to get wrong: only G05 and
G06 are architecturally new. G07 is the *same model as G02* in a different
conditioning mode, so it tests an unseen mode rather than an unseen
architecture. Audio-to-audio generators (G04, G07) start from the real recording
and preserve its structure, which is why they resemble real clips more closely.

## The answer, in two parts

### Part 1 — the honest answer to the research question: no

Feature fusion did not improve robustness. Combining the CNN and the
BEATs+AASIST branches produced a model that sits *between* its own two branches
on seen generators and is **worse than the plain waveform CNN** on unseen ones.
Two weak branches did not make a strong one.

The app must present this plainly. A negative result honestly reported is the
finding, not a failure to hide behind a neutral dashboard.

### Part 2 — why generalisation fails, from the model's own feature space

The best detector (a log-Mel CNN) reaches 0.0242 EER on seen generators and
0.0833 on unseen ones — roughly triple the error. Its 128-dimensional penultimate
layer shows where that comes from.

Take the axis running from the centroid of real clips to the centroid of
seen-fake clips. That axis is, in effect, what the detector learned "fake" means.
Project every test clip onto it, where **0.0 = the real centroid** and
**1.0 = the seen-fake centroid**:

| Generator | | Position on the axis | IQR |
|---|---|---|---|
| REAL | real | **0.000** | [−0.17, +0.13] |
| G01 AudioLDM | seen | +1.107 | [0.97, 1.25] |
| G02 AudioLDM 2 | seen | +1.003 | [0.89, 1.11] |
| G03 AudioGen | seen | +0.911 | [0.83, 1.01] |
| G04 AudioLDM (A2A) | seen | +0.979 | [0.90, 1.09] |
| G05 AudioLCM | **unseen** | **+0.683** | [0.55, 0.82] |
| G06 TangoFlux | **unseen** | **+0.695** | [0.59, 0.80] |
| G07 AudioLDM 2 (A2A) | **unseen** | **+0.646** | [0.50, 0.79] |

**Unseen generators land about 68% of the way along.** They are fake, and the
model pushes them in the right direction — but not far enough to clear the
decision boundary reliably. The detector did not learn "fake"; it learned "far
from real in this one specific direction", and unfamiliar generators only travel
part of the way. That single sentence is the thesis of the app.

Measured in the full 128-d space, not read off a 2-D plot, and the seen and
unseen interquartile ranges do not overlap. A 2-D t-SNE of the same space
separates the three groups at 83.9% under 5-fold kNN (33.3% is chance), stable
across random seeds.

## Naming and chrome — do not improvise these

The app is titled **"Robust Environmental Sound Deepfake Detection Against
Unseen Audio Generators"**, shortened in the header to **"Unseen-Generator
Deepfake Detection"** if space demands. A single line under it states the
research question verbatim.

Do **not** invent breadcrumb trails, section taxonomies, lab names, author
names, institution names, dates, logos, or category labels such as "Research
Brief", "Benchmark" or "Acoustic Forensics". Nothing of the sort is given here
because none of it should appear. The nav contains exactly the five views below
and nothing else.

## Screens

Six views in a persistent left nav (top bar under 900px). Open on view 0.

### 0. The question — the landing view

State the problem in the researcher's own terms, in this order:

1. The title and the research question, verbatim.
2. One short paragraph of setup: detectors are trained on four generators and
   tested on three they have never seen; the metric is EER; the dataset is
   EnvSDD under the ESDD 2026 protocol.
3. The answer, stated immediately rather than withheld: **fusion did not
   improve robustness**, and the best detector still roughly triples its error
   rate on unseen generators. Put the two numbers — 0.0242 seen, 0.0833
   unseen — in large type.
4. Three links into the views that substantiate it: "hear the problem
   yourself" (view 1), "see why it happens" (view 2), "see what it costs"
   (view 4).

Resist the urge to build a marketing hero. This is the abstract of a paper,
rendered well.

### 1. Test it yourself — the live detector

**The trained model actually runs here, in the browser.** This is not a replay
of stored results: the exported network is fetched and executed on whatever
audio the user provides.

```
model:  {baseUrl}model/logmel_cnn.onnx        5.31 MB, ONNX opset 17
input:  "waveform"   float32 [batch, 64000]   16 kHz mono, raw
outputs:
  "p_fake"     float32 [batch]        probability the clip is AI-generated
  "axis_pos"   float32 [batch]        position on the real -> seen-fake axis
  "embedding"  float32 [batch, 128]   penultimate features
```

Use `onnxruntime-web` from a CDN. Fetch the model lazily — only when the user
first opens this view — and show real progress while it downloads, because
5.31 MB is not instant on a phone.

**Audio handling.** Accept drag-and-drop, a file picker, and microphone
recording. Decode with `AudioContext.decodeAudioData`, downmix to mono, resample
to exactly 16 kHz with an `OfflineAudioContext`, then take exactly 64,000
samples (4.000 s). Pad with zeros if shorter. If longer, use the first 4 seconds
but say so, and offer a simple slider to choose which 4-second window to send.

**Do not normalise the audio in JavaScript.** Peak normalisation is baked into
the exported graph. Doing it again in JS would change the input the model sees
and silently degrade every prediction.

**The result panel** shows, in this order:

1. The verdict — "likely AI-generated" or "likely real" — with `p_fake` as a
   labelled bar, not a bare number.
2. The clip dropped onto the **same 0→1 axis as view 3**, using `axis_pos`, with
   the REAL / seen / unseen reference bands behind it. This is the payoff: a
   user's own audio placed in the same geometry as the research finding.
3. The log-Mel spectrogram the model computed, if you can render it.

**Three honesty requirements, all non-negotiable:**

- The model was trained on **environmental sound** — 4-second field recordings.
  Speech, music, or silence is out of distribution and the output is not
  meaningful. Detect the obvious cases where you can and warn; otherwise carry a
  standing note on this view.
- `axis_pos` is a linear projection, so it is exact for new audio. The **t-SNE
  map in view 4 cannot place new points** — t-SNE has no transform for unseen
  data. Never plot a user's clip on that map; place it on the axis instead, and
  say why if the user might expect otherwise.
- This detector reaches 0.0242 EER on generators it trained on and 0.0833 on
  ones it did not. It is a course research model, not a production tool. Say so.

### 2. "Can you tell?" — the hook

Present one clip at a time. The user guesses **Real** or **AI-generated**, then
the answer is revealed along with what the model thought and where that clip
sits on the axis. Track the user's running score across, say, 10 clips.

The reveal is the payload: show the true label, the model's probability, and a
marker dropping onto the 0→1 axis. After a few rounds the user should notice
unseen-generator clips clustering in the middle — before the app ever explains
it. End the round with "you scored 6/10; the detector scores 9.2/10 on
generators it trained on, and 7.7/10 on ones it didn't."

**Audio.** 80 real clips are hosted and fetchable. Build the URL as
`audioBaseUrl + point.f` — both fields are in the data file. The ONNX model in
view 1 sits alongside them, at `audioBaseUrl` with `audio/` swapped for
`model/logmel_cnn.onnx`. The host sends
`Access-Control-Allow-Origin: *`, so plain `<audio>` playback and Web Audio
decoding (for a waveform or spectrogram) both work; set `crossOrigin="anonymous"`
if you decode.

Only points carrying `"audio": true` have a file — 80 of the 760. The challenge
must draw its rounds **only** from those, and the map should mark them as
playable. If a fetch fails, degrade gracefully: show the clip's data with a
short "audio unavailable" note, never a broken player or an error state. The
app must stay fully usable if every fetch fails.

The 80 clips are 10 complete source groups — one real recording plus its seven
generated versions, drawn from all five source datasets in the test split. So
the challenge can offer the *same underlying recording* as real and as seven
different fakes, which is the sharpest version of the question.

### 3. The axis — the core explanation

The centrepiece. A horizontal axis from 0.0 to ~1.3, labelled **"real centroid"**
at 0 and **"seen-fake centroid"** at 1. For each of the eight groups (REAL,
G01–G07) draw a distribution along it — a box showing the IQR with a marker at
the mean, or a ridgeline/strip of the actual points, whichever you can make
legible. Order them by position, not by ID, so the ranking is visible.

Shade the region around 0.65–0.70 to show where all three unseen generators fall.
Let the user hover any distribution for its exact numbers.

Add a stepped walkthrough — four or five short captions the user advances
through — that builds the explanation: (1) here is real, (2) here is what the
model learned fake looks like, (3) here is where the unseen generators actually
landed, (4) that shortfall is the generalisation gap, (5) and this is why the
error rate triples.

### 4. The map — 2-D feature space

Scatter plot of `x`/`y` from the data, coloured by group. Filter chips for
real / seen / unseen and for individual generators. Clicking a point opens a
panel with its generator, real model name, axis position, model logit, and audio
if available. Show the three group centroids as distinct markers.

Be honest in a caption: t-SNE distances are not metric — this is a view of the
structure, and the numbers in view 3 are the actual evidence.

### 5. What it costs — results, and the answer to the question

Show the error rates, framed as the consequence of views 3 and 4, and as the
evidence for the negative answer stated on the landing view.

| Model | seen EER | unseen EER | gap |
|---|---|---|---|
| Log-Mel CNN | 0.0242 | 0.0833 | +0.0592 |
| Waveform CNN | 0.2133 | 0.2700 | +0.0567 |
| AASIST | 0.1133 | 0.2333 | +0.1200 |
| BEATs + AASIST | 0.2400 | 0.3433 | +0.1033 |
| Feature Fusion | 0.2033 | 0.2967 | +0.0933 |

Per-generator EER for the Log-Mel CNN, the model the embeddings come from:
G01 0.0100, G02 0.0133, G03 0.0367, G04 0.0300, G05 0.0967, G06 0.0667,
G07 0.0833.

Lay the fusion row against its own two branches explicitly — a small grouped
comparison of Waveform CNN, BEATs + AASIST, and Feature Fusion — so a reader can
see for themselves that the combination did not beat its parts. That comparison
is the direct answer to the research question and deserves its own labelled
block, not a row buried in a five-model table.

Three caveats to state plainly rather than bury:

- A small gap is not automatically good. Feature Fusion's gap looks competitive
  only because the model is weak everywhere, so the gap must always be read next
  to the seen EER.
- EER is an error rate, never "accuracy".
- G07 behaves inconsistently across models and the app should say so rather than
  smooth it over. For the Log-Mel CNN it is the *easiest* of the three unseen
  generators (0.0833), which fits it being an unseen conditioning mode of an
  architecture already seen. But the Waveform CNN, Feature Fusion and
  BEATs + AASIST all score about 0.51 on it — chance. The information is
  evidently there, and those three models fail to use it.

## Data shape

```ts
type Point = {
  x: number; y: number;      // 2-D t-SNE coordinates
  g: string;                 // "REAL" | "G01".."G07"
  grp: "real" | "seen" | "unseen";
  p: number;                 // position on the real->seen-fake axis
  logit: number;             // model output, higher = more fake
  sid: number;               // source recording id - clips sharing one are the
                             // same recording through different generators
  f: string;                 // filename, append to audioBaseUrl
  audio?: true;              // present only when the file is fetchable
};
```

760 points; 80 have audio. The scores were computed on these exact files, so
what a user hears is what the model was given.

`ui-data.json` also carries `axisPosition`, `groupMeans`, `centroids2d`,
`distanceToRealCentroid`, and `knn3wayAccuracy`. Use those rather than
recomputing from the sampled points.

Generator reference — show real names, never bare IDs:
G01 AudioLDM · G02 AudioLDM 2 · G03 AudioGen · G04 AudioLDM (audio-to-audio) ·
G05 AudioLCM · G06 TangoFlux · G07 AudioLDM 2 (audio-to-audio).
G04 and G07 are audio-to-audio, so they preserve the source recording's
structure and resemble real clips more than the text-to-audio ones do.

## Visual design

Calm and precise — a well-made explanatory piece, closer to a science article
than a SaaS dashboard. Generous whitespace, restrained colour, typography doing
the work. No gradient hero, no glassmorphism, no neon, no shadow on everything.

Use this palette exactly; it is validated for colour-vision deficiency:

```
real     #2a78d6   seen  #eb6834   unseen  #1baf7a
series 4 #eda100   series 5 #e87ba4
light: surface #fcfcfb  text #0b0b0b  secondary #52514e  grid #dedddb
dark:  surface #1a1a19  text #ffffff  secondary #c3c2b7  grid #33322f
```

A group keeps its colour everywhere and when others are filtered out. Never
encode meaning by colour alone — always pair with a label, shape or position.

Chart rules: never two y-scales on one plot; thin marks and 2px lines; recessive
grid; direct value labels rather than forcing an axis read; a legend whenever
two or more series show; a tooltip on every chart; axis labels that name the
unit; no pie or donut charts anywhere.

Light and dark mode from `prefers-color-scheme`, plus a manual toggle that
overrides it in both directions. Responsive to 400px — the scatter and axis stay
legible, wide elements scroll inside their own container, never the page.
Keyboard-navigable with visible focus rings; ARIA labels on all controls.

## Technical

- React + TypeScript, Recharts or D3 for the visuals, Tailwind if available.
- `onnxruntime-web` from a CDN for view 1. WASM backend; do not require WebGPU.
- Static data from the attached JSON. No backend and no API: the only network
  requests are the audio clips and the ONNX model, both from the URL above.
- No routing library; view switching is local state.
- Views 0, 2, 3, 4 and 5 render fully with no network. Views 1 and 2 need
  fetches; if either fails, show a clear recoverable message and leave the rest
  of the app working. A failed model download must never blank the page.
- Transitions under 200ms.

## Do not

- Do not invent data points, models, generators or numbers. What is given is
  exact and complete.
- Do not claim views 2-5 run the model. Only view 1 does; everything else
  displays precomputed outputs, and the app should be clear about which is which.
- Do not plot user-supplied audio on the t-SNE map. Use the axis.
- Do not peak-normalise audio in JavaScript; the model already does it.
- Do not present t-SNE distances as real distances.
- Do not add login, user accounts, settings pages or PDF export.
- Do not round the axis positions or EERs beyond the precision given.
- Do not invent branding, breadcrumbs, author or institution names, dates,
  version numbers, or section categories. If a label is not in this prompt, it
  does not belong in the app.
- Do not soften the negative result. Fusion did not improve robustness, and the
  app says so in plain words on the landing view.
- Do not describe G05, G06 and G07 as equivalent. G07 is an unseen conditioning
  mode of an architecture the model already saw; only G05 and G06 are new
  architectures.
