Build an interactive web app that explains **why AI audio detectors fail on
generators they have never seen** — using real measurements from a trained
detector, not a mock-up. React + TypeScript, single page, production quality.

Attach `ui-data.json` (720 real data points) alongside this prompt and import it.

## The finding the whole app exists to communicate

A CNN was trained to tell real environmental sound from AI-generated sound,
using four generators (G01–G04). It was then tested on three it had never seen
(G05–G07). It got worse — error rate roughly tripled. The interesting part is
*why*, and the model's own 128-dimensional feature space answers it.

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

This is measured in the full 128-d space, not read off a 2-D plot, and the seen
and unseen interquartile ranges do not overlap. A 2-D t-SNE of the same space
separates the three groups at 83.9% accuracy under 5-fold kNN (33.3% is chance),
and the layout is stable across random seeds.

## Screens

Four views in a persistent left nav (top bar under 900px). Open on view 1.

### 1. "Can you tell?" — the hook

Present one clip at a time. The user guesses **Real** or **AI-generated**, then
the answer is revealed along with what the model thought and where that clip
sits on the axis. Track the user's running score across, say, 10 clips.

The reveal is the payload: show the true label, the model's probability, and a
marker dropping onto the 0→1 axis. After a few rounds the user should notice
unseen-generator clips clustering in the middle — before the app ever explains
it. End the round with "you scored 6/10; the detector scores 9.2/10 on
generators it trained on, and 7.7/10 on ones it didn't."

**Audio.** 80 real clips are hosted and fetchable. Build the URL as
`audioBaseUrl + point.f` — both fields are in the data file. The host sends
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

### 2. The axis — the core explanation

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

### 3. The map — 2-D feature space

Scatter plot of `x`/`y` from the data, coloured by group. Filter chips for
real / seen / unseen and for individual generators. Clicking a point opens a
panel with its generator, real model name, axis position, model logit, and audio
if available. Show the three group centroids as distinct markers.

Be honest in a caption: t-SNE distances are not metric — this is a view of the
structure, and the numbers in view 2 are the actual evidence.

### 4. What it costs — results

Only now show the error rates, framed as the consequence of views 2 and 3.

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

Two caveats to state plainly rather than bury: a small gap is not automatically
good — Feature Fusion's gap looks competitive only because it is weak
everywhere, so the gap must be read next to the seen EER. And EER is an error
rate, never "accuracy".

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
- Data from the attached JSON only. No backend, no fetching, no mock API.
- No routing library; view switching is local state.
- Renders correctly with no network; audio is the only thing that needs one,
  and its absence must never break a view.
- Transitions under 200ms.

## Do not

- Do not invent data points, models, generators or numbers. What is given is
  exact and complete.
- Do not claim the app runs the detector. It displays precomputed outputs; say
  so where a user might assume otherwise.
- Do not present t-SNE distances as real distances.
- Do not add login, user accounts, settings pages or PDF export.
- Do not round the axis positions or EERs beyond the precision given.
