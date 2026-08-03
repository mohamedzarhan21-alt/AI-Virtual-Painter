# AI Virtual Painter
 production-quality desktop application that turns your index finger into a
paintbrush. A webcam tracks your hand in real time with MediaPipe, and a complete raster
painting engine converts the motion into digital artwork — fifteen brushes, fourteen colour
modes, unlimited undo, automatic shape recognition and resolution-independent export, all
behind a holographic heads-up display painted directly into the video frame.

![mode](https://img.shields.io/badge/python-3.11%2B-0a7f96) ![single file](https://img.shields.io/badge/deliverable-one%20file-22e6ff)

---

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [Gesture reference](#gesture-reference)
- [Keyboard and mouse reference](#keyboard-and-mouse-reference)
- [Folder structure](#folder-structure)
- [Technologies used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to run](#how-to-run)
- [Usage guide](#usage-guide)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Performance](#performance)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)

---

## Project overview

**What problem does it solve, and for whom?** It removes the tablet from digital drawing.
Anyone with a webcam can paint, sketch, annotate a scene or demo an idea in mid-air, with the
result exportable as a real image file.

**Tone.** One direction, committed to end to end: *Iron Man HUD read through Nothing OS
restraint*. A near-black cinematic void, a single electric-cyan instrument colour, a magenta
signal accent, thin one-pixel neon rules, monospaced telemetry, corner brackets and a rotating
graticule. Nothing is a stock widget — the title bar, the docks, the sliders and the palettes
are all rasterised into the video frame, which is what makes genuine frosted glass and additive
neon possible.

**The one thing you remember.** The **radial palette**: hold up two fingers and a three-ring
holographic instrument blooms around your fingertip — sizes on the inner ring, all fifteen
brushes on the middle ring, all fourteen colours on the outer ring — each one previewed with a
miniature stroke painted by the real engine. Rest your finger on a sector to select it; hold
perfectly still over the hub to save.

---

## Features

### Computer vision
- MediaPipe **HandLandmarker** (Tasks API), one hand, all 21 landmarks.
- Index-fingertip tracking through a **1-Euro filter** — motionless when you hover, no lag when
  you flick.
- Finger extension measured **radially from the wrist**, so gestures work with the hand
  rotated or upside down.
- Every gesture threshold is normalised by palm width, so it behaves the same near the lens or
  a metre away.
- Inference runs on a background thread against a downscaled frame; pose changes are debounced.

### Painting engine — 15 brushes
| Classic | Expressive |
|---|---|
| Brush · soft round with pressure taper | Particle · normal-scatter emitters |
| Pencil · graphite grain | Glow · wide additive halo |
| Neon · triple pass with a hot core | Fire · ember ramp with thermal rise |
| Marker · hard edge, flat ink | Spark · radial white-hot streaks |
| Watercolour · pigment bleed | Galaxy · nebula and starfield |
| Airbrush · 16-jet gaussian spray | Smoke · turbulent drifting volume |
| Calligraphy · 42° broad-edge nib | Magic trail · hue shift with glints |
| Highlighter · chisel, flat translucency | |

- Six stroke widths (5 / 10 / 15 / 20 / 30 / 40 px) and a continuous opacity control.
- Catmull-Rom spline smoothing with uniform arc-length resampling, antialiased throughout.
- Velocity-driven taper on the brushes where it reads as pressure.

### Colours
Red, orange, yellow, green, cyan, blue, purple, magenta, pink, brown, white and black, plus two
procedural modes: **Gradient** (a cyan → magenta → amber ramp that travels along the stroke) and
**Rainbow** (hue cycling with distance).

### Canvas and history
- A separate transparent BGRA drawing layer composited over the camera.
- Unlimited redraw: the document is a **vector stroke list**, and the raster is a cache.
- 100-state undo and redo covering strokes, erases, shape snapping and canvas clears.
- Eraser with a proportionally larger radius, applied as alpha subtraction.
- **Solo mode** hides the camera and shows the artwork on the void backdrop.

### Shape recognition
Draw a rough circle, rectangle, triangle, line or single-stroke arrow and it is replaced with
perfect geometry — re-rendered with whatever brush you were using, so a snapped circle is still
a fire circle. Every threshold is a fraction of the stroke's own length, so it works at any
scale. Can be toggled off.

### Export
PNG, JPG (composited over the live camera frame), transparent PNG, and a high-resolution PNG.
Because strokes are vectors, the high-resolution export **re-rasterises at scale** rather than
upscaling a bitmap — a 1280×720 canvas exports as a genuine 3840×2160 image.

### Interface
Frameless window with a painted title bar; frosted-glass panels with real background blur;
floating brush dock, colour rail and instrument column; the three-ring radial palette; a
morphing mode capsule; an animated brush preview painted by the engine itself; toasts;
hold-progress rings; a neon hand wireframe; a morphing reticle with a comet trail; a cinematic
grade with vignette, scanlines, grain and corner brackets; and a boot sequence that covers
start-up. Chrome fades back to 22 % while you paint, so the controls float rather than occupy.

---

## Gesture reference

| Gesture | Action |
|---|---|
| Index up, middle down | **Draw** |
| Index + middle up | **Select** — the radial palette blooms at your fingertip |
| Index + middle + ring up | **Erase** |
| Open palm, held 2 s | **Clear canvas** (undoable) |
| Thumb + index pinch | **Undo** |
| Thumb + middle pinch | **Redo** |
| Victory held still 2 s | **Save artwork** |

In Select mode, rest your fingertip on a sector for 0.55 s to choose it. A progress arc shows
the dwell. The three rings are sizes (inner), brushes (middle) and colours (outer); the hub
carries the save-hold ring.

## Keyboard and mouse reference

| Input | Action |
|---|---|
| Mouse | Click any control; drag the opacity slider; drag the title bar to move the window |
| Wheel | Cycle stroke width |
| `1`–`6` | Select stroke width |
| `←` / `→` | Previous / next brush |
| `↑` / `↓` | Previous / next colour |
| `[` / `]` | Opacity ∓ 5 % |
| `A` | Toggle shape recognition |
| `H` | Toggle the hand wireframe |
| `V` | Toggle solo (artwork-only) mode |
| `Tab` | Cycle export format |
| `Ctrl` + `S` | Export with a file dialog |
| `Ctrl` + `Shift` + `S` | Export immediately to the output folder |
| `Ctrl` + `Z` / `Ctrl` + `Y` | Undo / redo |
| `Ctrl` + `Delete` | Clear canvas |
| `F11` | Fullscreen |
| `Esc` | Leave fullscreen, or quit |

---

## Folder structure

```
AI Virtual Painter/
├── ai_virtual_painter.py          the entire application, one file
├── run.bat                        Windows launcher (double-click to start)
├── AI Virtual Painter.png         optional app icon (see below)
├── tests/
│   └── test_painter_engine.py     123 headless tests
├── pyproject.toml                 dependencies and tool configuration
└── README.md
```

The application is deliberately a single module, as specified. It carries no asset folder: the
typeface is resolved from the system's installed fonts, every brush icon is rasterised at
runtime by the painting engine itself, and the MediaPipe model bundle is cached outside the
project.

**The icon is entirely optional.** If `AI Virtual Painter.png` sits beside the script it is used
as the window icon and shown during the launch sequence; point `AIVP_ICON` somewhere else to
override it. Delete it and the application runs identically, with a text-only boot screen.

---

## Technologies used

| Component | Role |
|---|---|
| **MediaPipe Tasks** `0.10.30+` | Hand landmark detection (21 points, video mode) |
| **OpenCV** `4.9+` | Capture, all compositing, geometry and antialiased rasterisation |
| **NumPy** `1.26+` | Layer storage and brush stamping |
| **Pillow** `10+` | Glyph rasterisation and the Tk display bridge |
| **CustomTkinter** `5.2+` | Dark window shell and DPI handling |
| **Tkinter** | Event loop, input, file dialogs |
| **ruff · black · mypy · pytest** | Lint, format, type check, test |

---

## Installation

Requires **Python 3.11+**, a webcam, and roughly 8 MB of disk for the model cache.

```bash
pip install "opencv-python>=4.9" "mediapipe>=0.10.30" "numpy>=1.26" "Pillow>=10" "customtkinter>=5.2"
```

Or from the project file:

```bash
pip install -e .
```

### The hand-landmark model

MediaPipe's Tasks API needs a `hand_landmarker.task` bundle (~7.5 MB), which is not vendored
here. On first launch it is downloaded once from Google's official endpoint into a per-user
cache:

- Windows — `%LOCALAPPDATA%\AIVirtualPainter\models\`
- Linux / macOS — `~/.cache/ai-virtual-painter/models/`

To run entirely offline, fetch it yourself and point `AIVP_HAND_MODEL` at it:

```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Configuration

Every setting is an environment variable; nothing is hardcoded and there are no secrets.

| Variable | Default | Purpose |
|---|---|---|
| `AIVP_CAMERA_INDEX` | `0` | Capture device index |
| `AIVP_CAMERA_WIDTH` | `1280` | Requested capture width |
| `AIVP_CAMERA_HEIGHT` | `720` | Requested capture height |
| `AIVP_HAND_MODEL` | cache path | Path to an existing model bundle |
| `AIVP_MODEL_CACHE` | per-user cache | Directory for the downloaded bundle |
| `AIVP_OUTPUT_DIR` | `~/Pictures/AI Virtual Painter` | Where artwork is exported |
| `AIVP_ICON` | `./AI Virtual Painter.png` | Optional application icon |
| `AIVP_LOG_LEVEL` | `INFO` | Logging level |

Invalid values are logged and ignored rather than crashing the application.

---

## How to run

**Windows — double-click `run.bat`.** It locates a Python 3.11+ interpreter (trying `py -3.11`,
`py -3`, then `python`), runs from the project folder so the icon is found, offers to install any
missing dependencies with pip, and keeps the console open if something goes wrong so the error
stays readable.

Or start it directly on any platform:

```bash
python ai_virtual_painter.py
```

The window opens frameless and centred, plays its boot sequence while the camera and model come
online, and then shows the message `SEARCHING FOR A HAND`. Raise one hand, put your index finger
up, and paint.

If the camera cannot be opened — usually because another application holds it — a dialog names
the problem and suggests `AIVP_CAMERA_INDEX`.

---

## Usage guide

1. **Position yourself** about an arm's length from the camera, with your hand clearly lit. The
   cyan wireframe confirms tracking; press `H` to hide it.
2. **Paint** with your index finger up and your middle finger folded. The reticle fills with
   your current colour and trails a comet tail. The docks fade back so the canvas is clear.
3. **Change tools** by raising two fingers. The radial palette blooms where your finger is.
   Move onto a sector and hold for half a second.
4. **Erase** with three fingers up. The reticle becomes a dashed rotating circle at the eraser's
   true size.
5. **Undo and redo** by pinching your thumb to your index or middle finger.
6. **Recognise a shape** by drawing one roughly and lifting; a toast confirms what was snapped.
7. **Save** by holding a still victory sign for two seconds, or press `Ctrl` + `S`. `Tab` picks
   the format first.
8. **Clear** with an open palm held for two seconds. A red progress ring counts down, and undo
   brings the artwork back.

---

## Architecture

The file is organised as sixteen numbered sections, each a layer with one responsibility and no
upward dependencies.

```
1  Configuration      AppConfig, environment parsing
2  Theme              every colour in the product, defined once
3  Maths              easing, 1-Euro filtering, Catmull-Rom, resampling
4  Raster primitives  layers, blends, sprites, glass, batching
5  Typography         font resolution and the cached text renderer
6  Domain model       Tool, BrushId, ColorSlot, Stroke, BrushSpec
7  Path sampling      PathSampler — raw points to uniform spline samples
8  Brush engine       15 renderers + StrokeRasteriser
9  Shape recognition  ShapeRecogniser
10 Document           PaintDocument — strokes, layers, history
11 Export             ArtworkExporter
12 Capture & tracking CameraStream, HandTracker, model provisioning
13 Gestures           GestureEngine — poses, pinches, holds
14 Interface          HudRenderer, HitRegion, Command, toasts
15 Application        VirtualPainterApp — the loop and all input
16 Entry point        main()
```

### Threading

Three threads, coordinated by one-item drop-oldest mailboxes so nothing ever queues stale work:

```
camera thread  ──▶  latest frame  ──▶  Tk loop  ──▶  gesture engine ──▶ document ──▶ HUD ──▶ display
                                          │
                                          └──▶  inference mailbox  ──▶  hand thread
```

### The compositing model

```
camera frame ──▶ cinematic grade ──▶ + artwork layer (dirty rect only) ──▶ + HUD ──▶ Tk
```

Inside the document:

```
base          every committed stroke
stroke_buffer only the stroke in progress
canvas        base ⊕ stroke_buffer   ← what gets displayed
```

Because the live stroke never touches `base`, each frame only recomposites the few hundred
pixels the newest samples touched. A full canvas costs the same as an empty one.

---

## Design decisions

**Strokes are vectors; the raster is a cache.** This one decision buys unlimited undo without
storing a single bitmap snapshot, shape snapping that keeps the brush, and true
resolution-independent export. Its cost is that undo re-rasterises the document — acceptable for
an operation measured in tens of milliseconds and performed occasionally.

**Determinism through seeded indices.** Every expressive brush draws its randomness from
`(stroke seed, sample index)`. Because `PathSampler` is append-only and its indices are stable,
the incremental live render and a full rebuild are byte-identical — a property asserted directly
in the test suite. Without it, undo would visibly reshuffle every particle on the canvas.

**The interface is painted, not assembled.** Tk widgets can only sit *on top of* an image, never
inside it, so a widget-based interface could never have real frosted glass, additive neon or
per-element motion over live video. Painting the HUD into the frame gives complete control and
made the whole aesthetic possible. The trade-off is that accessibility relies on the keyboard
map rather than on native focus handling.

**Two compositing primitives, in 8-bit OpenCV.** Everything the HUD draws routes through
`blend_color_mask` and `blend_image_mask`. The obvious float32 NumPy expression measured about
**seven times slower**; the interface performs hundreds of blends per frame, so the primitive
was worth optimising and nothing else needed to be.

**Batched coverage masks.** The hand wireframe is 21 bones and the cursor trail is 26 segments.
Blended individually, per-call overhead dwarfed their pixels. `MaskBatch` collects marks into
one scratch mask and composites once, which alone moved the frame rate from 10 to 15 FPS.

**Radial extension, not screen-space Y.** Comparing fingertip and knuckle *distance from the
wrist* rather than their Y coordinates makes every pose rotation-invariant. Tilt your hand and
the gestures still work.

**The 1-Euro filter over a moving average.** Air drawing has contradictory requirements: dead
still when you hover, instant when you flick. A fixed filter must choose. The 1-Euro filter
adapts its cutoff to the signal's own speed and gets both.

**Physical pixels, not logical ones.** The window is a video surface, so both CustomTkinter
scaling factors are pinned and the geometry is corrected for monitor DPI. Left at the defaults on
a 125 % display, the widget came back 1600×900 and every frame was resampled on its way to the
screen for no visual gain — that fix alone was worth 10 FPS.

---

## Performance

Measured on the development machine (Windows 11, 16 logical cores) at 1280×720 with the full
effect stack, camera live, hand tracking active and the artwork on screen:

| Stage | Mean | p50 |
|---|---|---|
| Interface render | 9.7 ms | 8.9 ms |
| Tk display upload | 9.9 ms | 9.3 ms |
| Backdrop + artwork composite | 2.3 ms | 1.4 ms |
| Frame acquisition + gestures + model | 1.0 ms | 0.1 ms |
| **Full loop** | **22.9 ms** | **20.7 ms** |
| **Measured frame rate** | **~40–44 FPS** | |
| Hand inference *(background thread)* | ~20 ms | |

The optimisation pass took this from 10 FPS to 44: whole-frame overlay copies removed from line
drawing, all HUD blending moved from float32 NumPy to 8-bit OpenCV, small marks batched into
single composites, the animated brush preview turned into a lazily-built filmstrip, artwork
compositing restricted to its dirty rectangle, and the DPI-driven resample eliminated.

Two ceilings remain, both external: Tk's photo upload is a hard ~10 ms at 720p, and this
particular webcam only offers uncompressed YUY2 at 20 FPS. The application requests MJPG, which
most cameras honour for 30–60 FPS. The renderer runs ahead of the camera, so interface motion
stays smooth regardless.

---

## Testing

```bash
python -m pytest -q          # 123 tests
python -m ruff check .
python -m black --check .
python -m mypy ai_virtual_painter.py
```

All four pass. Everything except the Tk window runs headlessly: colour and easing maths, the
1-Euro filter's jitter/lag behaviour, path sampling determinism, all fifteen brushes, the
incremental-versus-rebuild equality property, undo/redo/clear round-trips, eraser behaviour, all
five shape classes plus rejection cases, every export format, the gesture state machine
(poses, pinch debouncing, two-second holds, staleness) and full HUD render passes in every tool
state.

---

## Known limitations

- **One hand.** The landmarker is configured for a single hand, as specified.
- **Tk display ceiling.** Roughly 10 ms per frame at 720p is Tk's own cost and cannot be
  optimised from Python.
- **Undo re-rasterises.** On a canvas with hundreds of expressive strokes an undo can take a
  noticeable moment. History is capped at 100 states.
- **High-resolution export blocks the loop.** A 3× re-render of a busy canvas pauses the window
  briefly.
- **Fonts are system-resolved.** Bahnschrift and Consolas are preferred, with a documented
  fallback chain; on a machine with neither, the interface falls back to a plain face.
- **Shape recognition is single-stroke** and covers the five specified primitives only.
- **Accessibility.** Because the interface is painted rather than assembled from widgets, it
  offers no native screen-reader semantics. Every control has a keyboard equivalent, contrast
  was checked against the dark palette, and no state is signalled by colour alone — but a
  screen reader cannot introspect the canvas.
- **First launch needs network access** once, to fetch the model bundle.

## Future improvements

- Layers with per-layer blend modes and opacity.
- Two-handed gestures: pinch-to-zoom and canvas panning.
- A GPU compositing path to lift the display ceiling.
- Background export on a worker thread with a document snapshot.
- Stroke stabilisation presets, and a pressure curve editor.
- Timelapse capture of the painting session.
- Persisting a session as a document that reopens with its history intact.
