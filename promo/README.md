# argus promo

Two films, built with [Remotion](https://www.remotion.dev/), 1920×1080, 30 fps,
H.264, no audio:

- [`argus-launch.mp4`](argus-launch.mp4): **the 90-second launch film** of the
  model-agnostic agent, every feature in eleven chapters (below).
- [`argus-promo.mp4`](argus-promo.mp4): the original 60-second promo of the
  local-model harness.

## The launch film (90 s)

| time | chapter |
|---|---|
| 0–5 s | cold open: "Every model. One agent." |
| 5–12.5 s | 01 any model: local servers and cloud APIs wired into one agent; `-m` specs and aliases |
| 12.5–22.5 s | 02 watch it think: the TUI streaming its reasoning, a push-in on the live token box, then todos, changed files, tests and the answer |
| 22.5–30.5 s | 03 switch mid-session: the ctrl+o picker (ready / needs a key), `/model opus`, the session carried over to Claude |
| 30.5–38.5 s | 04 every tool: the ten tools; a subagent, a dev server as a background job, then `fetch` of its page |
| 38.5–45.5 s | 05 safe by default: approval policies, bubblewrap / Landlock, the approval modal |
| 45.5–52.5 s | 06 plan · delegate · hook: plan mode, todos, subagents, hooks, skills |
| 52.5–58.5 s | 07 knows your code: pylsp flags an undefined name right after an edit; resume, checkpoints, SSH |
| 58.5–64.5 s | 08 tuned per model: `argus tune` ranking and the prompt-overhead cut |
| 64.5–70.5 s | 09 every run, measured: failure tags, the SQLite log, A/B statistics, the test count |
| 70.5–77.5 s | 10 make it yours: the same session in eight themes |
| 77.5–83.5 s | 11 one command: the installer and `argus doctor` |
| 83.5–90 s | close |

### Where every frame comes from

Nothing in the UI is drawn by hand. `scripts/capture_launch.py` drives the real
Textual app with Textual's Pilot against argus's own mock server (no real model)
and saves each state as an SVG screenshot in `public/launch/`; the film inlines
those SVGs, so they stay sharp through the camera moves. It also records the
numbers and terminal text in `src/launch/data.json`: real `argus doctor`,
`argus tune` (with the mock scripted to break native calls), `argus overhead`,
installer output, the test count, and where named strings sit in each capture
so the callouts land on them.

Three things are staged for the camera and said so on screen: cloud providers
show their real API host instead of the mock's address, the mock listens on the
local servers' usual ports so URLs look like a real machine's, and the model
replies are scripted.

Refresh after changing argus, then render:

```sh
.venv/bin/python promo/scripts/capture_launch.py       # from the repository root
cd promo && npm run render:launch                      # writes argus-launch.mp4
COMPOSITION=ArgusLaunch npm run stills -- /tmp/s 480 900   # single frames
```

Source: `src/launch/` (timeline, backdrop, SVG panels and callouts, one file per
group of chapters).

## The original promo (60 s)

| time | section |
|---|---|
| 0–5 s | hook: "Local models deserve a real harness." |
| 5–15 s | problem: cloud-tuned agents waste context and fail silently |
| 15–45 s | six features, 5 s each: constrained calls, failure tags, fuzzy edits, checkpoints, SSH, A/B suites |
| 45–55 s | terminal: `argus run -v`, then `argus show last` |
| 55–60 s | logo and tagline |

## Where the numbers come from

Nothing on screen is typed in by hand. `scripts/capture.py` runs argus against
its own mock server (no real model) and writes `src/data.json`:

- **prompt overhead** (1,181 → 294 tokens): `argus overhead --all` against the
  mock server, whose tokenizer is a word-level stand-in. The video labels these
  as a mock-server measurement; real Qwen token counts will differ.
- **terminal**: the real output of a scripted `argus run -v` and the
  `argus show last` that follows, captured line by line.
- **checkpoints**: the commits and change sets of that same run.
- **A/B report**: illustrative pass counts (labelled on screen), with the
  Wilson intervals and McNemar p-value computed by argus's own report code.

Refresh the data from the repository root after changing argus:

```sh
.venv/bin/python promo/scripts/capture.py
```

## Render

```sh
cd promo
npm install        # also copies the fonts and generates the film-grain tile
npm run studio     # live preview in the browser
npm run render     # writes argus-promo.mp4
npm run stills -- /tmp/stills 75 540 1760   # individual frames for review
```

Remotion downloads its own headless Chrome on first render. Where that
download is blocked (as it was in the sandbox this was made in, which does not
allow `remotion.media`), point it at an existing Chromium or
chrome-headless-shell instead:

```sh
REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell npm run render
```

## Layout

```
src/
  Promo.tsx          timeline: scenes cross-fade at the section boundaries
  theme.ts           colours, fonts, section timings
  lib/               animation helpers, font loading, scene timing
  components/        background, kinetic type, cards, feature layout, logo
  scenes/            one file per section
  data.json          generated by scripts/capture.py
scripts/
  capture.py         records the data shown in the video
  prepare-assets.mjs fonts + grain tile (npm postinstall)
  stills.mjs         renders single frames
```

Fonts are Inter and JetBrains Mono (both SIL Open Font License), taken from npm
packages at install time. Remotion is free for individuals and small teams;
see [remotion.dev/license](https://www.remotion.dev/license) for company use.
