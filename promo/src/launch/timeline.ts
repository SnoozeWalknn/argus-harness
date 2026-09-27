// The 90-second launch film: 30 fps, 2700 frames, 13 chapters.
export const LAUNCH_DURATION = 90 * 30;

export const CH = {
  open: [0, 150],
  models: [150, 375],
  tui: [375, 675],
  switch: [675, 915],
  tools: [915, 1155],
  safety: [1155, 1365],
  workflow: [1365, 1575],
  code: [1575, 1755],
  tuned: [1755, 1935],
  measured: [1935, 2115],
  themes: [2115, 2325],
  install: [2325, 2505],
  close: [2505, 2700],
} as const;

export type ChapterKey = keyof typeof CH;

/** Numbered chapters shown in the corner tag and on the rail. */
export const CHAPTERS: {key: ChapterKey; n: string; title: string; accent: string}[] = [
  {key: 'models', n: '01', title: 'Any model', accent: '#a78bfa'},
  {key: 'tui', n: '02', title: 'Watch it think', accent: '#818cf8'},
  {key: 'switch', n: '03', title: 'Switch mid-session', accent: '#7dd3fc'},
  {key: 'tools', n: '04', title: 'Every tool', accent: '#2dd4bf'},
  {key: 'safety', n: '05', title: 'Safe by default', accent: '#fbbf24'},
  {key: 'workflow', n: '06', title: 'Plan · delegate · hook', accent: '#f0abfc'},
  {key: 'code', n: '07', title: 'Knows your code', accent: '#4ade80'},
  {key: 'tuned', n: '08', title: 'Tuned per model', accent: '#60a5fa'},
  {key: 'measured', n: '09', title: 'Every run, measured', accent: '#fb923c'},
  {key: 'themes', n: '10', title: 'Make it yours', accent: '#f472b6'},
  {key: 'install', n: '11', title: 'One command', accent: '#a78bfa'},
];
