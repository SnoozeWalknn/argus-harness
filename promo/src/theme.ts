export const WIDTH = 1920;
export const HEIGHT = 1080;
export const FPS = 30;
export const DURATION = 60 * FPS;

export const colors = {
  bg: '#050507',
  text: '#f4f4f6',
  soft: '#d4d4d8',
  muted: '#8b8f98',
  faint: '#5d616b',
  line: 'rgba(255,255,255,0.08)',
  violet: '#a78bfa',
  violetDeep: '#7c3aed',
  indigo: '#818cf8',
  blue: '#60a5fa',
  sky: '#7dd3fc',
  teal: '#2dd4bf',
  green: '#4ade80',
  amber: '#fbbf24',
  orange: '#fb923c',
  rose: '#fb7185',
};

export const fonts = {
  sans: '"Inter", system-ui, sans-serif',
  mono: '"JetBrains Mono", "DejaVu Sans Mono", monospace',
};

// Section boundaries in frames (30 fps).
export const TIMELINE = {
  hook: [0, 150],
  context: [150, 300],
  silent: [300, 450],
  constrained: [450, 600],
  failures: [600, 750],
  edits: [750, 900],
  checkpoints: [900, 1050],
  ssh: [1050, 1200],
  ab: [1200, 1350],
  terminal: [1350, 1650],
  close: [1650, 1800],
} as const;
