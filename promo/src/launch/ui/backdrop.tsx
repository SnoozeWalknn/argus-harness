import React from 'react';
import {AbsoluteFill, interpolate, interpolateColors, useCurrentFrame} from 'remotion';
import {rgba} from '../../lib/anim';
import {CH} from '../timeline';

// Two slow aurora glows tinted per chapter: [frame, colour A, colour B, strength, A x, A y, grid].
const tint = (at: number, a: string, b: string, s: number, x: number, y: number, g: number) =>
  [at, a, b, s, x, y, g] as const;
const KEYS = [
  tint(0, '#1e1b4b', '#0c0a1d', 0.0, 960, 540, 0),
  tint(40, '#6d28d9', '#1e3a8a', 0.8, 960, 560, 0.1),
  tint(CH.models[0] + 10, '#6d28d9', '#1d4ed8', 0.85, 1340, 560, 0.35),
  tint(CH.tui[0] + 10, '#4338ca', '#6d28d9', 0.7, 960, 620, 0.2),
  tint(CH.switch[0] + 10, '#0e7490', '#6d28d9', 0.75, 1300, 560, 0.3),
  tint(CH.tools[0] + 10, '#0f766e', '#1e40af', 0.75, 1300, 560, 0.35),
  tint(CH.safety[0] + 10, '#b45309', '#7f1d1d', 0.65, 1300, 580, 0.3),
  tint(CH.workflow[0] + 10, '#a21caf', '#4338ca', 0.7, 1300, 560, 0.3),
  tint(CH.code[0] + 10, '#047857', '#1e3a8a', 0.7, 900, 560, 0.3),
  tint(CH.tuned[0] + 10, '#1d4ed8', '#6d28d9', 0.75, 1100, 560, 0.4),
  tint(CH.measured[0] + 10, '#c2410c', '#6d28d9', 0.65, 960, 560, 0.4),
  tint(CH.themes[0] + 10, '#be185d', '#4338ca', 0.7, 960, 600, 0.15),
  tint(CH.install[0] + 10, '#6d28d9', '#0e7490', 0.7, 960, 560, 0.3),
  tint(CH.close[0] + 10, '#7c3aed', '#2563eb', 0.95, 960, 540, 0.1),
  tint(CH.close[1], '#7c3aed', '#2563eb', 1.0, 960, 540, 0.05),
];

const frames = KEYS.map((k) => k[0]);
const at = (frame: number, i: number) =>
  interpolate(
    frame,
    frames,
    KEYS.map((k) => k[i] as number),
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
const toHex = (c: string) => {
  const m = c.match(/\d+(\.\d+)?/g) ?? ['0', '0', '0'];
  return '#' + m.slice(0, 3).map((v) => Math.round(Number(v)).toString(16).padStart(2, '0')).join('');
};

const Glow: React.FC<{x: number; y: number; r: number; color: string; alpha: number}> = ({x, y, r, color, alpha}) => (
  <div
    style={{
      position: 'absolute',
      left: x - r,
      top: y - r,
      width: r * 2,
      height: r * 2,
      background: `radial-gradient(closest-side, ${rgba(color, 0.5 * alpha)} 0%, ${rgba(color, 0.24 * alpha)} 40%, ${rgba(color, 0.07 * alpha)} 70%, ${rgba(color, 0)} 100%)`,
    }}
  />
);

export const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const colorA = toHex(interpolateColors(frame, frames, KEYS.map((k) => k[1])));
  const colorB = toHex(interpolateColors(frame, frames, KEYS.map((k) => k[2])));
  const strength = at(frame, 3);
  const ax = at(frame, 4) + Math.sin(frame / 80) * 70;
  const ay = at(frame, 5) + Math.cos(frame / 95) * 45;
  const grid = at(frame, 6);
  return (
    <AbsoluteFill style={{background: '#050507', overflow: 'hidden'}}>
      <Glow x={ax} y={ay} r={820} color={colorA} alpha={strength} />
      <Glow
        x={1920 - ax * 0.55 + Math.cos(frame / 70) * 90}
        y={1080 - ay * 0.4 + 220}
        r={960}
        color={colorB}
        alpha={strength * 0.75}
      />
      <AbsoluteFill
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.09) 1.2px, transparent 1.6px)',
          backgroundSize: '44px 44px',
          backgroundPosition: `22px ${22 - (frame * 0.15) % 44}px`,
          WebkitMaskImage: 'radial-gradient(ellipse 65% 60% at 50% 50%, black 10%, transparent 80%)',
          maskImage: 'radial-gradient(ellipse 65% 60% at 50% 50%, black 10%, transparent 80%)',
          opacity: grid,
        }}
      />
    </AbsoluteFill>
  );
};
