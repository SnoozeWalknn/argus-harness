import React from 'react';
import {AbsoluteFill, interpolate, interpolateColors, staticFile, useCurrentFrame} from 'remotion';
import {rgba} from '../lib/anim';

// Keyframes for the two aurora glows: [frame, colour A, colour B, strength, A x, A y, grid].
// Colours follow the story: violet hook, warm problem, a tint per feature, violet close.
const KEYS: [number, string, string, number, number, number, number][] = [
  [0, '#6d28d9', '#1e3a8a', 0.0, 960, 560, 0.0],
  [36, '#6d28d9', '#1e3a8a', 0.85, 960, 560, 0.15],
  [140, '#6d28d9', '#1e3a8a', 0.85, 960, 600, 0.2],
  [165, '#9a3412', '#7f1d1d', 0.5, 960, 700, 0.25],
  [440, '#9a3412', '#7f1d1d', 0.45, 960, 640, 0.25],
  [465, '#6d28d9', '#1d4ed8', 0.8, 1320, 540, 0.45],
  [615, '#b45309', '#6d28d9', 0.7, 1320, 520, 0.45],
  [765, '#047857', '#1e40af', 0.7, 1320, 560, 0.45],
  [915, '#6d28d9', '#4338ca', 0.75, 1320, 540, 0.45],
  [1065, '#0e7490', '#1e3a8a', 0.75, 1320, 540, 0.45],
  [1215, '#4338ca', '#6d28d9', 0.75, 1320, 540, 0.45],
  [1365, '#312e81', '#1f2937', 0.5, 960, 600, 0.3],
  [1640, '#312e81', '#1f2937', 0.45, 960, 600, 0.3],
  [1665, '#7c3aed', '#2563eb', 0.95, 960, 540, 0.12],
  [1800, '#7c3aed', '#2563eb', 1.0, 960, 540, 0.08],
];

const series = (i: number) => KEYS.map((k) => k[i] as number);
const frames = series(0);
const at = (frame: number, i: number) =>
  interpolate(frame, frames, series(i), {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

const Glow: React.FC<{x: number; y: number; r: number; color: string; alpha: number}> = ({x, y, r, color, alpha}) => (
  <div
    style={{
      position: 'absolute',
      left: x - r,
      top: y - r,
      width: r * 2,
      height: r * 2,
      background: `radial-gradient(closest-side, ${rgba(color, 0.55 * alpha)} 0%, ${rgba(color, 0.28 * alpha)} 38%, ${rgba(color, 0.09 * alpha)} 68%, ${rgba(color, 0)} 100%)`,
    }}
  />
);

export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const colorA = interpolateColors(frame, frames, KEYS.map((k) => k[1]));
  const colorB = interpolateColors(frame, frames, KEYS.map((k) => k[2]));
  const toHex = (c: string) => {
    const m = c.match(/\d+(\.\d+)?/g) ?? ['0', '0', '0'];
    return '#' + m.slice(0, 3).map((v) => Math.round(Number(v)).toString(16).padStart(2, '0')).join('');
  };
  const strength = at(frame, 3);
  const ax = at(frame, 4) + Math.sin(frame / 70) * 60;
  const ay = at(frame, 5) + Math.cos(frame / 90) * 40;
  const grid = at(frame, 6);
  return (
    <AbsoluteFill style={{background: '#050507', overflow: 'hidden'}}>
      <Glow x={ax} y={ay} r={760} color={toHex(colorA)} alpha={strength} />
      <Glow
        x={1920 - ax * 0.6 + Math.cos(frame / 60) * 80}
        y={1080 - ay * 0.4 + 200}
        r={900}
        color={toHex(colorB)}
        alpha={strength * 0.7}
      />
      <AbsoluteFill
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.10) 1.2px, transparent 1.6px)',
          backgroundSize: '44px 44px',
          backgroundPosition: '22px 22px',
          WebkitMaskImage: 'radial-gradient(ellipse 62% 58% at 50% 50%, black 10%, transparent 78%)',
          maskImage: 'radial-gradient(ellipse 62% 58% at 50% 50%, black 10%, transparent 78%)',
          opacity: grid,
        }}
      />
    </AbsoluteFill>
  );
};

/** Vignette and film grain on top of everything; the grain dithers dark gradients. */
export const Finish: React.FC = () => (
  <>
    <AbsoluteFill
      style={{background: 'radial-gradient(ellipse 80% 75% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%)'}}
    />
    <AbsoluteFill
      style={{backgroundImage: `url(${staticFile('grain.png')})`, backgroundSize: '256px 256px', opacity: 0.045}}
    />
  </>
);
