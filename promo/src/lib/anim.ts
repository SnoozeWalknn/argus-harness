import type {CSSProperties} from 'react';
import {Easing, interpolate, spring} from 'remotion';

export const EASE_OUT = Easing.bezier(0.16, 1, 0.3, 1);
export const EASE_IN_OUT = Easing.bezier(0.65, 0, 0.35, 1);

/** Clamped, eased interpolation of `frame` from [a, b] to [from, to]. */
export const ramp = (
  frame: number,
  a: number,
  b: number,
  from = 0,
  to = 1,
  easing: (t: number) => number = EASE_OUT,
): number =>
  interpolate(frame, [a, b], [from, to], {
    easing,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

/** Critically damped spring: smooth ease-out, no overshoot. */
export const settle = (frame: number, delay: number, fps: number, mass = 0.9): number =>
  spring({frame: frame - delay, fps, config: {damping: 200, mass}});

/** Springy pop with a little overshoot. */
export const pop = (frame: number, delay: number, fps: number): number =>
  spring({frame: frame - delay, fps, config: {damping: 13, stiffness: 170, mass: 0.7}});

/** Typewriter: the visible prefix of `text` at `frame`. */
export const typed = (text: string, frame: number, start: number, charsPerFrame = 1.5): string =>
  text.slice(0, Math.max(0, Math.floor((frame - start) * charsPerFrame)));

/** Blur/fade/rise entrance for any element; `p` is 0..1 progress. */
export const reveal = (p: number, rise = 24, blur = 10): CSSProperties => ({
  opacity: p,
  transform: `translateY(${(1 - p) * rise}px)`,
  filter: p < 0.999 ? `blur(${(1 - p) * blur}px)` : undefined,
});

export const gradientText = (from: string, to: string, angle = 90): CSSProperties => ({
  backgroundImage: `linear-gradient(${angle}deg, ${from}, ${to})`,
  WebkitBackgroundClip: 'text',
  backgroundClip: 'text',
  color: 'transparent',
  WebkitTextFillColor: 'transparent',
});

export const rgba = (hex: string, alpha: number): string => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

export const fmt = (n: number): string => Math.round(n).toLocaleString('en-US');
