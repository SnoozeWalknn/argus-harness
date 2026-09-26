import React from 'react';
import {useVideoConfig} from 'remotion';
import {pop, ramp, settle} from '../lib/anim';

/**
 * The argus mark: a ring of "eyes" around an iris (Argus Panoptes, the hundred-eyed watchman).
 * `f` is the local frame; everything is drawn by f ≈ 60.
 */
export const LogoMark: React.FC<{f: number; size?: number}> = ({f, size = 220}) => {
  const {fps} = useVideoConfig();
  const ring = ramp(f, 4, 40);
  const circumference = 2 * Math.PI * 76;
  const eyes = 12;
  const iris = pop(f, 26, fps);
  const glow = settle(f, 30, fps);
  return (
    <svg width={size} height={size} viewBox="-110 -110 220 220" style={{overflow: 'visible'}}>
      <defs>
        <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#c4b5fd" />
          <stop offset="55%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#60a5fa" />
        </linearGradient>
        <radialGradient id="iris" cx="0.4" cy="0.35" r="0.75">
          <stop offset="0%" stopColor="#ede9fe" />
          <stop offset="45%" stopColor="#a78bfa" />
          <stop offset="100%" stopColor="#5b21b6" />
        </radialGradient>
        <filter id="soft" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="10" />
        </filter>
      </defs>
      <circle r="58" fill="#7c3aed" opacity={0.35 * glow} filter="url(#soft)" />
      <circle
        r="76"
        fill="none"
        stroke="url(#ring)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - ring)}
        transform="rotate(-90)"
      />
      {Array.from({length: eyes}, (_, i) => {
        const a = (i / eyes) * Math.PI * 2 - Math.PI / 2;
        const open = settle(f, 16 + i * 2, fps, 0.6);
        return (
          <ellipse
            key={i}
            cx={Math.cos(a) * 98}
            cy={Math.sin(a) * 98}
            rx={5.5}
            ry={5.5 * open}
            fill="#e9e5ff"
            opacity={0.25 + 0.65 * open}
          />
        );
      })}
      <g transform={`scale(${iris})`}>
        <circle r="34" fill="url(#iris)" />
        <circle r="13" fill="#0b0a12" />
        <circle cx="-9" cy="-11" r="5" fill="#ffffff" opacity="0.9" />
      </g>
    </svg>
  );
};
