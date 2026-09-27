import React, {type CSSProperties} from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {pop, ramp, rgba, settle} from '../../lib/anim';
import {useSceneFrame} from '../../lib/scene';
import {fonts} from '../../theme';
import {CH, CHAPTERS, LAUNCH_DURATION} from '../timeline';

/**
 * A callout: a pulsing dot at the target, a leader line drawn out to a label pill.
 * Coordinates are in the parent's pixels; `dx`/`dy` place the label relative to the target.
 */
export const Callout: React.FC<{
  x: number;
  y: number;
  dx: number;
  dy: number;
  at: number;
  label: string;
  color?: string;
  out?: number;
}> = ({x, y, dx, dy, at, label, color = '#c4b5fd', out = 1e9}) => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const dot = pop(f, at, fps);
  const line = ramp(f, at + 4, at + 20);
  const text = settle(f, at + 14, fps);
  const gone = 1 - ramp(f, out, out + 12);
  const len = Math.hypot(dx, dy);
  const ring = ((f - at) % 36) / 36;
  const left = dx < 0;
  return (
    <div style={{position: 'absolute', left: 0, top: 0, opacity: gone, pointerEvents: 'none'}}>
      <svg style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}} width={1} height={1}>
        <line
          x1={x}
          y1={y}
          x2={x + dx}
          y2={y + dy}
          stroke={color}
          strokeWidth={2}
          strokeDasharray={len}
          strokeDashoffset={len * (1 - line)}
          opacity={0.85}
        />
        <circle cx={x} cy={y} r={7 * dot} fill={color} />
        {f > at ? <circle cx={x} cy={y} r={8 + ring * 22} fill="none" stroke={color} strokeWidth={2} opacity={(1 - ring) * 0.7} /> : null}
      </svg>
      <div
        style={{
          position: 'absolute',
          left: x + dx,
          top: y + dy,
          transform: `translate(${left ? '-100%' : '0'}, -50%) translateX(${(left ? 1 : -1) * (1 - text) * 16}px)`,
          opacity: text,
          whiteSpace: 'nowrap',
          padding: '12px 22px',
          borderRadius: 999,
          background: 'rgba(12,10,22,0.82)',
          border: `1.5px solid ${rgba(color, 0.7)}`,
          boxShadow: `0 12px 40px rgba(0,0,0,0.5), 0 0 30px ${rgba(color, 0.25)}`,
          color: '#f4f4f6',
          fontSize: 26,
          fontWeight: 600,
          letterSpacing: '-0.01em',
          fontFamily: fonts.sans,
        }}
      >
        {label}
      </div>
    </div>
  );
};

/** A glowing rounded box drawn around a region, e.g. a line in a capture. */
export const Highlight: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  at: number;
  color?: string;
  out?: number;
  pad?: number;
}> = ({x, y, w, h, at, color = '#a78bfa', out = 1e9, pad = 8}) => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const p = settle(f, at, fps);
  const gone = 1 - ramp(f, out, out + 12);
  return (
    <div
      style={{
        position: 'absolute',
        left: x - pad,
        top: y - pad,
        width: w + pad * 2,
        height: h + pad * 2,
        borderRadius: 10,
        border: `2px solid ${rgba(color, 0.95)}`,
        boxShadow: `0 0 0 6px ${rgba(color, 0.12)}, 0 0 40px ${rgba(color, 0.45)}`,
        background: rgba(color, 0.08),
        opacity: p * gone,
        transform: `scale(${1.08 - 0.08 * p})`,
      }}
    />
  );
};

/** The chapter tag in the top-left corner: number, rule, title. */
export const ChapterTag: React.FC<{n: string; title: string; accent: string}> = ({n, title, accent}) => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const p = settle(f, 4, fps);
  const rule = ramp(f, 8, 30);
  const t = settle(f, 12, fps);
  return (
    <div style={{position: 'absolute', left: 110, top: 78, display: 'flex', alignItems: 'center', gap: 20}}>
      <div
        style={{
          fontFamily: fonts.mono,
          fontWeight: 700,
          fontSize: 26,
          color: accent,
          opacity: p,
          transform: `translateY(${(1 - p) * 12}px)`,
          textShadow: `0 0 24px ${rgba(accent, 0.6)}`,
        }}
      >
        {n}
      </div>
      <div style={{width: 64 * rule, height: 2, background: `linear-gradient(90deg, ${accent}, transparent)`}} />
      <div
        style={{
          fontSize: 22,
          fontWeight: 600,
          letterSpacing: '0.22em',
          textTransform: 'uppercase',
          color: '#c9cad1',
          opacity: t,
          transform: `translateX(${(1 - t) * -10}px)`,
        }}
      >
        {title}
      </div>
    </div>
  );
};

/** Film progress along the top edge, one segment per chapter. */
export const Rail: React.FC = () => {
  const frame = useCurrentFrame();
  const show = ramp(frame, CH.models[0] - 10, CH.models[0] + 20) * (1 - ramp(frame, CH.close[0], CH.close[0] + 30));
  const x0 = 110;
  const width = 1700;
  const start = CH.models[0];
  const end = CH.close[0];
  return (
    <div style={{position: 'absolute', left: x0, top: 40, width, height: 3, opacity: show}}>
      {CHAPTERS.map((c) => {
        const [a, b] = CH[c.key];
        const l = ((a - start) / (end - start)) * width;
        const w = ((b - a) / (end - start)) * width - 6;
        const fill = Math.min(Math.max((frame - a) / (b - a), 0), 1);
        const active = frame >= a && frame < b;
        return (
          <div key={c.key} style={{position: 'absolute', left: l, width: w, height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.10)'}}>
            <div
              style={{
                width: `${fill * 100}%`,
                height: '100%',
                borderRadius: 2,
                background: active ? c.accent : 'rgba(255,255,255,0.45)',
                boxShadow: active ? `0 0 12px ${c.accent}` : undefined,
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

/** Rounded label chip. */
export const Chip: React.FC<{children: React.ReactNode; color?: string; style?: CSSProperties; mono?: boolean}> = ({
  children,
  color = '#a78bfa',
  style,
  mono = false,
}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      padding: '10px 20px',
      borderRadius: 999,
      background: rgba(color, 0.1),
      border: `1.5px solid ${rgba(color, 0.45)}`,
      color: '#f1f1f4',
      fontFamily: mono ? fonts.mono : fonts.sans,
      fontSize: mono ? 22 : 24,
      fontWeight: 600,
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Big headline + sub, left-aligned; words rise in on springs. */
export const Title: React.FC<{
  lines: string[];
  accent?: string;
  highlight?: number;
  sub?: string;
  at?: number;
  size?: number;
  style?: CSSProperties;
}> = ({lines, accent = '#a78bfa', highlight = -1, sub, at = 10, size = 84, style}) => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  let k = 0;
  return (
    <div style={{position: 'absolute', left: 110, top: 150, ...style}}>
      <div style={{fontSize: size, fontWeight: 750, letterSpacing: '-0.045em', lineHeight: 1.02, color: '#f5f5f7'}}>
        {lines.map((line, i) => (
          <div key={i} style={{whiteSpace: 'nowrap'}}>
            {line.split(' ').map((word, j) => {
              const p = settle(f, at + k++ * 3, fps);
              const hl = i === highlight;
              return (
                <span
                  key={j}
                  style={{
                    display: 'inline-block',
                    marginRight: '0.24em',
                    opacity: p,
                    transform: `translateY(${(1 - p) * 0.35}em)`,
                    filter: p < 0.999 ? `blur(${(1 - p) * 12}px)` : undefined,
                    ...(hl
                      ? {
                          backgroundImage: `linear-gradient(90deg, ${accent}, #f5f5f7 140%)`,
                          WebkitBackgroundClip: 'text',
                          backgroundClip: 'text',
                          color: 'transparent',
                        }
                      : {}),
                  }}
                >
                  {word}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      {sub ? (
        <div
          style={{
            marginTop: 26,
            fontSize: 30,
            lineHeight: 1.4,
            color: '#a1a4ad',
            maxWidth: 640,
            opacity: settle(f, at + 16, fps),
            transform: `translateY(${(1 - settle(f, at + 16, fps)) * 14}px)`,
          }}
        >
          {sub}
        </div>
      ) : null}
    </div>
  );
};
