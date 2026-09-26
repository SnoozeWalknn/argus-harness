import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {reveal, rgba, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors} from '../theme';
import {Kinetic, type Token} from './Kinetic';

/** Left: numbered eyebrow, kinetic headline, body. Right: the feature's animated visual. */
export const FeatureLayout: React.FC<{
  index: string;
  eyebrow: string;
  title: Token[];
  body: string;
  accent: string;
  children: React.ReactNode;
}> = ({index, eyebrow, title, body, accent, children}) => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const eyebrowP = settle(f, 2, fps);
  const bodyP = settle(f, 22, fps);
  const cardP = settle(f, 8, fps, 1.1);
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: 140, top: 0, bottom: 0, width: 620, display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 30, ...reveal(eyebrowP, 14, 6)}}>
          <span
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: 17,
              fontWeight: 500,
              color: accent,
              padding: '5px 11px',
              borderRadius: 999,
              border: `1px solid ${rgba(accent, 0.35)}`,
              background: rgba(accent, 0.08),
              boxShadow: `0 0 24px ${rgba(accent, 0.25)}`,
            }}
          >
            {index}
          </span>
          <span style={{fontSize: 18, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: colors.muted}}>
            {eyebrow}
          </span>
        </div>
        <Kinetic
          tokens={title}
          start={6}
          stagger={3}
          style={{fontSize: 74, fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1.04, color: colors.text}}
        />
        <div style={{marginTop: 30, fontSize: 28, lineHeight: 1.45, color: colors.muted, maxWidth: 580, ...reveal(bodyP, 16, 8)}}>
          {body}
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: 850,
          top: 0,
          bottom: 0,
          width: 940,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: cardP,
          transform: `translateY(${(1 - cardP) * 40}px) scale(${0.97 + 0.03 * cardP})`,
        }}
      >
        {children}
      </div>
    </AbsoluteFill>
  );
};
