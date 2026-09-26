import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {LogoMark} from '../components/Logo';
import {ramp, reveal, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors} from '../theme';

export const Close: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const word = settle(f, 44, fps);
  // The mark starts centred on its own, then slides left to make room for the wordmark.
  const slide = settle(f, 28, fps, 1.0);
  const markShift = (1 - slide) * 262;
  const sweep = ramp(f, 86, 128) * 140 - 20; // highlight position, %
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 54, marginTop: -30}}>
        <div style={{transform: `translateX(${markShift}px)`}}>
          <LogoMark f={f} size={240} />
        </div>
        <div
          style={{
            fontSize: 176,
            fontWeight: 800,
            letterSpacing: '-0.055em',
            lineHeight: 1,
            paddingBottom: '0.06em',
            backgroundImage: `linear-gradient(100deg, #f4f4f6 0%, #f4f4f6 ${sweep - 12}%, #ddd6fe ${sweep}%, #f4f4f6 ${sweep + 12}%, #d4d4d8 100%)`,
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            color: 'transparent',
            opacity: word,
            transform: `translateX(${(1 - word) * -24}px)`,
            filter: word < 0.999 ? `blur(${(1 - word) * 14}px)` : 'drop-shadow(0 0 40px rgba(139,92,246,0.35))',
          }}
        >
          Argus
        </div>
      </div>
      <div style={{marginTop: 58, fontSize: 52, fontWeight: 500, letterSpacing: '-0.02em', color: '#e4e4e7', ...reveal(settle(f, 62, fps), 18, 10)}}>
        See every turn.
      </div>
      <div style={{marginTop: 18, fontSize: 26, color: colors.muted, ...reveal(settle(f, 76, fps), 14, 8)}}>
        A headless coding-agent harness for local models
      </div>
    </AbsoluteFill>
  );
};
