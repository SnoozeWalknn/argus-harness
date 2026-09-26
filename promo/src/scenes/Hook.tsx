import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {Kinetic} from '../components/Kinetic';
import {gradientText, ramp} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors} from '../theme';

export const Hook: React.FC = () => {
  const f = useSceneFrame();
  const push = interpolate(f, [0, 160], [1, 1.03]);
  const sweep = ramp(f, 58, 96);
  const glow = 0.5 + 0.18 * Math.sin(f / 8);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div style={{transform: `scale(${push})`, textAlign: 'center'}}>
        <Kinetic
          start={10}
          stagger={5}
          style={{fontSize: 128, fontWeight: 700, letterSpacing: '-0.05em', lineHeight: 1.04, color: colors.text}}
          tokens={[
            'Local',
            'models',
            'deserve',
            '\n',
            {text: 'a', style: {color: colors.soft}},
            {text: 'real', style: {color: colors.soft}},
            {
              text: 'harness.',
              style: {...gradientText('#ffffff', '#b8a4ff', 100), paddingBottom: '0.08em'},
              glow: `0 0 36px rgba(139, 92, 246, ${glow})`,
            },
          ]}
        />
        <div
          style={{
            margin: '44px auto 0',
            height: 2,
            width: 560 * sweep,
            borderRadius: 2,
            background: 'linear-gradient(90deg, rgba(167,139,250,0), rgba(196,181,253,0.95), rgba(96,165,250,0))',
            boxShadow: '0 0 28px rgba(167,139,250,0.8)',
            opacity: ramp(f, 58, 70) * ramp(f, 120, 150, 1, 0.4),
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
