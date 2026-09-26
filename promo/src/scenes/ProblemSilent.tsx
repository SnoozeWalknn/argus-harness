import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {Card} from '../components/Card';
import {Kinetic} from '../components/Kinetic';
import {ramp, reveal, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

// Things that go wrong without anyone noticing: a loop, an unparsed call,
// a hallucinated next turn, a cut-off reply.
const LINES: [string, string][] = [
  ['t4', 'read(path="calc.py")'],
  ['t5', 'read(path="calc.py")'],
  ['t6', 'read(path="calc.py")'],
  ['t7', '<tool_call>{"name": "edit", "argum'],
  ['t8', 'The task is complete!<|im_start|>user'],
  ['t9', 'Let me think about this again. Let me th…'],
];

export const ProblemSilent: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{alignItems: 'center'}}>
      <Kinetic
        start={4}
        stagger={5}
        style={{marginTop: 170, fontSize: 88, fontWeight: 700, letterSpacing: '-0.045em', color: colors.text, textAlign: 'center'}}
        tokens={['And', 'fail', {text: 'silently.', style: {color: '#4b4f58'}}]}
      />
      <div style={{position: 'absolute', top: 360, ...reveal(settle(f, 10, fps), 26, 8)}}>
        <Card style={{width: 1160, padding: '30px 40px'}}>
          {LINES.map(([turn, text], i) => {
            const p = ramp(f, 18 + i * 9, 26 + i * 9);
            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  gap: 28,
                  fontFamily: fonts.mono,
                  fontSize: 25,
                  lineHeight: '48px',
                  opacity: p * 0.85,
                  transform: `translateX(${(1 - p) * 16}px)`,
                  whiteSpace: 'pre',
                }}
              >
                <span style={{color: '#50545d', width: 40}}>{turn}</span>
                <span style={{color: '#a1a5ad'}}>{text}</span>
              </div>
            );
          })}
          <div
            style={{
              marginTop: 18,
              paddingTop: 18,
              borderTop: '1px solid rgba(255,255,255,0.07)',
              fontFamily: fonts.mono,
              fontSize: 24,
              color: '#6f9c7a',
              opacity: ramp(f, 76, 86),
            }}
          >
            ✓ run finished · exit 0 · no errors reported
          </div>
        </Card>
      </div>
      <div style={{position: 'absolute', top: 850, fontSize: 30, color: colors.muted, ...reveal(settle(f, 92, fps), 14, 8)}}>
        Loops, broken calls and runaway output all look like success.
      </div>
    </AbsoluteFill>
  );
};
