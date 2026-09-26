import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {Label} from '../components/Card';
import {Kinetic} from '../components/Kinetic';
import {gradientText, ramp, reveal, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors} from '../theme';

const BAR_W = 1400;
const BAR_H = 78;

// A conceptual context window: overhead fills most of it before the task arrives.
const SEGMENTS = [
  {label: 'system prompt', from: 0, to: 0.24, t: 26, color: '#7c2d12'},
  {label: 'tool schemas', from: 0.24, to: 0.56, t: 38, color: '#9a3412'},
  {label: 'template', from: 0.56, to: 0.66, t: 52, color: '#7c2d12'},
  {label: 'task', from: 0.66, to: 0.7, t: 66, color: '#f4f4f6', hero: true},
  {label: 'turn 1', from: 0.7, to: 0.8, t: 78, color: '#2a2b31'},
  {label: 'turn 2', from: 0.8, to: 0.9, t: 88, color: '#2a2b31'},
  {label: 'turn 3', from: 0.9, to: 1.0, t: 98, color: '#2a2b31'},
];

export const ProblemContext: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const full = ramp(f, 106, 114);
  const shake = full > 0 ? Math.sin(f * 2.2) * 6 * ramp(f, 108, 132, 1, 0) : 0;
  const overheadP = ramp(f, 60, 74);
  return (
    <AbsoluteFill style={{alignItems: 'center'}}>
      <Kinetic
        start={4}
        stagger={4}
        style={{marginTop: 210, fontSize: 88, fontWeight: 700, letterSpacing: '-0.045em', color: colors.text, textAlign: 'center'}}
        tokens={[
          'Cloud-tuned',
          'agents',
          {text: 'waste', style: gradientText('#fdba74', '#f97316')},
          {text: 'context.', style: {...gradientText('#fb923c', '#ea580c'), paddingBottom: '0.08em'}},
        ]}
      />
      <div style={{position: 'absolute', top: 470, width: BAR_W, ...reveal(settle(f, 14, fps), 20, 8)}}>
        <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 16}}>
          <Label>Context window</Label>
          <Label color={full > 0 ? colors.rose : '#6b7079'} style={{opacity: 0.4 + 0.6 * full}}>
            {full > 0 ? 'context full' : 'local model'}
          </Label>
        </div>
        <div
          style={{
            position: 'relative',
            height: BAR_H,
            borderRadius: 18,
            background: 'rgba(255,255,255,0.03)',
            border: `1px solid ${full > 0 ? `rgba(251,113,133,${0.3 + 0.5 * full})` : 'rgba(255,255,255,0.1)'}`,
            boxShadow: full > 0 ? `0 0 ${50 * full}px rgba(244,63,94,${0.35 * full})` : undefined,
            overflow: 'hidden',
            transform: `translateX(${shake}px)`,
          }}
        >
          {SEGMENTS.map((s) => {
            const p = ramp(f, s.t, s.t + 12);
            const width = (s.to - s.from) * BAR_W * p;
            return (
              <div
                key={s.label}
                style={{
                  position: 'absolute',
                  left: s.from * BAR_W,
                  top: 0,
                  bottom: 0,
                  width,
                  background: s.hero
                    ? '#f4f4f6'
                    : `repeating-linear-gradient(135deg, rgba(255,255,255,0.05) 0 9px, transparent 9px 18px), ${s.color}`,
                  boxShadow: s.hero ? '0 0 30px rgba(255,255,255,0.55)' : undefined,
                  borderRight: '1px solid rgba(0,0,0,0.5)',
                  display: 'flex',
                  alignItems: 'center',
                  paddingLeft: 20,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  fontSize: 21,
                  fontWeight: 500,
                  color: s.hero ? '#0a0a0c' : 'rgba(255,255,255,0.72)',
                }}
              >
                {!s.hero && <span style={{opacity: ramp(f, s.t + 6, s.t + 14)}}>{s.label}</span>}
              </div>
            );
          })}
        </div>
        {/* bracket under the overhead */}
        <div style={{position: 'relative', height: 70, opacity: overheadP}}>
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 14,
              width: 0.66 * BAR_W * overheadP,
              height: 12,
              borderLeft: `1px solid ${colors.orange}`,
              borderRight: `1px solid ${colors.orange}`,
              borderBottom: `1px solid ${colors.orange}`,
              opacity: 0.8,
            }}
          />
          <div style={{position: 'absolute', left: 0, top: 36, fontSize: 22, color: colors.orange, fontWeight: 500}}>
            spent before your task arrives
          </div>
          <div
            style={{
              position: 'absolute',
              left: 0.66 * BAR_W,
              width: 0.04 * BAR_W,
              top: 36,
              textAlign: 'center',
              fontSize: 22,
              color: colors.text,
              fontWeight: 600,
              opacity: ramp(f, 72, 82),
            }}
          >
            task
          </div>
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          top: 760,
          fontSize: 30,
          color: colors.muted,
          textAlign: 'center',
          ...reveal(settle(f, 80, fps), 14, 8),
        }}
      >
        Prompts written for frontier APIs leave a local model little room to work.
      </div>
    </AbsoluteFill>
  );
};
