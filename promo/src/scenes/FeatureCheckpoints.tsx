import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import data from '../data.json';
import {pop, ramp, rgba, typed} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

// The checkpoints of the recorded demo run: baseline, before edit, before bash, final.
const POINTS = data.terminal.checkpoints;
const X0 = 70;
const X1 = 750;
const Y = 150;
const RESTORE = 'argus restore last --to baseline --yes';

export const FeatureCheckpoints: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const line = ramp(f, 18, 70);
  const step = (X1 - X0) / (POINTS.length - 1);
  const arc = ramp(f, 100, 124);
  const arcPath = `M ${X1} ${Y - 26} C ${X1 - 120} ${Y - 120}, ${X0 + 120} ${Y - 120}, ${X0} ${Y - 26}`;
  const arcLen = 820;
  const badge = pop(f, 120, fps);
  return (
    <FeatureLayout
      index="04"
      eyebrow="Shadow-git checkpoints"
      accent={colors.indigo}
      title={['Every', 'change,', 'reversible.']}
      body="A shadow repository snapshots the workspace before each edit and command. Your own git history is never touched."
    >
      <Card style={{width: 900, padding: '36px 40px 34px'}}>
        <div style={{display: 'flex', justifyContent: 'space-between'}}>
          <Label>refs/argus/&lt;run&gt;</Label>
          <div style={{fontFamily: fonts.mono, fontSize: 17, color: colors.faint}}>argus checkpoints last</div>
        </div>
        <div style={{position: 'relative', height: 300, marginTop: 10}}>
          <svg width={820} height={300} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}}>
            <defs>
              <linearGradient id="tl" gradientUnits="userSpaceOnUse" x1={X0} y1={Y} x2={X1} y2={Y}>
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#a78bfa" />
              </linearGradient>
              <marker id="head" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#c4b5fd" />
              </marker>
            </defs>
            <line x1={X0} y1={Y} x2={X1} y2={Y} stroke="rgba(255,255,255,0.08)" strokeWidth={3} />
            <line x1={X0} y1={Y} x2={X0 + (X1 - X0) * line} y2={Y} stroke="url(#tl)" strokeWidth={3} />
            <path
              d={arcPath}
              fill="none"
              stroke="#c4b5fd"
              strokeWidth={3}
              strokeDasharray={arcLen}
              strokeDashoffset={arcLen * (1 - arc)}
              markerEnd={arc > 0.97 ? 'url(#head)' : undefined}
              style={{filter: 'drop-shadow(0 0 10px rgba(167,139,250,0.9))', opacity: arc > 0 ? 1 : 0}}
            />
          </svg>
          {POINTS.map((p, i) => {
            const x = X0 + i * step;
            const reach = (x - X0) / (X1 - X0);
            const on = pop(f, 18 + reach * 52, fps);
            const changed = p.change.startsWith('M') || p.change.startsWith('A') || p.change.startsWith('D');
            return (
              <div key={i} style={{position: 'absolute', left: x - 90, top: Y - 14, width: 180, textAlign: 'center'}}>
                <div
                  style={{
                    width: 28,
                    height: 28,
                    margin: '0 auto',
                    borderRadius: 999,
                    background: i === 0 || i === POINTS.length - 1 ? '#a78bfa' : '#18181b',
                    border: '3px solid #a78bfa',
                    boxShadow: `0 0 ${24 * on}px ${rgba('#8b5cf6', 0.8)}`,
                    transform: `scale(${Math.max(on, 0)})`,
                  }}
                />
                <div style={{marginTop: 18, fontSize: 21, fontWeight: 600, color: colors.text, opacity: Math.min(1, on)}}>
                  {p.reason}
                </div>
                <div style={{fontFamily: fonts.mono, fontSize: 18, color: colors.faint, marginTop: 4, opacity: Math.min(1, on)}}>
                  {p.commit}
                </div>
                <div
                  style={{
                    fontFamily: fonts.mono,
                    fontSize: 17,
                    marginTop: 6,
                    color: changed ? colors.green : '#50545d',
                    opacity: Math.min(1, on),
                  }}
                >
                  {p.change || 'start'}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 6}}>
          <div
            style={{
              fontFamily: fonts.mono,
              fontSize: 21,
              color: colors.soft,
              padding: '14px 18px',
              borderRadius: 12,
              background: 'rgba(0,0,0,0.35)',
              border: '1px solid rgba(255,255,255,0.07)',
              minWidth: 520,
              opacity: ramp(f, 78, 86),
            }}
          >
            <span style={{color: colors.faint}}>$ </span>
            {typed(RESTORE, f, 82, 2.2)}
          </div>
          <div
            style={{
              fontSize: 19,
              fontWeight: 500,
              color: '#86efac',
              padding: '10px 16px',
              borderRadius: 999,
              background: rgba('#22c55e', 0.1),
              border: `1px solid ${rgba('#4ade80', 0.3)}`,
              transform: `scale(${Math.max(badge, 0)})`,
            }}
          >
            ✓ your repo untouched
          </div>
        </div>
      </Card>
    </FeatureLayout>
  );
};
