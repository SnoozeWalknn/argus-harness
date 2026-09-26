import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import data from '../data.json';
import {pop, ramp, rgba, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

// Illustrative numbers (labelled on screen); the intervals and p-value are computed by
// argus's own report code in scripts/capture.py.
const AB = data.ab;
const TRACK = 470;

export const FeatureAB: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const pP = pop(f, 92, fps);
  return (
    <FeatureLayout
      index="06"
      eyebrow="A/B suites"
      accent={colors.indigo}
      title={['Know', 'which', 'config', 'wins.']}
      body="Replay saved task suites under two configs, interleaved, with Wilson intervals and an exact McNemar test."
    >
      <Card style={{width: 900, padding: '34px 42px 30px'}}>
        <div style={{fontFamily: fonts.mono, fontSize: 18, color: colors.faint}}>
          $ argus ab native.toml grammar.toml --suite calc.toml -n 4
        </div>
        <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 26, marginBottom: 8}}>
          <Label>pass rate · 95% CI</Label>
          <div style={{fontSize: 17, color: colors.faint}}>illustrative numbers</div>
        </div>
        {AB.variants.map((v, i) => {
          const rate = v.passes / AB.runs;
          const grow = settle(f, 24 + i * 10, fps, 1.2);
          const ci = ramp(f, 58 + i * 6, 70 + i * 6);
          const [lo, hi] = v.ci95;
          const winner = i === 1;
          return (
            <div key={v.label} style={{display: 'grid', gridTemplateColumns: '130px 1fr 176px', alignItems: 'center', gap: 18, padding: '22px 0'}}>
              <div style={{fontSize: 24, fontWeight: 600, color: winner ? colors.text : colors.muted}}>{v.label}</div>
              <div style={{position: 'relative', width: TRACK, height: 30}}>
                <div style={{position: 'absolute', inset: '8px 0', borderRadius: 7, background: 'rgba(255,255,255,0.05)'}} />
                <div
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 8,
                    height: 14,
                    width: TRACK * rate * grow,
                    borderRadius: 7,
                    background: winner ? 'linear-gradient(90deg, #6366f1, #a78bfa)' : 'linear-gradient(90deg, #3f3f46, #71717a)',
                    boxShadow: winner ? `0 0 24px ${rgba('#8b5cf6', 0.6)}` : undefined,
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    left: TRACK * lo,
                    width: TRACK * (hi - lo),
                    top: 0,
                    height: 30,
                    borderLeft: '2px solid rgba(255,255,255,0.75)',
                    borderRight: '2px solid rgba(255,255,255,0.75)',
                    opacity: ci,
                  }}
                >
                  <div style={{position: 'absolute', left: 0, right: 0, top: 14, height: 2, background: 'rgba(255,255,255,0.75)'}} />
                </div>
              </div>
              <div style={{fontFamily: fonts.mono, fontSize: 21, color: winner ? colors.text : colors.muted, textAlign: 'right', opacity: ramp(f, 40 + i * 10, 52 + i * 10)}}>
                {v.passes}/{AB.runs} · {Math.round(rate * 100)}%
              </div>
            </div>
          );
        })}
        <div
          style={{
            marginTop: 20,
            paddingTop: 24,
            borderTop: '1px solid rgba(255,255,255,0.07)',
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
          }}
        >
          <div style={{fontSize: 20, color: colors.muted, lineHeight: 1.6, opacity: ramp(f, 76, 88)}}>
            {AB.runs} paired runs
            <br />
            only grammar passed: <span style={{color: colors.text}}>{AB.only_b}</span> · only native passed:{' '}
            <span style={{color: colors.text}}>{AB.only_a}</span>
          </div>
          <div style={{textAlign: 'right', transform: `scale(${Math.max(pP, 0)})`, transformOrigin: 'right bottom'}}>
            <div style={{fontSize: 17, color: colors.faint, letterSpacing: '0.12em', textTransform: 'uppercase'}}>exact McNemar</div>
            <div
              style={{
                fontSize: 64,
                fontWeight: 700,
                letterSpacing: '-0.03em',
                color: '#ddd6fe',
                textShadow: `0 0 30px ${rgba('#8b5cf6', 0.8)}`,
              }}
            >
              p = {AB.mcnemar_p.toFixed(3)}
            </div>
          </div>
        </div>
      </Card>
    </FeatureLayout>
  );
};
