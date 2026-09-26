import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import {pop, ramp, rgba, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

// argus's matching cascade, in order; the last step is the one that matches here.
const STEPS = ['exact', 'line numbers', 'whitespace', 'indentation', 'unicode', 'spacing'];
const FILE = ['def add(a, b):', '    return a - b', '', '', 'def mul(a, b):', '    return a * b'];
const STEP_T = 24;
const STEP_GAP = 8;

export const FeatureEdits: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const hitT = STEP_T + (STEPS.length - 1) * STEP_GAP;
  const matched = ramp(f, hitT + 4, hitT + 12);
  const replaced = ramp(f, hitT + 18, hitT + 26);
  const flash = ramp(f, hitT + 18, hitT + 20) * ramp(f, hitT + 20, hitT + 46, 1, 0); // 0 before the replace
  return (
    <FeatureLayout
      index="03"
      eyebrow="Fuzzy edits"
      accent={colors.green}
      title={['Edits', 'that', 'land.']}
      body="Search/replace that forgives whitespace, indentation and copied line numbers, and shows the closest text when nothing matches."
    >
      <Card style={{width: 900, padding: '36px 42px'}}>
        <Label>model sent</Label>
        <div style={{fontFamily: fonts.mono, fontSize: 22, color: colors.soft, marginTop: 12, whiteSpace: 'pre'}}>
          <span style={{color: colors.sky}}>edit</span>(path=<span style={{color: '#5eead4'}}>"calc.py"</span>, old=
          <span style={{color: '#fcd34d', background: rgba(colors.amber, 0.14), borderRadius: 4, padding: '0 3px'}}>
            "return a-b"
          </span>
          , new=<span style={{color: '#5eead4'}}>"return a + b"</span>)
        </div>
        <div style={{display: 'flex', gap: 8, marginTop: 28}}>
          {STEPS.map((step, i) => {
            const t = STEP_T + i * STEP_GAP;
            const hit = i === STEPS.length - 1;
            const checking = ramp(f, t, t + 3) * (hit ? 1 : ramp(f, t + 5, t + 8, 1, 0));
            const done = ramp(f, t + 5, t + 8);
            const scale = hit ? pop(f, t + 4, fps) : 1;
            const color = hit ? colors.green : colors.faint;
            return (
              <div
                key={step}
                style={{
                  fontFamily: fonts.mono,
                  fontSize: 16,
                  padding: '7px 12px',
                  whiteSpace: 'nowrap',
                  borderRadius: 10,
                  color: done ? color : checking > 0.5 ? colors.text : '#6b7079',
                  border: `1px solid ${hit && done ? rgba(colors.green, 0.5) : `rgba(255,255,255,${0.08 + 0.25 * checking})`}`,
                  background: hit && done ? rgba(colors.green, 0.12) : 'rgba(255,255,255,0.02)',
                  boxShadow: hit && done ? `0 0 26px ${rgba(colors.green, 0.35)}` : undefined,
                  transform: `scale(${hit ? Math.max(scale, 0.001) : 1})`,
                }}
              >
                {done ? (hit ? '✓ ' : '✗ ') : ''}
                {step}
              </div>
            );
          })}
        </div>
        <div
          style={{
            marginTop: 30,
            borderRadius: 16,
            background: 'rgba(0,0,0,0.38)',
            border: '1px solid rgba(255,255,255,0.06)',
            padding: '16px 0',
            fontFamily: fonts.mono,
            fontSize: 22,
            ...(() => {
              const p = settle(f, 12, fps);
              return {opacity: p};
            })(),
          }}
        >
          <div style={{padding: '0 22px 10px', fontSize: 16, color: colors.faint}}>calc.py</div>
          {FILE.map((line, i) => {
            const target = i === 1;
            return (
              <div
                key={i}
                style={{
                  position: 'relative',
                  display: 'flex',
                  lineHeight: '38px',
                  whiteSpace: 'pre',
                  background: target
                    ? `linear-gradient(90deg, ${rgba(replaced > 0 ? colors.green : colors.amber, 0.1 * matched + 0.25 * flash)}, transparent)`
                    : undefined,
                  boxShadow: target && matched > 0 ? `inset 3px 0 0 ${replaced > 0 ? colors.green : colors.amber}` : undefined,
                }}
              >
                <span style={{width: 60, textAlign: 'right', paddingRight: 22, color: '#4b4f58'}}>{i + 1}</span>
                {target ? (
                  <span style={{position: 'relative', color: colors.soft}}>
                    <span style={{opacity: 1 - replaced}}>{line}</span>
                    <span style={{position: 'absolute', left: 0, opacity: replaced, color: '#bbf7d0'}}>
                      {'    return a + b'}
                    </span>
                  </span>
                ) : (
                  <span style={{color: colors.soft}}>{line}</span>
                )}
              </div>
            );
          })}
        </div>
        <div style={{marginTop: 20, fontFamily: fonts.mono, fontSize: 16, color: colors.muted, opacity: ramp(f, hitT + 24, hitT + 34)}}>
          Edited calc.py (lines 2-2). Note: `old` was not exact; matched ignoring spacing.
        </div>
      </Card>
    </FeatureLayout>
  );
};
