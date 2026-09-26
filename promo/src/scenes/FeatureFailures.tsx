import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import {pop, ramp, rgba, settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

// Detail strings follow argus's real failure messages.
const ROWS = [
  {tag: 'loop', color: colors.amber, turn: 't5', detail: 'same call and result 3x in a row: grep(pattern=def)'},
  {tag: 'overrun', color: colors.violet, turn: 't6', detail: 'generation continued past the final answer (<|im_start|>)'},
  {tag: 'malformed_call', color: colors.rose, turn: 't2', detail: 'arguments are not valid JSON (Expecting \',\' delimiter)'},
  {tag: 'token_cap', color: colors.sky, turn: 't3', detail: 'reasoning exceeded 2048 tokens · retried, thinking off'},
];

export const FeatureFailures: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  return (
    <FeatureLayout
      index="02"
      eyebrow="Failure tagging"
      accent={colors.amber}
      title={['Every', 'failure', 'has', 'a', 'name.']}
      body="Loops, overruns, malformed calls and token caps are detected as they happen, tagged per turn and logged to SQLite."
    >
      <Card style={{width: 900, padding: '34px 40px 30px'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 14}}>
          <Label>failures</Label>
          <div style={{fontFamily: fonts.mono, fontSize: 17, color: colors.faint}}>argus failures</div>
        </div>
        {ROWS.map((r, i) => {
          const t = 20 + i * 13;
          const p = settle(f, t, fps);
          const flash = ramp(f, t + 4, t + 30, 1, 0);
          const pill = pop(f, t + 2, fps);
          return (
            <div
              key={r.tag}
              style={{
                display: 'grid',
                gridTemplateColumns: '215px 1fr 40px',
                alignItems: 'center',
                gap: 22,
                padding: '22px 0',
                borderTop: '1px solid rgba(255,255,255,0.06)',
                opacity: p,
                transform: `translateX(${(1 - p) * 50}px)`,
              }}
            >
              <div>
                <span
                  style={{
                    display: 'inline-block',
                    fontFamily: fonts.mono,
                    fontSize: 20,
                    fontWeight: 500,
                    color: r.color,
                    padding: '7px 14px',
                    borderRadius: 999,
                    background: rgba(r.color, 0.1),
                    border: `1px solid ${rgba(r.color, 0.4)}`,
                    boxShadow: `0 0 ${10 + 30 * flash}px ${rgba(r.color, 0.25 + 0.4 * flash)}`,
                    transform: `scale(${pill})`,
                  }}
                >
                  {r.tag}
                </span>
              </div>
              <div style={{fontFamily: fonts.mono, fontSize: 19, color: colors.soft, lineHeight: 1.4}}>{r.detail}</div>
              <div style={{fontFamily: fonts.mono, fontSize: 18, color: colors.faint, textAlign: 'right'}}>{r.turn}</div>
            </div>
          );
        })}
        <div
          style={{
            borderTop: '1px solid rgba(255,255,255,0.06)',
            paddingTop: 20,
            display: 'flex',
            gap: 14,
            fontSize: 19,
            color: colors.muted,
            opacity: ramp(f, 84, 98),
          }}
        >
          <span style={{color: colors.green}}>●</span> nudged, retried or aborted, and never silent
        </div>
      </Card>
    </FeatureLayout>
  );
};
