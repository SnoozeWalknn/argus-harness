import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import data from '../data.json';
import {EASE_IN_OUT, fmt, pop, ramp, reveal, rgba, settle, typed} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

const NATIVE = data.overhead.native.total;
const GRAMMAR = data.overhead.grammar.total;
const SAVED = Math.round((1 - GRAMMAR / NATIVE) * 100);
const ACTION = '{"tool":"edit","args":{"path":"calc.py","old":"a - b","new":"a + b"}}';

/** Colour a compact JSON action: keys violet, strings teal, punctuation grey. */
const Json: React.FC<{text: string}> = ({text}) => {
  const parts = [...text.matchAll(/"(?:[^"\\]|\\.)*"?|[{}:,]|[^"{}:,]+/g)];
  return (
    <>
      {parts.map((m, i) => {
        const p = m[0];
        const at = m.index ?? 0;
        const before = text[at - 1];
        // a key is followed by ':'; while still being typed, it follows '{' or ','
        const isKey = p.startsWith('"') && (text[at + p.length] === ':' || (at + p.length === text.length && (before === '{' || before === ',')));
        const color = /^[{}:,]$/.test(p) ? '#6b7079' : p.startsWith('"') ? (isKey ? '#c4b5fd' : '#5eead4') : colors.soft;
        return (
          <span key={i} style={{color}}>
            {p}
          </span>
        );
      })}
    </>
  );
};

const Bar: React.FC<{label: string; value: number; frac: number; color: string; glow?: boolean}> = ({
  label,
  value,
  frac,
  color,
  glow,
}) => (
  <div style={{display: 'grid', gridTemplateColumns: '150px 1fr 90px', alignItems: 'center', gap: 18}}>
    <div style={{fontSize: 21, color: colors.muted}}>{label}</div>
    <div style={{height: 16, borderRadius: 8, background: 'rgba(255,255,255,0.05)'}}>
      <div
        style={{
          width: `${frac * 100}%`,
          height: '100%',
          borderRadius: 8,
          background: color,
          boxShadow: glow ? `0 0 22px ${rgba('#8b5cf6', 0.7)}` : undefined,
        }}
      />
    </div>
    <div style={{fontFamily: fonts.mono, fontSize: 21, color: colors.soft, textAlign: 'right'}}>{fmt(value)}</div>
  </div>
);

export const FeatureConstrained: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const drop = ramp(f, 34, 84, 0, 1, EASE_IN_OUT);
  const value = NATIVE + (GRAMMAR - NATIVE) * drop;
  const chip = pop(f, 84, fps);
  return (
    <FeatureLayout
      index="01"
      eyebrow="Constrained tool calls"
      accent={colors.violet}
      title={['Calls', 'that', 'always', 'parse.']}
      body="Grammar and JSON-schema constraints enforce every argument, so the prompt carries one-line signatures instead of full schemas."
    >
      <Card style={{width: 900, padding: '40px 46px'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline'}}>
          <Label>Prompt overhead</Label>
          <div style={{fontSize: 17, color: colors.faint}}>tokens · mock-server measurement</div>
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 26, marginTop: 18}}>
          <div
            style={{
              fontSize: 136,
              fontWeight: 700,
              letterSpacing: '-0.05em',
              fontVariantNumeric: 'tabular-nums',
              color: colors.text,
              textShadow: `0 0 ${40 * drop}px ${rgba('#8b5cf6', 0.55)}`,
              lineHeight: 1,
            }}
          >
            {fmt(value)}
          </div>
          <div
            style={{
              fontSize: 30,
              fontWeight: 600,
              color: '#86efac',
              padding: '8px 18px',
              borderRadius: 999,
              background: rgba('#22c55e', 0.12),
              border: `1px solid ${rgba('#4ade80', 0.35)}`,
              transform: `scale(${chip})`,
              opacity: Math.min(1, chip),
            }}
          >
            −{SAVED}%
          </div>
        </div>
        <div style={{fontSize: 21, color: colors.muted, marginTop: 10}}>
          grammar protocol, vs {fmt(NATIVE)} with native tool schemas
        </div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 16, marginTop: 34}}>
          <Bar label="native" value={NATIVE} frac={1} color="linear-gradient(90deg, #52525b, #71717a)" />
          <Bar
            label="grammar"
            value={value}
            frac={value / NATIVE}
            color="linear-gradient(90deg, #7c3aed, #a78bfa)"
            glow
          />
        </div>
        <div
          style={{
            marginTop: 36,
            padding: '22px 26px',
            borderRadius: 16,
            background: 'rgba(0,0,0,0.35)',
            border: '1px solid rgba(255,255,255,0.06)',
            fontFamily: fonts.mono,
            fontSize: 21,
            lineHeight: 1.7,
            ...reveal(settle(f, 56, fps), 14, 6),
          }}
        >
          <div style={{color: '#5d616b'}}>root ::= think? action</div>
          <div style={{whiteSpace: 'pre'}}>
            <Json text={typed(ACTION, f, 66, 1.6)} />
            <span style={{color: colors.violet, opacity: Math.floor(f / 8) % 2 ? 1 : 0}}>▍</span>
          </div>
        </div>
      </Card>
    </FeatureLayout>
  );
};
