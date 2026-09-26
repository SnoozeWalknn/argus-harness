import React from 'react';
import {useVideoConfig} from 'remotion';
import {Card, Label} from '../components/Card';
import {FeatureLayout} from '../components/FeatureLayout';
import {LogoMark} from '../components/Logo';
import {EASE_IN_OUT, pop, ramp, rgba} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

const LEFT = 150;
const RIGHT = 670;
const LINK_Y = 150;
const PACKETS = [
  {tool: 'read', t: 26},
  {tool: 'edit', t: 42},
  {tool: 'bash', t: 58},
  {tool: 'grep', t: 74},
];
const TOOLS = ['read', 'edit', 'bash', 'glob', 'grep'];

const Node: React.FC<{x: number; title: string; sub: string; children: React.ReactNode; pulse: number}> = ({
  x,
  title,
  sub,
  children,
  pulse,
}) => (
  <div style={{position: 'absolute', left: x - 95, top: LINK_Y - 95, width: 190, textAlign: 'center'}}>
    <div
      style={{
        width: 190,
        height: 190,
        borderRadius: 34,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02))',
        border: `1px solid rgba(255,255,255,${0.1 + 0.25 * pulse})`,
        boxShadow: `0 0 ${50 * pulse}px ${rgba('#22d3ee', 0.35 * pulse)}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {children}
    </div>
    <div style={{marginTop: 18, fontSize: 24, fontWeight: 600, color: colors.text}}>{title}</div>
    <div style={{marginTop: 4, fontSize: 18, color: colors.muted}}>{sub}</div>
  </div>
);

export const FeatureSSH: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const arrivals = PACKETS.map((p) => ramp(f, p.t + 16, p.t + 20, 1, 0) * ramp(f, p.t + 14, p.t + 16));
  const serverPulse = Math.max(0, ...arrivals);
  return (
    <FeatureLayout
      index="05"
      eyebrow="Local or remote"
      accent={colors.teal}
      title={['Run', 'it', 'on', 'your', 'VM.']}
      body="Every tool goes through one executor, locally or over SSH, with a shared control socket and remote-side timeouts."
    >
      <Card style={{width: 900, padding: '34px 40px'}}>
        <div style={{position: 'relative', height: 380}}>
          <svg width={820} height={300} style={{position: 'absolute', left: 0, top: 0}}>
            <line
              x1={LEFT + 95}
              y1={LINK_Y}
              x2={RIGHT - 95}
              y2={LINK_Y}
              stroke="rgba(45,212,191,0.35)"
              strokeWidth={2}
              strokeDasharray="6 10"
              strokeDashoffset={-f * 1.5}
            />
          </svg>
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: LINK_Y - 58,
              textAlign: 'center',
              fontFamily: fonts.mono,
              fontSize: 17,
              color: '#5f9e97',
            }}
          >
            ssh · ControlMaster
          </div>
          <Node x={LEFT} title="argus" sub="your machine" pulse={0}>
            <LogoMark f={200} size={120} />
          </Node>
          <Node x={RIGHT} title="dev-vm" sub="remote workspace" pulse={serverPulse}>
            <div style={{display: 'flex', flexDirection: 'column', gap: 10}}>
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  style={{
                    width: 104,
                    height: 22,
                    borderRadius: 6,
                    border: '1px solid rgba(255,255,255,0.14)',
                    background: 'rgba(255,255,255,0.04)',
                    display: 'flex',
                    alignItems: 'center',
                    paddingLeft: 10,
                  }}
                >
                  <div style={{width: 7, height: 7, borderRadius: 7, background: i === 0 ? colors.teal : '#3f3f46'}} />
                </div>
              ))}
            </div>
          </Node>
          {PACKETS.map((p) => {
            const go = ramp(f, p.t, p.t + 16, 0, 1, EASE_IN_OUT);
            const back = ramp(f, p.t + 20, p.t + 34, 0, 1, EASE_IN_OUT);
            const visibleGo = go > 0 && go < 1;
            const visibleBack = back > 0 && back < 1;
            const x0 = LEFT + 110;
            const x1 = RIGHT - 110;
            return (
              <React.Fragment key={p.tool}>
                {visibleGo && (
                  <div
                    style={{
                      position: 'absolute',
                      left: x0 + (x1 - x0) * go - 36,
                      top: LINK_Y - 17,
                      width: 72,
                      textAlign: 'center',
                      fontFamily: fonts.mono,
                      fontSize: 17,
                      color: '#ccfbf1',
                      padding: '6px 0',
                      borderRadius: 999,
                      background: rgba('#14b8a6', 0.22),
                      border: `1px solid ${rgba('#2dd4bf', 0.6)}`,
                      boxShadow: `0 0 22px ${rgba('#2dd4bf', 0.6)}`,
                    }}
                  >
                    {p.tool}
                  </div>
                )}
                {visibleBack && (
                  <div
                    style={{
                      position: 'absolute',
                      left: x1 - (x1 - x0) * back - 14,
                      top: LINK_Y + 22,
                      fontSize: 20,
                      color: colors.green,
                      textShadow: `0 0 14px ${rgba('#4ade80', 0.9)}`,
                    }}
                  >
                    ✓
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
        <div style={{borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: 22}}>
          <Label>same tools · same output, local or ssh</Label>
          <div style={{display: 'flex', gap: 12, marginTop: 16}}>
            {TOOLS.map((tool, i) => {
              const ok = pop(f, 96 + i * 5, fps);
              return (
                <div
                  key={tool}
                  style={{
                    flex: 1,
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 10,
                    fontFamily: fonts.mono,
                    fontSize: 19,
                    color: colors.soft,
                    padding: '10px 0',
                    borderRadius: 12,
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'rgba(255,255,255,0.02)',
                  }}
                >
                  {tool}
                  <span style={{color: colors.green, transform: `scale(${Math.max(ok, 0)})`, display: 'inline-block'}}>✓</span>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
    </FeatureLayout>
  );
};
