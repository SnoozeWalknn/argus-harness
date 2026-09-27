import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {EASE_IN_OUT, ramp, reveal, rgba, settle} from '../../lib/anim';
import {useSceneFrame} from '../../lib/scene';
import {fonts} from '../../theme';
import {CHAPTERS} from '../timeline';
import {Callout, ChapterTag, Chip, Highlight, Title} from '../ui/overlay';
import {anchor, Panel} from '../ui/svg';

const chapter = (key: string) => CHAPTERS.find((c) => c.key === key)!;

// -- 04 · every tool -------------------------------------------------------------------------------

const TOOLS: [string, string][] = [
  ['read', 'files, with line numbers'],
  ['edit', 'search/replace that forgives'],
  ['write', 'create or replace a file'],
  ['bash', 'sandboxed shell commands'],
  ['job', 'background servers & watchers'],
  ['glob', 'find files by pattern'],
  ['grep', 'regex search (ripgrep)'],
  ['ls', 'the tree, .gitignore-aware'],
  ['fetch', 'web pages & localhost as text'],
  ['web_search', 'Brave or your SearXNG'],
];
const W_TOOLS = 940;

export const Tools: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('tools');
  const panel = settle(f, 70, fps, 1.2);
  const a = (key: string) => anchor('subagent-jobs', key, W_TOOLS);
  const sweepAt = ramp(f, 30, 70) * 12;
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title lines={['Every tool', 'you reach for.']} highlight={1} accent={c.accent} size={72} />
      <div
        style={{
          position: 'absolute',
          left: 110,
          top: 390,
          display: 'grid',
          gridTemplateColumns: '355px 355px',
          gap: 14,
        }}
      >
        {TOOLS.map(([name, desc], i) => {
          const p = settle(f, 14 + i * 4, fps);
          const lit = Math.max(0, 1 - Math.abs(sweepAt - i) / 1.5);
          return (
            <div
              key={name}
              style={{
                height: 86,
                borderRadius: 16,
                padding: '14px 20px',
                boxSizing: 'border-box',
                background: `linear-gradient(180deg, ${rgba(c.accent, 0.06 + 0.1 * lit)}, rgba(255,255,255,0.015))`,
                border: `1px solid ${rgba(c.accent, 0.18 + 0.5 * lit)}`,
                boxShadow: lit > 0.2 ? `0 0 30px ${rgba(c.accent, 0.25 * lit)}` : undefined,
                ...reveal(p, 18, 8),
              }}
            >
              <div style={{fontFamily: fonts.mono, fontWeight: 700, fontSize: 25, color: '#99f6e4'}}>{name}</div>
              <div style={{fontSize: 19, color: '#a1a4ad', marginTop: 4, whiteSpace: 'nowrap'}}>{desc}</div>
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 880, top: 250, perspective: 2200, opacity: panel}}>
        <div style={{transform: `rotateY(${(1 - panel) * -20}deg) translateX(${(1 - panel) * 140}px)`}}>
          <Panel name="subagent-jobs" width={W_TOOLS} sweep={ramp(f, 80, 116)} glow="rgba(15,118,110,0.45)">
            <Highlight x={a('job')[0] - 6} y={a('job')[1] - 10} w={570} h={48} at={112} color={c.accent} />
            <Callout x={a('job')[0] + 400} y={a('job')[1] - 10} dx={-10} dy={-250} at={116} label="A dev server, in the background" color={c.accent} />
            <Highlight x={a('fetch')[0] - 6} y={a('fetch')[1] - 10} w={400} h={28} at={142} color="#7dd3fc" />
            <Callout x={a('fetch')[0] + 400} y={a('fetch')[1]} dx={120} dy={80} at={146} label="…then it fetches the page" color="#7dd3fc" />
            <Callout x={a('explore')[0]} y={a('explore')[1]} dx={-30} dy={-60} at={172} label="A subagent, fresh context" color="#f0abfc" />
          </Panel>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// -- 05 · safe by default ------------------------------------------------------------------------------

const POLICIES: [string, string][] = [
  ['read-only', 'look, never touch'],
  ['ask', 'confirm each command'],
  ['auto', 'sandboxed, in the workspace'],
  ['full', 'no guard rails'],
];
const W_SAFE = 980;

export const Safety: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('safety');
  const panel = settle(f, 30, fps, 1.2);
  const sel = ramp(f, 40, 58, 0, 1, EASE_IN_OUT) + ramp(f, 150, 168, 0, 1, EASE_IN_OUT); // read-only → ask → auto
  const idx = Math.round(sel);
  const a = (key: string) => anchor('approval', key, W_SAFE);
  const segW = 168;
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['Sandboxed.', 'You approve.']}
        highlight={1}
        accent={c.accent}
        size={80}
        sub="Commands run under bubblewrap or Landlock. Nothing outside the workspace changes unless you say so."
      />
      <div style={{position: 'absolute', left: 110, top: 610, ...reveal(settle(f, 26, fps), 14, 8)}}>
        <div
          style={{
            position: 'relative',
            display: 'flex',
            width: segW * 4,
            height: 64,
            borderRadius: 18,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 4,
              left: 4 + sel * segW,
              width: segW - 8,
              height: 56,
              borderRadius: 14,
              background: rgba(c.accent, 0.22),
              border: `1.5px solid ${rgba(c.accent, 0.8)}`,
              boxShadow: `0 0 30px ${rgba(c.accent, 0.35)}`,
            }}
          />
          {POLICIES.map(([name], i) => (
            <div
              key={name}
              style={{
                position: 'relative',
                width: segW,
                lineHeight: '64px',
                textAlign: 'center',
                fontFamily: fonts.mono,
                fontWeight: 700,
                fontSize: 22,
                color: i === idx ? '#fff7ed' : '#8b8f98',
              }}
            >
              {name}
            </div>
          ))}
        </div>
        <div style={{marginTop: 16, fontSize: 24, color: '#d6d3d1', height: 32}}>
          {POLICIES[idx][0]}: {POLICIES[idx][1]}
          {idx === 2 ? <span style={{color: '#8b8f98'}}> · the default</span> : null}
        </div>
      </div>
      <div style={{position: 'absolute', left: 110, top: 790, display: 'flex', gap: 14, flexWrap: 'wrap', width: 700}}>
        {['bubblewrap ✓', 'Landlock ABI 7', 'network on/off', 'refusals keep the session'].map((t, i) => (
          <div key={t} style={{...reveal(settle(f, 80 + i * 8, fps), 12, 6)}}>
            <Chip color={c.accent} style={{fontSize: 21}}>
              {t}
            </Chip>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', left: 860, top: 260, perspective: 2200, opacity: panel}}>
        <div style={{transform: `rotateY(${(1 - panel) * -18}deg) translateX(${(1 - panel) * 120}px)`}}>
          <Panel name="approval" width={W_SAFE} sweep={ramp(f, 40, 76)} glow="rgba(180,83,9,0.45)">
            <Highlight x={a('cmd')[0] - 6} y={a('cmd')[1] - 16} w={a('cmd')[2] + 180} h={32} at={70} color={c.accent} />
            <Callout x={a('always')[0] + 60} y={a('always')[1] + 30} dx={80} dy={130} at={90} label="y · a · n, from the keyboard" color={c.accent} />
          </Panel>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// -- 06 · plan · delegate · hook ----------------------------------------------------------------------------

const FLOW: [string, string, string][] = [
  ['◇', 'Plan mode', 'tab: read-only, ends in a plan'],
  ['☑', 'Todos', 'the model keeps its own checklist'],
  ['⎇', 'Subagents', 'explore in a fresh context'],
  ['⚑', 'Hooks', 'pre_tool · post_tool · stop; exit 2 blocks'],
  ['✦', 'Skills', 'loaded only when needed'],
];
const W_PLAN = 980;

export const Workflow: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('workflow');
  const panel = settle(f, 24, fps, 1.2);
  const a = (key: string) => anchor('plan', key, W_PLAN);
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title lines={['Plan it.', 'Then build it.']} highlight={1} accent={c.accent} size={80} />
      <div style={{position: 'absolute', left: 110, top: 410, display: 'flex', flexDirection: 'column', gap: 18}}>
        {FLOW.map(([icon, name, desc], i) => {
          const p = settle(f, 30 + i * 9, fps);
          return (
            <div key={name} style={{display: 'flex', alignItems: 'center', gap: 22, ...reveal(p, 16, 8)}}>
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: 16,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 28,
                  color: '#fdf4ff',
                  background: rgba(c.accent, 0.14),
                  border: `1.5px solid ${rgba(c.accent, 0.45)}`,
                }}
              >
                {icon}
              </div>
              <div>
                <div style={{fontSize: 30, fontWeight: 700, color: '#f5f5f7', letterSpacing: '-0.02em'}}>{name}</div>
                <div style={{fontSize: 21, color: '#a1a4ad'}}>{desc}</div>
              </div>
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 860, top: 250, perspective: 2200, opacity: panel}}>
        <div style={{transform: `rotateY(${(1 - panel) * -18}deg) translateX(${(1 - panel) * 120}px)`}}>
          <Panel name="plan" width={W_PLAN} sweep={ramp(f, 34, 70)} glow="rgba(162,28,175,0.4)">
            <Highlight x={a('badge')[0] - 8} y={a('badge')[1] - 14} w={a('badge')[2] + 16} h={28} at={70} color={c.accent} />
            <Callout x={a('badge')[0] + a('badge')[2] + 12} y={a('badge')[1]} dx={90} dy={-80} at={76} label="Plan mode: nothing changes" color={c.accent} />
            <Highlight x={a('title')[0] - 8} y={a('title')[1] - 6} w={700} h={136} at={104} color="#c4b5fd" />
            <Callout x={a('title')[0] + 600} y={a('title')[1] + 130} dx={-20} dy={110} at={110} label="Then tab to build it" color="#c4b5fd" />
          </Panel>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// -- 07 · knows your code ----------------------------------------------------------------------------------

const W_LSP = 1560;

export const Code: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('code');
  const box = settle(f, 16, fps, 1.2);
  const [ex, ey] = anchor('lsp', 'error', W_LSP);
  const [wx, wy, ww] = anchor('lsp', 'warn', W_LSP);
  const drift = ramp(f, 0, 180, 0, 1) * 30;
  const viewW = 1000;
  const viewH = 520;
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['Knows your code.']}
        highlight={0}
        accent={c.accent}
        size={76}
        sub="A language server checks every edit; errors come straight back to the model."
      />
      <div style={{position: 'absolute', left: 110, top: 520, display: 'flex', flexDirection: 'column', gap: 20}}>
        {[
          ['argus resume', 'sessions continue, on any model'],
          ['argus diff last', 'every run is checkpointed; restore any'],
          ['executor.kind = "ssh"', 'the same tools on a remote box'],
        ].map(([cmd, what], i) => (
          <div key={cmd} style={{...reveal(settle(f, 50 + i * 10, fps), 14, 8)}}>
            <div style={{fontFamily: fonts.mono, fontSize: 25, fontWeight: 700, color: '#86efac'}}>{cmd}</div>
            <div style={{fontSize: 22, color: '#a1a4ad', marginTop: 2}}>{what}</div>
          </div>
        ))}
      </div>
      <div
        style={{
          position: 'absolute',
          left: 800,
          top: 250,
          width: viewW,
          height: viewH,
          borderRadius: 22,
          overflow: 'hidden',
          boxShadow: '0 50px 120px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.12), 0 0 80px rgba(4,120,87,0.35)',
          opacity: box,
          transform: `scale(${0.94 + 0.06 * box})`,
        }}
      >
        <div style={{position: 'absolute', left: viewW / 2 - (ex + 150) - drift, top: viewH / 2 - ey - 10}}>
          <Panel name="lsp" width={W_LSP}>
            <Highlight x={wx - 92} y={wy - 11} w={ww + 470} h={44} at={40} color="#fbbf24" pad={10} />
          </Panel>
        </div>
      </div>
      <div style={{position: 'absolute', left: 800, top: 250}}>
        <Callout x={viewW / 2 + 80 - drift} y={viewH / 2 + 12} dx={60} dy={290} at={64} label="pyflakes, right after the edit" color="#fbbf24" />
      </div>
    </AbsoluteFill>
  );
};
