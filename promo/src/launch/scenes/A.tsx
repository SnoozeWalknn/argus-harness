import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {LogoMark} from '../../components/Logo';
import {EASE_IN_OUT, gradientText, ramp, reveal, rgba, settle} from '../../lib/anim';
import {useSceneFrame} from '../../lib/scene';
import {fonts} from '../../theme';
import {CHAPTERS} from '../timeline';
import {Callout, ChapterTag, Chip, Highlight, Title} from '../ui/overlay';
import {anchor, Panel} from '../ui/svg';

const chapter = (key: string) => CHAPTERS.find((c) => c.key === key)!;

// -- 0 · cold open -------------------------------------------------------------------------------

export const Open: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const line = ramp(f, 0, 30);
  const a = settle(f, 12, fps);
  const b = settle(f, 34, fps);
  const lift = ramp(f, 78, 112, 0, 1, EASE_IN_OUT);
  const mark = settle(f, 84, fps);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div
        style={{
          position: 'absolute',
          top: 540 - lift * 150,
          left: 960 - 760 * line,
          width: 1520 * line,
          height: 2,
          background: 'linear-gradient(90deg, transparent, rgba(196,181,253,0.9), rgba(125,211,252,0.9), transparent)',
          boxShadow: '0 0 30px rgba(167,139,250,0.7)',
          opacity: 1 - ramp(f, 30, 60) * 0.85,
        }}
      />
      <div style={{textAlign: 'center', transform: `translateY(${-lift * 150}px) scale(${1 - lift * 0.18})`}}>
        <div style={{fontSize: 164, fontWeight: 800, letterSpacing: '-0.06em', lineHeight: 1, color: '#f5f5f7', ...reveal(a, 36, 20)}}>
          Every model.
        </div>
        <div
          style={{
            fontSize: 164,
            fontWeight: 800,
            letterSpacing: '-0.06em',
            lineHeight: 1.08,
            paddingBottom: '0.06em',
            ...gradientText('#ddd6fe', '#7dd3fc', 100),
            ...reveal(b, 36, 20),
            filter: b > 0.99 ? 'drop-shadow(0 0 50px rgba(139,92,246,0.45))' : `blur(${(1 - b) * 20}px)`,
          }}
        >
          One agent.
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          top: 650,
          display: 'flex',
          alignItems: 'center',
          gap: 26,
          opacity: mark,
          transform: `translateY(${(1 - mark) * 30}px)`,
        }}
      >
        <LogoMark f={(f - 80) * 1.4} size={120} />
        <div style={{fontSize: 76, fontWeight: 800, letterSpacing: '-0.05em', color: '#f4f4f6'}}>argus</div>
      </div>
      <div
        style={{
          position: 'absolute',
          top: 800,
          fontSize: 28,
          color: '#9a9ca6',
          letterSpacing: '0.02em',
          ...reveal(settle(f, 100, fps), 12, 8),
        }}
      >
        a coding agent for local and frontier models
      </div>
    </AbsoluteFill>
  );
};

// -- 01 · any model -------------------------------------------------------------------------------

const LOCAL = ['llama-server', 'Ollama', 'vLLM', 'LM Studio'];
const CLOUD = ['Anthropic', 'OpenAI', 'Gemini', 'OpenRouter'];
const SPECS = ['-m opus', '-m flash', '-m gpt', '-m ollama/qwen3-coder:30b', '-m llama'];

export const Models: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('models');
  const cx = 1345;
  const cy = 560;
  const column = (names: string[], side: -1 | 1, color: string, delay: number) =>
    names.map((name, i) => {
      const p = settle(f, delay + i * 6, fps, 1.1);
      const x = cx + side * 330;
      const y = cy - 195 + i * 130 + Math.sin((f + i * 23) / 28) * 5;
      const pulse = ((f * (side > 0 ? 1 : 1.2) + i * 13) % 46) / 46;
      const from = side < 0 ? {x: x + 12, y} : {x: cx + 70 * side, y: cy};
      const to = side < 0 ? {x: cx - 70, y: cy} : {x: x - 12, y};
      return {
        wire: (
          <g key={`w${name}`} opacity={p}>
            <path
              d={`M ${x} ${y} C ${(x + cx) / 2} ${y}, ${(x + cx) / 2} ${cy}, ${cx + side * 72} ${cy}`}
              fill="none"
              stroke={rgba(color, 0.35)}
              strokeWidth={1.6}
            />
            <circle
              cx={from.x + (to.x - from.x) * pulse}
              cy={from.y + (to.y - from.y) * pulse + Math.sin(pulse * Math.PI) * (y - cy) * (side < 0 ? -0.0 : 0)}
              r={4}
              fill={color}
              opacity={Math.sin(pulse * Math.PI) * 0.9}
            />
          </g>
        ),
        chip: (
          <div
            key={`c${name}`}
            style={{
              position: 'absolute',
              left: x,
              top: y,
              transform: `translate(${side < 0 ? '-100%' : '0'}, -50%) translateX(${(1 - p) * side * 60}px)`,
              opacity: p,
            }}
          >
            <Chip color={color} style={{fontSize: 26, padding: '12px 26px'}}>
              {name}
            </Chip>
          </div>
        ),
      };
    });
  const local = column(LOCAL, -1, '#2dd4bf', 20);
  const cloud = column(CLOUD, 1, '#a78bfa', 32);
  const heads = settle(f, 16, fps);
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['Local or cloud.', 'Same agent.']}
        highlight={1}
        accent={c.accent}
        sub="Native tool calls for frontier models, grammar-constrained calls for local ones. Keys come from your environment."
      />
      <svg style={{position: 'absolute', inset: 0}} width={1920} height={1080}>
        {local.map((o) => o.wire)}
        {cloud.map((o) => o.wire)}
      </svg>
      <div style={{position: 'absolute', left: cx, top: cy, transform: 'translate(-50%, -50%)'}}>
        <LogoMark f={f * 1.6 + 8} size={170} />
      </div>
      <div style={{position: 'absolute', left: cx - 330, top: cy - 300, transform: 'translateX(-100%)', fontSize: 20, letterSpacing: '0.26em', color: '#5eead4', opacity: heads}}>
        LOCAL
      </div>
      <div style={{position: 'absolute', left: cx + 330, top: cy - 300, fontSize: 20, letterSpacing: '0.26em', color: '#c4b5fd', opacity: heads}}>
        CLOUD
      </div>
      {local.map((o) => o.chip)}
      {cloud.map((o) => o.chip)}
      <div style={{position: 'absolute', left: 110, top: 640, display: 'flex', flexWrap: 'wrap', gap: 14, width: 640}}>
        {SPECS.map((spec, i) => (
          <div key={spec} style={{...reveal(settle(f, 80 + i * 7, fps), 14, 6)}}>
            <Chip mono color="#7dd3fc">
              {spec}
            </Chip>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', left: 110, top: 850, fontSize: 24, color: '#7c7f89', ...reveal(settle(f, 130, fps), 10, 6)}}>
        aliases: fable · opus · sonnet · haiku · gpt · gemini · flash · local
      </div>
    </AbsoluteFill>
  );
};

// -- 02 · watch it think ------------------------------------------------------------------------------

const W_TUI = 1320;

export const Tui: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('tui');
  const enter = settle(f, 0, fps, 1.3);
  const swap = ramp(f, 142, 164, 0, 1, EASE_IN_OUT);
  const zoom = ramp(f, 96, 126, 0, 1, EASE_IN_OUT) - ramp(f, 138, 166, 0, 1, EASE_IN_OUT);
  const t = (key: string) => anchor('hero-thinking', key, W_TUI);
  const d = (key: string) => anchor('hero-done', key, W_TUI);
  const [mx, my] = t('meter');
  // camera: scale about the panel's corner, moving the token box to the screen centre
  const left = 300;
  const top = 238;
  const s = 1 + 1.0 * zoom;
  const px = mx + 130;
  const py = my + 40;
  const tx = (960 - left - s * px) * zoom;
  const ty = (560 - top - s * py) * zoom;
  const titleOn = 1 - zoom;
  return (
    <AbsoluteFill>
      <div style={{opacity: titleOn}}>
        <ChapterTag n={c.n} title={c.title} accent={c.accent} />
        <div
          style={{
            position: 'absolute',
            left: 110,
            top: 124,
            fontSize: 58,
            fontWeight: 750,
            letterSpacing: '-0.04em',
            color: '#f5f5f7',
            ...reveal(settle(f, 8, fps), 18, 10),
          }}
        >
          Watch it <span style={gradientText(c.accent, '#f5f5f7', 90)}>think.</span>
        </div>
        <div style={{position: 'absolute', right: 110, top: 142, fontSize: 26, color: '#a1a4ad', ...reveal(settle(f, 20, fps), 12, 8)}}>
          Reasoning streams live · tokens and tok/s in the sidebar
        </div>
      </div>
      <div style={{position: 'absolute', left, top, transformOrigin: '0 0', transform: `translate(${tx}px, ${ty}px) scale(${s})`}}>
        <div style={{perspective: 2400}}>
          <div
            style={{
              transform: `rotateX(${(1 - enter) * 24}deg) translateY(${(1 - enter) * 140}px) scale(${0.9 + 0.1 * enter})`,
              opacity: enter,
            }}
          >
            <div style={{position: 'relative', width: W_TUI}}>
              <Panel name="hero-thinking" width={W_TUI} sweep={ramp(f, 20, 64)} style={{opacity: 1 - swap}} />
              <Panel name="hero-done" width={W_TUI} style={{position: 'absolute', left: 0, top: 0, opacity: swap}} />
              <div style={{position: 'absolute', inset: 0}}>
                <Highlight x={t('thinking')[0] - 10} y={t('thinking')[1] - 12} w={880} h={62} at={30} out={92} color={c.accent} />
                <Callout x={t('thinking')[0] + 300} y={t('thinking')[1] + 54} dx={40} dy={130} at={36} out={92} label="Thinking, as it streams" color={c.accent} />
                <Highlight x={mx - 30} y={my - 12} w={322} h={84} at={54} out={92} color="#f472b6" />
                <Callout x={mx + 150} y={my - 12} dx={-30} dy={-72} at={60} out={92} label="Tokens · tok/s · turn · time" color="#f472b6" />
                <Callout x={d('todo')[0] - 10} y={d('todo')[1]} dx={-80} dy={-40} at={180} label="Its todo list" color="#2dd4bf" />
                <Callout x={d('changed')[0] - 10} y={d('changed')[1]} dx={-80} dy={40} at={196} label="Files it changed" color="#fbbf24" />
                <Callout x={d('tests')[0] + d('tests')[2] + 10} y={d('tests')[1]} dx={120} dy={-50} at={212} label="It ran the tests" color="#4ade80" />
                <Callout x={d('final')[0] + 560} y={d('final')[1]} dx={70} dy={70} at={228} label="The answer, rendered" color="#a78bfa" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// -- 03 · switch mid-session -------------------------------------------------------------------------------

const W_SW = 1060;

export const Switch: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('switch');
  const enter = settle(f, 4, fps, 1.2);
  const filtered = ramp(f, 70, 82);
  const swap = ramp(f, 104, 124, 0, 1, EASE_IN_OUT);
  const p = (key: string) => anchor('picker', key, W_SW);
  const s = (key: string) => anchor('switched-done', key, W_SW);
  const typed = '/model opus'.slice(0, Math.max(0, Math.floor((f - 88) / 1.6)));
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['Switch models', 'mid-thought.']}
        highlight={1}
        accent={c.accent}
        size={76}
        sub="ctrl+o, or /model opus. History, files already read and background jobs carry over."
      />
      <div style={{position: 'absolute', left: 110, top: 620, display: 'flex', flexDirection: 'column', gap: 16}}>
        {['ctrl+o   pick from what is ready', '/model opus', '-m gemini/gemini-2.5-pro', '-m ollama/qwen3@http://gpu:11434/v1'].map((line, i) => (
          <div key={line} style={{...reveal(settle(f, 40 + i * 8, fps), 12, 6)}}>
            <Chip mono color={c.accent} style={{fontSize: 21}}>
              {line}
            </Chip>
          </div>
        ))}
      </div>
      <div
        style={{
          position: 'absolute',
          left: 790,
          top: 250,
          perspective: 2200,
          opacity: enter,
        }}
      >
        <div style={{transform: `rotateY(${(1 - enter) * -18}deg) translateX(${(1 - enter) * 120}px)`}}>
          <div style={{position: 'relative', width: W_SW}}>
            <Panel name="picker" width={W_SW} sweep={ramp(f, 14, 50)} glow="rgba(14,116,144,0.45)" style={{opacity: (1 - filtered) * (1 - swap)}} />
            <Panel name="picker-filtered" width={W_SW} style={{position: 'absolute', left: 0, top: 0, opacity: filtered * (1 - swap)}} />
            <Panel name="switched-done" width={W_SW} style={{position: 'absolute', left: 0, top: 0, opacity: swap}} />
            <div style={{position: 'absolute', inset: 0}}>
              <Callout x={p('opus')[0] + p('opus')[2] + 10} y={p('opus')[1]} dx={560} dy={-30} at={26} out={66} label="● ready: key found" color="#4ade80" />
              <Callout x={p('needs')[0] + 40} y={p('needs')[1] + 10} dx={-60} dy={150} at={40} out={66} label="○ needs a key" color="#fbbf24" />
              {f > 84 && f < 118 ? (
                <div
                  style={{
                    position: 'absolute',
                    left: 40,
                    bottom: 70,
                    padding: '14px 24px',
                    borderRadius: 12,
                    background: 'rgba(10,10,18,0.92)',
                    border: `1.5px solid ${rgba(c.accent, 0.6)}`,
                    fontFamily: fonts.mono,
                    fontSize: 28,
                    color: '#e5e7eb',
                    boxShadow: `0 0 40px ${rgba(c.accent, 0.3)}`,
                  }}
                >
                  › {typed}
                  <span style={{opacity: f % 16 < 8 ? 1 : 0}}>▍</span>
                </div>
              ) : null}
              <Highlight x={s('status')[0] - 4} y={s('status')[1] - 12} w={s('status')[2] + 300} h={24} at={128} color={c.accent} />
              <Callout x={s('continues')[0] + s('continues')[2]} y={s('continues')[1]} dx={40} dy={-150} at={140} label="New model, same session" color={c.accent} />
              <Callout x={s('edited')[0] + s('edited')[2] + 6} y={s('edited')[1]} dx={130} dy={40} at={160} label="Edits at once: no re-reading" color="#4ade80" />
            </div>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

