import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {LogoMark} from '../../components/Logo';
import {EASE_IN_OUT, fmt, gradientText, ramp, reveal, rgba, settle} from '../../lib/anim';
import {useSceneFrame} from '../../lib/scene';
import {fonts} from '../../theme';
import data from '../data.json';
import {CHAPTERS} from '../timeline';
import {ChapterTag, Chip, Title} from '../ui/overlay';
import {Panel} from '../ui/svg';

const chapter = (key: string) => CHAPTERS.find((c) => c.key === key)!;

const Card: React.FC<{style?: React.CSSProperties; children: React.ReactNode}> = ({style, children}) => (
  <div
    style={{
      position: 'absolute',
      borderRadius: 24,
      padding: '30px 34px',
      boxSizing: 'border-box',
      background: 'linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))',
      border: '1px solid rgba(255,255,255,0.10)',
      boxShadow: '0 40px 100px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08)',
      ...style,
    }}
  >
    {children}
  </div>
);

// -- 08 · tuned per model -------------------------------------------------------------------------------

type Row = {name: string; pass: number; n: number; lo: number; hi: number};
const ROWS: Row[] = data.tune.table.slice(1).map((line: string) => {
  const m = line.match(/^(\w+)\s+(\d+)\/(\d+)\s+\d+%\s+\[(\d+)-(\d+)%\]/)!;
  return {name: m[1], pass: +m[2], n: +m[3], lo: +m[4], hi: +m[5]};
});
const BEST = data.tune.best.replace(/^best: /, '').replace(/ \(.*/, '');

export const Tuned: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('tuned');
  const card = settle(f, 16, fps, 1.1);
  const barW = 560;
  const native = data.overhead.native;
  const grammar = data.overhead.grammar;
  const cut = Math.round((1 - grammar / native) * 100);
  const shrink = ramp(f, 110, 150, 0, 1, EASE_IN_OUT);
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['argus tune', 'finds what works.']}
        highlight={1}
        accent={c.accent}
        size={76}
        sub="Every protocol, on a small suite, ranked by the lower bound of its pass rate. The winner goes into the model's profile."
      />
      <Card style={{left: 1010, top: 240, width: 800, height: 400, opacity: card, transform: `translateY(${(1 - card) * 40}px)`}}>
        <div style={{fontFamily: fonts.mono, fontSize: 22, color: '#93c5fd'}}>$ argus tune llama/qwen3-coder-30b</div>
        <div style={{marginTop: 26, display: 'flex', flexDirection: 'column', gap: 24}}>
          {ROWS.map((r, i) => {
            const p = ramp(f, 30 + i * 12, 70 + i * 12, 0, 1, EASE_IN_OUT);
            const best = r.name === BEST;
            return (
              <div key={r.name} style={{display: 'flex', alignItems: 'center', gap: 20}}>
                <div style={{width: 180, fontFamily: fonts.mono, fontSize: 24, fontWeight: 700, color: best ? '#bfdbfe' : '#d4d4d8'}}>{r.name}</div>
                <div style={{position: 'relative', width: barW - 240, height: 30, borderRadius: 8, background: 'rgba(255,255,255,0.06)'}}>
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      height: 30,
                      borderRadius: 8,
                      width: `${(r.pass / r.n) * 100 * p}%`,
                      background: best ? 'linear-gradient(90deg, #3b82f6, #a78bfa)' : 'linear-gradient(90deg, #3b82f6aa, #60a5fa88)',
                      boxShadow: best ? '0 0 24px rgba(96,165,250,0.5)' : undefined,
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      top: 14,
                      height: 2,
                      left: `${r.lo}%`,
                      width: `${(r.hi - r.lo) * p}%`,
                      background: '#f5f5f7',
                      opacity: 0.7,
                      borderLeft: '2px solid #f5f5f7',
                      borderRight: '2px solid #f5f5f7',
                    }}
                  />
                </div>
                <div style={{fontFamily: fonts.mono, fontSize: 20, color: '#a1a4ad', whiteSpace: 'nowrap'}}>
                  {r.pass}/{r.n} · [{r.lo}–{r.hi}%]
                </div>
              </div>
            );
          })}
        </div>
        <div style={{marginTop: 30, ...reveal(settle(f, 80, fps), 10, 6)}}>
          <Chip color={c.accent} mono>
            → best: {BEST}, saved to its profile
          </Chip>
        </div>
      </Card>
      <Card style={{left: 1010, top: 670, width: 800, height: 250, ...reveal(settle(f, 96, fps), 30, 8)}}>
        <div style={{fontSize: 22, letterSpacing: '0.16em', textTransform: 'uppercase', color: '#8b8f98'}}>Prompt overhead</div>
        {[
          ['native tool schemas', native, '#94a3b8'],
          ['grammar-constrained', grammar, c.accent],
        ].map(([label, n, color], i) => {
          const w = i === 0 ? 1 : 1 - (1 - (n as number) / native) * shrink;
          return (
            <div key={label as string} style={{display: 'flex', alignItems: 'center', gap: 18, marginTop: 20}}>
              <div style={{width: 260, fontSize: 22, color: '#d4d4d8'}}>{label}</div>
              <div style={{height: 22, width: 330 * w, borderRadius: 6, background: color as string}} />
              <div style={{fontFamily: fonts.mono, fontSize: 22, color: '#e5e7eb'}}>{fmt(i === 0 ? native : native - (native - grammar) * shrink)}</div>
            </div>
          );
        })}
        <div style={{position: 'absolute', right: 34, top: 28, fontSize: 44, fontWeight: 800, ...gradientText('#93c5fd', '#c4b5fd'), opacity: shrink}}>
          −{cut}%
        </div>
      </Card>
      <Card style={{left: 110, top: 600, width: 860, height: 260, padding: '26px 30px', ...reveal(settle(f, 60, fps), 30, 8)}}>
        <div style={{fontFamily: fonts.mono, fontSize: 18, lineHeight: 1.8, color: '#a1a4ad', whiteSpace: 'pre'}}>
          {[...data.tune.ranking, data.tune.best].map((line: string, i: number) => (
            <div key={i} style={{color: line.startsWith('→') || line.startsWith('best') ? '#bfdbfe' : i === 0 ? '#6b7079' : '#c9cad1', opacity: ramp(f, 66 + i * 6, 76 + i * 6)}}>
              {line}
            </div>
          ))}
        </div>
      </Card>
      <div style={{position: 'absolute', left: 1010, top: 944, fontSize: 18, color: '#6b7079'}}>
        Measured with argus's mock server (scripted: native calls break); real counts differ.
      </div>
    </AbsoluteFill>
  );
};

// -- 09 · every run, measured --------------------------------------------------------------------------

const TAGS = ['loop', 'overrun', 'malformed_call', 'token_cap', 'refusal', 'fs_violation', 'timeout', 'blocked'];

export const Measured: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('measured');
  const n = Math.round(data.tests * ramp(f, 90, 130));
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <Title
        lines={['Every run,', 'measured.']}
        highlight={1}
        accent={c.accent}
        size={84}
        sub="Every turn goes into SQLite: prompt, reasoning, tool calls, timings, tokens, cost. Failures get a name."
      />
      <Card style={{left: 1000, top: 240, width: 810, height: 300, ...reveal(settle(f, 14, fps), 30, 8)}}>
        <div style={{fontSize: 22, letterSpacing: '0.16em', textTransform: 'uppercase', color: '#8b8f98'}}>Failure tags</div>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: 14, marginTop: 22}}>
          {TAGS.map((t, i) => (
            <div key={t} style={{...reveal(settle(f, 24 + i * 5, fps), 10, 6)}}>
              <Chip mono color={c.accent} style={{fontSize: 22}}>
                {t}
              </Chip>
            </div>
          ))}
        </div>
      </Card>
      <Card style={{left: 110, top: 600, width: 820, height: 240, padding: '26px 30px', ...reveal(settle(f, 40, fps), 30, 8)}}>
        <div style={{fontFamily: fonts.mono, fontSize: 21, lineHeight: 1.7, color: '#d4d4d8', whiteSpace: 'pre'}}>
          <span style={{color: '#fdba74'}}>SELECT</span> r.config_name, f.tag, <span style={{color: '#fdba74'}}>COUNT</span>(*){'\n'}
          <span style={{color: '#fdba74'}}>FROM</span> failures f <span style={{color: '#fdba74'}}>JOIN</span> runs r <span style={{color: '#fdba74'}}>ON</span> r.id = f.run_id{'\n'}
          <span style={{color: '#fdba74'}}>GROUP BY</span> 1, 2;{'\n'}
          <span style={{color: '#6b7079'}}>-- which failures does each config produce?</span>
        </div>
      </Card>
      <Card style={{left: 1000, top: 570, width: 390, height: 270, ...reveal(settle(f, 50, fps), 30, 8)}}>
        <div style={{fontSize: 22, letterSpacing: '0.16em', textTransform: 'uppercase', color: '#8b8f98'}}>A/B</div>
        <div style={{fontSize: 34, fontWeight: 700, color: '#f5f5f7', marginTop: 18, lineHeight: 1.2}}>Suites, Wilson intervals, McNemar</div>
        <div style={{fontFamily: fonts.mono, fontSize: 20, color: '#fdba74', marginTop: 16}}>argus ab a.toml b.toml</div>
      </Card>
      <Card style={{left: 1420, top: 570, width: 390, height: 270, ...reveal(settle(f, 64, fps), 30, 8)}}>
        <div style={{fontSize: 22, letterSpacing: '0.16em', textTransform: 'uppercase', color: '#8b8f98'}}>Tests</div>
        <div style={{fontSize: 96, fontWeight: 800, letterSpacing: '-0.04em', marginTop: 4, ...gradientText('#fdba74', '#c4b5fd')}}>{n}</div>
        <div style={{fontSize: 22, color: '#a1a4ad'}}>no model, no API key needed</div>
      </Card>
    </AbsoluteFill>
  );
};

// -- 10 · make it yours --------------------------------------------------------------------------------------

const THEMES = ['matte-black', 'kanagawa', 'everforest', 'osaka-jade', 'gruvbox', 'rose-pine', 'catppuccin-mocha', 'nord'];

export const Themes: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('themes');
  // step through the themes: hold on each, then a quick eased move to the next
  const step = 170 / (THEMES.length - 1);
  const k = Math.min(Math.max((f - 20) / step, 0), THEMES.length - 1);
  const base = Math.floor(k);
  const into = Math.min(Math.max((k - base - 0.45) / 0.55, 0), 1);
  const pos = base + EASE_IN_OUT(into);
  const W = 880;
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <div style={{position: 'absolute', left: 110, top: 124, fontSize: 64, fontWeight: 750, letterSpacing: '-0.04em', color: '#f5f5f7', ...reveal(settle(f, 6, fps), 18, 10)}}>
        Make it <span style={gradientText(c.accent, '#f5f5f7')}>yours.</span>
      </div>
      <div style={{position: 'absolute', right: 110, top: 146, fontSize: 26, color: '#a1a4ad', ...reveal(settle(f, 16, fps), 12, 8)}}>
        Follows your Omarchy theme · ctrl+t cycles
      </div>
      <div style={{position: 'absolute', left: 0, top: 300, width: 1920, height: 640, perspective: 2000}}>
        {THEMES.map((name, i) => {
          const d = i - pos;
          if (Math.abs(d) > 2.2) return null;
          const x = 960 + d * 660;
          const scale = 1 - Math.min(Math.abs(d), 2) * 0.2;
          const rot = Math.max(-1, Math.min(1, d)) * -42;
          const op = 1 - Math.min(Math.abs(d), 2.2) * 0.32;
          return (
            <div
              key={name}
              style={{
                position: 'absolute',
                left: x - W / 2,
                top: 0,
                zIndex: 100 - Math.round(Math.abs(d) * 10),
                transform: `rotateY(${rot}deg) scale(${scale})`,
                opacity: op,
                filter: Math.abs(d) > 0.5 ? `brightness(${1 - Math.min(Math.abs(d), 2) * 0.25})` : undefined,
              }}
            >
              <Panel name={`theme-${name}`} width={W} glow={rgba(c.accent, 0.25)} />
              <div
                style={{
                  marginTop: 26,
                  textAlign: 'center',
                  fontFamily: fonts.mono,
                  fontSize: 26,
                  color: '#e5e7eb',
                  opacity: Math.max(0, 1 - Math.abs(d) * 1.4),
                }}
              >
                {name}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// -- 11 · one command ----------------------------------------------------------------------------------------

const INSTALL = 'curl -fsSL https://raw.githubusercontent.com/SnoozeWalknn/argus-harness/main/install.sh | sh';
const pick = (prefixes: string[], lines: string[]) =>
  lines.filter((l) => prefixes.some((p) => l.startsWith(p)));
const OUTPUT = [
  ...pick(['==> installing', '==> setting up', '    wrote', '==> checking'], data.install),
  ...pick(['sandbox:', 'tui:', 'api keys:', 'model server:'], data.doctor),
  ...pick(['lsp python'], data.install),
  'done. Run argus in a project directory, or argus run "your task".',
];

export const Install: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const c = chapter('install');
  const card = settle(f, 10, fps, 1.1);
  const typed = INSTALL.slice(0, Math.max(0, Math.floor((f - 24) * 3.2)));
  const shown = Math.max(0, Math.floor((f - 62) / 7));
  return (
    <AbsoluteFill>
      <ChapterTag n={c.n} title={c.title} accent={c.accent} />
      <div style={{position: 'absolute', left: 110, top: 124, fontSize: 64, fontWeight: 750, letterSpacing: '-0.04em', color: '#f5f5f7', ...reveal(settle(f, 6, fps), 18, 10)}}>
        One command. <span style={gradientText(c.accent, '#7dd3fc')}>No sudo.</span>
      </div>
      <Card style={{left: 110, top: 250, width: 1700, height: 620, padding: 0, overflow: 'hidden', opacity: card, transform: `translateY(${(1 - card) * 40}px)`}}>
        <div style={{height: 50, display: 'flex', alignItems: 'center', gap: 10, paddingLeft: 22, borderBottom: '1px solid rgba(255,255,255,0.08)'}}>
          {['#ff5f57', '#febc2e', '#28c840'].map((col) => (
            <div key={col} style={{width: 14, height: 14, borderRadius: 7, background: col}} />
          ))}
          <div style={{marginLeft: 20, fontSize: 18, color: '#8b8f98'}}>~</div>
        </div>
        <div style={{padding: '28px 36px', fontFamily: fonts.mono, fontSize: 25, lineHeight: 1.62, color: '#d4d4d8'}}>
          <div>
            <span style={{color: '#86efac'}}>$ </span>
            {typed}
            {f < 62 ? <span style={{opacity: f % 16 < 8 ? 1 : 0}}>▍</span> : null}
          </div>
          {OUTPUT.slice(0, shown).map((line, i) => (
            <div
              key={i}
              style={{
                whiteSpace: 'pre',
                color: line.startsWith('==>') ? '#c4b5fd' : line.includes('✓') ? '#86efac' : line.startsWith('done') ? '#f5f5f7' : '#a1a4ad',
                fontWeight: line.startsWith('==>') || line.startsWith('done') ? 700 : 400,
              }}
            >
              {line}
            </div>
          ))}
        </div>
      </Card>
    </AbsoluteFill>
  );
};

// -- close ---------------------------------------------------------------------------------------------------

export const Close: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const word = settle(f, 40, fps);
  const slide = settle(f, 24, fps, 1.0);
  const sweep = ramp(f, 80, 124) * 140 - 20;
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 54, marginTop: -60}}>
        <div style={{transform: `translateX(${(1 - slide) * 250}px)`}}>
          <LogoMark f={f} size={230} />
        </div>
        <div
          style={{
            fontSize: 176,
            fontWeight: 800,
            letterSpacing: '-0.055em',
            lineHeight: 1,
            paddingBottom: '0.06em',
            backgroundImage: `linear-gradient(100deg, #f4f4f6 0%, #f4f4f6 ${sweep - 12}%, #ddd6fe ${sweep}%, #f4f4f6 ${sweep + 12}%, #d4d4d8 100%)`,
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            color: 'transparent',
            opacity: word,
            transform: `translateX(${(1 - word) * -24}px)`,
            filter: word < 0.999 ? `blur(${(1 - word) * 14}px)` : 'drop-shadow(0 0 40px rgba(139,92,246,0.35))',
          }}
        >
          argus
        </div>
      </div>
      <div style={{marginTop: 56, fontSize: 56, fontWeight: 650, letterSpacing: '-0.03em', color: '#ececf1', ...reveal(settle(f, 58, fps), 18, 10)}}>
        Every model. <span style={gradientText('#ddd6fe', '#7dd3fc')}>One agent.</span>
      </div>
      <div style={{marginTop: 22, fontSize: 26, color: '#9a9ca6', ...reveal(settle(f, 72, fps), 14, 8)}}>
        Open source · runs on your machine · every turn on the record
      </div>
      <div style={{marginTop: 40, ...reveal(settle(f, 88, fps), 14, 8)}}>
        <Chip mono color="#a78bfa" style={{fontSize: 24}}>
          github.com/SnoozeWalknn/argus-harness
        </Chip>
      </div>
    </AbsoluteFill>
  );
};
