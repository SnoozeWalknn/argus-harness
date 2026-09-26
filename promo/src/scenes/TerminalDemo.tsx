import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import data from '../data.json';
import {ramp, reveal, settle, typed} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';
import {colors, fonts} from '../theme';

type Line = {text: string; kind: string};
type Row = Line & {t: number; mode: 'cmd' | 'stream' | 'instant'; cps: number};

const LH = 34;
const VISIBLE = 22;
const SHOW_LINES = 26;

/** Schedule every terminal row: commands are typed, model text streams, the rest appears. */
const schedule = (): Row[] => {
  const rows: Row[] = [];
  let t = 14;
  const cmd = (text: string) => {
    rows.push({text, kind: 'cmd', t, mode: 'cmd', cps: 1.5});
    t += text.length / 1.5 + 8;
  };
  cmd(data.terminal.run.command);
  let streamedAnswer = false;
  for (const line of data.terminal.run.output as Line[]) {
    const streams = line.kind === 'reasoning' || (line.kind === 'answer' && !streamedAnswer);
    if (line.kind === 'answer') streamedAnswer = true;
    if (streams) {
      const dur = Math.min(16, Math.max(8, line.text.length / 3.5));
      rows.push({...line, t, mode: 'stream', cps: line.text.length / dur});
      t += dur + 3;
    } else {
      if (line.kind === 'ok') t += 5; // the tool runs
      rows.push({...line, t, mode: 'instant', cps: 0});
      t += 4;
    }
  }
  t += 10;
  cmd(data.terminal.show.command);
  for (const line of (data.terminal.show.output as Line[]).slice(0, SHOW_LINES)) {
    rows.push({...line, t, mode: 'instant', cps: 0});
    t += 1.6;
  }
  return rows;
};

const ROWS = schedule();

const Colored: React.FC<{line: Line; text: string}> = ({line, text}) => {
  const c = {
    dim: '#6b7280',
    body: '#c9d1d9',
    bright: '#f0f6fc',
  };
  switch (line.kind) {
    case 'tool': {
      const m = text.match(/^(\s*)(\w+)(.*)$/s);
      if (!m) return <span style={{color: c.body}}>{text}</span>;
      return (
        <>
          {m[1]}
          <span style={{color: colors.sky}}>{m[2]}</span>
          <span style={{color: c.body}}>{m[3]}</span>
        </>
      );
    }
    case 'ok': {
      const m = text.match(/^(\s*)(✓)(.*?)(\s*\(\d+ms\))?$/s);
      if (!m) return <span style={{color: c.body}}>{text}</span>;
      return (
        <>
          {m[1]}
          <span style={{color: colors.green}}>{m[2]}</span>
          <span style={{color: c.body}}>{m[3]}</span>
          <span style={{color: c.dim}}>{m[4] ?? ''}</span>
        </>
      );
    }
    case 'status':
    case 'title': {
      const m = text.match(/^(.*?)(completed)(.*)$/s);
      if (!m) return <span style={{color: c.bright, fontWeight: 700}}>{text}</span>;
      return (
        <>
          <span style={{color: line.kind === 'title' ? c.bright : c.dim, fontWeight: line.kind === 'title' ? 700 : 400}}>{m[1]}</span>
          <span style={{color: colors.green, fontWeight: 700}}>{m[2]}</span>
          <span style={{color: c.dim}}>{m[3]}</span>
        </>
      );
    }
    case 'field': {
      const i = text.indexOf(':');
      return (
        <>
          <span style={{color: c.dim}}>{text.slice(0, i + 1)}</span>
          <span style={{color: c.body}}>{text.slice(i + 1)}</span>
        </>
      );
    }
    case 'heading':
      return <span style={{color: '#b4a2fb'}}>{text}</span>;
    case 'answer':
      return <span style={{color: c.bright}}>{text}</span>;
    case 'reasoning':
      return <span style={{color: '#7d8590'}}>{text}</span>;
    default:
      return <span style={{color: c.dim}}>{text}</span>;
  }
};

export const TerminalDemo: React.FC = () => {
  const f = useSceneFrame();
  const {fps} = useVideoConfig();
  const win = settle(f, 0, fps, 1.1);
  const revealed = ROWS.reduce((n, r) => n + ramp(f, r.t, r.t + 5), 0);
  const scroll = Math.max(0, revealed - VISIBLE) * LH;
  const lastStarted = ROWS.reduce((idx, r, i) => (f >= r.t ? i : idx), -1);
  const blink = Math.floor(f / 9) % 2 === 0;
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div
        style={{
          width: 1560,
          height: 850,
          marginTop: -40,
          borderRadius: 20,
          background: 'rgba(10,11,15,0.94)',
          border: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 50px 140px rgba(0,0,0,0.7), 0 0 0 1px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)',
          overflow: 'hidden',
          opacity: win,
          transform: `translateY(${(1 - win) * 40}px) scale(${0.965 + 0.035 * win})`,
        }}
      >
        <div
          style={{
            height: 46,
            display: 'flex',
            alignItems: 'center',
            padding: '0 20px',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
            background: 'rgba(255,255,255,0.025)',
            position: 'relative',
          }}
        >
          {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
            <div key={c} style={{width: 13, height: 13, borderRadius: 13, background: c, opacity: 0.75, marginRight: 9}} />
          ))}
          <div style={{position: 'absolute', left: 0, right: 0, textAlign: 'center', fontSize: 16, color: '#6b7079'}}>
            /tmp/demo/calc — argus
          </div>
        </div>
        <div
          style={{
            position: 'relative',
            height: 850 - 46,
            overflow: 'hidden',
            padding: '26px 38px',
            WebkitMaskImage: 'linear-gradient(180deg, transparent 0px, black 34px, black calc(100% - 24px), transparent 100%)',
            maskImage: 'linear-gradient(180deg, transparent 0px, black 34px, black calc(100% - 24px), transparent 100%)',
          }}
        >
          <div style={{transform: `translateY(${-scroll}px)`}}>
            {ROWS.map((r, i) => {
              if (f < r.t) return <div key={i} style={{height: LH}} />;
              const text = r.mode === 'instant' ? r.text : typed(r.text, f, r.t, r.cps);
              const typing = r.mode === 'cmd' ? text.length < r.text.length : false;
              const showCursor = r.mode === 'cmd' && (i === lastStarted ? true : typing);
              return (
                <div
                  key={i}
                  style={{
                    height: LH,
                    lineHeight: `${LH}px`,
                    fontFamily: fonts.mono,
                    fontSize: 21,
                    whiteSpace: 'pre',
                    tabSize: 4,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    opacity: r.mode === 'instant' ? ramp(f, r.t, r.t + 3) : 1,
                  }}
                >
                  {r.mode === 'cmd' ? (
                    <>
                      <span style={{color: '#5eead4'}}>❯ </span>
                      <span style={{color: '#f0f6fc'}}>{text}</span>
                    </>
                  ) : (
                    <Colored line={r} text={text} />
                  )}
                  {showCursor && (
                    <span style={{display: 'inline-block', width: 11, height: 24, marginLeft: 2, verticalAlign: -4, background: blink || typing ? '#e5e7eb' : 'transparent'}} />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div style={{position: 'absolute', bottom: 62, fontSize: 19, color: colors.faint, ...reveal(settle(f, 20, fps), 8, 4)}}>
        Real argus output, recorded against the argus mock server
      </div>
    </AbsoluteFill>
  );
};
