import React, {type CSSProperties, useEffect, useState} from 'react';
import {cancelRender, continueRender, delayRender, staticFile} from 'remotion';
import data from '../data.json';

/** Real TUI screenshots (Textual SVG exports), inlined so they use the film's fonts. */
const cache = new Map<string, string>();
const pending = new Map<string, Promise<string>>();

const load = (name: string): Promise<string> => {
  let p = pending.get(name);
  if (!p) {
    p = fetch(staticFile(`launch/${name}.svg`))
      .then((r) => r.text())
      .then((t) => {
        const svg = t.replace(
          '<svg class="rich-terminal"',
          '<svg class="rich-terminal" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"',
        );
        cache.set(name, svg);
        return svg;
      });
    pending.set(name, p);
  }
  return p;
};

export const useSvg = (name: string): string | null => {
  const [svg, setSvg] = useState<string | null>(cache.get(name) ?? null);
  const [handle] = useState(() => (cache.has(name) ? null : delayRender(`TUI capture ${name}`)));
  useEffect(() => {
    if (svg !== null) return;
    load(name)
      .then((t) => {
        setSvg(t);
        if (handle !== null) continueRender(handle);
      })
      .catch((err) => cancelRender(err));
  }, [name, handle, svg]);
  return svg;
};

type Anchor = [number, number, number];
const anchors = data.anchors as unknown as Record<string, Record<string, number[]>>;

export const SIZE: [number, number] = [1726, 1026];

/** Where a named string sits in a capture, in panel pixels at `width`: [x, y, w]. */
export const anchor = (name: string, key: string, width: number): Anchor => {
  const a = anchors[name]?.[key];
  if (!a) throw new Error(`no anchor ${name}:${key}`);
  const k = width / SIZE[0];
  return [a[0] * k, a[1] * k, a[2] * k];
};

/**
 * A captured TUI state in a glass frame. `sweep` (0..1) runs a specular highlight across;
 * children are laid over it in panel pixels (use `anchor`).
 */
export const Panel: React.FC<{
  name: string;
  width: number;
  style?: CSSProperties;
  sweep?: number;
  glow?: string;
  children?: React.ReactNode;
}> = ({name, width, style, sweep = -1, glow = 'rgba(124,58,237,0.35)', children}) => {
  const svg = useSvg(name);
  const height = (width * SIZE[1]) / SIZE[0];
  const s = sweep * 160 - 30;
  return (
    <div style={{position: 'relative', width, height, ...style}}>
      <div
        style={{
          position: 'absolute',
          inset: '6% 4% -4% 4%',
          background: `radial-gradient(closest-side, ${glow}, transparent)`,
          filter: 'blur(40px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: 18,
          overflow: 'hidden',
          boxShadow: '0 50px 120px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.10)',
        }}
      >
        {svg ? <div style={{width: '100%', height: '100%'}} dangerouslySetInnerHTML={{__html: svg}} /> : null}
        {sweep > 0 && sweep < 1 ? (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: `linear-gradient(105deg, transparent ${s - 14}%, rgba(255,255,255,0.10) ${s}%, transparent ${s + 14}%)`,
              mixBlendMode: 'screen',
            }}
          />
        ) : null}
      </div>
      <div style={{position: 'absolute', inset: 0}}>{children}</div>
    </div>
  );
};
