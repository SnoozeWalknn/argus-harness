import React, {type CSSProperties} from 'react';
import {useVideoConfig} from 'remotion';
import {settle} from '../lib/anim';
import {useSceneFrame} from '../lib/scene';

export type Token = string | {text: string; style?: CSSProperties; glow?: string};

/**
 * Word-by-word kinetic headline: each word rises, un-blurs and fades in on a critically damped
 * spring. "\n" starts a new line.
 */
export const Kinetic: React.FC<{
  tokens: Token[];
  start: number;
  stagger?: number;
  rise?: number;
  blur?: number;
  style?: CSSProperties;
}> = ({tokens, start, stagger = 4, rise = 0.34, blur = 14, style}) => {
  const frame = useSceneFrame();
  const {fps} = useVideoConfig();
  let index = 0;
  return (
    <div style={style}>
      {tokens.map((token, k) => {
        if (token === '\n') return <br key={k} />;
        const t = typeof token === 'string' ? {text: token} : token;
        const p = settle(frame, start + index * stagger, fps);
        index += 1;
        const filters = [p < 0.999 ? `blur(${(1 - p) * blur}px)` : '', t.glow ? `drop-shadow(${t.glow})` : '']
          .filter(Boolean)
          .join(' ');
        return (
          <React.Fragment key={k}>
            <span
              style={{
                display: 'inline-block',
                opacity: p,
                transform: `translateY(${(1 - p) * rise}em)`,
                filter: filters || undefined,
                ...t.style,
              }}
            >
              {t.text}
            </span>{' '}
          </React.Fragment>
        );
      })}
    </div>
  );
};
