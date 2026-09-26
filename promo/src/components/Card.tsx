import React, {type CSSProperties} from 'react';

/** Glass panel: faint fill, hairline border, soft shadow. */
export const Card: React.FC<{style?: CSSProperties; children: React.ReactNode}> = ({style, children}) => (
  <div
    style={{
      position: 'relative',
      borderRadius: 26,
      background: 'linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018))',
      border: '1px solid rgba(255,255,255,0.09)',
      boxShadow: '0 40px 100px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.07)',
      overflow: 'hidden',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Small uppercase label. */
export const Label: React.FC<{children: React.ReactNode; color?: string; style?: CSSProperties}> = ({
  children,
  color = '#6b7079',
  style,
}) => (
  <div
    style={{
      fontSize: 16,
      fontWeight: 600,
      letterSpacing: '0.16em',
      textTransform: 'uppercase',
      color,
      ...style,
    }}
  >
    {children}
  </div>
);
