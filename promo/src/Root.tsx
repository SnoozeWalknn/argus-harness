import React from 'react';
import {Composition} from 'remotion';
import {loadFonts} from './lib/fonts';
import {Promo} from './Promo';
import {DURATION, FPS, HEIGHT, WIDTH} from './theme';

loadFonts();

export const RemotionRoot: React.FC = () => (
  <Composition id="ArgusPromo" component={Promo} durationInFrames={DURATION} fps={FPS} width={WIDTH} height={HEIGHT} />
);
