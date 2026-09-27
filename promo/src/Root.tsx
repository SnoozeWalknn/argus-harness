import React from 'react';
import {Composition} from 'remotion';
import {Launch} from './launch/Launch';
import {LAUNCH_DURATION} from './launch/timeline';
import {loadFonts} from './lib/fonts';
import {Promo} from './Promo';
import {DURATION, FPS, HEIGHT, WIDTH} from './theme';

loadFonts();

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="ArgusPromo" component={Promo} durationInFrames={DURATION} fps={FPS} width={WIDTH} height={HEIGHT} />
    <Composition id="ArgusLaunch" component={Launch} durationInFrames={LAUNCH_DURATION} fps={FPS} width={WIDTH} height={HEIGHT} />
  </>
);
