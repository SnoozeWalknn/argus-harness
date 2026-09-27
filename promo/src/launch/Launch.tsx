import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Finish} from '../components/Background';
import {ramp} from '../lib/anim';
import {Scene} from '../lib/scene';
import {colors, fonts} from '../theme';
import {Code, Safety, Tools, Workflow} from './scenes/B';
import {Close, Install, Measured, Themes, Tuned} from './scenes/C';
import {Models, Open, Switch, Tui} from './scenes/A';
import {CH} from './timeline';
import {Backdrop} from './ui/backdrop';
import {Rail} from './ui/overlay';

/** Where the UI on screen comes from, shown while captures are on screen. */
const Provenance: React.FC = () => {
  const f = useCurrentFrame();
  const on = ramp(f, CH.tui[0], CH.tui[0] + 20) * (1 - ramp(f, CH.themes[1] - 20, CH.themes[1]));
  return (
    <div
      style={{
        position: 'absolute',
        right: 110,
        bottom: 34,
        fontSize: 17,
        letterSpacing: '0.04em',
        color: '#6b7079',
        opacity: on,
      }}
    >
      Real argus UI · scripted runs against argus's mock server
    </div>
  );
};

export const Launch: React.FC = () => (
  <AbsoluteFill
    style={{
      background: colors.bg,
      fontFamily: fonts.sans,
      fontOpticalSizing: 'auto',
      fontVariantLigatures: 'none',
      fontFeatureSettings: '"liga" 0, "calt" 0',
      WebkitFontSmoothing: 'antialiased',
    }}
  >
    <Backdrop />
    <Scene name="Open" from={CH.open[0]} to={CH.open[1]} first>
      <Open />
    </Scene>
    <Scene name="01 Any model" from={CH.models[0]} to={CH.models[1]}>
      <Models />
    </Scene>
    <Scene name="02 Watch it think" from={CH.tui[0]} to={CH.tui[1]}>
      <Tui />
    </Scene>
    <Scene name="03 Switch" from={CH.switch[0]} to={CH.switch[1]}>
      <Switch />
    </Scene>
    <Scene name="04 Tools" from={CH.tools[0]} to={CH.tools[1]}>
      <Tools />
    </Scene>
    <Scene name="05 Safety" from={CH.safety[0]} to={CH.safety[1]}>
      <Safety />
    </Scene>
    <Scene name="06 Workflow" from={CH.workflow[0]} to={CH.workflow[1]}>
      <Workflow />
    </Scene>
    <Scene name="07 Code" from={CH.code[0]} to={CH.code[1]}>
      <Code />
    </Scene>
    <Scene name="08 Tuned" from={CH.tuned[0]} to={CH.tuned[1]}>
      <Tuned />
    </Scene>
    <Scene name="09 Measured" from={CH.measured[0]} to={CH.measured[1]}>
      <Measured />
    </Scene>
    <Scene name="10 Themes" from={CH.themes[0]} to={CH.themes[1]}>
      <Themes />
    </Scene>
    <Scene name="11 Install" from={CH.install[0]} to={CH.install[1]}>
      <Install />
    </Scene>
    <Scene name="Close" from={CH.close[0]} to={CH.close[1]} last>
      <Close />
    </Scene>
    <Rail />
    <Provenance />
    <Finish />
  </AbsoluteFill>
);
