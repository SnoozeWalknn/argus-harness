import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Background, Finish} from './components/Background';
import {Scene} from './lib/scene';
import {Close} from './scenes/Close';
import {FeatureAB} from './scenes/FeatureAB';
import {FeatureCheckpoints} from './scenes/FeatureCheckpoints';
import {FeatureConstrained} from './scenes/FeatureConstrained';
import {FeatureEdits} from './scenes/FeatureEdits';
import {FeatureFailures} from './scenes/FeatureFailures';
import {FeatureSSH} from './scenes/FeatureSSH';
import {Hook} from './scenes/Hook';
import {ProblemContext} from './scenes/ProblemContext';
import {ProblemSilent} from './scenes/ProblemSilent';
import {TerminalDemo} from './scenes/TerminalDemo';
import {colors, fonts, TIMELINE as T} from './theme';

export const Promo: React.FC = () => (
  <AbsoluteFill style={{
      background: colors.bg,
      fontFamily: fonts.sans,
      fontOpticalSizing: 'auto',
      fontVariantLigatures: 'none',
      fontFeatureSettings: '"liga" 0, "calt" 0',
      WebkitFontSmoothing: 'antialiased',
    }}>
    <Background />
    <Scene name="Hook" from={T.hook[0]} to={T.hook[1]} first>
      <Hook />
    </Scene>
    <Scene name="Problem: context" from={T.context[0]} to={T.context[1]}>
      <ProblemContext />
    </Scene>
    <Scene name="Problem: silent" from={T.silent[0]} to={T.silent[1]}>
      <ProblemSilent />
    </Scene>
    <Scene name="01 Constrained" from={T.constrained[0]} to={T.constrained[1]}>
      <FeatureConstrained />
    </Scene>
    <Scene name="02 Failures" from={T.failures[0]} to={T.failures[1]}>
      <FeatureFailures />
    </Scene>
    <Scene name="03 Edits" from={T.edits[0]} to={T.edits[1]}>
      <FeatureEdits />
    </Scene>
    <Scene name="04 Checkpoints" from={T.checkpoints[0]} to={T.checkpoints[1]}>
      <FeatureCheckpoints />
    </Scene>
    <Scene name="05 SSH" from={T.ssh[0]} to={T.ssh[1]}>
      <FeatureSSH />
    </Scene>
    <Scene name="06 A/B" from={T.ab[0]} to={T.ab[1]}>
      <FeatureAB />
    </Scene>
    <Scene name="Terminal" from={T.terminal[0]} to={T.terminal[1]}>
      <TerminalDemo />
    </Scene>
    <Scene name="Close" from={T.close[0]} to={T.close[1]} last>
      <Close />
    </Scene>
    <Finish />
  </AbsoluteFill>
);
