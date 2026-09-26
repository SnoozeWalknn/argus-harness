import React, {createContext, useContext} from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame} from 'remotion';
import {EASE_IN_OUT, ramp} from './anim';

/** Frames of cross-fade on each side of a section boundary. */
const HALF = 9;

const SceneTime = createContext({lead: 0, length: 150});

/** Frame relative to the scene's nominal start (negative during the incoming cross-fade). */
export const useSceneFrame = (): number => useCurrentFrame() - useContext(SceneTime).lead;
export const useSceneLength = (): number => useContext(SceneTime).length;

const Shell: React.FC<{duration: number; first: boolean; last: boolean; children: React.ReactNode}> = ({
  duration,
  first,
  last,
  children,
}) => {
  const f = useCurrentFrame();
  const fade = HALF * 2;
  const enter = first ? 1 : ramp(f, 0, fade, 0, 1, EASE_IN_OUT);
  const exit = last ? 1 : ramp(f, duration - fade, duration, 1, 0, EASE_IN_OUT);
  const drift = 1 + 0.018 * (f / duration); // slow continuous push-in, like a camera move
  const scale = drift * (first ? 1 : 1.02 - 0.02 * enter) * (last ? 1 : 0.985 + 0.015 * exit);
  const blur = (1 - enter) * 8 + (1 - exit) * 8;
  return (
    <AbsoluteFill
      style={{
        opacity: enter * exit,
        transform: `scale(${scale})`,
        filter: blur > 0.05 ? `blur(${blur}px)` : undefined,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

/** A section of the timeline [from, to) that cross-fades with its neighbours. */
export const Scene: React.FC<{
  from: number;
  to: number;
  name: string;
  first?: boolean;
  last?: boolean;
  children: React.ReactNode;
}> = ({from, to, name, first = false, last = false, children}) => {
  const start = first ? from : from - HALF;
  const end = last ? to : to + HALF;
  return (
    <Sequence from={start} durationInFrames={end - start} name={name}>
      <SceneTime.Provider value={{lead: from - start, length: to - from}}>
        <Shell duration={end - start} first={first} last={last}>
          {children}
        </Shell>
      </SceneTime.Provider>
    </Sequence>
  );
};
