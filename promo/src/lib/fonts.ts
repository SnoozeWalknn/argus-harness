import {cancelRender, continueRender, delayRender, staticFile} from 'remotion';

const FACES: [string, string, FontFaceDescriptors][] = [
  ['Inter', 'fonts/InterVariable.woff2', {weight: '100 900'}],
  ['JetBrains Mono', 'fonts/JetBrainsMono-Regular.woff2', {weight: '400'}],
  ['JetBrains Mono', 'fonts/JetBrainsMono-Medium.woff2', {weight: '500'}],
  ['JetBrains Mono', 'fonts/JetBrainsMono-Bold.woff2', {weight: '700'}],
];

let started = false;

/** Load the fonts before the first frame is captured (Remotion waits on delayRender). */
export const loadFonts = (): void => {
  if (started) return;
  started = true;
  const handle = delayRender('Loading fonts');
  Promise.all(
    FACES.map(async ([family, file, descriptors]) => {
      const face = new FontFace(family, `url('${staticFile(file)}') format('woff2')`, descriptors);
      await face.load();
      document.fonts.add(face);
    }),
  )
    .then(() => continueRender(handle))
    .catch((err) => cancelRender(err));
};
