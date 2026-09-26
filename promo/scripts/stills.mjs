// Render individual frames for review: node scripts/stills.mjs OUT_DIR FRAME [FRAME ...]
// Honours REMOTION_BROWSER_EXECUTABLE like remotion.config.ts.
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import {mkdirSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const [outDir, ...frames] = process.argv.slice(2);
if (!outDir || frames.length === 0) {
  console.error('usage: node scripts/stills.mjs OUT_DIR FRAME [FRAME ...]');
  process.exit(2);
}
mkdirSync(outDir, {recursive: true});
const serveUrl = await bundle({entryPoint: join(root, 'src/index.ts')});
const browserExecutable = process.env.REMOTION_BROWSER_EXECUTABLE ?? null;
const composition = await selectComposition({serveUrl, id: 'ArgusPromo', browserExecutable});
for (const f of frames.map(Number)) {
  const output = resolve(outDir, `frame-${String(f).padStart(4, '0')}.png`);
  await renderStill({composition, serveUrl, frame: f, output, browserExecutable});
  console.log(output);
}
