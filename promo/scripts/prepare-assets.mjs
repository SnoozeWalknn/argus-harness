// Copies the fonts the video uses out of node_modules and generates a film-grain tile.
// Runs on `npm install` (postinstall). Idempotent.
import {copyFileSync, existsSync, mkdirSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {deflateSync} from 'node:zlib';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const fonts = join(root, 'public', 'fonts');
mkdirSync(fonts, {recursive: true});

const copies = {
  'inter-ui/variable/InterVariable.woff2': 'InterVariable.woff2',
  'jetbrains-mono/fonts/webfonts/JetBrainsMono-Regular.woff2': 'JetBrainsMono-Regular.woff2',
  'jetbrains-mono/fonts/webfonts/JetBrainsMono-Medium.woff2': 'JetBrainsMono-Medium.woff2',
  'jetbrains-mono/fonts/webfonts/JetBrainsMono-Bold.woff2': 'JetBrainsMono-Bold.woff2',
};
for (const [from, to] of Object.entries(copies)) {
  copyFileSync(join(root, 'node_modules', from), join(fonts, to));
}

// 256x256 grey-alpha noise tile (PNG colour type 4). Overlaid at low opacity it dithers the
// dark gradients so H.264 does not band them.
const grain = join(root, 'public', 'grain.png');
if (!existsSync(grain)) {
  const size = 256;
  let seed = 0x2f6b1d3;
  const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 16) & 0xff;
  const raw = Buffer.alloc(size * (1 + size * 2));
  for (let y = 0; y < size; y++) {
    const row = y * (1 + size * 2);
    raw[row] = 0; // filter: none
    for (let x = 0; x < size; x++) {
      raw[row + 1 + x * 2] = rand() > 127 ? 255 : 0; // light or dark speck
      raw[row + 2 + x * 2] = rand(); // random strength
    }
  }
  const crcTable = Array.from({length: 256}, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc32 = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 4; // grey + alpha
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, {level: 9})),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  writeFileSync(grain, png);
}
console.log('promo assets ready');
