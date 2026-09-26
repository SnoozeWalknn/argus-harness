import {Config} from '@remotion/cli/config';

Config.setEntryPoint('src/index.ts');
Config.setVideoImageFormat('png'); // lossless frames: no JPEG artifacts in the dark gradients
Config.setCodec('h264');
Config.setCrf(18);
Config.setX264Preset('slow');
Config.setPixelFormat('yuv420p');
Config.setColorSpace('bt709');

// Remotion downloads its own headless Chrome. Where that download is blocked (sandboxes,
// locked-down CI), point it at an installed Chromium or chrome-headless-shell instead.
if (process.env.REMOTION_BROWSER_EXECUTABLE) {
  Config.setBrowserExecutable(process.env.REMOTION_BROWSER_EXECUTABLE);
}
