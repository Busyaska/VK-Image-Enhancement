// This worker deliberately mirrors image_processing.py: RGB/BGR ordering has no
// effect before the HSV step because brightness and contrast are channel-wise.
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
// numpy.round uses ties-to-even; Math.round would differ for an exact *.5 value.
function roundToEven(value) {
  const floor = Math.floor(value), fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
  let h = 0;
  if (delta) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  if (h < 0) h += 360;
  return [h, max === 0 ? 0 : delta / max, max];
}

function hsvToRgb(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g] = [c, x];
  else if (h < 120) [r, g] = [x, c];
  else if (h < 180) [g, b] = [c, x];
  else if (h < 240) [g, b] = [x, c];
  else if (h < 300) [r, b] = [x, c];
  else [r, b] = [c, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

self.onmessage = ({ data }) => {
  const { id, pixels, brightness, contrast, saturation } = data;
  const rgba = new Uint8ClampedArray(pixels);
  for (let i = 0; i < rgba.length; i += 4) {
    const r = clamp((rgba[i] * brightness - 128) * contrast + 128, 0, 255);
    const g = clamp((rgba[i + 1] * brightness - 128) * contrast + 128, 0, 255);
    const b = clamp((rgba[i + 2] * brightness - 128) * contrast + 128, 0, 255);
    const [h, s, v] = rgbToHsv(r, g, b);
    const [nr, ng, nb] = hsvToRgb(h, clamp(s * saturation, 0, 1), v);
    rgba[i] = clamp(roundToEven(nr), 0, 255);
    rgba[i + 1] = clamp(roundToEven(ng), 0, 255);
    rgba[i + 2] = clamp(roundToEven(nb), 0, 255);
  }
  self.postMessage({ id, pixels: rgba.buffer }, [rgba.buffer]);
};
