// Copy the six onnxruntime-web 1.31.1 runtime files the pages load from node_modules into web/vendor,
// then check each against the SHA-256 listed in web/vendor/README.md. Run after `npm ci`: npm run setup
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const vendor = join(root, 'web', 'vendor');
const expected = Object.fromEntries(
  [...readFileSync(join(vendor, 'README.md'), 'utf8').matchAll(/^\| (ort[\w.-]+) \| ([0-9a-f]{64}) \|$/gm)].map((m) => [m[1], m[2]]),
);
if (Object.keys(expected).length !== 6) throw new Error('web/vendor/README.md should list six runtime files with their SHA-256.');
if (!existsSync(dist)) {
  console.error('node_modules/onnxruntime-web is missing. Run npm ci first, then npm run setup.');
  process.exit(1);
}
for (const [name, sha] of Object.entries(expected)) {
  copyFileSync(join(dist, name), join(vendor, name));
  const actual = createHash('sha256').update(readFileSync(join(vendor, name))).digest('hex');
  if (actual !== sha) throw new Error(`${name}: SHA-256 ${actual} does not match ${sha} in web/vendor/README.md`);
}
console.log('Copied and checked the six onnxruntime-web 1.31.1 files into web/vendor/. Next: python -m http.server 8000 --bind 127.0.0.1 --directory web');
