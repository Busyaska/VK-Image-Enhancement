import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { createReadStream, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// onnxruntime-web does not expose its .wasm payload through package exports.
// This plugin makes exactly the required runtime file available in dev and build.
function onnxRuntimeWasm() {
  const files = [
    { name: 'ort-wasm-simd-threaded.mjs', type: 'text/javascript' },
    { name: 'ort-wasm-simd-threaded.wasm', type: 'application/wasm' },
  ];
  const runtimeDir = resolve('node_modules/onnxruntime-web/dist');
  return {
    name: 'serve-onnxruntime-wasm',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const requestPath = request.url?.split('?')[0];
        const file = files.find(({ name }) => requestPath === `${server.config.base}${name}`);
        if (!file) return next();
        response.setHeader('Content-Type', file.type);
        createReadStream(resolve(runtimeDir, file.name)).pipe(response);
      });
    },
    generateBundle() {
      for (const file of files) {
        this.emitFile({ type: 'asset', fileName: file.name, source: readFileSync(resolve(runtimeDir, file.name)) });
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), onnxRuntimeWasm()],
  base: process.env.GITHUB_ACTIONS ? '/VK-Image-Enhancement/' : '/',
  worker: { format: 'es' },
});
