import { useEffect, useRef, useState } from 'react';
// The /wasm entry point excludes optional WebGPU/WebNN JSEP modules.
import * as ort from 'onnxruntime-web/wasm';
import heic2any from 'heic2any';

const MAX_PIXELS = 15_000_000;
const worker = new Worker(new URL('./image.worker.js', import.meta.url), { type: 'module' });
const makeId = () => crypto.randomUUID();

function formatTime(ms) { return ms == null ? '—' : `${(ms / 1000).toFixed(2)} с`; }
function downloadableName(name) { return `${name.replace(/\.[^.]+$/, '') || 'enhanced'}-enhanced.png`; }

async function readableFile(file) {
  if (!/\.(heic|heif)$/i.test(file.name)) return file;
  const converted = await heic2any({ blob: file, toType: 'image/png', quality: 0.95 });
  return Array.isArray(converted) ? converted[0] : converted;
}

async function decode(file) {
  const source = await readableFile(file);
  const bitmap = await createImageBitmap(source);
  if (bitmap.width * bitmap.height > MAX_PIXELS) {
    bitmap.close();
    throw new Error('Максимальный размер изображения — 15 Мпк.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  canvas.getContext('2d', { willReadFrequently: true }).drawImage(bitmap, 0, 0);
  bitmap.close();
  return { width: canvas.width, height: canvas.height, imageData: canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height) };
}

async function predict(session, imageData, width, height) {
  const sample = document.createElement('canvas'); sample.width = sample.height = 256;
  const raw = document.createElement('canvas'); raw.width = width; raw.height = height;
  raw.getContext('2d').putImageData(imageData, 0, 0);
  sample.getContext('2d').drawImage(raw, 0, 0, 256, 256);
  const pixels = sample.getContext('2d').getImageData(0, 0, 256, 256).data;
  const input = new Float32Array(3 * 256 * 256);
  // Training loader keeps OpenCV's BGR order (load_image(..., to_rgb=false)).
  for (let i = 0; i < 256 * 256; i++) {
    input[i] = pixels[i * 4 + 2] / 255;
    input[65536 + i] = pixels[i * 4 + 1] / 255;
    input[131072 + i] = pixels[i * 4] / 255;
  }
  const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, 256, 256]) });
  const values = result[session.outputNames[0]].data;
  return { brightness: values[0], contrast: values[1], saturation: values[2] };
}

function App() {
  const [tasks, setTasks] = useState([]), [selectedId, setSelectedId] = useState(null), [ready, setReady] = useState(false);
  const sessionRef = useRef(null), taskRef = useRef(new Map());
  const selected = tasks.find((task) => task.id === selectedId);
  const update = (id, changes) => setTasks((all) => all.map((task) => task.id === id ? { ...task, ...changes } : task));

  useEffect(() => {
    // GitHub Pages cannot send COOP/COEP headers required for SharedArrayBuffer.
    // A single WASM thread is reliable in every supported browser; image pixels
    // themselves are still processed outside the UI thread by image.worker.js.
    ort.env.wasm.numThreads = 1;
    // Served by the Vite plugin as application/wasm in dev and production.
    ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;
    ort.InferenceSession.create(`${import.meta.env.BASE_URL}best_model.onnx`, { executionProviders: ['wasm'] })
      .then((session) => { sessionRef.current = session; setReady(true); })
      .catch((error) => setTasks([{ id: 'model-error', name: 'Модель', status: 'Ошибка', progress: 0, error: `Не удалось загрузить ONNX: ${error.message}` }]));
    worker.onmessage = ({ data }) => {
      const task = taskRef.current.get(data.id); if (!task || task.cancelled) return;
      const canvas = document.createElement('canvas'); canvas.width = task.width; canvas.height = task.height;
      canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(data.pixels), task.width, task.height), 0, 0);
      canvas.toBlob((blob) => {
        if (!blob || task.cancelled) return;
        const resultUrl = URL.createObjectURL(blob), elapsed = performance.now() - task.started;
        update(data.id, { status: 'Готово', progress: 100, resultUrl, elapsed }); taskRef.current.delete(data.id);
      }, 'image/png');
    };
    return () => { worker.terminate(); tasks.forEach((task) => task.resultUrl && URL.revokeObjectURL(task.resultUrl)); };
  }, []);

  async function enqueue(file) {
    const id = makeId(), sourceUrl = URL.createObjectURL(file), started = performance.now();
    const base = { id, name: file.name, sourceUrl, status: 'В очереди', progress: 0, elapsed: null, started };
    setTasks((all) => [...all, base]); setSelectedId(id);
    try {
      update(id, { status: 'Декодирование', progress: 10 }); const decoded = await decode(file);
      update(id, { status: 'Инференс модели', progress: 35 });
      const params = await predict(sessionRef.current, decoded.imageData, decoded.width, decoded.height);
      update(id, { status: 'Коррекция', progress: 65, params });
      const task = { id, width: decoded.width, height: decoded.height, started, cancelled: false };
      taskRef.current.set(id, task);
      worker.postMessage({ id, pixels: decoded.imageData.data.buffer, ...params }, [decoded.imageData.data.buffer]);
    } catch (error) { update(id, { status: 'Ошибка', progress: 0, error: error.message }); }
  }
  function cancel(id) { const task = taskRef.current.get(id); if (task) task.cancelled = true; update(id, { status: 'Отменено' }); }
  return <main>
    <header><div><h1>VK Image Enhancement</h1><p>Улучшение изображения с помощью локальной ML-модели</p></div><label className={`upload ${ready ? '' : 'disabled'}`}><input type="file" accept="image/jpeg,image/png,image/bmp,image/heic,image/heif" disabled={!ready} onChange={(e) => [...e.target.files].forEach(enqueue)} />Загрузить изображение</label></header>
    <div className="layout"><aside><h2>Задачи</h2>{tasks.length === 0 && <p className="muted">Загруженные изображения появятся здесь.</p>}{tasks.map((task) => <button className={`task ${task.id === selectedId ? 'active' : ''}`} key={task.id} onClick={() => setSelectedId(task.id)}><span>{task.name}</span><small>{task.status} · {task.progress}%</small></button>)}</aside>
    <section className="workspace">{selected ? <><div className="details"><div><b>{selected.name}</b><span className={`badge ${selected.status}`}>{selected.status}</span>{selected.error && <p className="error">{selected.error}</p>}<progress value={selected.progress} max="100" /></div><dl><div><dt>Время</dt><dd>{formatTime(selected.elapsed)}</dd></div>{selected.params && <><div><dt>Brightness</dt><dd>{selected.params.brightness.toFixed(4)}</dd></div><div><dt>Contrast</dt><dd>{selected.params.contrast.toFixed(4)}</dd></div><div><dt>Saturation</dt><dd>{selected.params.saturation.toFixed(4)}</dd></div></>}</dl><div className="actions">{selected.status !== 'Готово' && selected.status !== 'Ошибка' && selected.status !== 'Отменено' && <button onClick={() => cancel(selected.id)}>Прервать</button>}{selected.resultUrl && <a href={selected.resultUrl} download={downloadableName(selected.name)}>Скачать PNG</a>}</div></div><div className="images"><figure><figcaption>Исходное</figcaption><img src={selected.sourceUrl} alt="Исходное изображение" /></figure><figure><figcaption>Обработанное</figcaption>{selected.resultUrl ? <img src={selected.resultUrl} alt="Обработанное изображение" /> : <div className="placeholder">{selected.status}</div>}</figure></div></> : <div className="empty">Загрузите изображение, чтобы создать задачу.</div>}</section></div>
  </main>;
}
export default App;
