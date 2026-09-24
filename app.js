import { CameraCycle } from './camera-cycle.mjs?v=4';

const canvas = document.getElementById('stage');
const statusEl = document.getElementById('status');
const fail = msg => {
  statusEl.hidden = false; statusEl.textContent = msg;
  document.getElementById('start').hidden = true;
};

/* ---- a port of Kim Asendorf's ASDFPixelSort to WebGPU, fed by the camera.
   ASDF walks every column, then every row: it finds the first pixel that
   passes a threshold, runs on until one fails, sorts that span, and repeats.
   Here each line is one workgroup: a prefix scan numbers the spans, then one
   bitonic sort on (span, value, index) sorts every span in place at once. ---- */

// sort keys and threshold metrics share one list, indexed in the shader
const METRICS = ['luma', 'hue', 'saturation', 'asdf', 'max'];
const DIRECTIONS = {
  'columns → rows': [1, 0],   // ASDF's order
  'rows → columns': [0, 1],
  rows: [0],
  columns: [1],
};
// the longest line a workgroup can sort; indices and span ids pack into 11 bits
const MAX_LINE = 2048;

/* ---- starting values ---- */
const DEFAULTS = {
  direction: 'columns → rows', key: 'asdf', reverse: false,
  metric: 'max', lo: 0.235, hi: 1, invert: false,
  maxSpan: 0, breakChance: 0, jitter: false,
};

/* ---- presets: the guide's three ASDF modes, plus a couple of looks of our own.
   ASDF compares raw ARGB ints, so its black and white thresholds are really
   red-dominated cutoffs: -16000000 and -13000000 land at ~0.046 and ~0.225 ---- */
const PRESETS = {
  black:      { ...DEFAULTS, metric: 'asdf', lo: 0.046 },
  brightness: { ...DEFAULTS, metric: 'max', lo: 0.235 },
  white:      { ...DEFAULTS, metric: 'asdf', lo: 0.225 },
  melt: {
    ...DEFAULTS, direction: 'columns', key: 'luma', metric: 'luma',
    lo: 0.3, hi: 0.85, reverse: true,
  },
  shards: {
    ...DEFAULTS, direction: 'rows', key: 'hue', metric: 'luma',
    lo: 0.12, hi: 0.95, maxSpan: 140, breakChance: 0.004,
  },
};

/* ---- random look: every look-shaping setting, kept inside ranges that sort something ---- */
const pick = list => list[Math.floor(Math.random() * list.length)];
const chance = p => Math.random() < p;
const round = (v, step) => Math.round(v / step) * step;
const randomLook = () => {
  // a band at least 0.15 wide, so the mask never collapses to nothing
  const width = 0.15 + Math.random() * 0.85;
  const lo = round(Math.random() * (1 - width), 0.005);
  return {
    direction: pick(Object.keys(DIRECTIONS)),
    key: pick(METRICS),
    reverse: chance(0.5),
    metric: pick(METRICS),
    lo,
    hi: round(Math.min(lo + width, 1), 0.005),
    invert: chance(0.2),
    maxSpan: chance(0.5) ? 0 : Math.round(20 + Math.random() * 380),
    breakChance: chance(0.5) ? 0 : round(Math.random() * 0.01, 0.0005),
    jitter: chance(0.15),
  };
};

/* ---- live parameters shared by the controls and render loop ---- */
const params = {
  ...PRESETS.melt, resolution: 0.4, smooth: true, mirror: true, showMask: false, frozen: false,
  amount: 0.85, split: -1, grain: 0.035,
};
const actions = { save: false, useCamera: null };

/* ---- shared uniform block (64 bytes) ---- */
const PARAMS = /* wgsl */ `
  struct P {
    size: vec2f,       // work texture px
    lo: f32,
    hi: f32,
    invert: f32,
    key: f32,          // index into METRICS
    reverse: f32,
    maxSpan: f32,      // 0: unlimited
    breakChance: f32,  // per pixel
    seed: f32,
    showMask: f32,
    metric: f32,       // index into METRICS, for the threshold
    amount: f32,
    split: f32,
    padding: f32,
    grain: f32,
  };

  const INK = vec3f(244.0, 241.0, 234.0) / 255.0;

  fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

  fn hue(c: vec3f) -> f32 {
    let mx = max(c.r, max(c.g, c.b));
    let d = mx - min(c.r, min(c.g, c.b));
    if (d < 1e-5) { return 0.0; }
    var h: f32;
    if (mx == c.r) { h = (c.g - c.b) / d; }
    else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
    else { h = (c.r - c.g) / d + 4.0; }
    return fract(h / 6.0 + 1.0);
  }

  fn saturation(c: vec3f) -> f32 {
    let mx = max(c.r, max(c.g, c.b));
    return select((mx - min(c.r, min(c.g, c.b))) / mx, 0.0, mx < 1e-5);
  }

  // what Processing's sort() on ARGB ints orders by: red, then green, then blue
  fn asdf(c: vec3f) -> f32 {
    let q = floor(c * 255.0 + 0.5);
    return (q.r * 65536.0 + q.g * 256.0 + q.b) / 16777215.0;
  }

  fn metric(c: vec3f, which: u32) -> f32 {
    switch which {
      case 1u: { return hue(c); }
      case 2u: { return saturation(c); }
      case 3u: { return asdf(c); }
      case 4u: { return max(c.r, max(c.g, c.b)); }   // HSB brightness, as ASDF uses
      default: { return luma(c); }
    }
  }

  fn inMask(c: vec3f) -> bool {
    let v = metric(c, u32(p.metric));
    return (v >= p.lo && v <= p.hi) != (p.invert > 0.5);
  }
`;

/* ---- blit: the camera, cover-fit and mirrored, into the work texture ---- */
const BLIT = /* wgsl */ `
  struct B { size: vec2f, scale: vec2f, mirror: f32 };
  @group(0) @binding(0) var<uniform> b: B;
  @group(0) @binding(1) var samp: sampler;
  @group(0) @binding(2) var feed: texture_external;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  }

  @fragment
  fn fs(@builtin(position) fp: vec4f) -> @location(0) vec4f {
    var q = fp.xy / b.size;
    if (b.mirror > 0.5) { q.x = 1.0 - q.x; }
    let uv = (q - 0.5) * b.scale + 0.5;
    return vec4f(textureSampleBaseClampToEdge(feed, samp, uv).rgb, 1.0);
  }
`;

/* ---- sort: one workgroup per line ---- */
const SORT = /* wgsl */ `
  @group(0) @binding(0) var<uniform> p: P;
  @group(0) @binding(1) var src: texture_2d<f32>;
  @group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
  @group(0) @binding(3) var<uniform> pass_: vec4f;   // x: axis (0 rows, 1 columns)
  ${PARAMS}

  const WG = 256u;
  var<workgroup> keys: array<u32, ${MAX_LINE}>;
  var<workgroup> partial: array<u32, WG>;

  fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
  }

  fn coord(line: u32, i: u32) -> vec2i {
    return select(vec2i(i32(i), i32(line)), vec2i(i32(line), i32(i)), pass_.x > 0.5);
  }

  fn texel(line: u32, i: u32) -> vec4f {
    return textureLoad(src, coord(line, i), 0);
  }

  // a new span starts at i when the mask switches on, at every masked-out
  // pixel (so they stay put as spans of one), and at the optional breaks
  fn isBreak(line: u32, i: u32) -> bool {
    if (i == 0u) { return true; }
    if (!inMask(texel(line, i).rgb) || !inMask(texel(line, i - 1u).rgb)) { return true; }
    let seed = hash(u32(p.seed) ^ (u32(pass_.x) * 0x9e3779b9u));
    let ms = u32(p.maxSpan);
    if (ms > 0u && (i + hash(line ^ seed)) % ms == 0u) { return true; }
    if (p.breakChance > 0.0) {
      let r = f32(hash(i + hash(line + seed)) >> 8u) / 16777216.0;
      if (r < p.breakChance) { return true; }
    }
    return false;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
    let line = wg.x;
    let n = u32(select(p.size.x, p.size.y, pass_.x > 0.5));
    let p2 = 1u << (32u - countLeadingZeros(n - 1u));

    // number the spans: each thread counts breaks in its chunk, then a scan
    let chunk = (n + WG - 1u) / WG;
    let start = min(t * chunk, n);
    let end = min(start + chunk, n);
    var count = 0u;
    for (var i = start; i < end; i++) {
      if (isBreak(line, i)) { count++; }
    }
    partial[t] = count;
    workgroupBarrier();
    for (var off = 1u; off < WG; off <<= 1u) {
      var v = partial[t];
      if (t >= off) { v += partial[t - off]; }
      workgroupBarrier();
      partial[t] = v;
      workgroupBarrier();
    }

    // key: span (11 bits) | value (10 bits) | index (11 bits). Sorting the whole
    // line by it sorts each span in place; the index keeps it stable
    var span = partial[t] - count;
    for (var i = start; i < end; i++) {
      if (isBreak(line, i)) { span++; }
      var q = u32(clamp(metric(texel(line, i).rgb, u32(p.key)), 0.0, 1.0) * 1023.0 + 0.5);
      if (p.reverse > 0.5) { q = 1023u - q; }
      keys[i] = ((span - 1u) << 21u) | (q << 11u) | i;
    }
    for (var i = n + t; i < p2; i += WG) { keys[i] = 0xffffffffu; }
    workgroupBarrier();

    // bitonic sort in shared memory; pairs are disjoint, each owned by its lower index
    for (var k = 2u; k <= p2; k <<= 1u) {
      for (var j = k >> 1u; j > 0u; j >>= 1u) {
        for (var i = t; i < p2; i += WG) {
          let l = i ^ j;
          if (l > i) {
            let a = keys[i];
            let b = keys[l];
            if ((a > b) == ((i & k) == 0u)) {
              keys[i] = b;
              keys[l] = a;
            }
          }
        }
        workgroupBarrier();
      }
    }

    for (var i = t; i < n; i += WG) {
      textureStore(dst, coord(line, i), texel(line, keys[i] & 0x7ffu));
    }
  }
`;

/* ---- present: the sorted frame to the screen, or the mask over the source ---- */
const PRESENT = /* wgsl */ `
  @group(0) @binding(0) var<uniform> p: P;
  @group(0) @binding(1) var samp: sampler;
  @group(0) @binding(2) var sorted: texture_2d<f32>;
  @group(0) @binding(3) var source: texture_2d<f32>;
  ${PARAMS}

  struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let q = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VOut(vec4f(q * 2.0 - 1.0, 0.0, 1.0), vec2f(q.x, 1.0 - q.y));
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4f {
    let c = textureSample(sorted, samp, in.uv).rgb;
    let s = textureSample(source, samp, in.uv).rgb;
    if (p.split >= 0.0 && in.uv.x < p.split) { return vec4f(s, 1.0); }
    if (p.showMask < 0.5) {
      let noise = fract(sin(dot(in.pos.xy, vec2f(12.9898, 78.233))) * 43758.5453) - 0.5;
      return vec4f(mix(s, c, p.amount) + noise * p.grain, 1.0);
    }
    // spans that will sort in ink, the rest dimmed
    return vec4f(select(s * 0.25, mix(s, INK, 0.75), inMask(s)), 1.0);
  }
`;

async function main() {
  if (!navigator.gpu) return fail('WebGPU not available in this browser');
  // some Android drivers never answer, so each request reports where it stalled
  const within = (promise, label) => Promise.race([promise, new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`${label} timed out`)), 10000))]);
  statusEl.textContent = 'loading · gpu adapter';
  const adapter = await within(navigator.gpu.requestAdapter(), 'GPU adapter request');
  if (!adapter) return fail('No WebGPU adapter');
  const gpuName = [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.device].filter(Boolean).join(' ') || 'unknown GPU';
  statusEl.textContent = `loading · gpu device (${gpuName})`;
  const device = await within(adapter.requestDevice(), `GPU device request (${gpuName})`);
  device.lost.then(info => fail(`GPU lost: ${info.message} (${gpuName})`));
  device.addEventListener('uncapturederror', e => fail(`GPU error: ${e.error.message} (${gpuName})`));
  statusEl.textContent = `loading · shaders (${gpuName})`;

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  /* Still images and camera frames share the same GPU import path. */
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  let still = null, frozenFrame = null, cameraOk = false, sourceRequest = 0;
  const stopCamera = () => {
    video.srcObject?.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    cameraOk = false;
    frozenFrame?.close(); frozenFrame = null;
  };
  const setStill = (frame, name) => {
    stopCamera();
    still?.close(); still = frame;
    params.frozen = false;
    statusEl.hidden = true;
    $('camera-button').disabled = false;
    $('camera-button').textContent = 'start camera';
    document.body.dataset.source = name;
    document.body.dataset.phase = 'still';
  };
  actions.demo = name => {
    sourceRequest++;
    setStill(new VideoFrame(makeDemo(name), { timestamp: 0 }), `Study / ${name}`);
  };
  actions.openImage = async file => {
    if (!file) return;
    const request = ++sourceRequest;
    try {
      const bitmap = await createImageBitmap(file);
      if (request !== sourceRequest) { bitmap.close(); return; }
      const frame = new VideoFrame(bitmap, { timestamp: 0 });
      bitmap.close();
      setStill(frame, file.name);
      $('start').hidden = true;
      hint();
    } catch { toast('Could not open that image. Try PNG, JPEG or WebP.'); }
  };
  actions.useCamera = async () => {
    const request = ++sourceRequest;
    $('camera-button').disabled = true;
    $('camera-button').textContent = 'connecting…';
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      if (request !== sourceRequest) { stream.getTracks().forEach(t => t.stop()); return; }
      stopCamera();
      video.srcObject = stream;
      await video.play();
      if (request !== sourceRequest) { stream.getTracks().forEach(t => t.stop()); return; }
      cameraOk = true;
      still?.close(); still = null;
      params.frozen = false;
      statusEl.hidden = true;
      cycle.reset(performance.now());
      $('start').hidden = true;
      hint();
      document.body.dataset.source = 'camera';
      for (const track of stream.getVideoTracks()) track.addEventListener('ended', () => {
        if (video.srcObject !== stream) return;
        stopCamera();
        $('start').hidden = false;
        $('camera-button').textContent = 'restart camera';
      });
    } catch (err) {
      stream?.getTracks().forEach(t => t.stop());
      if (request === sourceRequest) {
        $('start').hidden = false;
        $('camera-button').textContent = 'retry camera';
        toast('Camera unavailable. Try again, or drop an image.');
      }
    } finally {
      if (request === sourceRequest) $('camera-button').disabled = false;
    }
  };
  const cycle = new CameraCycle();
  let envelope = 1, lastFrame = performance.now();
  actions.grab = () => { if (cameraOk) cycle.grab(); };
  actions.release = () => cycle.release(performance.now());
  actions.freeze = () => { if (cameraOk) cycle.toggle(performance.now()); };
  const updateCapture = now => {
    const dt = Math.min(100, now - lastFrame); lastFrame = now;
    if (!cameraOk || video.readyState < 2) { envelope = 1; return; }
    const held = cycle.tick(now);
    if (held && !frozenFrame) {
      frozenFrame = new VideoFrame(video);
      // Each automatic hold begins a different sort. A grab keeps your settings.
      if (!cycle.dragging && !cycle.pinned) mutate(false);
    } else if (!held && frozenFrame) {
      frozenFrame.close(); frozenFrame = null;
    }
    params.frozen = held;
    document.body.dataset.phase = held ? 'held' : 'live';
    const target = held ? 1 : .12;
    envelope += (target - envelope) * (1 - Math.exp(-dt / 650));
  };
  addEventListener('pagehide', () => { stopCamera(); still?.close(); });

  /* ---- pipelines ---- */
  const blitModule = device.createShaderModule({ code: BLIT });
  const sortModule = device.createShaderModule({ code: SORT });
  const presentModule = device.createShaderModule({ code: PRESENT });

  const blitPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: blitModule, entryPoint: 'vs' },
    fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  });
  const sortPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: sortModule, entryPoint: 'main' },
  });
  const presentPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: presentModule, entryPoint: 'vs' },
    fragment: { module: presentModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const paramData = new Float32Array(16);
  const paramBuf = device.createBuffer({ size: paramData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const blitData = new Float32Array(8);
  const blitBuf = device.createBuffer({ size: blitData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const axisBufs = [0, 1].map(v => {
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array([v, 0, 0, 0]));
    return buf;
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const nearest = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

  /* ---- layout: size the work textures, wire every pass that could run ---- */
  let W = 0, H = 0, dpr = 1, sw = 0, sh = 0;
  let laidOut = '';
  let textures = {}, sortBinds = {}, presentBinds;

  const layout = () => {
    W = Math.max(1, Math.round(canvas.clientWidth));
    H = Math.max(1, Math.round(canvas.clientHeight));
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);

    // the work texture keeps the screen's aspect; no line may outgrow a workgroup
    const k = Math.min(params.resolution, MAX_LINE / Math.max(canvas.width, canvas.height));
    sw = Math.max(1, Math.round(canvas.width * k));
    sh = Math.max(1, Math.round(canvas.height * k));

    for (const t of Object.values(textures)) t.destroy();
    const make = () => device.createTexture({
      size: [sw, sh],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    textures = { source: make(), tmp: make(), out: make() };

    // one pass reads source and writes out; two go source -> tmp -> out
    sortBinds = {};
    for (const [from, to] of [['source', 'out'], ['source', 'tmp'], ['tmp', 'out']]) {
      for (const axis of [0, 1]) {
        sortBinds[`${from}>${to}:${axis}`] = device.createBindGroup({
          layout: sortPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramBuf } },
            { binding: 1, resource: textures[from].createView() },
            { binding: 2, resource: textures[to].createView() },
            { binding: 3, resource: { buffer: axisBufs[axis] } },
          ],
        });
      }
    }
    presentBinds = [nearest, linear].map(sampler => device.createBindGroup({
      layout: presentPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: textures.out.createView() },
        { binding: 3, resource: textures.source.createView() },
      ],
    }));
    laidOut = layoutKey();
  };
  const layoutKey = () => `${canvas.clientWidth}x${canvas.clientHeight}@${Math.min(devicePixelRatio || 1, 2)}:${params.resolution}`;

  const save = () => canvas.toBlob(blob => {
    const a = document.createElement('a');
    if (!blob) { toast('Export failed. Please try again.'); return; }
    a.href = URL.createObjectURL(blob);
    a.download = `pixel-sort-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast('PNG exported');
  }, 'image/png');

  /* ---- frame ---- */
  let seed = 1;
  const frame = () => {
    try { draw(); } catch (err) { console.error(err); fail(`render error: ${err.message} (${gpuName})`); return; }
    requestAnimationFrame(frame);
  };
  const draw = () => {
    updateCapture(performance.now());
    if (layoutKey() !== laidOut) layout();

    const feedSource = still ?? frozenFrame ?? (cameraOk && video.readyState >= 2 && video.videoWidth > 0 ? video : null);
    const fw = (still ?? frozenFrame)?.displayWidth ?? video.videoWidth;
    const fh = (still ?? frozenFrame)?.displayHeight ?? video.videoHeight;

    if (params.jitter) seed = (seed + 1) % 16777216;
    paramData.set([
      sw, sh,
      params.lo, params.hi,
      params.invert ? 1 : 0,
      METRICS.indexOf(params.key),
      params.reverse ? 1 : 0,
      params.maxSpan,
      params.breakChance,
      seed,
      params.showMask ? 1 : 0,
      METRICS.indexOf(params.metric),
      params.amount * envelope, params.split, 0, params.grain,
    ]);
    device.queue.writeBuffer(paramBuf, 0, paramData);

    const enc = device.createCommandEncoder();

    // A captured VideoFrame keeps a frozen camera frame intact across resizes.
    if (feedSource) {
      const a = sw / sh, v = fw / fh;
      // stills aren't mirrored: they're not a selfie
      blitData.set([sw, sh, a > v ? 1 : a / v, a > v ? v / a : 1, params.mirror && !still ? 1 : 0]);
      device.queue.writeBuffer(blitBuf, 0, blitData);
      const feed = device.importExternalTexture({ source: feedSource });
      const bp = enc.beginRenderPass({
        colorAttachments: [{
          view: textures.source.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      bp.setPipeline(blitPipe);
      // external textures expire every frame, so their bind groups do too
      bp.setBindGroup(0, device.createBindGroup({
        layout: blitPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitBuf } },
          { binding: 1, resource: linear },
          { binding: 2, resource: feed },
        ],
      }));
      bp.draw(3);
      bp.end();
    }

    const axes = DIRECTIONS[params.direction] ?? DIRECTIONS['columns → rows'];
    const cp = enc.beginComputePass();
    cp.setPipeline(sortPipe);
    axes.forEach((axis, k) => {
      const from = k === 0 ? 'source' : 'tmp';
      const to = k === axes.length - 1 ? 'out' : 'tmp';
      cp.setBindGroup(0, sortBinds[`${from}>${to}:${axis}`]);
      // one workgroup per line: rows are as many as the height, columns the width
      cp.dispatchWorkgroups(axis === 0 ? sh : sw);
    });
    cp.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    rp.setPipeline(presentPipe);
    rp.setBindGroup(0, presentBinds[params.smooth ? 1 : 0]);
    rp.draw(3);
    rp.end();

    device.queue.submit([enc.finish()]);
    // the canvas can only be read in the task that drew it
    if (actions.save) { actions.save = false; save(); }
  };

  layout();
  requestAnimationFrame(frame);
  actions.demo('signal');
  mutate();
  $('camera-button').disabled = false;
}



function makeDemo(name) {
  const art = document.createElement('canvas'); art.width = 1600; art.height = 1200;
  const c = art.getContext('2d'); const w = art.width, h = art.height;
  let seed = name === 'signal' ? 1 + Math.floor(Math.random() * 2147483645) : 173;
  const rand = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  if (name === 'signal') {
    const palette = pick([
      ['#eaf5a2', '#e08bce', '#ed512f', '#665de7'],
      ['#fe734e', '#ffc6d9', '#759dec', '#f1ee93'],
      ['#cded8c', '#66cdc0', '#bb70f1', '#fca478'],
    ]);
    c.fillStyle = '#171325'; c.fillRect(0, 0, w, h);
    const phase = rand() * 6;
    // Broad, overlapping ribbons give the sorter colour, edges and empty space.
    for (let layer = 0; layer < 22; layer++) {
      const y = -230 + layer * 76;
      const gradient = c.createLinearGradient(0, y, w, y + 320);
      gradient.addColorStop(0, palette[layer % 4]);
      gradient.addColorStop(.48, palette[(layer + 1) % 4]);
      gradient.addColorStop(1, '#21192e');
      c.fillStyle = gradient;
      c.beginPath();
      for (let x = -100; x <= w + 100; x += 8) {
        const yy = y + Math.sin(x / 310 + phase + layer * .12) * 240 + Math.cos(x / 570 + layer * .19) * 140;
        if (x === -100) c.moveTo(x, yy); else c.lineTo(x, yy);
      }
      for (let x = w + 100; x >= -100; x -= 8) {
        const yy = y + 52 + Math.sin(x / 310 + phase + layer * .12 + .04) * 240 + Math.cos(x / 570 + layer * .19) * 140;
        c.lineTo(x, yy);
      }
      c.closePath(); c.fill();
    }
    const glow = c.createRadialGradient(650, 480, 10, 650, 480, 880);
    glow.addColorStop(0, '#ffffff00'); glow.addColorStop(1, '#10091eb0');
    c.fillStyle = glow; c.fillRect(0, 0, w, h);
  } else if (name === 'dunes') {
    const sky = c.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, '#242537'); sky.addColorStop(.3, '#766073'); sky.addColorStop(.57, '#e2a384'); sky.addColorStop(1, '#e9ad72');
    c.fillStyle = sky; c.fillRect(0, 0, w, h);
    const glow = c.createRadialGradient(1080, 370, 20, 1080, 370, 390);
    glow.addColorStop(0, '#f9c79366'); glow.addColorStop(1, '#ffc59200'); c.fillStyle = glow; c.fillRect(0, 0, w, h);
    c.fillStyle = '#ffe3a9'; c.beginPath(); c.arc(1080, 365, 120, 0, Math.PI * 2); c.fill();
    for (let layer = 0; layer < 7; layer++) {
      const y = 540 + layer * 100;
      const g = c.createLinearGradient(0, y - 140, 100, y + 290);
      g.addColorStop(0, ['#6d5968', '#bc8070', '#683f52', '#d27d52', '#482d42', '#b26444', '#302639'][layer]);
      g.addColorStop(1, ['#4f485b', '#785460', '#462e46', '#713c3d', '#252339', '#6a393a', '#151e2b'][layer]);
      c.fillStyle = g; c.beginPath(); c.moveTo(0, h);
      for (let x = 0; x <= w; x += 4) {
        const wave = Math.sin(x / (290 + layer * 23) + layer * 1.9) * (70 + layer * 10) + Math.cos(x / 620 + layer) * 45;
        c.lineTo(x, y + wave);
      }
      c.lineTo(w, h); c.closePath(); c.fill();
    }
  } else if (name === 'prism') {
    c.fillStyle = '#141625'; c.fillRect(0, 0, w, h);
    c.translate(w / 2, h / 2); c.rotate(-.3);
    for (let i = 0; i < 15; i++) {
      const x = (i - 7) * 120;
      const g = c.createLinearGradient(x, -h / 2, x + 100, h / 2);
      g.addColorStop(0, `hsl(${(i * 21 + 215) % 360} 62% 40%)`);
      g.addColorStop(.42, '#f2cf9e'); g.addColorStop(.5, '#c2e0d0');
      g.addColorStop(.63, `hsl(${(i * 21 + 240) % 360} 45% 38%)`); g.addColorStop(1, '#11172e');
      c.fillStyle = g; c.fillRect(x, -h, 112, h * 2);
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
    const shade = c.createRadialGradient(800, 530, 130, 800, 530, 920);
    shade.addColorStop(0, '#11111100'); shade.addColorStop(1, '#0c0a1acd'); c.fillStyle = shade; c.fillRect(0, 0, w, h);
  } else {
    const sea = c.createLinearGradient(0, 0, w, h); sea.addColorStop(0, '#b2c7af'); sea.addColorStop(.45, '#337c80'); sea.addColorStop(1, '#102b3b'); c.fillStyle = sea; c.fillRect(0, 0, w, h);
    for (let j = 0; j < 95; j++) {
      const y = j * 19 - 260;
      c.beginPath();
      for (let x = -20; x < w + 20; x += 8) {
        const yy = y + x * .3 + Math.sin(x / 160 + j * .14) * 45 + Math.sin(x / 48 + j * .3) * 9;
        if (x === -20) c.moveTo(x, yy); else c.lineTo(x, yy);
      }
      c.strokeStyle = j % 9 < 3 ? `rgba(210,229,197,${.3 + rand() * .4})` : `rgba(10,47,60,${.15 + rand() * .4})`;
      c.lineWidth = 2 + rand() * 15; c.stroke();
    }
  }
  // A little deterministic texture makes the threshold respond to the studies.
  const pixels = c.getImageData(0, 0, w, h);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const n = (rand() - .5) * 7;
    pixels.data[i] += n; pixels.data[i + 1] += n; pixels.data[i + 2] += n;
  }
  c.putImageData(pixels, 0, 0); return art;
}

const $ = id => document.getElementById(id);
let toastTimer, hintTimer;
const history = [];
function toast(text) {
  $('toast').textContent = text; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 2800);
}
function hint() {
  $('hint').hidden = false;
  clearTimeout(hintTimer); hintTimer = setTimeout(() => $('hint').hidden = true, 4500);
}
function remember() {
  history.push({ ...params });
  if (history.length > 60) history.shift();
}
function mutate(saveHistory = true) {
  if (saveHistory) remember();
  const base = pick([PRESETS.melt, PRESETS.shards, { ...DEFAULTS, direction: 'rows', key: 'luma' }]);
  Object.assign(params, base, {
    lo: .08 + Math.random() * .3, hi: .72 + Math.random() * .28,
    reverse: chance(.5), maxSpan: chance(.5) ? 0 : 100 + Math.floor(Math.random() * 450),
    amount: .9, grain: .01,
  });
}
function mountInteraction() {
  $('camera-button').onclick = () => actions.useCamera?.();
  $('file-input').onchange = e => { actions.openImage?.(e.target.files[0]); e.target.value = ''; };
  let gesture = null;
  canvas.addEventListener('pointerdown', e => {
    if (e.button !== 0 || gesture) return;
    remember();
    canvas.focus({ preventScroll: true }); canvas.setPointerCapture(e.pointerId);
    gesture = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false, start: { ...params } };
    $('hint').hidden = true;
  });
  canvas.addEventListener('pointermove', e => {
    if (!gesture || gesture.id !== e.pointerId) return;
    const dx = (e.clientX - gesture.x) / canvas.clientWidth;
    const dy = (e.clientY - gesture.y) / canvas.clientHeight;
    if (Math.hypot(e.clientX - gesture.x, e.clientY - gesture.y) < 6 && !gesture.moved) return;
    if (!gesture.moved) actions.grab?.();
    gesture.moved = true;
    const start = gesture.start;
    params.lo = Math.max(0, Math.min(.88, start.lo + dx * .85));
    params.hi = Math.min(1, Math.max(params.lo + .1, start.hi + dy * .6));
    params.amount = Math.min(1, Math.max(.2, start.amount - dy * .8));
    params.maxSpan = Math.abs(dy) < .06 ? 0 : Math.round(30 + (1 - Math.min(1, Math.abs(dy) * 2)) * 570);
    if (Math.abs(dx) > .12 || Math.abs(dy) > .12) params.direction = Math.abs(dx) > Math.abs(dy) ? 'rows' : 'columns';
  });
  const release = (e, cancelled = false) => {
    if (!gesture || gesture.id !== e.pointerId) return;
    const moved = gesture.moved; gesture = null;
    if (moved) actions.release?.();
    else if (!cancelled) {
      if (document.body.dataset.source === 'camera') actions.freeze?.();
      else mutate(false);
    }
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener('pointerup', e => release(e));
  canvas.addEventListener('pointercancel', e => release(e, true));
  canvas.addEventListener('lostpointercapture', e => release(e, true));
  addEventListener('blur', () => { if (gesture?.moved) actions.release?.(); gesture = null; });
  addEventListener('keydown', e => {
    if (e.repeat || e.target.closest?.('button, input') || e.altKey) return;
    const key = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && key !== 'z') return;
    if (key === ' ') { e.preventDefault(); if (document.body.dataset.source === 'camera') actions.freeze?.(); else mutate(); }
    else if (key === 'r') mutate();
    else if (key === 's') actions.save = true;
    else if (key === 'c') actions.useCamera?.();
    else if (key === 'o') $('file-input').click();
    else if (key === 'z') { e.preventDefault(); const old = history.pop(); if (old) Object.assign(params, old); }
  });
  addEventListener('dragover', e => e.preventDefault());
  addEventListener('drop', e => {
    e.preventDefault();
    const file = [...e.dataTransfer.files].find(f => f.type.startsWith('image/'));
    if (file) actions.openImage?.(file);
  });
}
mountInteraction();
main().catch(err => { console.error(err); fail(err.message); });
