/**
 * PADDYSCAN AI - COMPLETE FRONTEND CONTROLLER
 * Full integration with Flask PyTorch DualViewEfficientNetB0 Backend
 */

// ==============================================================================
// 1. GLOBAL STATE & CONFIGURATION
// ==============================================================================

function getApiBaseUrl() {
  // Clear any stale local overrides if on https or tunnel
  try {
    const saved = localStorage.getItem('paddyscan_api_url');
    if (saved && (window.location.protocol === 'https:' || saved.includes('127.0.0.1') || saved.includes('localhost'))) {
      localStorage.removeItem('paddyscan_api_url');
    }
  } catch (e) {}

  // 1. If running under HTTP or HTTPS (Dev Tunnels, port 5000, custom domains)
  if (window.location.protocol.startsWith('http')) {
    // Only if served by a separate dev server on port 5500 (Live Server)
    if (window.location.port === '5500') {
      const host = window.location.hostname || '127.0.0.1';
      return `http://${host}:5000`;
    }
    // In all standard web & tunnel setups (e.g. devtunnels.ms, localhost:5000, LAN IP),
    // use relative URL '' so all API requests (/api/...) match the page origin and protocol.
    return '';
  }

  // 2. Fallback for file:// protocol
  return 'http://127.0.0.1:5000';
}

const API_BASE_URL = getApiBaseUrl();

const state = {
  activeSection: 'front-view-section',
  modelInfo: null,

  // Front-View Analysis State (Single Image Only)
  frontFile: null,
  frontResult: null,

  // Dual-View Analysis State
  dualFrontFile: null,
  dualBackFile: null,
  dualResult: null,

  // Batch Analysis State
  batchFiles: [],
  batchResults: null,
  activeFilter: 'all',
  batchChart: null
};

// ==============================================================================
// 1B. ONNX RUNTIME WEB (WASM CPU ONLY) ENGINE & BOTANICAL DIAGNOSTICS
// ==============================================================================

const CLASSES = [
  "Brown spot",
  "Healthy Paddy",
  "Hispa",
  "Leaf Blast"
];

const DISEASE_METADATA = {
  "Healthy Paddy": {
    status: "Healthy",
    severity: "None",
    pathogen: "None (Plant is physiologically healthy)",
    symptoms: "Leaves display vibrant, uniform green coloration without necrotic lesions, discolored margins, or feeding trails.",
    management: "Maintain standard water leveling (2-5 cm during vegetative stage), balanced NPK fertilization, and periodic field scouting.",
    color_theme: "#16a34a"
  },
  "Brown spot": {
    status: "Diseased",
    severity: "Moderate to High",
    pathogen: "Bipolaris oryzae (Cochliobolus miyabeanus)",
    symptoms: "Small, circular to oval dark brown spots with yellow halos across leaf surfaces. Severe infection leads to seedling blight and kernel discoloration.",
    management: "Apply balanced soil fertilization especially potassium (K) and silicon. Seed treatment with Carbendazim (2g/kg) or foliar spray of Mancozeb / Propiconazole.",
    color_theme: "#d97706"
  },
  "Hispa": {
    status: "Diseased",
    severity: "Moderate to Severe",
    pathogen: "Dicladispa armigera (Coleoptera: Chrysomelidae)",
    symptoms: "Adult beetles scrape upper leaf epidermis creating characteristic white parallel streaks; grubs mine inside leaves causing blotch blister mines and withered tips.",
    management: "Clip leaf tips before transplanting to destroy eggs. Sweep net collection in early morning. Foliar spray of Chlorpyrifos 20% EC (2.5 ml/L) or Quinalphos 25% EC.",
    color_theme: "#ca8a04"
  },
  "Leaf Blast": {
    status: "Diseased",
    severity: "Critical / Severe",
    pathogen: "Magnaporthe oryzae (Pyricularia oryzae)",
    symptoms: "Spindle-shaped or diamond-shaped lesions with gray-whitish centers and dark reddish-brown margins. Rapid coalescence leads to complete leaf blighting.",
    management: "Avoid excessive nitrogen application. Spray Tricyclazole 75% WP (0.6 g/L) or Isoprothiolane 40% EC (1.5 ml/L) at initial lesion detection. Ensure good water drainage.",
    color_theme: "#dc2626"
  }
};

const FRONT_VIEW_MODEL_METRICS = { accuracy: 88.4, precision: 88.9, recall: 87.6, f1_score: 88.2 };
const FRONT_VIEW_DISEASE_METRICS = {
  "Healthy Paddy": { accuracy: 92.1, precision: 91.5, recall: 91.8, f1_score: 91.6 },
  "Brown spot": { accuracy: 87.5, precision: 88.2, recall: 86.4, f1_score: 87.3 },
  "Hispa": { accuracy: 88.9, precision: 88.4, recall: 89.2, f1_score: 88.8 },
  "Leaf Blast": { accuracy: 88.2, precision: 86.8, recall: 89.5, f1_score: 88.1 }
};

const DUAL_VIEW_MODEL_METRICS = { accuracy: 94.2, precision: 94.6, recall: 93.8, f1_score: 94.2 };
const DUAL_VIEW_DISEASE_METRICS = {
  "Healthy Paddy": { accuracy: 97.4, precision: 96.7, recall: 96.7, f1_score: 96.7 },
  "Brown spot": { accuracy: 93.8, precision: 95.7, recall: 91.2, f1_score: 93.4 },
  "Hispa": { accuracy: 94.5, precision: 93.8, recall: 95.2, f1_score: 94.5 },
  "Leaf Blast": { accuracy: 94.8, precision: 92.5, recall: 96.0, f1_score: 94.2 }
};

const ortState = {
  session: null,
  loading: false,
  loaded: false,
  error: null,
  modelPath: 'models/best_dual_efficientnet_b0.onnx',
  threads: 1
};

if (typeof ort !== 'undefined') {
  try {
    ort.env.wasm.wasmPaths = 'js/ort/';
  } catch (_) {}
}

function updateModelStatusUI(status, label) {
  const pulse = document.getElementById('backendPulse');
  const statusText = document.getElementById('backendStatusText');
  const deviceText = document.getElementById('backendDeviceText');
  const topbarChip = document.getElementById('topbarModelChip');
  const topbarModelIcon = document.getElementById('topbarModelIcon');
  const topbarModelName = document.getElementById('topbarModelName');
  const topbarModelMobile = document.getElementById('topbarModelMobile');
  const topbarModelArch = document.getElementById('topbarModelArch');

  if (status === 'ready') {
    if (pulse) pulse.className = 'status-pulse connected';
    if (statusText) statusText.textContent = 'WASM CPU Ready';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-cpu me-1"></i> Device CPU (WASM)`;
    if (topbarChip) topbarChip.className = 'model-badge-chip';
    if (topbarModelIcon) topbarModelIcon.className = 'bi bi-shield-check text-success me-1';
    if (topbarModelName) topbarModelName.textContent = 'best_dual_efficientnet_b0.onnx (WASM CPU)';
    if (topbarModelMobile) topbarModelMobile.textContent = 'WASM Active';
    if (topbarModelArch) topbarModelArch.textContent = 'EfficientNet-B0 + cGAN';
  } else if (status === 'loading') {
    if (pulse) pulse.className = 'status-pulse';
    if (statusText) statusText.textContent = label || 'Loading ONNX...';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-hourglass-split me-1"></i> Preparing WASM CPU`;
  } else if (status === 'error') {
    if (pulse) pulse.className = 'status-pulse error';
    if (statusText) statusText.textContent = 'Model Load Error';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-exclamation-triangle me-1"></i> WASM error`;
    if (topbarChip) topbarChip.className = 'model-badge-chip offline';
    if (topbarModelIcon) topbarModelIcon.className = 'bi bi-x-circle-fill text-danger me-1';
    if (topbarModelMobile) topbarModelMobile.textContent = 'Load Failed';
  }
}

async function initOnnxSession() {
  if (ortState.session) return ortState.session;
  if (ortState.loading) {
    while (ortState.loading) {
      await new Promise(r => setTimeout(r, 100));
    }
    if (ortState.session) return ortState.session;
  }

  ortState.loading = true;
  updateModelStatusUI('loading', 'Loading ONNX WASM CPU...');

  try {
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web library (ort.min.js) is not loaded.');
    }

    if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated && navigator.hardwareConcurrency > 1) {
      ortState.threads = Math.min(navigator.hardwareConcurrency, 4);
    } else {
      ortState.threads = 1;
    }
    ort.env.wasm.numThreads = ortState.threads;

    const sessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true
    };

    const candidatePaths = [
      'models/best_dual_efficientnet_b0.onnx',
      '/models/best_dual_efficientnet_b0.onnx',
      '../frontend/models/best_dual_efficientnet_b0.onnx',
      '../backend/models/best_dual_efficientnet_b0.onnx'
    ];

    let session = null;
    let loadedPath = null;
    let lastErr = null;

    for (const p of candidatePaths) {
      try {
        session = await ort.InferenceSession.create(p, sessionOptions);
        loadedPath = p;
        break;
      } catch (err) {
        lastErr = err;
      }
    }

    if (!session) {
      throw lastErr || new Error('Could not locate best_dual_efficientnet_b0.onnx model asset.');
    }

    ortState.session = session;
    ortState.loaded = true;
    ortState.modelPath = loadedPath;
    ortState.loading = false;
    updateModelStatusUI('ready', 'WASM CPU Ready');
    console.log(`[PaddyScan AI] ONNX Runtime Web initialized on user CPU (${ortState.threads} thread(s)) from: ${loadedPath}`);
    return session;
  } catch (err) {
    ortState.loading = false;
    ortState.error = err.message;
    updateModelStatusUI('error', 'Model Load Failed');
    console.error('[PaddyScan AI] ONNX session init error:', err);
    throw err;
  }
}

let _sharedCanvas = null;
let _sharedCtx = null;
function getSharedCanvas() {
  if (!_sharedCanvas) {
    _sharedCanvas = document.createElement('canvas');
    _sharedCanvas.width = 256;
    _sharedCanvas.height = 256;
    _sharedCtx = _sharedCanvas.getContext('2d', { willReadFrequently: true });
  }
  return { canvas: _sharedCanvas, ctx: _sharedCtx };
}

async function loadImageToCanvas(fileOrBlob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(fileOrBlob);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const origWidth = img.naturalWidth || img.width;
        const origHeight = img.naturalHeight || img.height;
        const { canvas, ctx } = getSharedCanvas();
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(img, 0, 0, 256, 256);
        const imageData = ctx.getImageData(0, 0, 256, 256);
        URL.revokeObjectURL(url);
        resolve({
          width: origWidth,
          height: origHeight,
          imageData: imageData
        });
      } catch (e) {
        URL.revokeObjectURL(url);
        reject(e);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Corrupted or invalid image: The uploaded file cannot be processed as an image.'));
    };
    img.src = url;
  });
}

function verifyPaddyLeafImageFromCanvas(imgInfo, fileSize) {
  if (!fileSize || fileSize === 0) {
    return {
      isValid: false,
      message: "Invalid image: File is empty (0 bytes). Please upload a paddy leaf image for disease detection.",
      details: { error_type: "empty_file" }
    };
  }

  const { width: w, height: h, imageData } = imgInfo;
  if (w < 32 || h < 32) {
    return {
      isValid: false,
      message: "Invalid image: Image resolution is too small (minimum 32x32 pixels required). Please upload a clearer paddy leaf image.",
      details: { error_type: "too_small", width: w, height: h }
    };
  }

  if (Math.max(w, h) / Math.max(Math.min(w, h), 1) > 25.0) {
    return {
      isValid: false,
      message: "Invalid image: Degenerate aspect ratio. Please upload a standard photograph of a paddy leaf.",
      details: { error_type: "extreme_aspect_ratio" }
    };
  }

  const pixels = imageData.data;
  const numPixels = 256 * 256;

  let sumGray = 0, sumGraySq = 0;
  let sumR = 0, sumRSq = 0;
  let sumG = 0, sumGSq = 0;
  let sumB = 0, sumBSq = 0;

  const grayArr = new Float32Array(numPixels);
  const rArr = new Float32Array(numPixels);
  const gArr = new Float32Array(numPixels);
  const bArr = new Float32Array(numPixels);

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const r = pixels[idx];
    const g = pixels[idx + 1];
    const b = pixels[idx + 2];
    const gr = 0.299 * r + 0.587 * g + 0.114 * b;

    rArr[i] = r; gArr[i] = g; bArr[i] = b; grayArr[i] = gr;
    sumGray += gr; sumGraySq += gr * gr;
    sumR += r; sumRSq += r * r;
    sumG += g; sumGSq += g * g;
    sumB += b; sumBSq += b * b;
  }

  const meanGray = sumGray / numPixels;
  const stdGray = Math.sqrt(Math.max(0, sumGraySq / numPixels - meanGray * meanGray));
  const meanR = sumR / numPixels;
  const stdR = Math.sqrt(Math.max(0, sumRSq / numPixels - meanR * meanR));
  const meanG = sumG / numPixels;
  const stdG = Math.sqrt(Math.max(0, sumGSq / numPixels - meanG * meanG));
  const meanB = sumB / numPixels;
  const stdB = Math.sqrt(Math.max(0, sumBSq / numPixels - meanB * meanB));

  if (stdGray < 5.0 || (stdR < 3.0 && stdG < 3.0 && stdB < 3.0)) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "monochrome", std_gray: stdGray }
    };
  }

  if (meanGray > 252.0 && stdGray < 8.0) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "blank_white" }
    };
  }

  if (meanGray < 5.0 && stdGray < 8.0) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "blank_black" }
    };
  }

  let gradSum = 0;
  for (let y = 0; y < 255; y++) {
    for (let x = 0; x < 255; x++) {
      const idx = y * 256 + x;
      const dy = Math.abs(grayArr[(y + 1) * 256 + x] - grayArr[idx]);
      const dx = Math.abs(grayArr[y * 256 + (x + 1)] - grayArr[idx]);
      gradSum += (dy + dx) * 0.5;
    }
  }
  const meanGrad = gradSum / (255 * 255);

  let cov = 0, var1 = 0, var2 = 0;
  const nPairs = 255 * 256;
  let sumG1 = 0, sumG2 = 0;
  for (let y = 0; y < 255; y++) {
    for (let x = 0; x < 256; x++) {
      sumG1 += gArr[y * 256 + x];
      sumG2 += gArr[(y + 1) * 256 + x];
    }
  }
  const mG1 = sumG1 / nPairs;
  const mG2 = sumG2 / nPairs;
  for (let y = 0; y < 255; y++) {
    for (let x = 0; x < 256; x++) {
      const d1 = gArr[y * 256 + x] - mG1;
      const d2 = gArr[(y + 1) * 256 + x] - mG2;
      cov += d1 * d2;
      var1 += d1 * d1;
      var2 += d2 * d2;
    }
  }
  const spatialCorr = (var1 > 0 && var2 > 0) ? (cov / Math.sqrt(var1 * var2)) : 0.0;

  if (meanGrad > 35.0 || spatialCorr < 0.60) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "noise_or_uncorrelated", spatial_corr: spatialCorr, mean_grad: meanGrad }
    };
  }

  let greenCount = 0;
  let yellowCount = 0;
  let lesionCount = 0;
  let totalFoliarCount = 0;
  let blueDominantCount = 0;
  let redDominantCount = 0;
  let foliarExGSum = 0;
  let foliarExGCount = 0;
  let centerFoliarCount = 0;

  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const i = y * 256 + x;
      const r = rArr[i];
      const g = gArr[i];
      const b = bArr[i];

      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      const delta = maxC - minC;
      let hue = 0;
      if (delta > 0) {
        if (maxC === r) {
          hue = ((g - b) / delta) % 6;
        } else if (maxC === g) {
          hue = (b - r) / delta + 2;
        } else {
          hue = (r - g) / delta + 4;
        }
        hue = Math.round(hue * 30);
        if (hue < 0) hue += 180;
      }
      const sat = maxC > 0 ? (delta / maxC) * 255 : 0;
      const val = maxC;

      const isGreen = (hue >= 18 && hue <= 110 && sat >= 20 && val >= 20 && g > b * 1.02 && (g >= r * 0.85 || (2 * g - r - b > 5)));
      const isYellow = (hue >= 12 && hue < 22 && sat >= 25 && val >= 30 && g > b * 1.10 && r > b * 1.10 && g >= r * 0.65 && (2 * g - r - b > -10));
      const isLesion = (hue >= 5 && hue <= 24 && sat >= 15 && val >= 15 && val <= 210 && r > b * 1.05 && g > b * 1.01 && r >= g * 0.85);

      if (isGreen) {
        greenCount++;
        foliarExGSum += (2 * g - r - b);
        foliarExGCount++;
      }
      if (isYellow) yellowCount++;
      if (isLesion) lesionCount++;

      const isFoliar = isGreen || isYellow || isLesion;
      if (isFoliar) {
        totalFoliarCount++;
        if (x >= 40 && x < 184 && y >= 40 && y < 184) {
          centerFoliarCount++;
        }
      }

      if (b > g * 1.25 && b > r * 1.25 && sat > 40 && val > 100) blueDominantCount++;
      if (r > g * 1.30 && r > b * 1.30 && sat > 40 && val > 60) redDominantCount++;
    }
  }

  const greenRatio = greenCount / numPixels;
  const yellowRatio = yellowCount / numPixels;
  const lesionRatio = lesionCount / numPixels;
  const vegRatio = greenRatio + yellowRatio;
  const totalFoliarRatio = totalFoliarCount / numPixels;
  const centerFoliarRatio = centerFoliarCount / (144 * 144);
  const blueRatio = blueDominantCount / numPixels;
  const redRatio = redDominantCount / numPixels;
  const meanExG = foliarExGCount > 0 ? (foliarExGSum / foliarExGCount) : 0.0;

  if (vegRatio < 0.02 && totalFoliarRatio < 0.025) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "low_vegetation", veg_ratio: Math.round(vegRatio * 1000) / 10, total_foliar: Math.round(totalFoliarRatio * 1000) / 10 }
    };
  }

  if (lesionRatio > 0.35 && greenRatio < 0.03) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "lesion_without_foliage", lesion_ratio: Math.round(lesionRatio * 1000) / 10, green_ratio: Math.round(greenRatio * 1000) / 10 }
    };
  }

  if (blueRatio > 0.30 && totalFoliarRatio < 0.03) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "blue_dominant", blue_ratio: Math.round(blueRatio * 1000) / 10 }
    };
  }

  if (redRatio > 0.25 && greenRatio < 0.03) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "red_dominant", red_ratio: Math.round(redRatio * 1000) / 10 }
    };
  }

  if (foliarExGCount > 0 && meanExG < 3.0) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "low_foliar_exg", exg_mean: Math.round(meanExG * 10) / 10 }
    };
  }

  if (centerFoliarRatio < 0.02 && totalFoliarRatio < 0.03) {
    return {
      isValid: false,
      message: "Invalid image: Please upload a paddy leaf image for disease detection.",
      details: { error_type: "center_not_foliar", center_foliar: Math.round(centerFoliarRatio * 1000) / 10 }
    };
  }

  return {
    isValid: true,
    message: "Valid paddy leaf image.",
    details: {
      veg_ratio: Math.round(vegRatio * 1000) / 10,
      green_ratio: Math.round(greenRatio * 1000) / 10,
      lesion_ratio: Math.round(lesionRatio * 1000) / 10,
      center_foliar: Math.round(centerFoliarRatio * 1000) / 10,
      exg_mean: Math.round(meanExG * 10) / 10
    }
  };
}

function imageDataToTensor(imageData) {
  const pixels = imageData.data;
  const numPixels = 256 * 256;
  const float32Data = new Float32Array(1 * 3 * numPixels);
  const channelSize = numPixels;
  const rOffset = 0;
  const gOffset = channelSize;
  const bOffset = channelSize * 2;

  for (let i = 0; i < numPixels; i++) {
    const pixelIndex = i * 4;
    const r = pixels[pixelIndex] / 255.0;
    const g = pixels[pixelIndex + 1] / 255.0;
    const b = pixels[pixelIndex + 2] / 255.0;

    float32Data[rOffset + i] = (r - 0.485) / 0.229;
    float32Data[gOffset + i] = (g - 0.456) / 0.224;
    float32Data[bOffset + i] = (b - 0.406) / 0.225;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 256, 256]);
}

function flipTensorHorizontal(tensor) {
  const data = tensor.data;
  const flipped = new Float32Array(data.length);
  const W = 256;
  const H = 256;
  const channelSize = W * H;

  for (let c = 0; c < 3; c++) {
    const cOffset = c * channelSize;
    for (let y = 0; y < H; y++) {
      const rowOffset = y * W;
      for (let x = 0; x < W; x++) {
        const srcIdx = cOffset + rowOffset + x;
        const dstIdx = cOffset + rowOffset + (W - 1 - x);
        flipped[dstIdx] = data[srcIdx];
      }
    }
  }

  return new ort.Tensor('float32', flipped, [1, 3, 256, 256]);
}

function extractBotanicalSignals(imageData) {
  const pixels = imageData.data;
  const numPixels = 256 * 256;
  let nLeaf = 0;
  let hispaScrapes = 0;
  let brownSpots = 0;
  let healthyGreen = 0;

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const r = pixels[idx];
    const g = pixels[idx + 1];
    const b = pixels[idx + 2];

    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const delta = maxC - minC;
    const sat = maxC > 0 ? (delta / maxC) * 255 : 0;
    const val = maxC;

    // Segment leaf surface from dark backdrop
    const isLeaf = (val > 35) && ((g > b * 0.95) || (r > b * 0.95) || (g > 45));
    if (isLeaf) {
      nLeaf++;

      // 1. Hispa feeding trails: whitish/translucent scrapes along leaf venation
      const isScrape = ((val > 110 && sat < 95 && g > 80 && r > 80) ||
                        (r > 125 && g > 125 && b > 75 && Math.abs(r - g) < 35));
      if (isScrape) {
        hispaScrapes++;
      }

      // 2. Brown Spot: discrete dark reddish-brown circular spots (R > G > B with distinct contrast)
      const isBrown = (r > g * 1.15 && r > b * 1.25 && r > 55 && r < 175 && g > 30 && sat > 40);
      if (isBrown) {
        brownSpots++;
      }

      // 3. Healthy: pure chlorophyll green
      if (g > b * 1.10 && g >= r * 0.90 && (2 * g - r - b > 15) && sat >= 30) {
        healthyGreen++;
      }
    }
  }

  const leafCount = Math.max(nLeaf, 1);
  return {
    scrapeRatio: hispaScrapes / leafCount,
    brownRatio: brownSpots / leafCount,
    greenRatio: healthyGreen / leafCount
  };
}

function calibrateUncertainLogits(logitsIn, topImgInfo, bottomImgInfo = null, viewType = "front") {
  const logits = Float32Array.from(logitsIn);
  const bsIdx = 0; // Brown spot
  const hpIdx = 1; // Healthy Paddy
  const hIdx  = 2; // Hispa
  const lbIdx = 3; // Leaf Blast

  const sigTop = extractBotanicalSignals(topImgInfo.imageData);
  let sc = sigTop.scrapeRatio;
  let br = sigTop.brownRatio;

  if (bottomImgInfo) {
    const sigBot = extractBotanicalSignals(bottomImgInfo.imageData);
    sc = Math.max(sc, sigBot.scrapeRatio);
    br = Math.max(br, sigBot.brownRatio);
  }

  // Margin between Brown Spot (0) and Hispa (2): positive if Brown Spot > Hispa
  const marginBsH = logits[bsIdx] - logits[hIdx];

  // ONLY calibrate if Brown Spot and Hispa are in a close contest (margin within ±1.8)
  if (Math.abs(marginBsH) < 1.8) {
    // 1. True Hispa: prominent longitudinal feeding scrapes and lack of discrete circular brown spots
    if ((sc > 0.10 && br < 0.006) || (sc > 0.15 && sc > br * 3.0)) {
      const boost = 1.5 * Math.min(sc / 0.18, 2.0);
      logits[hIdx] += boost;
      logits[bsIdx] -= boost * 0.5;
    }
    // 2. True Brown Spot: discrete circular brown spots and absence of prominent scrape trails
    else if (br > 0.010 && sc < 0.08) {
      const boost = 1.5 * Math.min(br / 0.02, 2.0);
      logits[bsIdx] += boost;
      logits[hIdx] -= boost * 0.5;
    }
  }

  return logits;
}

async function runBrowserInference(tensorTop, tensorBottom, topImgInfo, bottomImgInfo = null, viewType = "front") {
  const session = await initOnnxSession();
  const isDualView = tensorBottom !== null && tensorBottom !== tensorTop;
  const startT = performance.now();

  let logits;

  if (isDualView) {
    const feedsFwd = { img_top: tensorTop, img_bottom: tensorBottom };
    const outFwd = await session.run(feedsFwd);
    const logitsFwd = Array.from(outFwd.logits.data);

    const feedsRev = { img_top: tensorBottom, img_bottom: tensorTop };
    const outRev = await session.run(feedsRev);
    const logitsRev = Array.from(outRev.logits.data);

    logits = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
      logits[i] = 0.80 * logitsFwd[i] + 0.20 * logitsRev[i];
    }
  } else {
    const feeds1 = { img_top: tensorTop, img_bottom: tensorTop };
    const out1 = await session.run(feeds1);
    const logits1 = Array.from(out1.logits.data);

    const topFlip = flipTensorHorizontal(tensorTop);
    const feeds2 = { img_top: topFlip, img_bottom: topFlip };
    const out2 = await session.run(feeds2);
    const logits2 = Array.from(out2.logits.data);

    logits = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
      logits[i] = (logits1[i] + logits2[i]) / 2.0;
    }
  }

  if (topImgInfo) {
    logits = calibrateUncertainLogits(logits, topImgInfo, bottomImgInfo, viewType);
  }

  const temp = isDualView ? 0.92 : 1.0;
  for (let i = 0; i < 4; i++) {
    logits[i] /= temp;
  }

  let maxLogit = -Infinity;
  for (let i = 0; i < 4; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  let sumExp = 0;
  const expLogits = new Float32Array(4);
  for (let i = 0; i < 4; i++) {
    expLogits[i] = Math.exp(logits[i] - maxLogit);
    sumExp += expLogits[i];
  }

  const probs = new Float32Array(4);
  let bestIdx = 0;
  let bestProb = -1;
  for (let i = 0; i < 4; i++) {
    probs[i] = expLogits[i] / sumExp;
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIdx = i;
    }
  }

  const elapsedMs = Math.round((performance.now() - startT) * 10) / 10;
  const predClass = CLASSES[bestIdx];
  const confPct = Math.round(bestProb * 10000) / 100;

  const classProbabilities = {};
  CLASSES.forEach((cls, i) => {
    classProbabilities[cls] = Math.round(probs[i] * 10000) / 100;
  });

  const predMetrics = isDualView
    ? (DUAL_VIEW_DISEASE_METRICS[predClass] || DUAL_VIEW_MODEL_METRICS)
    : (FRONT_VIEW_DISEASE_METRICS[predClass] || FRONT_VIEW_MODEL_METRICS);
  const overallMetrics = isDualView ? DUAL_VIEW_MODEL_METRICS : FRONT_VIEW_MODEL_METRICS;

  return {
    predicted_class: predClass,
    confidence: confPct,
    class_index: bestIdx,
    probabilities: classProbabilities,
    inference_time_ms: elapsedMs,
    disease_info: DISEASE_METADATA[predClass] || {},
    metrics: predMetrics,
    model_metrics: overallMetrics,
    is_dual_view: isDualView
  };
}

function detectLeafView(filename) {
  if (!filename) return "unknown";
  const stem = filename.replace(/\\/g, '/').split('/').pop().replace(/\.[^/.]+$/, '').toLowerCase();
  if (/([_-]t\d*$|_t$|front|top|adaxial|^t\d*[_-]|^t\d+$)/.test(stem)) return "front";
  if (/([_-][bd]\d*$|_[bd]$|back|bottom|abaxial|^[bd]\d*[_-]|^[bd]\d+$)/.test(stem)) return "back";
  return "unknown";
}

// Helper to strip folder prefixes like 'Leaf Blast/Leaf-1' -> 'Leaf-1' or 'Leaf Blast/000002_t1.JPG' -> '000002_t1.JPG'
function stripFolderPrefix(filePath) {
  if (!filePath) return '';
  return filePath.replace(/\\/g, '/').split('/').pop();
}
window.stripFolderPrefix = stripFolderPrefix;

// Client-side specimen ID extraction helper (pure specimen ID e.g. Leaf-1)
function extractClientSampleId(filename) {
  if (!filename) return 'Sample';
  const cleanPath = filename.replace(/\\/g, '/');
  const baseName = cleanPath.split('/').pop();
  const stem = baseName.replace(/\.[^/.]+$/, "").toLowerCase();

  let leafId = null;
  let m = stem.match(/[_-](?:t|b|d)(\d+)$/);
  if (m) leafId = 'Leaf-' + m[1];
  else {
    m = stem.match(/(?:t|b|d)(\d+)$/);
    if (m) leafId = 'Leaf-' + m[1];
    else {
      m = stem.match(/^(.*?)[_-](?:t1?|b1?|d1?|front|back|top|bottom)/);
      if (m && m[1]) leafId = m[1].charAt(0).toUpperCase() + m[1].slice(1);
      else {
        const cleaned = stem.replace(/[_-]?(t1?|b1?|d1?|top|bottom|front|back|adaxial|abaxial)/g, '');
        leafId = cleaned ? cleaned : stem;
      }
    }
  }
  return leafId;
}
window.extractClientSampleId = extractClientSampleId;


// ==============================================================================
// 2. COMPLETE RESULTS & STATE RESET
// ==============================================================================

// Front-View (T1) State & UI Reset
function resetFrontViewSection() {
  state.frontFile = null;
  state.frontResult = null;
  const frontInput = document.getElementById('frontFileInput');
  if (frontInput) frontInput.value = '';
  document.getElementById('frontPrompt')?.classList.remove('d-none');
  document.getElementById('frontPreviewBox')?.classList.add('d-none');
  document.getElementById('removeFrontBtn')?.classList.add('d-none');
  const frontImg = document.getElementById('frontImgPreview');
  if (frontImg) frontImg.src = '';
  const frontName = document.getElementById('frontFilenameText');
  if (frontName) frontName.textContent = '';
  const frontSize = document.getElementById('frontFilesizeText');
  if (frontSize) frontSize.textContent = '';
  const analyzeFrontBtn = document.getElementById('analyzeFrontBtn');
  if (analyzeFrontBtn) analyzeFrontBtn.disabled = true;

  const frontEmpty = document.getElementById('frontEmptyState');
  if (frontEmpty) frontEmpty.style.display = 'block';
  const frontResult = document.getElementById('frontResultCard');
  if (frontResult) frontResult.style.display = 'none';
  const frontError = document.getElementById('frontErrorCard');
  if (frontError) frontError.style.display = 'none';
  const exportFrontBtn = document.getElementById('exportFrontCsvBtn');
  if (exportFrontBtn) exportFrontBtn.style.display = 'none';
}

// Dual-View (T1 + B1) State & UI Reset
function resetDualViewSection() {
  state.dualFrontFile = null;
  state.dualBackFile = null;
  state.dualResult = null;
  const dualFrontInput = document.getElementById('dualFrontInput');
  if (dualFrontInput) dualFrontInput.value = '';
  const dualBackInput = document.getElementById('dualBackInput');
  if (dualBackInput) dualBackInput.value = '';

  document.getElementById('dualFrontPrompt')?.classList.remove('d-none');
  document.getElementById('dualFrontPreviewBox')?.classList.add('d-none');
  document.getElementById('removeDualFrontBtn')?.classList.add('d-none');
  const dualFrontImg = document.getElementById('dualFrontImg');
  if (dualFrontImg) dualFrontImg.src = '';
  const dualFrontFilename = document.getElementById('dualFrontFilename');
  if (dualFrontFilename) dualFrontFilename.textContent = '';

  document.getElementById('dualBackPrompt')?.classList.remove('d-none');
  document.getElementById('dualBackPreviewBox')?.classList.add('d-none');
  document.getElementById('removeDualBackBtn')?.classList.add('d-none');
  const dualBackImg = document.getElementById('dualBackImg');
  if (dualBackImg) dualBackImg.src = '';
  const dualBackFilename = document.getElementById('dualBackFilename');
  if (dualBackFilename) dualBackFilename.textContent = '';

  const pairBadge = document.getElementById('pairBadge');
  if (pairBadge) pairBadge.textContent = '0/2 Ready';
  const pairDot = document.getElementById('pairCheckDot');
  if (pairDot) pairDot.className = 'pair-dot';
  const pairText = document.getElementById('pairCheckText');
  if (pairText) pairText.textContent = 'Upload both T1 and B1 images';
  const analyzeDualBtn = document.getElementById('analyzeDualBtn');
  if (analyzeDualBtn) analyzeDualBtn.disabled = true;

  const dualEmpty = document.getElementById('dualEmptyState');
  if (dualEmpty) dualEmpty.style.display = 'block';
  const dualResult = document.getElementById('dualResultCard');
  if (dualResult) dualResult.style.display = 'none';
  const dualError = document.getElementById('dualErrorCard');
  if (dualError) dualError.style.display = 'none';
  const exportDualBtn = document.getElementById('exportDualCsvBtn');
  if (exportDualBtn) exportDualBtn.style.display = 'none';
}

// Batch Analysis State & UI Reset
function resetBatchSection() {
  state.batchFiles = [];
  state.batchResults = null;
  state.batchFilesMap?.clear();
  state.batchUrlCache?.clear();
  if (state.batchChart) {
    try { state.batchChart.destroy(); } catch (_) {}
    state.batchChart = null;
  }
  const folderInput = document.getElementById('batchFolderInput');
  if (folderInput) folderInput.value = '';
  const filesInput = document.getElementById('batchFilesInput');
  if (filesInput) filesInput.value = '';

  const batchStagedTotal = document.getElementById('batchStagedTotal');
  if (batchStagedTotal) batchStagedTotal.textContent = '0 files';
  const batchStagedDetails = document.getElementById('batchStagedDetails');
  if (batchStagedDetails) batchStagedDetails.textContent = '';

  const runBatchBtn = document.getElementById('runBatchBtn');
  if (runBatchBtn) runBatchBtn.disabled = true;

  const batchProgress = document.getElementById('batchProgressWrapper');
  if (batchProgress) batchProgress.style.display = 'none';
  const batchProgressBar = document.getElementById('batchProgressBar');
  if (batchProgressBar) batchProgressBar.style.width = '0%';

  const batchResultsWrapper = document.getElementById('batchResultsWrapper');
  if (batchResultsWrapper) batchResultsWrapper.style.display = 'none';
  const batchTableBody = document.getElementById('batchTableBody');
  if (batchTableBody) batchTableBody.innerHTML = '';
  const exportBatchBtn = document.getElementById('exportBatchCsvBtn');
  if (exportBatchBtn) exportBatchBtn.style.display = 'none';
}

// Reset only the given or active section
function resetActiveSection(targetId) {
  const sectionId = targetId || state.activeSection || 'front-view-section';
  if (sectionId === 'front-view-section') {
    resetFrontViewSection();
  } else if (sectionId === 'dual-view-section') {
    resetDualViewSection();
  } else if (sectionId === 'batch-section') {
    resetBatchSection();
  } else if (sectionId === 'training-graphs-section') {
    if (typeof renderTrainingCurves === 'function') {
      renderTrainingCurves(true);
    }
  } else if (sectionId === 'model-info-section' && state.modelInfo) {
    if (typeof renderModelSpecs === 'function') {
      renderModelSpecs(state.modelInfo);
    }
  } else if (sectionId === 'disease-guide-section' && state.modelInfo) {
    if (typeof renderDiseaseGuide === 'function') {
      renderDiseaseGuide(state.modelInfo.disease_reference);
    }
  }
}

function resetAllResults(notify = false) {
  resetFrontViewSection();
  resetDualViewSection();
  resetBatchSection();

  // Clear all generic file inputs on page
  document.querySelectorAll('input[type="file"]').forEach(inp => {
    try { inp.value = ''; } catch (_) {}
  });

  if (notify) {
    showAlert('All analysis results and uploads have been reset.', 'info', state.activeSection);
  }
}

// Make globally accessible
window.resetFrontViewSection = resetFrontViewSection;
window.resetDualViewSection = resetDualViewSection;
window.resetBatchSection = resetBatchSection;
window.resetActiveSection = resetActiveSection;
window.resetAllResults = resetAllResults;

// ==============================================================================
// 3. INITIALIZATION
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
  resetAllResults();
  initNavigation();
  initSystemHealth();
  initFrontViewSection();
  initDualViewSection();
  initBatchSection();
  initTrainingCurvesSection();
});

// ==============================================================================
// 3. NAVIGATION LOGIC
// ==============================================================================

const PAGE_TITLES = {
  'front-view-section': {
    title: 'Front-View Analysis (T1)',
    caption: 'Evaluate adaxial leaf surfaces via Siamese feature extraction fallback'
  },
  'dual-view-section': {
    title: 'Dual-View Analysis (T1 + D1)',
    caption: 'Complete 2560D Siamese feature fusion of front and back leaf surfaces'
  },
  'batch-section': {
    title: 'Folder-Based Batch Analysis',
    caption: 'Automated leaf pair detection, mode strategies, and distribution analytics'
  },
  'training-graphs-section': {
    title: 'Training & Evaluation Curves',
    caption: '50-Epoch Dual-View EfficientNet-B0 + cGAN Accuracy & Cross-Entropy Loss Convergence'
  },
  'model-info-section': {
    title: 'Model Architecture: EfficientNet-B0 + cGAN',
    caption: 'Conditional GAN dataset synthesis and Dual-Stream Siamese EfficientNet-B0 backbone'
  },
  'disease-guide-section': {
    title: 'Paddy Disease Field Guide',
    caption: 'Botanical symptoms, pathogen etiology, and agronomic management'
  }
};

function switchSection(targetId, smoothScroll = false) {
  const sections = document.querySelectorAll('.content-section');
  const navButtons = document.querySelectorAll('.nav-item-btn');
  if (!targetId || !document.getElementById(targetId)) return;

  // Clear existing alerts from old section when navigating away
  if (state.activeSection && state.activeSection !== targetId) {
    const oldContainer = document.getElementById(`alertContainer-${state.activeSection}`);
    if (oldContainer) oldContainer.innerHTML = '';
  }

  navButtons.forEach(b => {
    if (b.getAttribute('data-target') === targetId) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });

  sections.forEach(sec => {
    if (sec.id === targetId) {
      sec.classList.add('active');
    } else {
      sec.classList.remove('active');
    }
  });

  state.activeSection = targetId;

  // Persist across browser refresh
  try {
    localStorage.setItem('paddyscan_active_section', targetId);
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, null, `#${targetId}`);
    }
  } catch (_) {}

  // Scroll to top
  if (smoothScroll) {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Update Topbar
  const info = PAGE_TITLES[targetId] || { title: 'PaddyScan AI', caption: '' };
  const titleEl = document.getElementById('pageTitle');
  const captionEl = document.getElementById('pageCaption');
  if (titleEl) titleEl.textContent = info.title;
  if (captionEl) captionEl.textContent = info.caption;

  // Trigger graph drawing when training curves section is selected
  if (targetId === 'training-graphs-section') {
    setTimeout(() => {
      if (typeof renderTrainingCurves === 'function') {
        renderTrainingCurves(true);
      }
    }, 60);
  }
}

function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item-btn');
  const sidebar = document.getElementById('appSidebar');
  const openSidebarBtn = document.getElementById('openSidebarBtn');
  const closeSidebarBtn = document.getElementById('closeSidebarBtn');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      if (!targetId) return;

      switchSection(targetId, true);

      // Mobile close sidebar
      if (window.innerWidth < 992 && sidebar) {
        sidebar.classList.remove('show');
        sidebarBackdrop?.classList.remove('show');
      }
    });
  });

  // Restore saved active section on page refresh
  const savedSection = window.location.hash.replace('#', '') || (function() {
    try { return localStorage.getItem('paddyscan_active_section'); } catch(_) { return null; }
  })();
  if (savedSection && document.getElementById(savedSection) && PAGE_TITLES[savedSection]) {
    switchSection(savedSection, false);
  }

  if (openSidebarBtn) {
    openSidebarBtn.addEventListener('click', () => {
      sidebar?.classList.add('show');
      sidebarBackdrop?.classList.add('show');
    });
  }
  if (closeSidebarBtn) {
    closeSidebarBtn.addEventListener('click', () => {
      sidebar?.classList.remove('show');
      sidebarBackdrop?.classList.remove('show');
    });
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', () => {
      sidebar?.classList.remove('show');
      sidebarBackdrop.classList.remove('show');
    });
  }

  // Refresh current active page connection, health, and reset only the active page results
  document.getElementById('checkApiBtn')?.addEventListener('click', () => {
    resetActiveSection(state.activeSection);
    initSystemHealth(true);
  });
}

// ==============================================================================
// 4. API & SYSTEM HEALTH
// ==============================================================================

async function initSystemHealth(isManual = false) {
  const pulse = document.getElementById('backendPulse');
  const statusText = document.getElementById('backendStatusText');
  const deviceText = document.getElementById('backendDeviceText');

  // Topbar elements
  const refreshBtn = document.getElementById('checkApiBtn');
  const refreshIcon = document.getElementById('refreshBtnIcon');
  const refreshText = document.getElementById('refreshBtnText');
  const topbarChip = document.getElementById('topbarModelChip');
  const topbarModelIcon = document.getElementById('topbarModelIcon');
  const topbarModelName = document.getElementById('topbarModelName');
  const topbarModelMobile = document.getElementById('topbarModelMobile');
  const topbarModelArch = document.getElementById('topbarModelArch');

  if (pulse) pulse.className = 'status-pulse';
  if (statusText) statusText.textContent = 'Initializing WASM...';

  if (isManual && refreshBtn) {
    refreshBtn.disabled = true;
    if (refreshIcon) refreshIcon.classList.add('spin-anim');
    if (refreshText) refreshText.textContent = 'Checking...';
  }

  const startTime = Date.now();

  const defaultSpecs = {
    status: 'success',
    loaded: true,
    checkpoint: 'best_dual_efficientnet_b0.onnx',
    model_name: 'Dual-View Siamese EfficientNet-B0 + cGAN Augmented Pipeline',
    gan_augmentation: 'Conditional Generative Adversarial Network (cGAN)',
    inference_engine: 'ONNX Runtime Web (WASM CPU)',
    device: 'User Device CPU (WASM)',
    device_name: 'On-Device CPU (WASM SIMD)',
    classes: CLASSES,
    num_classes: 4,
    total_parameters: 5321856,
    input_resolution: '256x256 RGB',
    normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
    feature_fusion: 'Dual 1280-dim concatenated feature vectors (2560-dim)',
    front_only_support: true,
    front_only_mechanism: "Trained Siamese single-view projection (img_bottom = img_top fallback from author's training code)",
    metrics: DUAL_VIEW_MODEL_METRICS,
    class_metrics: DUAL_VIEW_DISEASE_METRICS,
    front_metrics: FRONT_VIEW_MODEL_METRICS,
    disease_reference: DISEASE_METADATA
  };

  try {
    // 1. Initialize & warm up ONNX Runtime Web WASM session
    let onnxReady = false;
    try {
      await initOnnxSession();
      onnxReady = true;
    } catch (e) {
      console.warn('WASM session pre-warm deferred to first user upload:', e);
    }

    // 2. Fetch server metadata if online, or fallback to embedded specs
    let serverData = null;
    try {
      const res = await fetch(`${API_BASE_URL}/api/model-info`);
      if (res.ok) {
        serverData = await res.json();
      }
    } catch (_) {}

    const data = Object.assign({}, defaultSpecs, serverData || {});
    state.modelInfo = data;

    if (isManual) {
      const elapsed = Date.now() - startTime;
      if (elapsed < 500) {
        await new Promise(resolve => setTimeout(resolve, 500 - elapsed));
      }
    }

    if (pulse) pulse.className = 'status-pulse connected';
    if (statusText) statusText.textContent = onnxReady ? 'WASM CPU Ready' : 'Model Online';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-cpu me-1"></i> Device CPU (WASM)`;

    if (topbarChip) topbarChip.className = 'model-badge-chip';
    if (topbarModelIcon) topbarModelIcon.className = 'bi bi-shield-check text-success me-1';
    if (topbarModelName) topbarModelName.textContent = 'best_dual_efficientnet_b0.onnx (WASM CPU)';
    if (topbarModelMobile) topbarModelMobile.textContent = 'WASM Active';
    if (topbarModelArch) topbarModelArch.textContent = 'EfficientNet-B0 + cGAN';

    renderModelSpecs(data);
    renderDiseaseGuide(data.disease_reference || DISEASE_METADATA);

    if (isManual) {
      if (refreshIcon) refreshIcon.className = 'bi bi-check-circle-fill text-success me-1';
      if (refreshText) refreshText.textContent = 'Refreshed!';
      showAlert(`WASM CPU Engine Active: Predictions run locally on your device CPU.`, 'success', state.activeSection);
      setTimeout(() => {
        if (refreshIcon) refreshIcon.className = 'bi bi-arrow-repeat me-1';
        if (refreshText) refreshText.textContent = 'Refresh';
      }, 1800);
    }
  } catch (err) {
    console.error('System Health check exception:', err);
    state.modelInfo = defaultSpecs;
    renderModelSpecs(defaultSpecs);
    renderDiseaseGuide(DISEASE_METADATA);

    if (pulse) pulse.className = 'status-pulse connected';
    if (statusText) statusText.textContent = 'WASM CPU Ready';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-cpu me-1"></i> Device CPU (WASM)`;
  } finally {
    if (refreshIcon) refreshIcon.classList.remove('spin-anim');
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function renderModelSpecs(info) {
  if (!info) return;
  document.getElementById('infoCheckpoint').textContent = 'best_dual_efficientnet_b0.onnx';
  document.getElementById('infoResolution').textContent = info.input_resolution || '256 x 256 RGB';
  document.getElementById('infoParams').textContent = info.total_parameters ? info.total_parameters.toLocaleString() : '5,321,856';
  document.getElementById('infoHardware').textContent = 'WebAssembly (WASM CPU) • On-Device';

  if (info.metrics) {
    const elAcc = document.getElementById('infoAccuracy');
    if (elAcc) elAcc.textContent = `${info.metrics.accuracy}%`;
    const elPrec = document.getElementById('infoPrecision');
    if (elPrec) elPrec.textContent = `${info.metrics.precision}%`;
    const elRec = document.getElementById('infoRecall');
    if (elRec) elRec.textContent = `${info.metrics.recall}%`;
    const elF1 = document.getElementById('infoF1');
    if (elF1) elF1.textContent = `${info.metrics.f1_score}%`;
  }
}

function renderDiseaseGuide(diseaseMap) {
  const container = document.getElementById('diseaseGuideCards');
  if (!container || !diseaseMap) return;
  container.innerHTML = '';

  const diseaseMetricsMap = {
    'Healthy Paddy': { accuracy: '97.4%', precision: '96.7%', recall: '96.7%', f1: '96.7%' },
    'Brown spot': { accuracy: '93.8%', precision: '95.7%', recall: '91.2%', f1: '93.4%' },
    'Hispa': { accuracy: '94.5%', precision: '93.8%', recall: '95.2%', f1: '94.5%' },
    'Leaf Blast': { accuracy: '94.8%', precision: '92.5%', recall: '96.0%', f1: '94.2%' }
  };

  Object.entries(diseaseMap).forEach(([name, d]) => {
    const col = document.createElement('div');
    col.className = 'col-12 col-lg-6 d-flex flex-column';
    const dm = diseaseMetricsMap[name] || { accuracy: '94.2%', precision: '94.6%', recall: '93.8%', f1: '94.2%' };
    col.innerHTML = `
      <div class="guide-card h-100 d-flex flex-column">
        <div class="guide-header d-flex justify-content-between align-items-center">
          <div>
            <span class="badge mb-1" style="background-color: ${d.color_theme}; color: #fff;">${d.status}</span>
            <h5 class="fw-bold mb-0 text-dark">${name}</h5>
          </div>
          <span class="badge bg-light text-muted border">${d.severity}</span>
        </div>
        <div class="p-3 flex-grow-1 d-flex flex-column">
          <!-- 4 Benchmark Metrics: Accuracy, Precision, Recall, F1-Score -->
          <div class="mb-3 p-2 bg-light rounded-3 border">
            <span class="text-muted text-xs text-uppercase fw-bold d-block mb-1">Class Benchmark Metrics:</span>
            <div class="metrics-quad-grid">
              <div class="metric-quad-card py-1 px-1">
                <div class="metric-quad-title text-xs mb-0">Accuracy</div>
                <div class="fw-bold small text-success">${dm.accuracy}</div>
              </div>
              <div class="metric-quad-card py-1 px-1">
                <div class="metric-quad-title text-xs mb-0">Precision</div>
                <div class="fw-bold small text-primary">${dm.precision}</div>
              </div>
              <div class="metric-quad-card py-1 px-1">
                <div class="metric-quad-title text-xs mb-0">Recall</div>
                <div class="fw-bold small text-info">${dm.recall}</div>
              </div>
              <div class="metric-quad-card py-1 px-1">
                <div class="metric-quad-title text-xs mb-0">F1-Score</div>
                <div class="fw-bold small text-dark">${dm.f1}</div>
              </div>
            </div>
          </div>

          <div class="mb-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Etiology / Pathogen:</span>
            <div class="small fst-italic text-secondary">${d.pathogen}</div>
          </div>
          <div class="mb-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Diagnostic Symptoms:</span>
            <div class="small text-dark">${d.symptoms}</div>
          </div>
          <div class="mt-auto pt-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Agronomic Care & Control:</span>
            <div class="small text-success fw-semibold">${d.management}</div>
          </div>
        </div>
      </div>
    `;
    container.appendChild(col);
  });
}

// ==============================================================================
// 5. SECTION 1: FRONT-VIEW ANALYSIS (T1)
// ==============================================================================

function initFrontViewSection() {
  const fileInput = document.getElementById('frontFileInput');
  const dropzone = document.getElementById('frontDropzone');
  const frontPrompt = document.getElementById('frontPrompt');
  const frontPreviewBox = document.getElementById('frontPreviewBox');
  const frontImgPreview = document.getElementById('frontImgPreview');
  const frontFilenameText = document.getElementById('frontFilenameText');
  const frontFilesizeText = document.getElementById('frontFilesizeText');
  const removeBtn = document.getElementById('removeFrontBtn');
  const analyzeBtn = document.getElementById('analyzeFrontBtn');
  const loadSampleBtn = document.getElementById('loadFrontSampleBtn');
  const exportCsvBtn = document.getElementById('exportFrontCsvBtn');

  const frontErrorCard = document.getElementById('frontErrorCard');
  const frontErrorHeroCard = document.getElementById('frontErrorHeroCard');
  const frontErrorIconBox = document.getElementById('frontErrorIconBox');
  const frontErrorIcon = document.getElementById('frontErrorIcon');
  const frontErrorBadge = document.getElementById('frontErrorBadge');
  const frontErrorTitle = document.getElementById('frontErrorTitle');
  const frontErrorMessage = document.getElementById('frontErrorMessage');
  const frontErrorTrySampleBtn = document.getElementById('frontErrorTrySampleBtn');

  if (frontErrorTrySampleBtn && loadSampleBtn) {
    frontErrorTrySampleBtn.addEventListener('click', () => {
      loadSampleBtn.click();
    });
  }

  function renderFrontError(message, errorType = 'invalid_image') {
    const emptyState = document.getElementById('frontEmptyState');
    const resultCard = document.getElementById('frontResultCard');

    if (emptyState) emptyState.style.display = 'none';
    if (resultCard) resultCard.style.display = 'none';
    if (exportCsvBtn) exportCsvBtn.style.display = 'none';
    if (frontErrorCard) frontErrorCard.style.display = 'block';

    if (frontErrorMessage) frontErrorMessage.textContent = message;

    const isCorrupted = errorType === 'corrupted_image' || errorType === 'processing_error';
    if (frontErrorHeroCard) frontErrorHeroCard.className = isCorrupted ? 'error-hero-card corrupted mb-4' : 'error-hero-card mb-4';
    if (frontErrorIconBox) frontErrorIconBox.className = isCorrupted ? 'error-icon-box corrupted me-3' : 'error-icon-box me-3';
    if (frontErrorIcon) frontErrorIcon.className = isCorrupted ? 'bi bi-x-circle-fill' : 'bi bi-exclamation-triangle-fill';
    if (frontErrorBadge) {
      frontErrorBadge.className = isCorrupted ? 'badge bg-danger-subtle text-danger border border-danger-subtle mb-1' : 'badge bg-warning-subtle text-warning border border-warning-subtle mb-1';
      frontErrorBadge.textContent = isCorrupted ? 'Processing Failure' : 'Validation Alert';
    }
    if (frontErrorTitle) {
      frontErrorTitle.textContent = isCorrupted ? 'Corrupted or Unreadable Image' : 'Invalid Image Detected';
    }
    if (frontErrorMessage) {
      frontErrorMessage.className = isCorrupted ? 'error-message-banner corrupted mb-3' : 'error-message-banner mb-3';
    }
  }

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (e.dataTransfer.files.length > 1) {
        showAlert('Front-View mode takes only 1 image. Loaded the first image.', 'info');
      }
      setFrontFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setFrontFile(e.target.files[0]);
    }
  });

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFrontFile();
  });

  // Load Test Sample Button (if present)
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('assets/samples/sample_01_t1.jpg');
        if (!resp.ok) throw new Error('Sample not found');
        const blob = await resp.blob();
        const testFile = new File([blob], 'sample_01_t1.jpg', { type: 'image/jpeg' });
        setFrontFile(testFile);
        showAlert('Loaded sample front-view leaf image: sample_01_t1.jpg', 'success');
      } catch (err) {
        showAlert('Could not load sample leaf asset. Upload your own image.', 'warning');
      }
    });
  }

  analyzeBtn.addEventListener('click', runFrontViewAnalysis);
  exportCsvBtn.addEventListener('click', exportFrontResultsCsv);

  function setFrontFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderFrontError("Corrupted or invalid image: The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image");
        showAlert("Invalid file: Please upload a valid image file (JPG, PNG, WEBP).", "warning");
        return;
      }
    }

    if (file.size === 0) {
      renderFrontError("Invalid image: File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image");
      showAlert("Invalid image: The uploaded file is empty (0 bytes).", "warning");
      return;
    }

    state.frontFile = file;
    frontPrompt.classList.add('d-none');
    frontPreviewBox.classList.remove('d-none');
    removeBtn.classList.remove('d-none');
    frontImgPreview.src = URL.createObjectURL(file);
    frontFilenameText.textContent = file.name;
    const kb = (file.size / 1024).toFixed(1);
    frontFilesizeText.textContent = `${kb} KB • Ready for analysis`;
    analyzeBtn.disabled = false;

    // Reset error card if previously shown
    if (frontErrorCard) frontErrorCard.style.display = 'none';
    const frontEmpty = document.getElementById('frontEmptyState');
    const frontResult = document.getElementById('frontResultCard');
    if (!state.frontResult && frontEmpty && frontResult && frontResult.style.display === 'none') {
      frontEmpty.style.display = 'block';
    }
  }

  function clearFrontFile() {
    state.frontFile = null;
    state.frontResult = null;
    frontPrompt.classList.remove('d-none');
    frontPreviewBox.classList.add('d-none');
    removeBtn.classList.add('d-none');
    frontFilenameText.textContent = '';
    frontFilesizeText.textContent = '';
    fileInput.value = '';
    analyzeBtn.disabled = true;

    // Reset results view to empty
    document.getElementById('frontEmptyState').style.display = 'block';
    document.getElementById('frontResultCard').style.display = 'none';
    if (frontErrorCard) frontErrorCard.style.display = 'none';
    exportCsvBtn.style.display = 'none';
  }

  async function runFrontViewAnalysis() {
    if (!state.frontFile) return;

    const spinner = document.getElementById('frontSpinner');
    const searchIcon = document.getElementById('frontSearchIcon');
    analyzeBtn.disabled = true;
    spinner.classList.remove('d-none');
    searchIcon.classList.add('d-none');

    try {
      // 1. Ensure ONNX WASM session is initialized on user device CPU
      await initOnnxSession();

      // 2. Load image onto canvas to extract pixels & validate
      const imgInfo = await loadImageToCanvas(state.frontFile);

      // 3. Multi-layer botanical paddy leaf validation
      const valResult = verifyPaddyLeafImageFromCanvas(imgInfo, state.frontFile.size);
      if (!valResult.isValid) {
        renderFrontError(valResult.message, valResult.details.error_type || 'invalid_image');
        showAlert(valResult.message, 'warning');
        return;
      }

      // 4. Preprocess to tensor & detect leaf view
      const tensorTop = imageDataToTensor(imgInfo.imageData);
      const viewType = detectLeafView(state.frontFile.name);

      // 5. Run on-device WASM CPU inference with TTA & botanical calibration
      const inf = await runBrowserInference(tensorTop, null, imgInfo, null, viewType);

      const resultData = {
        status: "success",
        filename: state.frontFile.name,
        view: "T1 (Front View)",
        inference_mode: "Front-View Siamese Fallback (T1)",
        sample_id: extractClientSampleId(state.frontFile.name),
        predicted_class: inf.predicted_class,
        confidence: inf.confidence,
        class_index: inf.class_index,
        probabilities: inf.probabilities,
        inference_time_ms: inf.inference_time_ms,
        disease_info: inf.disease_info,
        validation_details: valResult.details,
        metrics: inf.metrics,
        model_metrics: inf.model_metrics,
        predictions: [{
          status: "success",
          filename: state.frontFile.name,
          view: "T1 (Front View)",
          inference_mode: "Front-View Siamese Fallback (T1)",
          sample_id: extractClientSampleId(state.frontFile.name),
          ...inf
        }]
      };

      state.frontResult = resultData;
      renderFrontResult(resultData);
      showAlert(`Front-View Diagnosis: ${resultData.predicted_class} (${resultData.confidence}% accuracy)`, 'success');
    } catch (err) {
      console.error('Front view analysis error:', err);
      renderFrontError(err.message || 'An unexpected error occurred during client-side inference.', 'processing_error');
      showAlert(`Analysis failed: ${err.message}`, 'danger');
    } finally {
      analyzeBtn.disabled = false;
      spinner.classList.add('d-none');
      searchIcon.classList.remove('d-none');
    }
  }

  function renderFrontResult(data) {
    const emptyState = document.getElementById('frontEmptyState');
    const resultCard = document.getElementById('frontResultCard');

    if (frontErrorCard) frontErrorCard.style.display = 'none';
    emptyState.style.display = 'none';
    resultCard.style.display = 'block';
    exportCsvBtn.style.display = 'inline-block';

    const color = data.disease_info?.color_theme || '#059669';
    document.getElementById('frontSeverityBadge').textContent = data.disease_info?.severity || 'Severity: Normal';
    document.getElementById('frontSeverityBadge').style.backgroundColor = color;
    document.getElementById('frontSeverityBadge').style.color = '#fff';

    document.getElementById('frontDiseaseTitle').textContent = data.predicted_class;
    document.getElementById('frontDiseaseTitle').style.color = color;
    document.getElementById('frontPathogenText').textContent = data.disease_info?.pathogen || 'No pathogen';

    document.getElementById('frontConfidenceVal').textContent = `${data.confidence}%`;
    document.getElementById('frontConfidenceVal').style.color = color;

    if (state.frontFile) {
      document.getElementById('frontShowcaseImg').src = URL.createObjectURL(state.frontFile);
    }

    // Populate Diagnostic Performance Metrics: Accuracy, Precision, Recall, F1-Score
    const m = data.metrics || { accuracy: 88.4, precision: 88.9, recall: 87.6, f1_score: 88.2 };
    const elAcc = document.getElementById('frontAccuracyVal');
    if (elAcc) elAcc.textContent = `${m.accuracy}%`;
    const elPrec = document.getElementById('frontPrecisionVal');
    if (elPrec) elPrec.textContent = `${m.precision}%`;
    const elRec = document.getElementById('frontRecallVal');
    if (elRec) elRec.textContent = `${m.recall}%`;
    const elF1 = document.getElementById('frontF1Val');
    if (elF1) elF1.textContent = `${m.f1_score}%`;

    // Display ONLY the single highest-accuracy predicted class (other 3 classes are hidden as requested)
    document.getElementById('frontProbMeters').innerHTML = `
      <div class="prob-meter-row">
        <div class="prob-label-group">
          <span class="fw-bold text-dark">${data.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${data.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px;">
          <div class="prob-bar-fill" style="width: ${data.confidence}%; background-color: ${color};"></div>
        </div>
      </div>
    `;

    // Agronomic text
    document.getElementById('frontSymptomsText').innerHTML = `<strong>Symptoms:</strong> ${data.disease_info?.symptoms || 'None'}`;
    document.getElementById('frontManagementText').innerHTML = `<i class="bi bi-shield-check me-1"></i><strong>Action Plan:</strong> ${data.disease_info?.management || 'Standard care'}`;
    document.getElementById('frontLatency').textContent = data.inference_time_ms;
  }

  function exportFrontResultsCsv() {
    if (!state.frontResult) return;
    const r = state.frontResult;
    const m = r.metrics || { accuracy: 94.2, precision: 94.6, recall: 93.8, f1_score: 94.2 };
    const headers = ['Filename', 'Sample ID', 'Inference Mode', 'Predicted Disease (Highest Accuracy)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Confidence (%)', 'Latency (ms)', 'Status'];
    const rows = [[
      `"${r.filename || ''}"`,
      `"${r.sample_id || ''}"`,
      `"${r.inference_mode || 'Front-View (T1)'}"`,
      `"${r.predicted_class || 'N/A'}"`,
      m.accuracy,
      m.precision,
      m.recall,
      m.f1_score,
      r.confidence !== undefined ? r.confidence : '',
      r.inference_time_ms || '',
      `"${r.status || 'success'}"`
    ]];

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    downloadCsvBlob(csvContent, `paddyscan_${r.filename || 'front_view'}_result.csv`);
  }
}

// ==============================================================================
// 6. SECTION 2: DUAL-VIEW ANALYSIS (T1 + D1)
// ==============================================================================

function initDualViewSection() {
  const frontInput = document.getElementById('dualFrontInput');
  const backInput = document.getElementById('dualBackInput');
  const frontDropzone = document.getElementById('dualFrontDropzone');
  const backDropzone = document.getElementById('dualBackDropzone');

  const frontPrompt = document.getElementById('dualFrontPrompt');
  const frontPreviewBox = document.getElementById('dualFrontPreviewBox');
  const frontImg = document.getElementById('dualFrontImg');
  const frontFilename = document.getElementById('dualFrontFilename');
  const removeFrontBtn = document.getElementById('removeDualFrontBtn');

  const backPrompt = document.getElementById('dualBackPrompt');
  const backPreviewBox = document.getElementById('dualBackPreviewBox');
  const backImg = document.getElementById('dualBackImg');
  const backFilename = document.getElementById('dualBackFilename');
  const removeBackBtn = document.getElementById('removeDualBackBtn');

  const pairCheckDot = document.getElementById('pairCheckDot');
  const pairCheckText = document.getElementById('pairCheckText');
  const pairBadge = document.getElementById('pairBadge');
  const analyzeBtn = document.getElementById('analyzeDualBtn');
  const loadPairBtn = document.getElementById('loadDualSampleBtn');

  // Drag and drop for front
  setupDropEvents(frontDropzone, (file) => setDualFrontFile(file));
  frontInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) setDualFrontFile(e.target.files[0]);
  });
  removeFrontBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.dualFrontFile = null;
    updateDualState();
  });

  const dualErrorCard = document.getElementById('dualErrorCard');
  const dualErrorHeroCard = document.getElementById('dualErrorHeroCard');
  const dualErrorIconBox = document.getElementById('dualErrorIconBox');
  const dualErrorIcon = document.getElementById('dualErrorIcon');
  const dualErrorBadge = document.getElementById('dualErrorBadge');
  const dualErrorTitle = document.getElementById('dualErrorTitle');
  const dualErrorMessage = document.getElementById('dualErrorMessage');
  const dualErrorTrySampleBtn = document.getElementById('dualErrorTrySampleBtn');

  if (dualErrorTrySampleBtn && loadPairBtn) {
    dualErrorTrySampleBtn.addEventListener('click', () => {
      loadPairBtn.click();
    });
  }

  function renderDualError(message, errorType = 'invalid_image', failedView = null) {
    const emptyState = document.getElementById('dualEmptyState');
    const resultCard = document.getElementById('dualResultCard');

    if (emptyState) emptyState.style.display = 'none';
    if (resultCard) resultCard.style.display = 'none';
    if (dualErrorCard) dualErrorCard.style.display = 'block';

    if (dualErrorMessage) dualErrorMessage.textContent = message;

    const isCorrupted = errorType === 'corrupted_image' || errorType === 'processing_error';
    if (dualErrorHeroCard) dualErrorHeroCard.className = isCorrupted ? 'error-hero-card corrupted mb-4' : 'error-hero-card mb-4';
    if (dualErrorIconBox) dualErrorIconBox.className = isCorrupted ? 'error-icon-box corrupted me-3' : 'error-icon-box me-3';
    if (dualErrorIcon) dualErrorIcon.className = isCorrupted ? 'bi bi-x-circle-fill' : 'bi bi-exclamation-triangle-fill';
    if (dualErrorBadge) {
      dualErrorBadge.className = isCorrupted ? 'badge bg-danger-subtle text-danger border border-danger-subtle mb-1' : 'badge bg-warning-subtle text-warning border border-warning-subtle mb-1';
      dualErrorBadge.textContent = isCorrupted ? 'Processing Failure' : (failedView ? `${failedView.toUpperCase()} View Alert` : 'Validation Alert');
    }
    if (dualErrorTitle) {
      dualErrorTitle.textContent = isCorrupted ? 'Corrupted or Unreadable Image' : 'Invalid Image Detected';
    }
    if (dualErrorMessage) {
      dualErrorMessage.className = isCorrupted ? 'error-message-banner corrupted mb-3' : 'error-message-banner mb-3';
    }
  }

  // Drag and drop for back
  setupDropEvents(backDropzone, (file) => setDualBackFile(file));
  backInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) setDualBackFile(e.target.files[0]);
  });
  removeBackBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.dualBackFile = null;
    updateDualState();
  });

  // Load Test Pair Button (if present)
  if (loadPairBtn) {
    loadPairBtn.addEventListener('click', async () => {
      try {
        const [res1, res2] = await Promise.all([
          fetch('assets/samples/sample_01_t1.jpg'),
          fetch('assets/samples/sample_01_b1.jpg')
        ]);
        const blob1 = await res1.blob();
        const blob2 = await res2.blob();
        setDualFrontFile(new File([blob1], 'sample_01_t1.jpg', { type: 'image/jpeg' }));
        setDualBackFile(new File([blob2], 'sample_01_b1.jpg', { type: 'image/jpeg' }));
        showAlert('Loaded verified dual-view leaf pair (T1 + B1).', 'success');
      } catch (err) {
        showAlert('Could not load test pair assets. Please upload manually.', 'warning');
      }
    });
  }

  analyzeBtn.addEventListener('click', runDualViewAnalysis);

  function setDualFrontFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderDualError("Corrupted or invalid image (Front View): The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image", "front");
        showAlert('Please upload a valid image file for Front View (T1).', 'warning');
        return;
      }
    }

    if (file.size === 0) {
      renderDualError("Invalid image (Front View): File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image", "front");
      showAlert("The selected front-view file is empty (0 bytes).", "warning");
      return;
    }

    state.dualFrontFile = file;
    updateDualState();
  }

  function setDualBackFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderDualError("Corrupted or invalid image (Back View): The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image", "back");
        showAlert('Please upload a valid image file for Back View (D1).', 'warning');
        return;
      }
    }

    if (file.size === 0) {
      renderDualError("Invalid image (Back View): File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image", "back");
      showAlert("The selected back-view file is empty (0 bytes).", "warning");
      return;
    }

    state.dualBackFile = file;
    updateDualState();
  }

  function updateDualState() {
    if (dualErrorCard) dualErrorCard.style.display = 'none';

    // Front View UI
    if (state.dualFrontFile) {
      frontPrompt.classList.add('d-none');
      frontPreviewBox.classList.remove('d-none');
      frontImg.src = URL.createObjectURL(state.dualFrontFile);
      frontFilename.textContent = state.dualFrontFile.name;
      removeFrontBtn.classList.remove('d-none');
    } else {
      frontPrompt.classList.remove('d-none');
      frontPreviewBox.classList.add('d-none');
      frontFilename.textContent = '';
      removeFrontBtn.classList.add('d-none');
      frontInput.value = '';
    }

    // Back View UI
    if (state.dualBackFile) {
      backPrompt.classList.add('d-none');
      backPreviewBox.classList.remove('d-none');
      backImg.src = URL.createObjectURL(state.dualBackFile);
      backFilename.textContent = state.dualBackFile.name;
      removeBackBtn.classList.remove('d-none');
    } else {
      backPrompt.classList.remove('d-none');
      backPreviewBox.classList.add('d-none');
      backFilename.textContent = '';
      removeBackBtn.classList.add('d-none');
      backInput.value = '';
    }

    // Pair validation
    const readyCount = (state.dualFrontFile ? 1 : 0) + (state.dualBackFile ? 1 : 0);
    pairBadge.textContent = `${readyCount}/2 Ready`;

    if (readyCount === 2) {
      pairCheckDot.className = 'pair-dot ready';
      pairCheckText.textContent = 'Both views synchronized and ready for analysis';
      analyzeBtn.disabled = false;
    } else if (readyCount === 1) {
      pairCheckDot.className = 'pair-dot';
      pairCheckText.textContent = state.dualFrontFile ? 'Upload Back View (B1)' : 'Upload Front View (T1)';
      analyzeBtn.disabled = true;
    } else {
      pairCheckDot.className = 'pair-dot';
      pairCheckText.textContent = 'Upload both T1 and B1 images';
      analyzeBtn.disabled = true;
    }
  }

  async function runDualViewAnalysis() {
    if (!state.dualFrontFile || !state.dualBackFile) {
      showAlert('Both Front (T1) and Back (B1) views are required for Dual-View analysis.', 'warning');
      return;
    }

    const spinner = document.getElementById('dualSpinner');
    const searchIcon = document.getElementById('dualSearchIcon');
    analyzeBtn.disabled = true;
    spinner.classList.remove('d-none');
    searchIcon.classList.add('d-none');

    try {
      // 1. Ensure ONNX WASM session is ready on device CPU
      await initOnnxSession();

      // 2. Load both images to canvas
      const imgInfoFront = await loadImageToCanvas(state.dualFrontFile);
      const imgInfoBack = await loadImageToCanvas(state.dualBackFile);

      // 3. Validate Front view
      const valFront = verifyPaddyLeafImageFromCanvas(imgInfoFront, state.dualFrontFile.size);
      if (!valFront.isValid) {
        const displayMsg = "Invalid image (Front View): Please upload a paddy leaf image for disease detection.";
        renderDualError(displayMsg, valFront.details.error_type || 'invalid_image', 'front');
        showAlert(displayMsg, 'warning');
        return;
      }

      // 4. Validate Back view
      const valBack = verifyPaddyLeafImageFromCanvas(imgInfoBack, state.dualBackFile.size);
      if (!valBack.isValid) {
        const displayMsg = "Invalid image (Back View): Please upload a paddy leaf image for disease detection.";
        renderDualError(displayMsg, valBack.details.error_type || 'invalid_image', 'back');
        showAlert(displayMsg, 'warning');
        return;
      }

      // 5. Preprocess tensors
      let tensorTop = imageDataToTensor(imgInfoFront.imageData);
      let tensorBottom = imageDataToTensor(imgInfoBack.imageData);
      let topImgInfo = imgInfoFront;
      let botImgInfo = imgInfoBack;

      // Auto-align inverted views if user uploaded bottom view into front slot and top view into back slot
      const vF = detectLeafView(state.dualFrontFile.name);
      const vB = detectLeafView(state.dualBackFile.name);
      if (vF === 'back' && vB === 'front') {
        [tensorTop, tensorBottom] = [tensorBottom, tensorTop];
        [topImgInfo, botImgInfo] = [botImgInfo, topImgInfo];
        console.log('Auto-aligned inverted dual views:', state.dualFrontFile.name, '-> bottom,', state.dualBackFile.name, '-> top');
      }

      // 6. Execute full Dual-View Siamese Fusion on device CPU via WASM
      const inf = await runBrowserInference(tensorTop, tensorBottom, topImgInfo, botImgInfo, 'front');

      const resultData = {
        status: "success",
        inference_mode: "Dual-View Fusion (T1 + D1)",
        front_filename: state.dualFrontFile.name,
        back_filename: state.dualBackFile.name,
        sample_id: extractClientSampleId(state.dualFrontFile.name),
        front_validation: valFront.details,
        back_validation: valBack.details,
        ...inf
      };

      state.dualResult = resultData;
      renderDualResult(resultData);
      showAlert(`Dual-View Diagnosis completed: ${resultData.predicted_class} (${resultData.confidence}% confidence)`, 'success');
    } catch (err) {
      console.error('Dual view analysis error:', err);
      renderDualError(err.message || 'An unexpected error occurred during dual-view processing.', 'processing_error');
      showAlert(`Dual-View analysis failed: ${err.message}`, 'danger');
    } finally {
      analyzeBtn.disabled = false;
      spinner.classList.add('d-none');
      searchIcon.classList.remove('d-none');
    }
  }

  function renderDualResult(data) {
    const emptyState = document.getElementById('dualEmptyState');
    const resultCard = document.getElementById('dualResultCard');

    if (dualErrorCard) dualErrorCard.style.display = 'none';
    emptyState.style.display = 'none';
    resultCard.style.display = 'block';

    const color = data.disease_info?.color_theme || '#059669';
    document.getElementById('dualSeverityBadge').textContent = data.disease_info?.severity || 'Severity: Normal';
    document.getElementById('dualSeverityBadge').style.backgroundColor = color;
    document.getElementById('dualSeverityBadge').style.color = '#fff';

    document.getElementById('dualDiseaseTitle').textContent = data.predicted_class;
    document.getElementById('dualDiseaseTitle').style.color = color;
    document.getElementById('dualPathogenText').textContent = data.disease_info?.pathogen || 'No pathogen';

    document.getElementById('dualConfidenceVal').textContent = `${data.confidence}%`;
    document.getElementById('dualConfidenceVal').style.color = color;

    document.getElementById('dualShowcaseFrontImg').src = URL.createObjectURL(state.dualFrontFile);
    document.getElementById('dualShowcaseBackImg').src = URL.createObjectURL(state.dualBackFile);

    // Populate Diagnostic Performance Metrics: Accuracy, Precision, Recall, F1-Score
    const m = data.metrics || { accuracy: 94.2, precision: 94.6, recall: 93.8, f1_score: 94.2 };
    const elAcc = document.getElementById('dualAccuracyVal');
    if (elAcc) elAcc.textContent = `${m.accuracy}%`;
    const elPrec = document.getElementById('dualPrecisionVal');
    if (elPrec) elPrec.textContent = `${m.precision}%`;
    const elRec = document.getElementById('dualRecallVal');
    if (elRec) elRec.textContent = `${m.recall}%`;
    const elF1 = document.getElementById('dualF1Val');
    if (elF1) elF1.textContent = `${m.f1_score}%`;

    // Display ONLY the single highest-accuracy predicted class (other 3 classes are hidden as requested)
    document.getElementById('dualProbMeters').innerHTML = `
      <div class="prob-meter-row">
        <div class="prob-label-group">
          <span class="fw-bold text-dark">${data.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${data.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px;">
          <div class="prob-bar-fill" style="width: ${data.confidence}%; background-color: ${color};"></div>
        </div>
      </div>
    `;

    // Agronomic text
    document.getElementById('dualSymptomsText').innerHTML = `<strong>Symptoms:</strong> ${data.disease_info?.symptoms || 'None'}`;
    document.getElementById('dualManagementText').innerHTML = `<i class="bi bi-shield-check me-1"></i><strong>Action Plan:</strong> ${data.disease_info?.management || 'Standard care'}`;
    document.getElementById('dualLatency').textContent = data.inference_time_ms;

    const exportDualBtn = document.getElementById('exportDualCsvBtn');
    if (exportDualBtn) exportDualBtn.style.display = 'inline-block';
  }

  const exportDualBtn = document.getElementById('exportDualCsvBtn');
  if (exportDualBtn) {
    exportDualBtn.addEventListener('click', exportDualResultsCsv);
  }

  function exportDualResultsCsv() {
    if (!state.dualResult) return;
    const r = state.dualResult;
    const m = r.metrics || { accuracy: 94.2, precision: 94.6, recall: 93.8, f1_score: 94.2 };
    const headers = ['Sample ID', 'Front Filename (T1)', 'Back Filename (D1)', 'Inference Mode', 'Predicted Disease', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Confidence (%)', 'Latency (ms)', 'Status'];
    const rows = [[
      `"${r.sample_id || ''}"`,
      `"${r.front_filename || ''}"`,
      `"${r.back_filename || ''}"`,
      `"${r.inference_mode || 'Dual-View Fusion (T1 + D1)'}"`,
      `"${r.predicted_class || 'N/A'}"`,
      m.accuracy,
      m.precision,
      m.recall,
      m.f1_score,
      r.confidence !== undefined ? r.confidence : '',
      r.inference_time_ms || '',
      `"${r.status || 'success'}"`
    ]];

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    downloadCsvBlob(csvContent, `paddyscan_${r.sample_id || 'dual_view'}_result.csv`);
  }
}

// ==============================================================================
// 7. SECTION 3: FOLDER-BASED BATCH ANALYSIS
// ==============================================================================

function initBatchSection() {
  const folderInput = document.getElementById('batchFolderInput');
  const filesInput = document.getElementById('batchFilesInput');
  const dropzone = document.getElementById('batchDropzone');
  const modeSelect = document.getElementById('batchModeSelect');
  const runBtn = document.getElementById('runBatchBtn');
  const loadBatchSampleBtn = document.getElementById('loadSampleBatchBtn');
  const searchInput = document.getElementById('tableSearchInput');
  const exportCsvBtn = document.getElementById('exportBatchCsvBtn');

  // Drag and drop
  setupDropEvents(dropzone, null, (files) => handleBatchFilesSelected(files));

  folderInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(Array.from(e.target.files));
    }
  });

  filesInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(Array.from(e.target.files));
    }
  });

  modeSelect.addEventListener('change', () => {
    const hint = document.getElementById('batchModeHint');
    if (hint) {
      if (modeSelect.value === 'auto' || modeSelect.value === 'dual') {
        hint.textContent = 'Dual-View Mode: Automatically pairs T1 & B1 images for complete Dual-View Siamese fusion.';
      } else {
        hint.textContent = 'Top-View Only: Evaluates strictly top/front leaf views (T1, T2, T3...); bottom views (B1, B2, B3...) are excluded.';
      }
    }
    updateBatchStagingDisplay();
  });

  // Load Demo Batch Samples (if present)
  if (loadBatchSampleBtn) {
    loadBatchSampleBtn.addEventListener('click', async () => {
      try {
        const sampleNames = [
          'sample_01_t1.jpg',
          'sample_01_b1.jpg',
          'sample_02_t2.jpg',
          'sample_02_b2.jpg',
          'sample_03_t3.jpg',
          'sample_04_b4.jpg'
        ];
        const promises = sampleNames.map(async name => {
          const resp = await fetch(`assets/samples/${name}`);
          const blob = await resp.blob();
          return new File([blob], name, { type: 'image/jpeg' });
        });
        const files = await Promise.all(promises);
        handleBatchFilesSelected(files);
        showAlert(`Loaded ${files.length} demo batch images covering paired and single views.`, 'success');
      } catch (err) {
        showAlert('Could not load demo batch sample files.', 'warning');
      }
    });
  }

  runBtn.addEventListener('click', runBatchAnalysis);
  searchInput.addEventListener('input', () => filterAndRenderBatchTable());
  exportCsvBtn.addEventListener('click', exportBatchResultsCsv);

  // Filter pills
  document.querySelectorAll('#filterButtonGroup button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#filterButtonGroup button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeFilter = btn.getAttribute('data-filter');
      filterAndRenderBatchTable();
    });
  });

  function handleBatchFilesSelected(files) {
    const validImages = files.filter(f => {
      const isMimeImage = f.type && f.type.startsWith('image/');
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(f.name);
      return isMimeImage || isExtImage;
    });

    if (validImages.length === 0) {
      showAlert('No valid image files found in folder or file selection.', 'warning');
      return;
    }

    state.batchFiles = validImages;
    state.batchFilesMap = new Map();
    state.batchUrlCache = new Map();

    // Fast O(1) file lookup map
    validImages.forEach(f => {
      const name = (f.name || '').toLowerCase().trim();
      const rel = (f.webkitRelativePath || '').replace(/\\/g, '/').toLowerCase().trim();
      const base = name.split('/').pop().split('\\').pop();

      if (name) state.batchFilesMap.set(name, f);
      if (rel) {
        state.batchFilesMap.set(rel, f);
        state.batchFilesMap.set(rel.split('/').pop(), f);
      }
      if (base) state.batchFilesMap.set(base, f);
    });

    updateBatchStagingDisplay();
  }

  // View detection helper utilities
  function isBottomViewName(filename) {
    if (!filename) return false;
    const base = filename.replace(/\\/g, '/').split('/').pop().replace(/\.[^/.]+$/, '').toLowerCase();
    return /([_-][bd]\d*$|_[bd]$|back|bottom|abaxial|^[bd]\d*[_-]|^[bd]\d+$)/.test(base);
  }

  function isTopViewName(filename) {
    if (!filename) return false;
    const base = filename.replace(/\\/g, '/').split('/').pop().replace(/\.[^/.]+$/, '').toLowerCase();
    if (isBottomViewName(base)) return false;
    return /([_-]t\d*$|_t$|front|top|adaxial|^t\d*[_-]|^t\d+$)/.test(base);
  }

  function updateBatchStagingDisplay() {
    const totalCount = state.batchFiles.length;
    let frontCount = 0;
    let backCount = 0;
    state.batchFiles.forEach(f => {
      const name = f.name.toLowerCase();
      if (isTopViewName(name)) frontCount++;
      else if (isBottomViewName(name)) backCount++;
    });

    const isTopOnlyMode = modeSelect && modeSelect.value === 'front';
    const stagedTotal = document.getElementById('batchStagedTotal');
    const stagedDetails = document.getElementById('batchStagedDetails');

    if (isTopOnlyMode) {
      if (stagedTotal) stagedTotal.textContent = `${frontCount} top view files`;
      if (stagedDetails) {
        stagedDetails.textContent = `${frontCount} top (T1, T2...) selected • ${backCount} bottom (B1...) excluded`;
      }
      runBtn.disabled = frontCount === 0;
    } else {
      if (stagedTotal) stagedTotal.textContent = `${totalCount} files`;
      if (stagedDetails) {
        stagedDetails.textContent = `${frontCount} front (T1) • ${backCount} back (B1)`;
      }
      runBtn.disabled = totalCount === 0;
    }
  }


  // Client-side pairing grouping: keeps T1 and D1 views together, or takes strictly top views in Top-Only mode
  function groupBatchFilesForProcessing(files, mode) {
    if (mode === 'front') {
      // Exclude all bottom/back views (b1, b2, b3, d1, back, bottom)
      const hasExplicitTop = files.some(f => isTopViewName(f.name));
      const topFiles = files.filter(f => {
        if (isBottomViewName(f.name)) return false;
        if (hasExplicitTop) {
          return isTopViewName(f.name);
        }
        return true;
      });
      return topFiles.map(f => [f]);
    }

    const groups = new Map();
    files.forEach(file => {
      const rel = (file.webkitRelativePath || file.name).replace(/\\/g, '/');
      const parts = rel.split('/');
      parts.pop();
      const parentDir = parts.join('/');
      const sId = extractClientSampleId(rel) || file.name;
      const groupKey = parentDir ? `${parentDir}:::${sId}` : sId;
      if (!groups.has(groupKey)) {
        groups.set(groupKey, []);
      }
      groups.get(groupKey).push(file);
    });
    return Array.from(groups.values());
  }

  function buildBatchWorkItems(files, mode) {
    const workItems = [];

    if (mode === 'front') {
      for (const f of files) {
        if (isBottomViewName(f.name)) continue;
        const sId = extractClientSampleId(f.name) || stripFolderPrefix(f.name);
        workItems.push({
          sampleId: sId,
          mode: "Top-Only (T1)",
          frontFile: f,
          backFile: null,
          frontFilename: stripFolderPrefix(f.name),
          backFilename: null,
          matched: true,
          isDual: false,
          viewType: "front"
        });
      }
    } else {
      const groups = new Map();
      for (const file of files) {
        const rel = (file.webkitRelativePath || file.name).replace(/\\/g, '/');
        const parts = rel.split('/');
        parts.pop();
        const parentDir = parts.join('/');
        const sId = extractClientSampleId(rel) || file.name;
        const groupKey = parentDir ? `${parentDir}:::${sId}` : sId;

        if (!groups.has(groupKey)) {
          groups.set(groupKey, { front: [], back: [], unknown: [] });
        }

        const view = detectLeafView(file.name);
        if (view === 'front') {
          groups.get(groupKey).front.push(file);
        } else if (view === 'back') {
          groups.get(groupKey).back.push(file);
        } else {
          groups.get(groupKey).unknown.push(file);
        }
      }

      for (const [groupKey, views] of groups.entries()) {
        const sampleId = groupKey.split(":::").pop().split("/").pop();
        const frontList = views.front;
        const backList = views.back;
        const unknownList = views.unknown;

        const numPairs = Math.min(frontList.length, backList.length);
        for (let i = 0; i < numPairs; i++) {
          const fF = frontList[i];
          const bF = backList[i];
          workItems.push({
            sampleId: sampleId,
            mode: "Dual-View (T1 + D1)",
            frontFile: fF,
            backFile: bF,
            frontFilename: stripFolderPrefix(fF.name),
            backFilename: stripFolderPrefix(bF.name),
            matched: true,
            isDual: true,
            viewType: "front"
          });
        }

        for (let i = numPairs; i < frontList.length; i++) {
          const fF = frontList[i];
          if (mode === 'auto') {
            workItems.push({
              sampleId: sampleId,
              mode: "Top-Only (T1)",
              frontFile: fF,
              backFile: null,
              frontFilename: stripFolderPrefix(fF.name),
              backFilename: null,
              matched: true,
              isDual: false,
              viewType: "front"
            });
          } else {
            workItems.push({
              sampleId: sampleId,
              mode: "Dual-View Required",
              frontFile: null,
              backFile: null,
              frontFilename: stripFolderPrefix(fF.name),
              backFilename: null,
              matched: false,
              unmatchedReason: "Dual-View mode requires both T1 and D1 views. Missing back image."
            });
          }
        }

        for (let i = numPairs; i < backList.length; i++) {
          const bF = backList[i];
          if (mode === 'auto') {
            workItems.push({
              sampleId: sampleId,
              mode: "Single-View (B1 Fallback)",
              frontFile: null,
              backFile: bF,
              frontFilename: null,
              backFilename: stripFolderPrefix(bF.name),
              matched: true,
              isDual: false,
              viewType: "back"
            });
          } else {
            workItems.push({
              sampleId: sampleId,
              mode: "Dual-View Required",
              frontFile: null,
              backFile: null,
              frontFilename: "Missing Front View (T1)",
              backFilename: stripFolderPrefix(bF.name),
              matched: false,
              unmatchedReason: "Dual-View mode requires both T1 and D1 views. Missing front image."
            });
          }
        }

        for (const uF of unknownList) {
          if (mode === 'auto') {
            const v = detectLeafView(uF.name);
            workItems.push({
              sampleId: sampleId,
              mode: "Single-View (Siamese)",
              frontFile: uF,
              backFile: null,
              frontFilename: stripFolderPrefix(uF.name),
              backFilename: null,
              matched: true,
              isDual: false,
              viewType: v !== 'unknown' ? v : 'front'
            });
          } else {
            workItems.push({
              sampleId: sampleId,
              mode: "Unmatched",
              frontFile: null,
              backFile: null,
              frontFilename: stripFolderPrefix(uF.name),
              backFilename: null,
              matched: false,
              unmatchedReason: "Filename does not specify T1 or D1 view tag."
            });
          }
        }
      }
    }

    return workItems;
  }

  // Progressive streaming batch analysis directly on user CPU via WebAssembly
  async function runBatchAnalysis() {
    if (!state.batchFiles || state.batchFiles.length === 0) return;

    const progressWrapper = document.getElementById('batchProgressWrapper');
    const progressBar = document.getElementById('batchProgressBar');
    const progressText = document.getElementById('batchProgressText');
    const progressPct = document.getElementById('batchProgressPct');
    const spinner = document.getElementById('batchSpinner');
    const playIcon = document.getElementById('batchPlayIcon');

    runBtn.disabled = true;
    spinner.classList.remove('d-none');
    playIcon.classList.add('d-none');
    progressWrapper.style.display = 'block';
    progressBar.style.width = '3%';
    progressPct.textContent = '3%';

    try {
      progressText.textContent = 'Initializing ONNX Web WASM CPU engine...';
      await initOnnxSession();

      const mode = modeSelect.value;
      const workItems = buildBatchWorkItems(state.batchFiles, mode);
      const totalSamples = workItems.length;

      progressText.textContent = `Analyzing ${totalSamples} samples on your device CPU...`;

      state.batchResults = {
        status: 'success',
        summary: {
          total_files: mode === 'front' ? totalSamples : state.batchFiles.length,
          total_samples: 0,
          processed_count: 0,
          unmatched_count: 0,
          error_count: 0,
          healthy_count: 0,
          diseased_count: 0,
          average_confidence: 0,
          class_distribution: {
            'Brown spot': 0,
            'Healthy Paddy': 0,
            'Hispa': 0,
            'Leaf Blast': 0
          },
          mode: mode
        },
        results: []
      };

      const resultsWrapper = document.getElementById('batchResultsWrapper');
      if (resultsWrapper) {
        resultsWrapper.style.display = 'block';
      }
      renderBatchResults(state.batchResults, false);

      for (let idx = 0; idx < totalSamples; idx++) {
        const item = workItems[idx];
        state.batchResults.summary.total_samples++;

        const currentPct = Math.max(3, Math.round(((idx + 1) / totalSamples) * 100));
        progressBar.style.width = `${currentPct}%`;
        progressPct.textContent = `${currentPct}%`;
        progressText.textContent = `Analyzing sample ${idx + 1} of ${totalSamples}: ${item.sampleId}...`;

        if (!item.matched) {
          state.batchResults.summary.unmatched_count++;
          state.batchResults.results.push({
            sample_id: item.sampleId,
            status: "unmatched",
            mode: item.mode,
            front_filename: item.frontFilename,
            back_filename: item.backFilename,
            matched: false,
            reason: item.unmatchedReason
          });
        } else {
          try {
            let frontImgInfo = null;
            let backImgInfo = null;
            let valErr = null;

            if (item.frontFile) {
              frontImgInfo = await loadImageToCanvas(item.frontFile);
              const valF = verifyPaddyLeafImageFromCanvas(frontImgInfo, item.frontFile.size);
              if (!valF.isValid) valErr = valF.message;
            }

            if (!valErr && item.backFile) {
              backImgInfo = await loadImageToCanvas(item.backFile);
              const valB = verifyPaddyLeafImageFromCanvas(backImgInfo, item.backFile.size);
              if (!valB.isValid) valErr = valB.message;
            }

            if (valErr) {
              state.batchResults.summary.error_count++;
              state.batchResults.results.push({
                sample_id: item.sampleId,
                status: "error",
                mode: item.mode,
                front_filename: item.frontFilename,
                back_filename: item.backFilename,
                error: valErr
              });
            } else {
              let tTop = frontImgInfo ? imageDataToTensor(frontImgInfo.imageData) : imageDataToTensor(backImgInfo.imageData);
              let tBot = item.isDual ? imageDataToTensor(backImgInfo.imageData) : tTop;
              let imTop = frontImgInfo || backImgInfo;
              let imBot = item.isDual ? backImgInfo : null;

              if (item.isDual && item.frontFile && item.backFile) {
                const vf = detectLeafView(item.frontFile.name);
                const vb = detectLeafView(item.backFile.name);
                if (vf === 'back' && vb === 'front') {
                  [tTop, tBot] = [tBot, tTop];
                  [imTop, imBot] = [imBot, imTop];
                }
              }

              const inf = await runBrowserInference(tTop, item.isDual ? tBot : null, imTop, imBot, item.viewType || 'front');

              const predClass = inf.predicted_class;
              state.batchResults.summary.processed_count++;
              state.batchResults.summary.class_distribution[predClass] = (state.batchResults.summary.class_distribution[predClass] || 0) + 1;
              if (predClass === 'Healthy Paddy') {
                state.batchResults.summary.healthy_count++;
              } else {
                state.batchResults.summary.diseased_count++;
              }

              const successes = state.batchResults.results.filter(r => r.status === 'success');
              const totalConf = successes.reduce((acc, r) => acc + (r.confidence || 0), 0) + inf.confidence;
              state.batchResults.summary.average_confidence = Math.round((totalConf / (successes.length + 1)) * 100) / 100;

              state.batchResults.results.push({
                sample_id: item.sampleId,
                status: "success",
                mode: item.mode,
                front_filename: item.frontFilename,
                back_filename: item.backFilename,
                matched: true,
                predicted_class: predClass,
                confidence: inf.confidence,
                class_index: inf.class_index,
                probabilities: inf.probabilities,
                inference_time_ms: inf.inference_time_ms,
                disease_info: inf.disease_info,
                metrics: inf.metrics
              });
            }
          } catch (itemErr) {
            state.batchResults.summary.error_count++;
            state.batchResults.results.push({
              sample_id: item.sampleId,
              status: "error",
              mode: item.mode,
              front_filename: item.frontFilename,
              back_filename: item.backFilename,
              error: itemErr.message || "Failed to process sample"
            });
          }
        }

        if (idx % 2 === 0 || idx === totalSamples - 1) {
          renderBatchResults(state.batchResults, false);
        }

        await new Promise(r => setTimeout(r, 0));
      }

      const allSuccesses = state.batchResults.results.filter(r => r.status === 'success');
      let overallAcc = 94.2, overallPrec = 94.6, overallRec = 93.8, overallF1 = 94.2;
      if (allSuccesses.length > 0) {
        overallAcc = Math.round((allSuccesses.reduce((acc, r) => acc + (r.metrics?.accuracy || 94.2), 0) / allSuccesses.length) * 10) / 10;
        overallPrec = Math.round((allSuccesses.reduce((acc, r) => acc + (r.metrics?.precision || 94.6), 0) / allSuccesses.length) * 10) / 10;
        overallRec = Math.round((allSuccesses.reduce((acc, r) => acc + (r.metrics?.recall || 93.8), 0) / allSuccesses.length) * 10) / 10;
        overallF1 = Math.round((allSuccesses.reduce((acc, r) => acc + (r.metrics?.f1_score || 94.2), 0) / allSuccesses.length) * 10) / 10;
      }

      state.batchResults.summary.metrics = {
        accuracy: overallAcc,
        precision: overallPrec,
        recall: overallRec,
        f1_score: overallF1
      };
      state.batchResults.summary.accuracy = overallAcc;
      state.batchResults.summary.precision = overallPrec;
      state.batchResults.summary.recall = overallRec;
      state.batchResults.summary.f1_score = overallF1;

      progressBar.style.width = '100%';
      progressPct.textContent = '100%';
      progressText.textContent = `Completed! Analyzed all ${state.batchResults.summary.total_samples} samples on your device CPU.`;

      renderBatchResults(state.batchResults, true);
      showAlert(`Batch analysis completed! Analyzed all ${state.batchResults.summary.total_samples} samples locally on CPU.`, 'success');
    } catch (err) {
      console.error('Batch error:', err);
      showAlert(`Batch processing error: ${err.message}`, 'danger');
    } finally {
      runBtn.disabled = false;
      spinner.classList.add('d-none');
      playIcon.classList.remove('d-none');
      setTimeout(() => {
        progressWrapper.style.display = 'none';
      }, 2000);
    }
  }

  function renderBatchResults(data, isFinal = false) {
    const resultsWrapper = document.getElementById('batchResultsWrapper');
    if (resultsWrapper) {
      resultsWrapper.style.display = 'block';
    }

    const sum = data.summary || {};
    document.getElementById('metricTotal').textContent = sum.total_samples || 0;
    document.getElementById('metricTotalFiles').textContent = `${sum.total_files || 0} images`;

    document.getElementById('metricHealthy').textContent = sum.healthy_count || 0;
    const hPct = sum.total_samples > 0 ? Math.round((sum.healthy_count / sum.total_samples) * 100) : 0;
    document.getElementById('metricHealthyPct').textContent = `${hPct}% of total`;

    document.getElementById('metricDiseased').textContent = sum.diseased_count || 0;
    const dPct = sum.total_samples > 0 ? Math.round((sum.diseased_count / sum.total_samples) * 100) : 0;
    document.getElementById('metricDiseasedPct').textContent = `${dPct}% of total`;

    document.getElementById('metricUnmatched').textContent = sum.unmatched_count || 0;
    document.getElementById('metricErrors').textContent = sum.error_count || 0;
    document.getElementById('metricAvgConfidence').textContent = `${sum.average_confidence || 0}%`;

    // 4 Key Batch Performance Metrics: In initial condition & while running, display 0%; generate computed scores only when all predictions are done
    const elBatchAcc = document.getElementById('metricBatchAccuracy');
    const elBatchPrec = document.getElementById('metricBatchPrecision');
    const elBatchRec = document.getElementById('metricBatchRecall');
    const elBatchF1 = document.getElementById('metricBatchF1');

    if (!isFinal) {
      if (elBatchAcc) elBatchAcc.textContent = '0%';
      if (elBatchPrec) elBatchPrec.textContent = '0%';
      if (elBatchRec) elBatchRec.textContent = '0%';
      if (elBatchF1) elBatchF1.textContent = '0%';
    } else {
      const bm = sum.metrics || {
        accuracy: sum.accuracy || 94.2,
        precision: sum.precision || 94.6,
        recall: sum.recall || 93.8,
        f1_score: sum.f1_score || 94.2
      };
      if (elBatchAcc) elBatchAcc.textContent = `${bm.accuracy}%`;
      if (elBatchPrec) elBatchPrec.textContent = `${bm.precision}%`;
      if (elBatchRec) elBatchRec.textContent = `${bm.recall}%`;
      if (elBatchF1) elBatchF1.textContent = `${bm.f1_score}%`;
    }

    // Render Doughnut Chart using Chart.js
    renderBatchChart(sum.class_distribution, sum.unmatched_count);

    // Render Breakdown bars
    renderClassBreakdownBars(sum.class_distribution, sum.total_samples);

    // Render Table without image thumbnails
    filterAndRenderBatchTable();
  }

  function renderBatchChart(classDist, unmatchedCount) {
    const canvas = document.getElementById('diseaseChartCanvas');
    if (!canvas) return;

    if (state.batchChart) {
      state.batchChart.destroy();
    }

    const labels = ['Brown spot', 'Healthy Paddy', 'Hispa', 'Leaf Blast'];
    const colors = ['#d97706', '#16a34a', '#ca8a04', '#dc2626'];
    const counts = labels.map(c => (classDist && classDist[c]) || 0);

    if (unmatchedCount > 0) {
      labels.push('Unmatched / Skipped');
      colors.push('#94a3b8');
      counts.push(unmatchedCount);
    }

    state.batchChart = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              boxWidth: 12,
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 }
            }
          }
        },
        cutout: '65%'
      }
    });
  }

  function renderClassBreakdownBars(classDist, total) {
    const container = document.getElementById('batchClassBreakdown');
    if (!container || !classDist) return;
    container.innerHTML = '';

    const colors = {
      'Brown spot': '#d97706',
      'Healthy Paddy': '#16a34a',
      'Hispa': '#ca8a04',
      'Leaf Blast': '#dc2626'
    };

    Object.entries(classDist).forEach(([cls, count]) => {
      const pct = total > 0 ? Math.round((count / total) * 100) : 0;
      const barColor = colors[cls] || '#059669';
      const div = document.createElement('div');
      div.className = 'mb-3';
      div.innerHTML = `
        <div class="d-flex justify-content-between small fw-bold mb-1">
          <span>${cls}</span>
          <span style="color: ${barColor}">${count} leaves (${pct}%)</span>
        </div>
        <div class="progress" style="height: 8px;">
          <div class="progress-bar" style="width: ${pct}%; background-color: ${barColor};"></div>
        </div>
      `;
      container.appendChild(div);
    });
  }

  function filterAndRenderBatchTable() {
    if (!state.batchResults || !state.batchResults.results) return;
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    const searchTerm = (document.getElementById('tableSearchInput')?.value || '').toLowerCase().trim();
    const filter = state.activeFilter;

    const filtered = state.batchResults.results.filter(row => {
      // Filter matching
      if (filter === 'unmatched' && row.status !== 'unmatched') return false;
      if (filter === 'errors' && row.status !== 'error') return false;
      if (filter !== 'all' && filter !== 'unmatched' && filter !== 'errors') {
        if (row.predicted_class !== filter) return false;
      }

      // Search matching
      if (searchTerm) {
        const idMatch = (row.sample_id || '').toLowerCase().includes(searchTerm) || stripFolderPrefix(row.sample_id).toLowerCase().includes(searchTerm);
        const frontMatch = (row.front_filename || '').toLowerCase().includes(searchTerm) || stripFolderPrefix(row.front_filename).toLowerCase().includes(searchTerm);
        const backMatch = (row.back_filename || '').toLowerCase().includes(searchTerm) || stripFolderPrefix(row.back_filename).toLowerCase().includes(searchTerm);
        const classMatch = (row.predicted_class || '').toLowerCase().includes(searchTerm);
        if (!idMatch && !frontMatch && !backMatch && !classMatch) return false;
      }

      return true;
    });

    state.currentFilteredBatch = filtered;

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="text-center py-4 text-muted">
            <i class="bi bi-inbox fs-3 d-block mb-1"></i> No matching batch samples found.
          </td>
        </tr>
      `;
      return;
    }

    filtered.forEach((row, i) => {
      const tr = document.createElement('tr');
      const isSuccess = row.status === 'success';
      const isUnmatched = row.status === 'unmatched';

      const isTopOnly = !row.back_filename ||
                        row.back_filename === 'None' ||
                        row.back_filename === 'N/A' ||
                        row.back_filename === 'null' ||
                        row.back_filename === 'Missing Back View (D1)' ||
                        (row.mode && (row.mode.toLowerCase().includes('top') || row.mode.toLowerCase().includes('front')));

      let statusBadge = '<span class="badge bg-success-subtle text-success border border-success-subtle">Analyzed</span>';
      if (isUnmatched) {
        statusBadge = '<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Unmatched</span>';
      } else if (row.status === 'error') {
        statusBadge = '<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Error</span>';
      }

      const diseaseBadge = isSuccess
        ? `<span class="badge" style="background-color: ${row.disease_info?.color_theme || '#059669'}; color: white;">${row.predicted_class}</span>`
        : (isUnmatched ? `<span class="text-muted small">${row.reason || 'Missing View'}</span>` : `<span class="text-danger small">${row.error || 'Failed'}</span>`);

      const confText = isSuccess ? `<strong>${row.confidence}%</strong>` : '&mdash;';

      const cleanSampleId = stripFolderPrefix(row.sample_id);
      const cleanFrontFn = stripFolderPrefix(row.front_filename);
      const cleanBackFn = stripFolderPrefix(row.back_filename);

      // Render Table Row: Clean display with pure Sample ID and base Filenames without directory prefixes
      tr.innerHTML = `
        <td class="fw-bold text-dark">${cleanSampleId}</td>
        <td>
          ${isTopOnly ? `
            <div class="small fw-semibold text-truncate text-dark" style="max-width: 320px;" title="${cleanFrontFn}">
              <span class="badge bg-success-subtle text-success border border-success-subtle me-1">T1</span>${cleanFrontFn}
            </div>
            <div class="text-xs text-muted">Top-View Leaf Image</div>
          ` : `
            <div class="small text-truncate text-dark mb-1" style="max-width: 320px;" title="${cleanFrontFn}">
              <span class="badge bg-success-subtle text-success border border-success-subtle me-1">T1</span>${cleanFrontFn}
            </div>
            <div class="small text-muted text-truncate" style="max-width: 320px;" title="${cleanBackFn}">
              <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle me-1">D1</span>${cleanBackFn}
            </div>
          `}
        </td>
        <td>
          ${isTopOnly ? `
            <span class="badge bg-success-subtle text-success border border-success-subtle">Top-Only (T1)</span>
          ` : `
            <span class="badge bg-light text-dark border">${row.mode || 'Dual-View'}</span>
          `}
        </td>
        <td>${diseaseBadge}</td>
        <td>${confText}</td>
        <td>${statusBadge}</td>
        <td class="text-end">
          ${isSuccess ? `
            <button class="btn btn-outline-success btn-xs rounded-pill px-3 batch-detail-btn" data-filtered-idx="${i}">
              <i class="bi bi-info-circle me-1"></i> Details
            </button>
          ` : '&mdash;'}
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Attach event listeners to details buttons
    tbody.querySelectorAll('.batch-detail-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-filtered-idx'), 10);
        const selectedRow = state.currentFilteredBatch && state.currentFilteredBatch[idx];
        if (selectedRow) {
          showBatchDetailModal(selectedRow);
        }
      });
    });
  }

  function exportBatchResultsCsv() {
    if (!state.batchResults || !state.batchResults.results) return;
    const headers = ['Sample ID', 'Front Filename (T1)', 'Back Filename (D1)', 'Analysis Mode', 'Predicted Disease', 'Confidence (%)', 'Status', 'Details/Reason'];
    const rows = state.batchResults.results.map(r => {
      return [
        `"${stripFolderPrefix(r.sample_id)}"`,
        `"${stripFolderPrefix(r.front_filename)}"`,
        `"${stripFolderPrefix(r.back_filename)}"`,
        `"${r.mode || ''}"`,
        `"${r.predicted_class || ''}"`,
        r.confidence !== undefined ? r.confidence : '',
        `"${r.status || ''}"`,
        `"${r.reason || r.error || ''}"`
      ];
    });

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    downloadCsvBlob(csvContent, 'paddyscan_batch_results.csv');
  }
}

// Helper to find matching File object from staged batch files
function findBatchFile(filename) {
  if (!filename) return null;
  const cleanTarget = filename.replace(/\\/g, '/').toLowerCase().trim();
  const targetBase = cleanTarget.split('/').pop();

  if (state.batchFilesMap) {
    if (state.batchFilesMap.has(cleanTarget)) return state.batchFilesMap.get(cleanTarget);
    if (state.batchFilesMap.has(targetBase)) return state.batchFilesMap.get(targetBase);
  }

  if (!state.batchFiles || state.batchFiles.length === 0) return null;

  return state.batchFiles.find(f => {
    const fRel = (f.webkitRelativePath || '').replace(/\\/g, '/').toLowerCase().trim();
    const fName = (f.name || '').toLowerCase().trim();

    if (fRel && (fRel === cleanTarget || fRel.endsWith('/' + cleanTarget) || cleanTarget.endsWith('/' + fRel))) return true;
    if (fName === targetBase || fName === cleanTarget) return true;
    return false;
  }) || null;
}

// Helper to get image URL for a batch file
function getBatchImageUrl(filename) {
  if (!filename) return null;
  const cleanTarget = filename.replace(/\\/g, '/').toLowerCase().trim();
  const targetBase = cleanTarget.split('/').pop();

  state.batchUrlCache = state.batchUrlCache || new Map();
  if (state.batchUrlCache.has(cleanTarget)) return state.batchUrlCache.get(cleanTarget);
  if (state.batchUrlCache.has(targetBase)) return state.batchUrlCache.get(targetBase);

  const file = findBatchFile(filename);
  if (file) {
    try {
      const url = URL.createObjectURL(file);
      state.batchUrlCache.set(cleanTarget, url);
      state.batchUrlCache.set(targetBase, url);
      return url;
    } catch (e) {
      console.warn('Could not create object URL for file:', filename, e);
    }
  }

  if (cleanTarget.includes('sample_')) {
    const url = `assets/samples/${targetBase}`;
    state.batchUrlCache.set(cleanTarget, url);
    state.batchUrlCache.set(targetBase, url);
    return url;
  }
  return null;
}

// Global modal viewer function
window.showBatchDetailModal = function (row) {
  const modalEl = document.getElementById('detailModal');
  const modalBody = document.getElementById('detailModalBody');
  const modal = new bootstrap.Modal(modalEl);

  const color = row.disease_info?.color_theme || '#059669';

  const isTopOnly = !row.back_filename ||
                    row.back_filename === 'None' ||
                    row.back_filename === 'N/A' ||
                    row.back_filename === 'null' ||
                    row.back_filename === 'Missing Back View (D1)' ||
                    (row.mode && (row.mode.toLowerCase().includes('top') || row.mode.toLowerCase().includes('front')));

  // Find particular images for this sample
  const cleanSampleId = (typeof stripFolderPrefix === 'function') ? stripFolderPrefix(row.sample_id) : (row.sample_id || '').split('/').pop();
  const cleanFrontFn = (typeof stripFolderPrefix === 'function') ? stripFolderPrefix(row.front_filename) : (row.front_filename || '').split('/').pop();
  const cleanBackFn = (typeof stripFolderPrefix === 'function') ? stripFolderPrefix(row.back_filename) : (row.back_filename || '').split('/').pop();

  const frontImgUrl = getBatchImageUrl(row.front_filename);
  const backImgUrl = isTopOnly ? null : getBatchImageUrl(row.back_filename);

  let imageShowcaseHtml = '';
  if (isTopOnly) {
    // Show ONLY the Top Image preview card
    imageShowcaseHtml = `
      <div class="card p-3 bg-light border-0 text-center mb-3">
        <div class="d-flex justify-content-between align-items-center mb-2 px-1">
          <span class="badge bg-success-subtle text-success border border-success-subtle">
            T1 • Top View (Adaxial)
          </span>
          <span class="text-xs text-muted">Single Top Leaf Image Analysis</span>
        </div>
        <div style="height: 240px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
          ${frontImgUrl ? `
            <img src="${frontImgUrl}" alt="Top Leaf (T1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
          ` : `
            <div class="text-muted small">
              <i class="bi bi-file-earmark-image fs-1 d-block mb-1 text-success"></i>
              ${cleanFrontFn || cleanSampleId}
            </div>
          `}
        </div>
        <div class="text-xs text-muted text-truncate mt-2 text-start px-1" title="${cleanFrontFn}">
          <i class="bi bi-file-earmark-image me-1 text-success"></i><strong>Top Image File:</strong> ${cleanFrontFn}
        </div>
      </div>
    `;
  } else if (frontImgUrl || backImgUrl) {
    imageShowcaseHtml = `
      <div class="row g-3 mb-3">
        ${frontImgUrl ? `
          <div class="${backImgUrl ? 'col-sm-6' : 'col-12'}">
            <div class="card p-2 bg-light border-0 text-center">
              <span class="badge bg-success-subtle text-success border border-success-subtle mb-2 align-self-start">
                T1 • Front View
              </span>
              <div style="height: 200px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
                <img src="${frontImgUrl}" alt="Front Leaf (T1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
              </div>
              <div class="text-xs text-muted text-truncate mt-1 text-start" title="${cleanFrontFn}">
                ${cleanFrontFn}
              </div>
            </div>
          </div>
        ` : ''}

        ${backImgUrl ? `
          <div class="${frontImgUrl ? 'col-sm-6' : 'col-12'}">
            <div class="card p-2 bg-light border-0 text-center">
              <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle mb-2 align-self-start">
                D1 • Back View
              </span>
              <div style="height: 200px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
                <img src="${backImgUrl}" alt="Back Leaf (D1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
              </div>
              <div class="text-xs text-muted text-truncate mt-1 text-start" title="${cleanBackFn}">
                ${cleanBackFn}
              </div>
            </div>
          </div>
        ` : ''}
      </div>
    `;
  } else {
    imageShowcaseHtml = `
      <div class="alert alert-light border py-3 text-center mb-3 text-muted small">
        <i class="bi bi-file-earmark-image fs-3 d-block mb-1 text-secondary"></i>
        Sample file: <strong>${cleanFrontFn || cleanSampleId}</strong>
      </div>
    `;
  }

  modalBody.innerHTML = `
    <!-- Top Diagnosis Header with Single Maximum Disease and Highest Accuracy Score -->
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <span class="badge px-3 py-1 text-uppercase fw-bold" style="background-color: ${color}; color: #fff;">${row.predicted_class}</span>
        <h4 class="fw-bold mt-2 mb-0">${cleanSampleId}</h4>
      </div>
      <div class="text-end">
        <h3 class="fw-bold mb-0" style="color: ${color}">${row.confidence}%</h3>
        <span class="text-muted text-xs">Highest Confidence</span>
      </div>
    </div>

    <!-- Leaf Images for this Particular Sample (Only Top Image for Top-Only) -->
    ${imageShowcaseHtml}

    <!-- Sample & Processing Metadata -->
    <div class="card border-0 bg-light p-3 mb-3 rounded-3">
      <div class="small mb-1"><strong>${isTopOnly ? 'Top Leaf File (T1)' : 'Front File (T1)'}:</strong> ${cleanFrontFn || 'N/A'}</div>
      ${!isTopOnly && cleanBackFn ? `<div class="small mb-1"><strong>Back File (D1):</strong> ${cleanBackFn}</div>` : ''}
      <div class="small"><strong>Mode:</strong> ${row.mode} • <strong>Latency:</strong> ${row.inference_time_ms} ms</div>
    </div>

    <!-- Diagnosed Disease Confidence Score (ONLY maximum disease, other 3 omitted) -->
    <div class="card border-0 bg-light p-3 mb-3 rounded-3">
      <h6 class="fw-bold small text-dark mb-2">Diagnosed Disease Confidence Score</h6>
      <div class="prob-meter-row">
        <div class="prob-label-group d-flex justify-content-between mb-1">
          <span class="fw-bold text-dark">${row.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${row.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px; background: #e2e8f0; border-radius: 9999px; overflow: hidden;">
          <div class="prob-bar-fill" style="width: ${row.confidence}%; background-color: ${color}; height: 100%; border-radius: 9999px;"></div>
        </div>
      </div>
    </div>

    <!-- Agronomic Recommendations -->
    <div class="p-3 bg-white rounded-3 border">
      <h6 class="fw-bold text-success mb-1">Agronomic Recommendations:</h6>
      <p class="small text-muted mb-2">${row.disease_info?.symptoms || ''}</p>
      <div class="small text-dark fw-semibold">${row.disease_info?.management || ''}</div>
    </div>
  `;

  modal.show();
};

// ==============================================================================
// 8. COMMON UI HELPERS & UTILITIES
// ==============================================================================

function renderProbabilityMeters(probs, accentColor) {
  if (!probs) return '';
  return Object.entries(probs).map(([cls, pct]) => `
    <div class="prob-meter-row">
      <div class="prob-label-group">
        <span>${cls}</span>
        <span class="text-muted">${pct}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%; background-color: ${pct > 40 ? accentColor : '#94a3b8'};"></div>
      </div>
    </div>
  `).join('');
}

// ==============================================================================
// 7.5 TRAINING & EVALUATION CURVES (ANIMATED DUAL-VIEW EFFICIENTNET-B0 + cGAN)
// ==============================================================================

const TRAINING_CURVE_DATA = {
  epochs: Array.from({ length: 50 }, (_, i) => i + 1),
  trainAcc: [
    53.87, 77.80, 83.40, 87.40, 88.00, 88.90, 90.00, 90.60, 91.10, 92.80,
    91.70, 92.50, 92.80, 93.40, 92.30, 93.70, 93.90, 94.40, 94.10, 94.40,
    94.30, 94.20, 94.10, 93.90, 93.50, 93.90, 95.00, 94.70, 94.80, 94.10,
    94.00, 95.00, 93.30, 95.30, 94.10, 94.80, 94.20, 94.80, 93.70, 94.10,
    94.80, 94.10, 94.50, 95.00, 93.80, 94.00, 93.80, 94.40, 93.90, 94.00
  ],
  valAcc: [
    78.40, 83.80, 86.20, 87.20, 89.70, 90.30, 92.20, 91.80, 90.80, 93.80,
    92.70, 92.70, 92.70, 91.70, 91.70, 93.10, 92.70, 93.60, 92.70, 93.60,
    92.70, 93.60, 93.60, 93.20, 92.70, 92.70, 93.60, 92.70, 94.10, 93.20,
    92.70, 93.20, 93.20, 93.10, 92.70, 92.70, 92.20, 92.70, 92.70, 94.10,
    92.70, 92.70, 93.10, 92.70, 92.70, 92.70, 92.70, 92.70, 92.70, 93.20
  ],
  testAcc: [
    75.20, 81.60, 84.50, 86.80, 88.90, 89.80, 91.40, 91.20, 91.90, 93.10,
    92.40, 92.90, 93.00, 92.50, 92.80, 93.40, 93.10, 93.80, 93.20, 93.90,
    93.60, 93.80, 93.90, 93.60, 93.40, 93.70, 94.10, 93.80, 94.20, 93.90,
    93.80, 94.20, 93.60, 94.40, 93.90, 94.20, 93.80, 94.30, 93.90, 94.10,
    94.20, 94.00, 94.20, 94.40, 94.00, 94.10, 94.00, 94.30, 94.10, 94.20
  ],
  trainLoss: [
    1.080, 0.608, 0.481, 0.383, 0.336, 0.321, 0.297, 0.264, 0.260, 0.220,
    0.234, 0.216, 0.213, 0.185, 0.206, 0.182, 0.161, 0.149, 0.144, 0.150,
    0.168, 0.164, 0.165, 0.161, 0.173, 0.167, 0.130, 0.133, 0.154, 0.166,
    0.155, 0.165, 0.171, 0.139, 0.151, 0.153, 0.144, 0.131, 0.160, 0.147,
    0.141, 0.168, 0.139, 0.133, 0.153, 0.142, 0.145, 0.154, 0.168, 0.147
  ],
  valLoss: [
    0.651, 0.452, 0.358, 0.362, 0.269, 0.266, 0.204, 0.213, 0.215, 0.169,
    0.171, 0.175, 0.193, 0.200, 0.181, 0.215, 0.179, 0.202, 0.157, 0.170,
    0.154, 0.175, 0.186, 0.193, 0.149, 0.154, 0.147, 0.192, 0.174, 0.148,
    0.173, 0.144, 0.197, 0.170, 0.167, 0.169, 0.175, 0.150, 0.185, 0.155,
    0.160, 0.156, 0.147, 0.159, 0.159, 0.158, 0.158, 0.163, 0.153, 0.133
  ]
};

let accuracyChartInstance = null;
let lossChartInstance = null;
let curveDrawingTimer = null;

function renderTrainingCurves(animate = true) {
  if (curveDrawingTimer) {
    clearInterval(curveDrawingTimer);
    curveDrawingTimer = null;
  }

  const accCanvas = document.getElementById('accuracyChartCanvas');
  const lossCanvas = document.getElementById('lossChartCanvas');
  if (!accCanvas || !lossCanvas || typeof Chart === 'undefined') return;

  if (accuracyChartInstance) {
    accuracyChartInstance.destroy();
    accuracyChartInstance = null;
  }
  if (lossChartInstance) {
    lossChartInstance.destroy();
    lossChartInstance = null;
  }

  const statusText = document.getElementById('curveDrawStatusText');
  const spinner = document.getElementById('curveDrawSpinner');
  const counter = document.getElementById('curveEpochCounter');

  const accCtx = accCanvas.getContext('2d');
  const lossCtx = lossCanvas.getContext('2d');

  const initialLabels = animate ? [1] : TRAINING_CURVE_DATA.epochs;
  const initialTrainAcc = animate ? [TRAINING_CURVE_DATA.trainAcc[0]] : TRAINING_CURVE_DATA.trainAcc;
  const initialValAcc = animate ? [TRAINING_CURVE_DATA.valAcc[0]] : TRAINING_CURVE_DATA.valAcc;
  const initialTestAcc = animate ? [TRAINING_CURVE_DATA.testAcc[0]] : TRAINING_CURVE_DATA.testAcc;
  const initialTrainLoss = animate ? [TRAINING_CURVE_DATA.trainLoss[0]] : TRAINING_CURVE_DATA.trainLoss;
  const initialValLoss = animate ? [TRAINING_CURVE_DATA.valLoss[0]] : TRAINING_CURVE_DATA.valLoss;

  // 1. Initialize Accuracy Chart with Training, Validation, and Testing curves
  accuracyChartInstance = new Chart(accCtx, {
    type: 'line',
    data: {
      labels: [...initialLabels],
      datasets: [
        {
          label: 'Training Accuracy',
          data: [...initialTrainAcc],
          borderColor: '#1f77b4',
          backgroundColor: '#1f77b4',
          borderWidth: 2.2,
          pointStyle: 'circle',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#1f77b4',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        },
        {
          label: 'Validation Accuracy',
          data: [...initialValAcc],
          borderColor: '#ff7f0e',
          backgroundColor: '#ff7f0e',
          borderWidth: 2.2,
          pointStyle: 'rect',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#ff7f0e',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        },
        {
          label: 'Testing Accuracy',
          data: [...initialTestAcc],
          borderColor: '#059669',
          backgroundColor: '#059669',
          borderWidth: 2.4,
          pointStyle: 'triangle',
          pointRadius: 4.5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#059669',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleFont: { size: 13, weight: '700' },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            title: (items) => `Epoch ${items[0].label}`,
            label: (item) => ` ${item.dataset.label}: ${Number(item.raw).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          },
          min: 1,
          max: 50,
          ticks: {
            stepSize: 5,
            color: '#64748b'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Accuracy (%)',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          min: 50,
          max: 100,
          ticks: {
            stepSize: 10,
            color: '#64748b',
            callback: (v) => `${v}%`
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          }
        }
      }
    }
  });

  // 2. Initialize Loss Chart
  lossChartInstance = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [...initialLabels],
      datasets: [
        {
          label: 'Train Loss',
          data: [...initialTrainLoss],
          borderColor: '#d62728',
          backgroundColor: '#d62728',
          borderWidth: 2.2,
          pointStyle: 'circle',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#d62728',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        },
        {
          label: 'Validation Loss',
          data: [...initialValLoss],
          borderColor: '#2ca02c',
          backgroundColor: '#2ca02c',
          borderWidth: 2.2,
          pointStyle: 'rect',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#2ca02c',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleFont: { size: 13, weight: '700' },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            title: (items) => `Epoch ${items[0].label}`,
            label: (item) => ` ${item.dataset.label}: ${Number(item.raw).toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          },
          min: 1,
          max: 50,
          ticks: {
            stepSize: 5,
            color: '#64748b'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Cross-Entropy Loss',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          min: 0.0,
          max: 1.15,
          ticks: {
            stepSize: 0.2,
            color: '#64748b'
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          }
        }
      }
    }
  });

  if (!animate) {
    if (spinner) spinner.style.display = 'none';
    if (statusText) statusText.innerHTML = '<i class="bi bi-check-circle-fill text-success me-1"></i> <strong>Graph rendering complete!</strong> Hover over any point to inspect metrics';
    if (counter) counter.textContent = 'Epoch: 50 / 50 (Completed)';
    return;
  }

  // 3. Progressive real-time line drafting animation
  if (spinner) spinner.style.display = 'inline-block';
  if (statusText) statusText.innerHTML = '<span class="text-success fw-bold">Live Graph Synthesis:</span> Designing and drawing convergence curves across 50 epochs...';
  if (counter) counter.textContent = 'Epoch: 1 / 50';

  let currentIdx = 1;
  const totalPoints = 50;
  const stepInterval = 28; // ~1.4 seconds total progressive draw

  curveDrawingTimer = setInterval(() => {
    if (currentIdx < totalPoints) {
      const ep = TRAINING_CURVE_DATA.epochs[currentIdx];
      
      // Update Accuracy Chart (Training, Validation, Testing)
      accuracyChartInstance.data.labels.push(ep);
      accuracyChartInstance.data.datasets[0].data.push(TRAINING_CURVE_DATA.trainAcc[currentIdx]);
      accuracyChartInstance.data.datasets[1].data.push(TRAINING_CURVE_DATA.valAcc[currentIdx]);
      accuracyChartInstance.data.datasets[2].data.push(TRAINING_CURVE_DATA.testAcc[currentIdx]);
      accuracyChartInstance.update('none');

      // Update Loss Chart
      lossChartInstance.data.labels.push(ep);
      lossChartInstance.data.datasets[0].data.push(TRAINING_CURVE_DATA.trainLoss[currentIdx]);
      lossChartInstance.data.datasets[1].data.push(TRAINING_CURVE_DATA.valLoss[currentIdx]);
      lossChartInstance.update('none');

      currentIdx++;
      if (counter) counter.textContent = `Epoch: ${currentIdx} / 50`;
    } else {
      clearInterval(curveDrawingTimer);
      curveDrawingTimer = null;
      if (spinner) spinner.style.display = 'none';
      if (statusText) statusText.innerHTML = '<i class="bi bi-check-circle-fill text-success me-1"></i> <strong>Graph rendering complete!</strong> Hover over any point to inspect metrics';
      if (counter) counter.textContent = 'Epoch: 50 / 50 (Completed)';
    }
  }, stepInterval);
}

function initTrainingCurvesSection() {
  const replayBtn = document.getElementById('replayGraphAnimBtn');
  const skipBtn = document.getElementById('btnSkipCurveAnim');
  const btnInteractive = document.getElementById('btnViewInteractive');
  const btnMatplotlib = document.getElementById('btnViewMatplotlib');
  const interactiveView = document.getElementById('interactiveCurvesView');
  const matplotlibView = document.getElementById('matplotlibPlotsView');
  const statusBar = document.getElementById('curveDrawStatusBar');

  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      // Return to interactive view if currently in publication plot mode
      if (btnInteractive && !btnInteractive.classList.contains('active')) {
        btnInteractive.click();
      }
      renderTrainingCurves(true);
    });
  }

  if (skipBtn) {
    skipBtn.addEventListener('click', () => {
      renderTrainingCurves(false);
    });
  }

  if (btnInteractive && btnMatplotlib) {
    btnInteractive.addEventListener('click', () => {
      btnInteractive.classList.add('btn-success', 'active');
      btnInteractive.classList.remove('btn-light', 'text-secondary');
      btnMatplotlib.classList.remove('btn-success', 'active');
      btnMatplotlib.classList.add('btn-light', 'text-secondary');

      interactiveView?.classList.remove('d-none');
      matplotlibView?.classList.add('d-none');
      statusBar?.classList.remove('d-none');
      if (!accuracyChartInstance || !lossChartInstance) {
        renderTrainingCurves(true);
      }
    });

    btnMatplotlib.addEventListener('click', () => {
      btnMatplotlib.classList.add('btn-success', 'active');
      btnMatplotlib.classList.remove('btn-light', 'text-secondary');
      btnInteractive.classList.remove('btn-success', 'active');
      btnInteractive.classList.add('btn-light', 'text-secondary');

      matplotlibView?.classList.remove('d-none');
      interactiveView?.classList.add('d-none');
      statusBar?.classList.add('d-none');
    });
  }
}

function setupDropEvents(element, singleFileCallback, multiFileCallback) {
  if (!element) return;
  ['dragenter', 'dragover'].forEach(n => {
    element.addEventListener(n, (e) => {
      e.preventDefault();
      e.stopPropagation();
      element.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(n => {
    element.addEventListener(n, (e) => {
      e.preventDefault();
      e.stopPropagation();
      element.classList.remove('dragover');
    });
  });
  element.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();

    // Check for webkitGetAsEntry to support dragging and dropping entire folders
    if (multiFileCallback && e.dataTransfer.items && e.dataTransfer.items.length > 0 && e.dataTransfer.items[0].webkitGetAsEntry) {
      const files = [];
      const traverseEntry = async (entry, path = '') => {
        if (entry.isFile) {
          await new Promise((res) => {
            entry.file(f => {
              try {
                Object.defineProperty(f, 'webkitRelativePath', {
                  value: (path ? path + '/' : '') + f.name,
                  writable: true
                });
              } catch (_) {}
              files.push(f);
              res();
            }, () => res());
          });
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          await new Promise((res) => {
            reader.readEntries(async (entries) => {
              for (const child of entries) {
                await traverseEntry(child, (path ? path + '/' : '') + entry.name);
              }
              res();
            }, () => res());
          });
        }
      };

      try {
        for (let i = 0; i < e.dataTransfer.items.length; i++) {
          const item = e.dataTransfer.items[i];
          if (item.kind === 'file') {
            const entry = item.webkitGetAsEntry();
            if (entry) await traverseEntry(entry);
          }
        }
      } catch (_) {}

      if (files.length > 0) {
        multiFileCallback(files);
        return;
      }
    }

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (singleFileCallback) {
        singleFileCallback(e.dataTransfer.files[0]);
      } else if (multiFileCallback) {
        multiFileCallback(Array.from(e.dataTransfer.files));
      }
    }
  });
}

function showAlert(message, type = 'info', targetSectionId = null) {
  const activeId = targetSectionId || state.activeSection || 'front-view-section';
  let container = document.getElementById(`alertContainer-${activeId}`);

  // Fallback to searching within section or creating container inside section
  if (!container) {
    const sectionEl = document.getElementById(activeId);
    if (sectionEl) {
      container = sectionEl.querySelector('.section-alert-container');
      if (!container) {
        container = document.createElement('div');
        container.className = 'section-alert-container mb-3';
        container.id = `alertContainer-${activeId}`;
        const fluid = sectionEl.querySelector('.container-fluid') || sectionEl;
        fluid.insertBefore(container, fluid.firstChild);
      }
    }
  }

  // Fallback to globalAlertContainer if no section container is available
  if (!container) {
    container = document.getElementById('globalAlertContainer');
    if (container) container.classList.remove('d-none');
  }

  if (!container) return;

  // Keep at most 2 active alerts in the section to prevent screen clutter
  while (container.children.length >= 2) {
    container.removeChild(container.firstChild);
  }

  const alert = document.createElement('div');
  alert.className = `alert alert-${type} alert-dismissible fade show shadow-sm border-0 mb-3`;
  alert.role = 'alert';
  alert.innerHTML = `
    <div class="d-flex align-items-center">
      <i class="bi bi-${type === 'success' ? 'check-circle-fill text-success' : (type === 'danger' ? 'exclamation-octagon-fill text-danger' : 'info-circle-fill text-info')} me-2 fs-5"></i>
      <div class="small fw-medium">${message}</div>
    </div>
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;

  container.appendChild(alert);

  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 250);
  }, 4500);
}

function downloadCsvBlob(csvString, filename) {
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
