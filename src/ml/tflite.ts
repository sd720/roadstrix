/**
 * ═══════════════════════════════════════════════════════════════════════
 *  TFLite Model Loader & Inference Engine
 *  ─────────────────────────────────────────────────────────────────────
 *  Handles loading the bundled Autoencoder TFLite model and running
 *  real-time inference on suspension (IMU) sensor data for pothole
 *  detection via anomaly reconstruction error.
 * ═══════════════════════════════════════════════════════════════════════
 */

export type BundledModelLoadResult = {
  model: any;
  state: 'loaded' | 'unavailable' | 'error';
  message: string;
};

export type InferenceResult = {
  mse: number;
  isPothole: boolean;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
};

// ── Model configuration (synced with train_suspension_model.py) ──────
export const MODEL_CONFIG = {
  WINDOW_SIZE: 50,       // 50 samples = 0.5s @ 100Hz
  NUM_CHANNELS: 6,       // accel(x,y,z) + gyro(x,y,z)

  // Normalisation constants (updated after training)
  // These will be overridden by model_config.json at runtime
  mean: [0.0, 0.0, 9.81, 0.0, 0.0, 0.0],
  std:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

  // Detection thresholds (updated after training calibration)
  threshold_default: 0.5,
  threshold_high_precision: 0.8,
  threshold_high_recall: 0.3,
};

// ── Dynamic config loader ────────────────────────────────────────────
let configLoaded = false;

export async function loadModelConfig(): Promise<void> {
  try {
    const config = require('../../assets/model_config.json');
    if (config.mean) MODEL_CONFIG.mean = config.mean;
    if (config.std) MODEL_CONFIG.std = config.std;
    if (config.threshold_default) MODEL_CONFIG.threshold_default = config.threshold_default;
    if (config.threshold_high_precision) MODEL_CONFIG.threshold_high_precision = config.threshold_high_precision;
    if (config.threshold_high_recall) MODEL_CONFIG.threshold_high_recall = config.threshold_high_recall;
    if (config.window_size) MODEL_CONFIG.WINDOW_SIZE = config.window_size;
    if (config.num_channels) MODEL_CONFIG.NUM_CHANNELS = config.num_channels;
    configLoaded = true;
    console.log('[TFLite] Model config loaded successfully');
  } catch (e) {
    console.log('[TFLite] model_config.json not found, using defaults');
  }
}

// ── Dynamic Baseline State (Self-Calibrating) ────────────────────────
// This allows the model to dynamically adapt to different road surfaces
// (e.g., gravel vs smooth highway). A pothole is detected if the current
// impact is significantly worse than the *recent* road surface.
let recentMseHistory: number[] = [];
const HISTORY_MAX_LEN = 10; // keep last 10 readings (5 seconds)

// ── Signal DSP State ─────────────────────────────────────────────────
// Exponential Moving Average (EMA) to act as a low-pass filter.
// This strips out high-frequency engine vibrations so the neural net
// only sees the actual suspension movement.
let emaState: number[] | null = null;
const EMA_ALPHA = 0.6; // lower = heavier filtering
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return 'Unknown TFLite error';
}

// ── Model loader ─────────────────────────────────────────────────────
export async function loadBundledTfliteModel(): Promise<BundledModelLoadResult> {
  if (process.env.NODE_ENV === 'test') {
    return {
      model: undefined,
      state: 'unavailable',
      message: 'Skipped model loading during tests.',
    };
  }

  // Load config first
  await loadModelConfig();

  try {
    const { loadTensorflowModel } = require('react-native-fast-tflite') as {
      loadTensorflowModel: (source: number, delegates: string[]) => Promise<any>;
    };

    const model = await loadTensorflowModel(require('../../assets/model.tflite'), []);
    return {
      model,
      state: 'loaded',
      message: 'Suspension Autoencoder loaded — real-time pothole detection active.',
    };
  } catch (error) {
    const message = getErrorMessage(error);
    const isCustomBuildIssue =
      message.includes('Nitro') ||
      message.includes('native module') ||
      message.includes('TfliteModule') ||
      message.includes('AssetLoader');

    return {
      model: undefined,
      state: isCustomBuildIssue ? 'unavailable' : 'error',
      message: isCustomBuildIssue
        ? 'TFLite runtime needs a custom Expo dev build. Expo Go cannot run the bundled model.'
        : message,
    };
  }
}

// ── Normalise & Filter a single window of sensor data ─────────────────
export function normaliseSensorWindow(
  window: number[][]
): Float32Array {
  const { WINDOW_SIZE, NUM_CHANNELS, mean, std } = MODEL_CONFIG;
  const flat = new Float32Array(WINDOW_SIZE * NUM_CHANNELS);

  if (!emaState) {
    emaState = new Array(NUM_CHANNELS).fill(0);
  }

  for (let t = 0; t < WINDOW_SIZE; t++) {
    for (let c = 0; c < NUM_CHANNELS; c++) {
      const rawVal = window[t]?.[c] ?? 0;
      
      // 1. Digital Signal Processing: Low-pass EMA Filter
      // Smooths out micro-vibrations (engine/road noise)
      emaState[c] = (EMA_ALPHA * rawVal) + ((1 - EMA_ALPHA) * emaState[c]);
      const filteredVal = emaState[c];

      // 2. Z-Score Normalisation
      flat[t * NUM_CHANNELS + c] = (filteredVal - mean[c]) / std[c];
    }
  }
  return flat;
}

// ── Run inference & compute MSE ──────────────────────────────────────
export function runPotholeInference(
  model: any,
  sensorWindow: number[][],
  speedKmh: number = 0,
): InferenceResult | null {
  if (!model) return null;

  try {
    const inputData = normaliseSensorWindow(sensorWindow);
    const { WINDOW_SIZE, NUM_CHANNELS } = MODEL_CONFIG;

    // Run model: input shape [1, 50, 6], output shape [1, 50, 6]
    const outputData = model.runSync([inputData]);
    if (!outputData || !outputData[0]) return null;

    const output = outputData[0] as Float32Array;

    // Compute MSE between input and reconstructed output
    let mse = 0;
    const totalElements = WINDOW_SIZE * NUM_CHANNELS;
    for (let i = 0; i < totalElements; i++) {
      const diff = inputData[i] - output[i];
      mse += diff * diff;
    }
    mse /= totalElements;

    // ── Dynamic Baseline Adaptation (Z-Score Anomaly) ──────────────
    // Instead of a pure static threshold, we compare the current MSE
    // against the recent moving average of the road.
    let baselineMse = MODEL_CONFIG.threshold_default * 0.5; // fallback
    if (recentMseHistory.length > 0) {
      baselineMse = recentMseHistory.reduce((a, b) => a + b, 0) / recentMseHistory.length;
    }

    // A pothole is an extreme deviation from the current baseline
    // (e.g., 2.5x worse than the recent road surface)
    let dynamicThreshold = Math.max(
      MODEL_CONFIG.threshold_default,
      baselineMse * 2.5 
    );

    // ── Speed-adaptive scaling ─────────────────────────────────────
    // At higher speeds (>80km/h), impacts are faster. The suspension
    // absorbs them differently, causing a sharper but briefer spike.
    if (speedKmh > 100) {
      dynamicThreshold *= 0.75; // 25% more sensitive at extreme speed
    } else if (speedKmh > 70) {
      dynamicThreshold *= 0.85; // 15% more sensitive at high speed
    } else if (speedKmh < 20) {
      dynamicThreshold *= 1.20; // 20% less sensitive when crawling (ignores speed bumps)
    }

    const isPothole = mse > dynamicThreshold;

    // Update baseline history (only with normal road data!)
    if (!isPothole) {
      recentMseHistory.push(mse);
      if (recentMseHistory.length > HISTORY_MAX_LEN) {
        recentMseHistory.shift();
      }
    }

    // ── Severity classification ────────────────────────────────────
    let severity: 'low' | 'medium' | 'high' = 'low';
    if (isPothole) {
      const ratio = mse / dynamicThreshold;
      if (ratio > 3.0) severity = 'high';
      else if (ratio > 1.8) severity = 'medium';
      else severity = 'low';
    }

    // ── Confidence score (0-1) ─────────────────────────────────────
    const confidence = isPothole
      ? Math.min(1.0, (mse - dynamicThreshold) / (dynamicThreshold * 2))
      : 0;

    return { mse, isPothole, severity, confidence };
  } catch (e) {
    console.log('[TFLite] Inference error:', e);
    return null;
  }
}
