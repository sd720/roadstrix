"""
=============================================================================
  POTHOLE DETECTION — 1D Convolutional Autoencoder (Suspension Data)
=============================================================================
  Architecture:  Multi-Scale 1D Conv Autoencoder (AEE)
  Input:         Sliding window of 6-axis IMU data
                 [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
  Window:        50 samples @ 100 Hz  =  0.5 seconds of road data
  Detection:     Reconstruction MSE > threshold  →  pothole anomaly
  Output:        Quantized TFLite model (~50-80 KB, < 2 ms inference)

  Why Autoencoder?
  ────────────────
  We train ONLY on "normal" smooth-road driving data.  When the car hits
  a pothole the sensor pattern deviates sharply from anything the model
  has seen → the reconstruction error spikes → we flag it as an anomaly.
  This is unsupervised — no labelled pothole data needed for training.

  Why this works at HIGH speed?
  ─────────────────────────────
  • 100 Hz sampling captures impulses as short as 10 ms.
  • A 0.5 s sliding window catches even brief jolts at 120+ km/h.
  • Speed-adaptive thresholding in the app lowers the bar at high speed
    (faster = harder suspension = smaller absolute spike, but the
    autoencoder still sees a distributional shift).

  Run:
      python src/ml/train_suspension_model.py
  Produces:
      assets/model.tflite   (quantised INT8 — fastest on mobile)
      assets/model_float.tflite  (FP32 fallback)
=============================================================================
"""

import os
import sys
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
WINDOW_SIZE   = 50       # 50 samples  =  0.5 s @ 100 Hz
NUM_CHANNELS  = 6        # accel(x,y,z) + gyro(x,y,z)
LATENT_DIM    = 8        # bottleneck size
EPOCHS        = 120      # training epochs
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3

# Number of synthetic training samples
N_NORMAL_TRAIN   = 12_000
N_NORMAL_VAL     = 2_000
N_POTHOLE_TEST   = 1_000   # anomaly samples for threshold calibration

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
ASSETS_DIR   = os.path.join(PROJECT_ROOT, 'assets')


# =============================================================================
#  1.  SYNTHETIC DATA GENERATION
# =============================================================================
def generate_normal_driving(n_samples: int) -> np.ndarray:
    """
    Simulate 6-axis IMU readings for SMOOTH road driving.

    Realistic characteristics modelled:
      • Gravity bias on accel_z ≈ 9.81 m/s²
      • Small vibrations (engine, tyre noise)  σ ≈ 0.3–0.6
      • Gentle turns  →  lateral accel & yaw rate drift
      • Speed bumps (gentle, periodic humps)
      • Road texture noise (band-limited)
    """
    data = np.zeros((n_samples, WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)

    for i in range(n_samples):
        t = np.linspace(0, 0.5, WINDOW_SIZE)

        # Base vibrations (engine + road texture)
        vibration_freq = np.random.uniform(5, 25)          # Hz
        vibration_amp  = np.random.uniform(0.1, 0.4)       # m/s²

        # Accelerometer  (gravity on Z, small noise on X/Y)
        ax = np.random.normal(0, 0.25, WINDOW_SIZE) + vibration_amp * np.sin(2 * np.pi * vibration_freq * t)
        ay = np.random.normal(0, 0.20, WINDOW_SIZE) + np.random.uniform(-0.3, 0.3)  # slight lateral
        az = 9.81 + np.random.normal(0, 0.30, WINDOW_SIZE) + vibration_amp * 0.5 * np.sin(2 * np.pi * vibration_freq * t + np.random.uniform(0, 2*np.pi))

        # Gyroscope (small rotational drift)
        gx = np.random.normal(0, 0.05, WINDOW_SIZE)
        gy = np.random.normal(0, 0.04, WINDOW_SIZE)
        gz = np.random.normal(0, 0.06, WINDOW_SIZE)  # yaw

        # Occasional gentle speed bump (smooth sinusoidal)
        if np.random.random() < 0.15:
            bump_center = np.random.randint(10, 40)
            bump_width  = np.random.randint(6, 14)
            bump_amp    = np.random.uniform(0.3, 0.8)
            bump = bump_amp * np.exp(-0.5 * ((np.arange(WINDOW_SIZE) - bump_center) / (bump_width / 2.5)) ** 2)
            az += bump
            gy += bump * np.random.uniform(0.02, 0.06)

        data[i] = np.stack([ax, ay, az, gx, gy, gz], axis=-1)

    return data


def generate_pothole_events(n_samples: int) -> np.ndarray:
    """
    Simulate 6-axis IMU readings for roads WITH potholes.

    Pothole signature:
      • Sharp downward jolt on accel_z  (drop into hole)
      • Immediate upward rebound        (exit from hole)
      • Lateral instability on X/Y
      • High-frequency transient on gyro (pitch + roll)
      • Duration: 30–100 ms  =  3–10 samples @ 100 Hz
    """
    data = generate_normal_driving(n_samples)  # start with normal baseline

    for i in range(n_samples):
        # Pothole impact parameters
        impact_center = np.random.randint(12, 38)
        impact_width  = np.random.randint(2, 6)             # 20–60 ms
        severity      = np.random.choice([1.0, 2.0, 3.5])   # low / med / high

        drop_amp    = severity * np.random.uniform(2.5, 6.0)    # m/s²
        rebound_amp = severity * np.random.uniform(1.5, 4.0)
        lateral_amp = severity * np.random.uniform(0.8, 2.5)

        idx = np.arange(WINDOW_SIZE)
        # Sharp V-shaped jolt on Z
        drop    = -drop_amp * np.exp(-0.5 * ((idx - impact_center) / max(impact_width * 0.4, 0.5)) ** 2)
        rebound =  rebound_amp * np.exp(-0.5 * ((idx - impact_center - impact_width) / max(impact_width * 0.5, 0.5)) ** 2)
        data[i, :, 2] += drop + rebound  # accel_z

        # Lateral instability
        data[i, :, 0] += lateral_amp * np.random.uniform(-1, 1) * np.exp(-0.5 * ((idx - impact_center) / max(impact_width, 1)) ** 2)
        data[i, :, 1] += lateral_amp * np.random.uniform(-1, 1) * np.exp(-0.5 * ((idx - impact_center) / max(impact_width, 1)) ** 2)

        # Gyroscope transient (pitch + roll spike)
        gyro_spike_amp = severity * np.random.uniform(0.3, 1.2)
        data[i, :, 3] += gyro_spike_amp * np.exp(-0.5 * ((idx - impact_center) / max(impact_width * 0.6, 0.5)) ** 2)  # pitch
        data[i, :, 4] += gyro_spike_amp * 0.7 * np.exp(-0.5 * ((idx - impact_center + 1) / max(impact_width * 0.5, 0.5)) ** 2)  # roll

        # High-freq ringing after impact
        ring_freq = np.random.uniform(30, 60)
        ring_decay = np.exp(-8 * np.maximum(t := np.linspace(0, 0.5, WINDOW_SIZE) - impact_center / 100.0, 0))
        data[i, :, 2] += severity * 0.4 * np.sin(2 * np.pi * ring_freq * np.linspace(0, 0.5, WINDOW_SIZE)) * ring_decay

    return data


# =============================================================================
#  2.  NORMALISATION
# =============================================================================
def compute_normalisation(data: np.ndarray):
    """Per-channel mean/std for z-score normalisation."""
    mean = data.reshape(-1, NUM_CHANNELS).mean(axis=0)
    std  = data.reshape(-1, NUM_CHANNELS).std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((data - mean) / std).astype(np.float32)


# =============================================================================
#  3.  BUILD AUTOENCODER MODEL
# =============================================================================
def build_autoencoder():
    """
    Multi-scale 1D Convolutional Autoencoder.

    Encoder:
      Conv1D(32, k=3) → Conv1D(64, k=5) → Conv1D(32, k=3) → Dense(latent)
    Decoder:
      Dense → Conv1DTranspose(32) → Conv1DTranspose(64) → Conv1D(6)

    Total params: ~25K  →  tiny, fast, mobile-optimised.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)

    inp = keras.Input(shape=(WINDOW_SIZE, NUM_CHANNELS), name='sensor_input')

    # ── Encoder ──────────────────────────────────────────────────────────
    x = layers.Conv1D(32, 3, activation='relu', padding='same', name='enc_conv1')(inp)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = layers.MaxPooling1D(2, name='enc_pool1')(x)                  # 50 → 25

    x = layers.Conv1D(64, 5, activation='relu', padding='same', name='enc_conv2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = layers.MaxPooling1D(2, name='enc_pool2')(x)                  # 25 → 12

    x = layers.Conv1D(32, 3, activation='relu', padding='same', name='enc_conv3')(x)
    x = layers.BatchNormalization(name='enc_bn3')(x)

    # Flatten → bottleneck
    x = layers.Flatten(name='enc_flatten')(x)
    x = layers.Dense(LATENT_DIM, activation='relu', name='bottleneck')(x)

    # ── Decoder ──────────────────────────────────────────────────────────
    x = layers.Dense(12 * 32, activation='relu', name='dec_dense')(x)
    x = layers.Reshape((12, 32), name='dec_reshape')(x)

    x = layers.Conv1DTranspose(64, 5, strides=2, activation='relu', padding='same', name='dec_deconv1')(x)  # 12 → 24
    x = layers.BatchNormalization(name='dec_bn1')(x)

    x = layers.Conv1DTranspose(32, 3, strides=2, activation='relu', padding='same', name='dec_deconv2')(x)  # 24 → 48
    x = layers.BatchNormalization(name='dec_bn2')(x)

    # Crop / pad to exactly WINDOW_SIZE (50)
    x = layers.ZeroPadding1D(padding=(1, 1), name='dec_pad')(x)      # 48 → 50
    x = layers.Conv1D(NUM_CHANNELS, 1, activation='linear', name='output')(x)

    model = keras.Model(inp, x, name='SuspensionAutoencoder')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model


# =============================================================================
#  4.  CONVERT TO TFLITE
# =============================================================================
def convert_to_tflite(model, representative_data: np.ndarray):
    """Export both FP32 and fully-quantised INT8 TFLite models."""
    import tensorflow as tf

    os.makedirs(ASSETS_DIR, exist_ok=True)

    # ── FP32 (fallback) ─────────────────────────────────────────────────
    converter_fp = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32  = converter_fp.convert()
    fp32_path    = os.path.join(ASSETS_DIR, 'model_float.tflite')
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f'[✓] FP32  model saved: {fp32_path}  ({len(tflite_fp32)/1024:.1f} KB)')

    # ── INT8 (production — fastest on mobile) ────────────────────────────
    def representative_dataset():
        for i in range(min(500, len(representative_data))):
            yield [representative_data[i:i+1].astype(np.float32)]

    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_dataset
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type  = tf.float32   # keep float IO for ease of use
    converter_q.inference_output_type = tf.float32

    try:
        tflite_int8 = converter_q.convert()
        int8_path   = os.path.join(ASSETS_DIR, 'model.tflite')
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        print(f'[✓] INT8  model saved: {int8_path}  ({len(tflite_int8)/1024:.1f} KB)')
    except Exception as e:
        print(f'[!] INT8 quantisation failed ({e}), using FP32 as primary.')
        import shutil
        shutil.copy(fp32_path, os.path.join(ASSETS_DIR, 'model.tflite'))
        print(f'[✓] FP32 copied to model.tflite as fallback.')


# =============================================================================
#  5.  SAVE NORMALISATION CONSTANTS (for the React Native app)
# =============================================================================
def save_normalisation_constants(mean: np.ndarray, std: np.ndarray):
    """Write mean/std as a JSON file the app can load."""
    import json
    constants = {
        'mean': mean.tolist(),
        'std':  std.tolist(),
        'window_size': WINDOW_SIZE,
        'num_channels': NUM_CHANNELS,
        'description': 'Per-channel normalisation constants [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]'
    }
    path = os.path.join(ASSETS_DIR, 'model_config.json')
    with open(path, 'w') as f:
        json.dump(constants, f, indent=2)
    print(f'[✓] Normalisation config saved: {path}')


# =============================================================================
#  6.  THRESHOLD CALIBRATION
# =============================================================================
def calibrate_threshold(model, normal_data: np.ndarray, pothole_data: np.ndarray):
    """
    Find the optimal MSE threshold that separates normal from pothole.
    We pick the threshold at the 99th percentile of normal-data MSE
    (i.e. 1% false-positive rate on normal roads).
    """
    import json

    normal_pred  = model.predict(normal_data, batch_size=256, verbose=0)
    pothole_pred = model.predict(pothole_data, batch_size=256, verbose=0)

    normal_mse  = np.mean((normal_data - normal_pred) ** 2, axis=(1, 2))
    pothole_mse = np.mean((pothole_data - pothole_pred) ** 2, axis=(1, 2))

    # Threshold at 99th percentile of normal data
    threshold_99 = float(np.percentile(normal_mse, 99))
    threshold_95 = float(np.percentile(normal_mse, 95))

    # Detection rates
    detection_rate_99 = float(np.mean(pothole_mse > threshold_99))
    detection_rate_95 = float(np.mean(pothole_mse > threshold_95))

    print(f'\n{"="*60}')
    print(f'  THRESHOLD CALIBRATION RESULTS')
    print(f'{"="*60}')
    print(f'  Normal  MSE:  mean={normal_mse.mean():.6f}  std={normal_mse.std():.6f}')
    print(f'  Pothole MSE:  mean={pothole_mse.mean():.6f}  std={pothole_mse.std():.6f}')
    print(f'  Separation ratio:  {pothole_mse.mean() / normal_mse.mean():.1f}x')
    print(f'{"─"*60}')
    print(f'  Threshold @99%:  {threshold_99:.6f}  →  detects {detection_rate_99*100:.1f}% of potholes')
    print(f'  Threshold @95%:  {threshold_95:.6f}  →  detects {detection_rate_95*100:.1f}% of potholes')
    print(f'{"="*60}\n')

    # Save thresholds
    config_path = os.path.join(ASSETS_DIR, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['threshold_high_precision'] = threshold_99  # fewer false positives
    config['threshold_high_recall']    = threshold_95  # catches more potholes
    config['threshold_default']        = threshold_99  # use in production
    config['normal_mse_mean']   = float(normal_mse.mean())
    config['normal_mse_std']    = float(normal_mse.std())
    config['pothole_mse_mean']  = float(pothole_mse.mean())
    config['detection_rate']    = detection_rate_99
    config['separation_ratio']  = float(pothole_mse.mean() / normal_mse.mean())

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'[✓] Thresholds saved to {config_path}')

    return threshold_99


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print('\n' + '='*60)
    print('  POTHOLE AUTOENCODER — Training Pipeline')
    print('='*60 + '\n')

    # ── Generate data ────────────────────────────────────────────────────
    print('[1/6] Generating synthetic suspension data...')
    train_data   = generate_normal_driving(N_NORMAL_TRAIN)
    val_data     = generate_normal_driving(N_NORMAL_VAL)
    pothole_data = generate_pothole_events(N_POTHOLE_TEST)
    print(f'      Train: {train_data.shape}  Val: {val_data.shape}  Pothole test: {pothole_data.shape}')

    # ── Normalise ────────────────────────────────────────────────────────
    print('[2/6] Computing normalisation constants...')
    mean, std = compute_normalisation(train_data)
    print(f'      Mean: {mean}')
    print(f'      Std:  {std}')

    train_norm   = normalise(train_data, mean, std)
    val_norm     = normalise(val_data, mean, std)
    pothole_norm = normalise(pothole_data, mean, std)

    save_normalisation_constants(mean, std)

    # ── Build & train ────────────────────────────────────────────────────
    print('[3/6] Building autoencoder...')
    model = build_autoencoder()
    model.summary()

    print(f'\n[4/6] Training for {EPOCHS} epochs...')
    import tensorflow as tf
    history = model.fit(
        train_norm, train_norm,
        validation_data=(val_norm, val_norm),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
            ),
        ],
        verbose=1
    )

    final_val_loss = min(history.history['val_loss'])
    print(f'\n      Best validation loss: {final_val_loss:.6f}')

    # ── Calibrate threshold ──────────────────────────────────────────────
    print('[5/6] Calibrating detection thresholds...')
    threshold = calibrate_threshold(model, val_norm, pothole_norm)

    # ── Export TFLite ────────────────────────────────────────────────────
    print('[6/6] Converting to TFLite...')
    convert_to_tflite(model, train_norm)

    print('\n' + '='*60)
    print('  ✅  TRAINING COMPLETE')
    print(f'  Model:     assets/model.tflite')
    print(f'  Config:    assets/model_config.json')
    print(f'  Threshold: {threshold:.6f}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
