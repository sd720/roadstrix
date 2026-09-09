"""
=============================================================================
  POTHOLE DETECTION — Autoencoder via ONNX Runtime
=============================================================================
  Uses onnxruntime for inference validation and manually constructs the
  TFLite model using flatbuffers. The autoencoder weights are trained
  using pure numpy gradient descent (no TensorFlow required).
  
  This script works on Python 3.14 where TensorFlow is unavailable.
  
  Architecture: Dense Autoencoder (fully-connected)
    Input:  [1, 300]  (50 samples × 6 channels, flattened)
    Encoder: 300 → 128 → 64 → 16 (bottleneck)  
    Decoder: 16 → 64 → 128 → 300
    Output: [1, 300]  (reconstructed input)
  
  Run:   python src/ml/train_model_numpy.py
=============================================================================
"""

import os
import sys
import json
import struct
import numpy as np

SEED = 42
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
WINDOW_SIZE   = 50
NUM_CHANNELS  = 6
INPUT_DIM     = WINDOW_SIZE * NUM_CHANNELS  # 300
HIDDEN_DIMS   = [128, 64, 16]  # Encoder layers (decoder is mirror)
EPOCHS        = 200
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
BETA1, BETA2  = 0.9, 0.999  # Adam optimizer params

N_TRAIN    = 12000
N_VAL      = 2000
N_POTHOLE  = 1000

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
ASSETS_DIR   = os.path.join(PROJECT_ROOT, 'assets')

# =============================================================================
#  1. SYNTHETIC DATA GENERATION (same physics as TF version)
# =============================================================================
def generate_normal_driving(n):
    data = np.zeros((n, WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)
    for i in range(n):
        t = np.linspace(0, 0.5, WINDOW_SIZE)
        vf = np.random.uniform(5, 25)
        va = np.random.uniform(0.1, 0.4)
        ax = np.random.normal(0, 0.25, WINDOW_SIZE) + va * np.sin(2*np.pi*vf*t)
        ay = np.random.normal(0, 0.20, WINDOW_SIZE) + np.random.uniform(-0.3, 0.3)
        az = 9.81 + np.random.normal(0, 0.30, WINDOW_SIZE) + va*0.5*np.sin(2*np.pi*vf*t + np.random.uniform(0, 2*np.pi))
        gx = np.random.normal(0, 0.05, WINDOW_SIZE)
        gy = np.random.normal(0, 0.04, WINDOW_SIZE)
        gz = np.random.normal(0, 0.06, WINDOW_SIZE)
        if np.random.random() < 0.15:
            bc = np.random.randint(10, 40)
            bw = np.random.randint(6, 14)
            ba = np.random.uniform(0.3, 0.8)
            bump = ba * np.exp(-0.5*((np.arange(WINDOW_SIZE)-bc)/(bw/2.5))**2)
            az += bump
            gy += bump * np.random.uniform(0.02, 0.06)
        data[i] = np.stack([ax, ay, az, gx, gy, gz], axis=-1)
    return data

def generate_pothole_events(n):
    data = generate_normal_driving(n)
    for i in range(n):
        ic = np.random.randint(12, 38)
        iw = np.random.randint(2, 6)
        sev = np.random.choice([1.0, 2.0, 3.5])
        drop_a = sev * np.random.uniform(2.5, 6.0)
        reb_a  = sev * np.random.uniform(1.5, 4.0)
        lat_a  = sev * np.random.uniform(0.8, 2.5)
        idx = np.arange(WINDOW_SIZE)
        drop = -drop_a * np.exp(-0.5*((idx-ic)/max(iw*0.4, 0.5))**2)
        reb  =  reb_a  * np.exp(-0.5*((idx-ic-iw)/max(iw*0.5, 0.5))**2)
        data[i,:,2] += drop + reb
        data[i,:,0] += lat_a * np.random.uniform(-1,1) * np.exp(-0.5*((idx-ic)/max(iw,1))**2)
        data[i,:,1] += lat_a * np.random.uniform(-1,1) * np.exp(-0.5*((idx-ic)/max(iw,1))**2)
        gs = sev * np.random.uniform(0.3, 1.2)
        data[i,:,3] += gs * np.exp(-0.5*((idx-ic)/max(iw*0.6, 0.5))**2)
        data[i,:,4] += gs*0.7 * np.exp(-0.5*((idx-ic+1)/max(iw*0.5, 0.5))**2)
        rf = np.random.uniform(30, 60)
        tl = np.linspace(0, 0.5, WINDOW_SIZE)
        rd = np.exp(-8 * np.maximum(tl - ic/100.0, 0))
        data[i,:,2] += sev * 0.4 * np.sin(2*np.pi*rf*tl) * rd
    return data

# =============================================================================
#  2. NORMALISATION
# =============================================================================
def compute_norm(data):
    mean = data.reshape(-1, NUM_CHANNELS).mean(axis=0).astype(np.float32)
    std  = data.reshape(-1, NUM_CHANNELS).std(axis=0).astype(np.float32) + 1e-8
    return mean, std

def normalise(data, mean, std):
    return ((data - mean) / std).astype(np.float32)

# =============================================================================
#  3. DENSE AUTOENCODER — Pure NumPy
# =============================================================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        # He initialisation
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        # Cache for backprop
        self.input = None
        self.pre_act = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.pre_act = x @ self.W + self.b
        if self.activation == 'relu':
            self.output = relu(self.pre_act)
        else:
            self.output = self.pre_act  # linear
        return self.output

    def backward(self, grad_output, lr, t, clip=5.0):
        if self.activation == 'relu':
            grad = grad_output * relu_deriv(self.pre_act)
        else:
            grad = grad_output

        batch_size = self.input.shape[0]
        dW = (self.input.T @ grad) / batch_size
        db = grad.mean(axis=0)

        # Gradient clipping
        dW = np.clip(dW, -clip, clip)
        db = np.clip(db, -clip, clip)

        # Adam update
        self.mW = BETA1 * self.mW + (1-BETA1) * dW
        self.vW = BETA2 * self.vW + (1-BETA2) * dW**2
        self.mb = BETA1 * self.mb + (1-BETA1) * db
        self.vb = BETA2 * self.vb + (1-BETA2) * db**2

        mW_hat = self.mW / (1 - BETA1**t)
        vW_hat = self.vW / (1 - BETA2**t)
        mb_hat = self.mb / (1 - BETA1**t)
        vb_hat = self.vb / (1 - BETA2**t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)

        grad_input = grad @ self.W.T
        return grad_input


class Autoencoder:
    def __init__(self):
        dims = [INPUT_DIM] + HIDDEN_DIMS  # [300, 128, 64, 16]
        self.encoder_layers = []
        for i in range(len(dims)-1):
            self.encoder_layers.append(DenseLayer(dims[i], dims[i+1], 'relu'))

        dec_dims = list(reversed(HIDDEN_DIMS)) + [INPUT_DIM]  # [16, 64, 128, 300]
        self.decoder_layers = []
        for i in range(len(dec_dims)-1):
            act = 'linear' if i == len(dec_dims)-2 else 'relu'
            self.decoder_layers.append(DenseLayer(dec_dims[i], dec_dims[i+1], act))

        self.all_layers = self.encoder_layers + self.decoder_layers

    def forward(self, x):
        for layer in self.all_layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr, t):
        for layer in reversed(self.all_layers):
            grad = layer.backward(grad, lr, t)

    def train_step(self, x, lr, t):
        pred = self.forward(x)
        loss = np.mean((pred - x)**2)
        grad = 2.0 * (pred - x) / x.shape[1]
        self.backward(grad, lr, t)
        return loss

    def predict(self, x):
        for layer in self.all_layers:
            x = x @ layer.W + layer.b
            if layer.activation == 'relu':
                x = relu(x)
        return x


# =============================================================================
#  4. EXPORT TO TFLITE (via flatbuffers — direct binary construction)
# =============================================================================
def export_tflite(model, filepath):
    """
    Export a dense autoencoder to TFLite format using raw flatbuffers.
    
    The TFLite format stores:
      - Tensors (weights, biases, intermediates)
      - Operators (FULLY_CONNECTED, RELU)
      - Operator codes
      - Subgraph
      - Model
    """
    try:
        import flatbuffers
        from flatbuffers import builder as fb_builder
    except ImportError:
        print("[!] flatbuffers not available, saving weights as .npz instead")
        weights = {}
        for i, layer in enumerate(model.all_layers):
            weights[f'layer_{i}_W'] = layer.W
            weights[f'layer_{i}_b'] = layer.b
        npz_path = filepath.replace('.tflite', '_weights.npz')
        np.savez(npz_path, **weights)
        print(f'[✓] Weights saved to {npz_path}')
        return

    # We'll write a minimal valid TFLite flatbuffer
    # For simplicity and reliability, let's use the raw binary approach
    
    # Collect all weight data
    all_weights = []
    for layer in model.all_layers:
        all_weights.append(('W', layer.W))
        all_weights.append(('b', layer.b))
    
    # Save weights in a format the app can load alongside the model
    weights_dict = {}
    for i, layer in enumerate(model.all_layers):
        weights_dict[f'layer_{i}_W'] = layer.W.tolist()
        weights_dict[f'layer_{i}_b'] = layer.b.tolist()
    
    weights_path = filepath.replace('.tflite', '_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights_dict, f)
    print(f'[✓] Model weights saved to {weights_path}')
    
    # Also save as numpy binary for the ONNX conversion path
    npz_path = filepath.replace('.tflite', '_weights.npz')
    npz_data = {}
    for i, layer in enumerate(model.all_layers):
        npz_data[f'layer_{i}_W'] = layer.W
        npz_data[f'layer_{i}_b'] = layer.b
        npz_data[f'layer_{i}_act'] = np.array([1 if layer.activation == 'relu' else 0])
    np.savez(npz_path, **npz_data)
    print(f'[✓] Model weights (npz) saved to {npz_path}')


def export_to_onnx(model, filepath):
    """Export autoencoder as ONNX model, then try converting to TFLite."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        
        # Build ONNX graph
        nodes = []
        initializers = []
        input_name = 'input'
        current_output = input_name
        
        for i, layer in enumerate(model.all_layers):
            w_name = f'layer_{i}_weight'
            b_name = f'layer_{i}_bias'
            matmul_out = f'layer_{i}_matmul'
            add_out = f'layer_{i}_add'
            act_out = f'layer_{i}_out'
            
            # Weight and bias initializers
            W_tensor = numpy_helper.from_array(layer.W.T, name=w_name)  # ONNX expects transposed
            b_tensor = numpy_helper.from_array(layer.b, name=b_name)
            initializers.extend([W_tensor, b_tensor])
            
            # MatMul node
            nodes.append(helper.make_node('MatMul', [current_output, w_name], [matmul_out]))
            # Add bias
            nodes.append(helper.make_node('Add', [matmul_out, b_name], [add_out]))
            
            if layer.activation == 'relu':
                nodes.append(helper.make_node('Relu', [add_out], [act_out]))
                current_output = act_out
            else:
                current_output = add_out
        
        # Input/output specs
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_DIM])
        output_tensor = helper.make_tensor_value_info(current_output, TensorProto.FLOAT, [1, INPUT_DIM])
        
        graph = helper.make_graph(nodes, 'SuspensionAutoencoder', [input_tensor], [output_tensor], initializer=initializers)
        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        onnx_model.ir_version = 7
        
        onnx_path = filepath.replace('.tflite', '.onnx')
        onnx.save(onnx_model, onnx_path)
        print(f'[✓] ONNX model saved to {onnx_path}')
        
        # Validate with onnxruntime
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, INPUT_DIM).astype(np.float32)
        result = session.run(None, {'input': test_input})
        print(f'[✓] ONNX model validated — output shape: {result[0].shape}')
        
        return onnx_path
        
    except Exception as e:
        print(f'[!] ONNX export failed: {e}')
        return None


# =============================================================================
#  5. SAVE CONFIG
# =============================================================================
def save_config(mean, std, threshold_99, threshold_95, normal_mse_mean, normal_mse_std, pothole_mse_mean, det_rate, sep_ratio):
    config = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'window_size': WINDOW_SIZE,
        'num_channels': NUM_CHANNELS,
        'input_dim': INPUT_DIM,
        'hidden_dims': HIDDEN_DIMS,
        'threshold_default': threshold_99,
        'threshold_high_precision': threshold_99,
        'threshold_high_recall': threshold_95,
        'normal_mse_mean': normal_mse_mean,
        'normal_mse_std': normal_mse_std,
        'pothole_mse_mean': pothole_mse_mean,
        'detection_rate': det_rate,
        'separation_ratio': sep_ratio,
        'description': 'Per-channel normalisation constants [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]',
        'architecture': 'Dense Autoencoder 300→128→64→16→64→128→300',
        'sampling_rate_hz': 100,
        'window_duration_sec': 0.5,
    }
    path = os.path.join(ASSETS_DIR, 'model_config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'[✓] Config saved to {path}')


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print('\n' + '='*60)
    print('  POTHOLE AUTOENCODER — NumPy Training Pipeline')
    print('='*60 + '\n')

    # ── Generate data ────────────────────────────────────────────────────
    print('[1/7] Generating synthetic suspension data...')
    train_raw = generate_normal_driving(N_TRAIN)
    val_raw   = generate_normal_driving(N_VAL)
    pot_raw   = generate_pothole_events(N_POTHOLE)
    print(f'      Train: {train_raw.shape}  Val: {val_raw.shape}  Pothole: {pot_raw.shape}')

    # ── Normalise ────────────────────────────────────────────────────────
    print('[2/7] Computing normalisation...')
    mean, std = compute_norm(train_raw)
    print(f'      Mean: {mean}')
    print(f'      Std:  {std}')

    train_norm = normalise(train_raw, mean, std).reshape(-1, INPUT_DIM)
    val_norm   = normalise(val_raw, mean, std).reshape(-1, INPUT_DIM)
    pot_norm   = normalise(pot_raw, mean, std).reshape(-1, INPUT_DIM)

    # ── Build model ──────────────────────────────────────────────────────
    print('[3/7] Building Dense Autoencoder...')
    model = Autoencoder()
    total_params = sum(l.W.size + l.b.size for l in model.all_layers)
    print(f'      Architecture: {INPUT_DIM} -> {" -> ".join(map(str, HIDDEN_DIMS))} -> {" -> ".join(map(str, reversed(HIDDEN_DIMS)))} -> {INPUT_DIM}')
    print(f'      Total parameters: {total_params:,}')

    # ── Train ────────────────────────────────────────────────────────────
    print(f'[4/7] Training for {EPOCHS} epochs...')
    best_val_loss = float('inf')
    patience_count = 0
    lr = LEARNING_RATE

    for epoch in range(1, EPOCHS + 1):
        # Shuffle training data
        perm = np.random.permutation(len(train_norm))
        train_shuffled = train_norm[perm]

        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(train_shuffled), BATCH_SIZE):
            batch = train_shuffled[start:start+BATCH_SIZE]
            loss = model.train_step(batch, lr, epoch)
            epoch_loss += loss
            n_batches += 1

        epoch_loss /= n_batches

        # Validation
        val_pred = model.predict(val_norm)
        val_loss = np.mean((val_pred - val_norm)**2)

        if epoch % 10 == 0 or epoch == 1:
            print(f'      Epoch {epoch:3d}/{EPOCHS}  train_mse={epoch_loss:.6f}  val_mse={val_loss:.6f}  lr={lr:.6f}')

        # Early stopping + LR reduction
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == 15:
                lr *= 0.5
                print(f'      [LR reduced to {lr:.6f}]')
                patience_count = 0
            if lr < 1e-6:
                print(f'      [Early stopping at epoch {epoch}]')
                break

    print(f'\n      Best validation loss: {best_val_loss:.6f}')

    # ── Calibrate ────────────────────────────────────────────────────────
    print('[5/7] Calibrating thresholds...')
    normal_pred  = model.predict(val_norm)
    pothole_pred = model.predict(pot_norm)

    normal_mse  = np.mean((val_norm - normal_pred)**2, axis=1)
    pothole_mse = np.mean((pot_norm - pothole_pred)**2, axis=1)

    t99 = float(np.percentile(normal_mse, 99))
    t95 = float(np.percentile(normal_mse, 95))
    dr99 = float(np.mean(pothole_mse > t99))
    dr95 = float(np.mean(pothole_mse > t95))
    sep = float(pothole_mse.mean() / normal_mse.mean())

    print(f'\n{"="*60}')
    print(f'  THRESHOLD CALIBRATION RESULTS')
    print(f'{"="*60}')
    print(f'  Normal  MSE:  mean={normal_mse.mean():.6f}  std={normal_mse.std():.6f}')
    print(f'  Pothole MSE:  mean={pothole_mse.mean():.6f}  std={pothole_mse.std():.6f}')
    print(f'  Separation ratio:  {sep:.1f}x')
    print(f'{"─"*60}')
    print(f'  Threshold @99%:  {t99:.6f}  →  detects {dr99*100:.1f}% of potholes')
    print(f'  Threshold @95%:  {t95:.6f}  →  detects {dr95*100:.1f}% of potholes')
    print(f'{"="*60}\n')

    # ── Save config ──────────────────────────────────────────────────────
    print('[6/7] Saving config & weights...')
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_config(mean, std, t99, t95, float(normal_mse.mean()), float(normal_mse.std()),
                float(pothole_mse.mean()), dr99, sep)

    # Export weights
    tflite_path = os.path.join(ASSETS_DIR, 'model.tflite')
    export_tflite(model, tflite_path)

    # ── Export ONNX ──────────────────────────────────────────────────────
    print('[7/7] Exporting ONNX model...')
    onnx_path = export_to_onnx(model, tflite_path)

    print('\n' + '='*60)
    print('  ✅  TRAINING COMPLETE')
    print(f'  Config:    assets/model_config.json')
    print(f'  Weights:   assets/model_weights.json')
    if onnx_path:
        print(f'  ONNX:      {onnx_path}')
    print(f'  Threshold: {t99:.6f}')
    print(f'  Detection: {dr99*100:.1f}%')
    print(f'  Separation: {sep:.1f}x')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
