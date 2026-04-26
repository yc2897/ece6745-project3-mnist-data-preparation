import numpy as np
import json
import os
import sys

# --------------------------------------------------------------------------
# Input/output directories + precision (passed as arguments)
#   argv[1] = input_dir  (has weights.json + test_data.json)
#   argv[2] = output_dir
#   argv[3] = precision (8 or 16, default 8)
# --------------------------------------------------------------------------

input_dir  = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
precision  = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(output_dir, exist_ok=True)

if precision not in (8, 16):
    sys.exit(f"ERROR: precision must be 8 or 16, got {precision}")

with open(os.path.join(input_dir, "weights.json")) as f:
    weights = json.load(f)
with open(os.path.join(input_dir, "test_data.json")) as f:
    test_data = json.load(f)

W1_f = np.array(weights["W1"], dtype=np.float32)  # (HIDDEN, INPUT)
b1_f = np.array(weights["b1"], dtype=np.float32)  # (HIDDEN,)
W2_f = np.array(weights["W2"], dtype=np.float32)  # (OUTPUT, HIDDEN)
b2_f = np.array(weights["b2"], dtype=np.float32)  # (OUTPUT,)

HIDDEN_DIM, INPUT_DIM = W1_f.shape
OUTPUT_DIM, _         = W2_f.shape

X_test_f = np.array(test_data["X_test_raw"], dtype=np.float32)  # (N, INPUT)
y_test   = np.array(test_data["y_test"], dtype=np.int64)        # (N,)

# --------------------------------------------------------------------------
# Fixed-point format
# --------------------------------------------------------------------------
# Signed fixed-point with FRAC_BITS fractional bits.
# A float value v is represented as: int(round(v * 2^FRAC_BITS))
#
# 8-bit  -> Q3.4  (1 sign + 3 integer + 4 fractional)
#           range [-8.0, +7.9375]
#           stored as int8, accumulator int32
#
# 16-bit -> Q7.8  (1 sign + 7 integer + 8 fractional)
#           range [-128.0, +127.996]
#           stored as int16, accumulator int32
#
# After a multiply of two Q-values, the result has 2*FRAC_BITS fractional
# bits. After accumulation, we right-shift by FRAC_BITS to realign.
#
# Accumulator headroom (MNIST 784 -> 128 -> 10):
#   layer1: <= 784 products of two int8/int16 values -> safely fits int32
#   layer2: <= 128 products of two int8/int16 values -> safely fits int32

if precision == 8:
    FRAC_BITS   = 4
    WEIGHT_BITS = 8
    ACC_BITS    = 32
    STORE_DTYPE = np.int8
    INT_MIN, INT_MAX = -128, 127
    Q_LABEL     = "Q3.4"
else:  # precision == 16
    FRAC_BITS   = 8
    WEIGHT_BITS = 16
    ACC_BITS    = 32
    STORE_DTYPE = np.int16
    INT_MIN, INT_MAX = -32768, 32767
    Q_LABEL     = "Q7.8"

SCALE = (1 << FRAC_BITS)
Q_MAX = INT_MAX / SCALE

def float_to_fixed(arr):
    """Convert float array to fixed-point integer in STORE_DTYPE."""
    fixed = np.round(arr * SCALE).astype(np.int32)
    fixed = np.clip(fixed, INT_MIN, INT_MAX).astype(STORE_DTYPE)
    return fixed

def fixed_to_float(arr):
    return arr.astype(np.float32) / SCALE

# --------------------------------------------------------------------------
# Quantize weights and inputs
# --------------------------------------------------------------------------

W1 = float_to_fixed(W1_f)
b1 = float_to_fixed(b1_f)
W2 = float_to_fixed(W2_f)
b2 = float_to_fixed(b2_f)
X_test = float_to_fixed(X_test_f)

print(f"Precision: {WEIGHT_BITS}-bit {Q_LABEL} (frac_bits={FRAC_BITS}, range +-{Q_MAX:.4f})")
print(f"Network shape: {INPUT_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM}")

print("Quantization error (max abs difference from original):")
print(f"  W1: {np.max(np.abs(fixed_to_float(W1) - W1_f)):.6f}")
print(f"  b1: {np.max(np.abs(fixed_to_float(b1) - b1_f)):.6f}")
print(f"  W2: {np.max(np.abs(fixed_to_float(W2) - W2_f)):.6f}")
print(f"  b2: {np.max(np.abs(fixed_to_float(b2) - b2_f)):.6f}")
print(f"  X:  {np.max(np.abs(fixed_to_float(X_test) - X_test_f)):.6f}")

for name, orig in [("W1", W1_f), ("b1", b1_f),
                   ("W2", W2_f), ("b2", b2_f),
                   ("X",  X_test_f)]:
    max_val = np.max(np.abs(orig))
    if max_val > Q_MAX:
        print(f"  WARNING: {name} has values up to {max_val:.2f}, exceeds {Q_LABEL} range (+-{Q_MAX:.4f})!")

# --------------------------------------------------------------------------
# Fixed-point forward pass (mimics hardware exactly)
# --------------------------------------------------------------------------
# Accumulate products at full int32 precision, right-shift by FRAC_BITS to
# realign, then add the bias. ReLU at hidden, raw logits at output.

def forward_fixed(x):
    """x: (INPUT_DIM,) int. returns (OUTPUT_DIM,) int32 logits."""
    x32 = x.astype(np.int32)

    h = np.zeros(HIDDEN_DIM, dtype=np.int32)
    for i in range(HIDDEN_DIM):
        acc = np.int32(0)
        for j in range(INPUT_DIM):
            acc += np.int32(W1[i, j]) * x32[j]
        acc = acc >> FRAC_BITS
        acc += np.int32(b1[i])
        h[i] = acc

    h = np.maximum(h, 0)

    y = np.zeros(OUTPUT_DIM, dtype=np.int32)
    for i in range(OUTPUT_DIM):
        acc = np.int32(0)
        for j in range(HIDDEN_DIM):
            acc += np.int32(W2[i, j]) * np.int32(h[j])
        acc = acc >> FRAC_BITS
        acc += np.int32(b2[i])
        y[i] = acc

    return y

# --------------------------------------------------------------------------
# Run inference and compare
# --------------------------------------------------------------------------

print(f"\nFixed-point forward pass ({Q_LABEL})")
print("=" * 50)

# Reference predictions from the float (PyTorch) model. Pipeline correctness
# is judged by whether quantization preserves these decisions, NOT by raw
# label accuracy (the float model itself isn't 100% on MNIST).
float_preds = test_data.get("pytorch_preds")
if float_preds is None:
    sys.exit("ERROR: test_data.json missing 'pytorch_preds'; rerun step1.")

correct = 0
matches_float = 0
all_logits = []
for i in range(len(X_test)):
    logits = forward_fixed(X_test[i])
    all_logits.append(logits)
    pred = int(np.argmax(logits))
    label = int(y_test[i])
    fp = int(float_preds[i])
    label_status = "OK"   if pred == label else "MISS"
    float_status = "SAME" if pred == fp    else "DIFF"
    if float_status == "DIFF" or label_status == "MISS" or i < 5:
        print(f"  sample {i:2d}: pred={pred} true={label} float={fp} "
              f"logits={logits.tolist()} {label_status} {float_status}")
    correct       += (pred == label)
    matches_float += (pred == fp)

acc = correct / len(X_test)
match_rate = matches_float / len(X_test)
print(f"\nLabel accuracy:        {correct}/{len(X_test)} = {acc:.4f}")
print(f"Matches float (step1): {matches_float}/{len(X_test)} = {match_rate:.4f}")
if matches_float == len(X_test):
    print("PASS: Fixed-point preserves every float decision (no quantization-induced flips).")
else:
    flips = len(X_test) - matches_float
    print(f"FAIL: {flips} prediction(s) changed due to quantization.")

# --------------------------------------------------------------------------
# Export fixed-point weights + test vectors
# --------------------------------------------------------------------------

fixed_weights = {
    "FRAC_BITS":   FRAC_BITS,
    "WEIGHT_BITS": WEIGHT_BITS,
    "ACC_BITS":    ACC_BITS,
    "INPUT_DIM":   INPUT_DIM,
    "HIDDEN_DIM":  HIDDEN_DIM,
    "OUTPUT_DIM":  OUTPUT_DIM,
    "W1": W1.tolist(),
    "b1": b1.tolist(),
    "W2": W2.tolist(),
    "b2": b2.tolist(),
}
with open(os.path.join(output_dir, "weights_fixed.json"), "w") as f:
    json.dump(fixed_weights, f)

test_vectors = {
    "FRAC_BITS":     FRAC_BITS,
    "WEIGHT_BITS":   WEIGHT_BITS,
    "ACC_BITS":      ACC_BITS,
    "INPUT_DIM":     INPUT_DIM,
    "HIDDEN_DIM":    HIDDEN_DIM,
    "OUTPUT_DIM":    OUTPUT_DIM,
    "matches_float": int(matches_float),
    "vectors":       [],
}
for i in range(len(X_test)):
    logits = all_logits[i]
    test_vectors["vectors"].append({
        "x":          X_test[i].tolist(),
        "logits":     logits.tolist(),
        "pred":       int(np.argmax(logits)),
        "label":      int(y_test[i]),
        "float_pred": int(float_preds[i]),
    })
with open(os.path.join(output_dir, "test_vectors.json"), "w") as f:
    json.dump(test_vectors, f)

print(f"\nFixed-point weights saved to weights_fixed.json")
print(f"Test vectors saved to test_vectors.json")
