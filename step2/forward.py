import numpy as np
import json
import os
import sys

# --------------------------------------------------------------------------
# step2: NumPy float forward pass
# --------------------------------------------------------------------------
# Reimplements step1's PyTorch model in pure numpy (float32) and verifies
# it matches PyTorch bit-exactly. This is the reference for step3's
# fixed-point version.
# --------------------------------------------------------------------------

input_dir  = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(input_dir, "weights.json")) as f:
    weights = json.load(f)
with open(os.path.join(input_dir, "test_data.json")) as f:
    test_data = json.load(f)

W1 = np.array(weights["W1"], dtype=np.float32)  # (128, 784)
b1 = np.array(weights["b1"], dtype=np.float32)  # (128,)
W2 = np.array(weights["W2"], dtype=np.float32)  # (10, 128)
b2 = np.array(weights["b2"], dtype=np.float32)  # (10,)

X_test = np.array(test_data["X_test_raw"], dtype=np.float32)
y_test = np.array(test_data["y_test"], dtype=np.int64)

def forward(x):
    h = np.maximum(W1 @ x + b1, 0)
    return W2 @ h + b2

print("NumPy float forward pass")
print("=" * 50)

results = []
correct = 0
for i in range(len(X_test)):
    logits = forward(X_test[i])
    pred = int(np.argmax(logits))
    label = int(y_test[i])
    status = "OK" if pred == label else "MISS"
    if status == "MISS" or i < 5:
        # Print only first 5 logit dims to keep the line short for MNIST.
        preview = np.array2string(logits, precision=2, separator=", ", max_line_width=200)
        print(f"  sample {i:2d}: pred={pred} true={label} logits={preview} {status}")
    correct += (pred == label)
    results.append({
        "x": X_test[i].tolist(),
        "logits": logits.tolist(),
        "pred": pred,
        "label": label,
    })

acc = correct / len(X_test)
print(f"\nAccuracy: {correct}/{len(X_test)} = {acc:.4f}")

pytorch_preds = test_data.get("pytorch_preds")
matches_pytorch = None
if pytorch_preds is not None:
    matches_pytorch = all(r["pred"] == p for r, p in zip(results, pytorch_preds))
    print(f"Matches PyTorch: {matches_pytorch}")

out = {
    "num_correct": correct,
    "num_tests": len(X_test),
    "accuracy": acc,
    "matches_pytorch": matches_pytorch,
    "vectors": results,
}
with open(os.path.join(output_dir, "test_vectors_float.json"), "w") as f:
    json.dump(out, f)
