#!/usr/bin/env python3
"""step3 summary: numpy fixed-point forward pass."""

import json
import os
import sys
from datetime import datetime

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output")

with open(os.path.join(out_dir, "weights_fixed.json")) as f:
    wf = json.load(f)
with open(os.path.join(out_dir, "test_vectors.json")) as f:
    tv = json.load(f)

vectors = tv["vectors"]
num_tests = len(vectors)
num_correct = sum(1 for v in vectors if v["pred"] == v["label"])
acc = num_correct / num_tests if num_tests else 0
# Pipeline-correctness check: did fixed-point preserve every float decision?
matches_float = tv.get("matches_float")
if matches_float is None and vectors and "float_pred" in vectors[0]:
    matches_float = sum(1 for v in vectors if v["pred"] == v["float_pred"])
match_rate = (matches_float / num_tests) if (matches_float is not None and num_tests) else None

W = 24
lines = []
def emit(s=""):
    lines.append(s)
    print(s)

emit("=" * 60)
emit(" step3: NumPy Fixed-Point Forward Pass")
emit("=" * 60)
emit(f" {'timestamp':<{W}} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
weight_bits = wf.get("WEIGHT_BITS", 16)
acc_bits    = wf.get("ACC_BITS", 32)
int_bits    = weight_bits - 1 - wf['FRAC_BITS']
emit(f" {'format':<{W}} = Q{int_bits}.{wf['FRAC_BITS']} signed")
emit(f" {'data_width':<{W}} = {weight_bits}-bit")
emit(f" {'accum_width':<{W}} = {acc_bits}-bit")
emit(f" {'frac_bits':<{W}} = {wf['FRAC_BITS']}")
in_dim  = wf.get("INPUT_DIM")
hid_dim = wf.get("HIDDEN_DIM")
out_dim = wf.get("OUTPUT_DIM")
if in_dim is not None and hid_dim is not None and out_dim is not None:
    emit(f" {'network_shape':<{W}} = {in_dim} -> {hid_dim} -> {out_dim}")
emit(f" {'reads from':<{W}} = step1/output/")
emit(f" {'num_test_vectors':<{W}} = {num_tests}")
emit(f" {'fixed_point_accuracy':<{W}} = {num_correct}/{num_tests} ({100*acc:.1f}%)")
if matches_float is not None:
    emit(f" {'matches_float':<{W}} = {matches_float}/{num_tests} ({100*match_rate:.1f}%)")
emit(f" {'outputs':<{W}} = weights_fixed.json, test_vectors.json")
emit("")
# step3 PASS = quantization preserved every float decision (label accuracy
# is bounded by the float model itself, so we don't gate on it).
if matches_float is not None:
    status = "PASS" if matches_float == num_tests else "FAIL"
    emit(f" STATUS = {status} (fixed-point == float on every exported sample)")
else:
    status = "PASS" if num_correct == num_tests else "FAIL"
    emit(f" STATUS = {status}")
emit("")

with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
