#!/usr/bin/env python3
"""step2 summary: numpy float reproduction."""

import json
import os
import sys
from datetime import datetime

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output")

with open(os.path.join(out_dir, "test_vectors_float.json")) as f:
    tv = json.load(f)

W = 24
lines = []
def emit(s=""):
    lines.append(s)
    print(s)

emit("=" * 60)
emit(" step2: NumPy Float Reproduction")
emit("=" * 60)
emit(f" {'timestamp':<{W}} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
emit(f" {'framework':<{W}} = NumPy (float32)")
emit(f" {'reads from':<{W}} = step1/output/")
emit(f" {'num_test_vectors':<{W}} = {tv['num_tests']}")
emit(f" {'numpy_accuracy':<{W}} = {tv['num_correct']}/{tv['num_tests']} ({100*tv['accuracy']:.1f}%)")
emit(f" {'matches_pytorch':<{W}} = {'yes' if tv['matches_pytorch'] else 'no'}")
emit(f" {'outputs':<{W}} = test_vectors_float.json")
emit("")
# step2 is a reproduction check: did NumPy float reproduce PyTorch exactly?
# Classifier accuracy is irrelevant here; what matters is bit-for-bit
# decision parity with step1.
status = "PASS" if tv["matches_pytorch"] else "FAIL"
emit(f" STATUS = {status} (numpy float == pytorch on every exported sample)")
emit("")

with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
