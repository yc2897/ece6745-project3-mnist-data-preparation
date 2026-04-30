#!/usr/bin/env python3
"""step1 summary: read PyTorch training outputs and write summary.txt."""

import json
import os
import sys
from datetime import datetime

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output")

with open(os.path.join(out_dir, "weights.json")) as f:
    w = json.load(f)
with open(os.path.join(out_dir, "test_data.json")) as f:
    td = json.load(f)

W = 24
lines = []
def emit(s=""):
    lines.append(s)
    print(s)

in_dim  = w.get("INPUT_DIM",  len(w['W1'][0]))
hid_dim = w.get("HIDDEN_DIM", len(w['W1']))
out_dim = w.get("OUTPUT_DIM", len(w['W2']))
res     = w.get("RESOLUTION", 28)

emit("=" * 60)
emit(" step1: PyTorch Train (float)")
emit("=" * 60)
emit(f" {'timestamp':<{W}} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
emit(f" {'model':<{W}} = {in_dim} -> {hid_dim} (ReLU) -> {out_dim}")
emit(f" {'resolution':<{W}} = {res}x{res}")
emit(f" {'framework':<{W}} = PyTorch (float32)")
emit(f" {'seed':<{W}} = 42")
emit(f" {'W1 shape':<{W}} = {len(w['W1'])}x{len(w['W1'][0])}")
emit(f" {'b1 shape':<{W}} = {len(w['b1'])}")
emit(f" {'W2 shape':<{W}} = {len(w['W2'])}x{len(w['W2'][0])}")
emit(f" {'b2 shape':<{W}} = {len(w['b2'])}")
num_params = (len(w['W1']) * len(w['W1'][0]) + len(w['b1'])
              + len(w['W2']) * len(w['W2'][0]) + len(w['b2']))
emit(f" {'num_params':<{W}} = {num_params}")
n_full = td.get("pytorch_full_num_tests")
acc_full = td.get("pytorch_full_accuracy")
if n_full is not None and acc_full is not None:
    emit(f" {'full_test_set':<{W}} = {n_full}")
    emit(f" {'full_test_accuracy':<{W}} = {100*acc_full:.2f}%")
emit(f" {'exported_vectors':<{W}} = {len(td['y_test'])}")
n_c = td.get("pytorch_num_correct", 0)
n_t = td.get("pytorch_num_tests", len(td["y_test"]))
emit(f" {'exported_accuracy':<{W}} = {n_c}/{n_t} ({100*n_c/n_t:.1f}%)")
emit(f" {'outputs':<{W}} = weights.json, test_data.json")
emit("")
# Sanity check: did the model actually learn? MNIST baseline target is
# ~97%; we accept anything >= 95% as a successful train.
ACC_THRESHOLD = 0.95
status = "PASS" if (acc_full is not None and acc_full >= ACC_THRESHOLD) else "FAIL"
emit(f" STATUS = {status} (full-test acc threshold {100*ACC_THRESHOLD:.0f}%)")
emit("")

with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
