# ece6745 mnist data preparation

This repo prepares data for an ECE 6745 hardware accelerator that classifies
MNIST handwritten digits. It mirrors the iris repo's pipeline — three steps
run by `./run`:

- **Step 1:** Train an `INPUT_DIM → HIDDEN_DIM → 10` neural net in PyTorch
  and save the weights (float32). Default `784 → 128 → 10` matches the
  canonical TF/Keras MNIST tutorial baseline (~97% test accuracy on the
  full 10k test set). Resolution and hidden width are now CLI knobs (see
  below).
- **Step 2:** Redo the forward pass in NumPy floats (float32) to confirm it
  matches PyTorch on the exported subset.
- **Step 3:** Convert everything (float32) to signed fixed-point integers
  and check predictions still match — both on the 30-sample exported subset
  and on the full 10,000-sample MNIST test set.

The final fixed-point weights and a 30-sample golden test vector subset
become the reference for the C and Verilog versions built later.

## Configurable knobs

`./run` now takes three knobs as named flags or env vars. All have defaults
matching the original behavior.

| Knob | Default | Meaning | Affects |
|---|---|---|---|
| `--precision=4` / `=8` / `=16` | 8 | Fixed-point format (Q2.1 / Q3.4 / Q7.8) | step3 |
| `--resolution=R` | 28 | Image side, in pixels (R×R = INPUT_DIM) | step1 (data pipeline + model) |
| `--hidden=H` | 128 | Hidden-layer width | step1 (model) |

Examples:

```sh
./run                                                # 28x28, h=128, 8-bit  (default)
./run --precision=16                                 # same shape, 16-bit
./run --precision=8 --resolution=14 --hidden=64      # 14x14, h=64, 8-bit
PRECISION=8 RESOLUTION=14 HIDDEN=64 ./run            # equivalent
./run --help
```

For smaller resolutions, the input pipeline applies a bilinear `Resize` to
each MNIST image *before* normalizing, then flattens to a length-`R*R`
input vector. So `--resolution=14` produces a 196-dim input; the rest of
the flow (training, fixed-point conversion, golden vector export) just
sees a smaller flat input.

## Variant-named results

Each `./run` produces a self-contained directory under `results/` whose
name encodes the topology + precision:

```
results/<INPUT_DIM>x<HIDDEN>_<precision>bits/
```

For example: `results/784x128_8bits/`, `results/196x64_4bits/`,
`results/49x8_8bits/`. Re-running the same combination overwrites that
directory; running a different combination adds a sibling directory. The
`results/` tree therefore accumulates one directory per (resolution,
hidden, precision) combo you've trained.

Each `results/.../` contains `weights_fixed.json`, `test_vectors.json`,
`summary.txt`, and the auto-generated C headers `weights_<bits>.h` /
`test_vectors_<bits>.h` (see [Generate C headers](#generate-c-headers)).

## Network shape

By default `784 → 128 (ReLU) → 10`. Drop-in for the parameterized hardwired
accelerator: feed `INPUT_DIM`, `HIDDEN_DIM`, `OUTPUT_DIM` from
`weights_fixed.json` into the SystemVerilog parameters of `Proj3Xcel.v`.

Param count for any `R×R → H → 10` MLP is `R²·H + H + H·10 + 10`. The 15
topologies in the current sweep:

| | h=128 | h=64 | h=32 | h=16 | h=8 |
|---|---|---|---|---|---|
| **28×28** (input=784) | 101,770 | 50,890 | 25,450 | 12,730 | 6,370 |
| **14×14** (input=196) |  26,506 | 13,258 |  6,634 |  3,322 | 1,666 |
| **7×7**   (input=49)  |   7,690 |  3,850 |  1,930 |    970 |   490 |

## Dataset and split

MNIST ships pre-split (Yann LeCun's 1998 partition): **60,000 train +
10,000 test**, used as-is. We do not re-split — every published MNIST
accuracy number uses the same partition, so ours are directly comparable.

## Preprocessing

Standard PyTorch MNIST normalization: `(pixel/255 − 0.1307) / 0.3081`,
where 0.1307 / 0.3081 are the global MNIST mean/std. Conceptually the same
as iris's StandardScaler (zero-mean, ~unit-variance), and preferred over
plain `/255` because it spreads pixel values across more of the
fixed-point range — giving much better resolution under 8-bit Q3.4
quantization.

For `--resolution < 28`, a `transforms.Resize((R,R), bilinear,
antialias=True)` is inserted **before** normalize so the per-pixel
distribution stays comparable across resolutions.

## Test vector export

The full 10,000-sample test set is too large to embed in a C/RTL
testbench. Step 1 exports the **first 30** test images (in dataset order,
deterministic across runs) as the hardware golden vectors. Step 3 also
runs fixed-point inference on **all 10,000 samples** and reports accuracy
there — that's where you see the real precision and resolution
accuracy gaps (the 30-sample subset is too coarse to resolve them; see
[Sweep results](#sweep-results-float-vs-fixed-point-accuracy-on-full-10k)).

## Precision (step 3)

Step 3 supports two fixed-point formats:

| Value | Format | Range | Frac bits | Storage | Accumulator |
|---|---|---|---|---|---|
| `4`                | Q2.1 signed | ±3.5       | 1 | int4 (in int8) | int32 |
| **`8` (default)**  | Q3.4 signed | ±7.9375    | 4 | int8  | int32 |
| `16`               | Q7.8 signed | ±127.996   | 8 | int16 | int32 |

The 4-bit case uses Q2.1 (not Q1.2) because MNIST normalized pixels reach
~3.0; Q1.2's ±1.75 range would saturate every brightish pixel.

Accumulator headroom: layer 1 sums up to `INPUT_DIM` int8/int16 products,
layer 2 sums up to `HIDDEN_DIM`. Both fit comfortably in int32 for the
trained weight magnitudes; step 3 also asserts this in the vectorized
full-10k pass.

## Sweep results: float vs fixed-point accuracy on full 10k

The full sweep covers 3 resolutions × 5 hidden widths × 2 active
precisions (8 and 4) = 30 cells. All numbers below are full 10,000-sample
MNIST test accuracy. The float column is the PyTorch model; the 8-bit /
4-bit columns are the same model after fixed-point conversion in step3.

Run with seed = 42, 6 epochs, Adam(lr=0.001), batch=128.

| Topology | Params | **Float** | **8-bit FP** | **4-bit FP** |
|---|---|---|---|---|
| **— hidden = 128 —** | | | | |
| 784 → 128 → 10 | 101,770 | 97.69 % | 97.39 % | 18.09 % |
| 196 → 128 → 10 |  26,506 | 97.18 % | 97.13 % | 37.89 % |
| 49  → 128 → 10 |   7,690 | 95.07 % | 94.89 % | 74.09 % |
| **— hidden = 64 —** | | | | |
| 784 → 64  → 10 |  50,890 | 97.28 % | 96.96 % | 11.81 % |
| 196 → 64  → 10 |  13,258 | 96.41 % | 96.35 % | 47.85 % |
| 49  → 64  → 10 |   3,850 | 93.82 % | 93.48 % | 84.12 % |
| **— hidden = 32 —** | | | | |
| 784 → 32  → 10 |  25,450 | 96.31 % | 96.14 % | 16.33 % |
| 196 → 32  → 10 |   6,634 | 94.94 % | 94.93 % | 52.28 % |
| 49  → 32  → 10 |   1,930 | 92.53 % | 92.20 % | 78.50 % |
| **— hidden = 16 —** | | | | |
| 784 → 16  → 10 |  12,730 | 94.46 % | 94.23 % | 18.30 % |
| 196 → 16  → 10 |   3,322 | 93.60 % | 93.46 % | 64.99 % |
| 49  → 16  → 10 |     970 | 90.78 % | 90.42 % | 69.52 % |
| **— hidden = 8 —** | | | | |
| 784 → 8   → 10 |   6,370 | 92.21 % | 91.04 % | 24.05 % |
| 196 → 8   → 10 |   1,666 | 90.47 % | 90.45 % | 62.14 % |
| 49  → 8   → 10 |     490 | 87.04 % | 86.80 % | 72.98 % |

### Reading this

**8-bit Q3.4 is essentially free across the entire matrix.** The largest
fixed-vs-float gap in any cell is 1.17 % (784→8); the median gap is under
0.3 %. For ASIC purposes treat 8-bit FP accuracy as ≈ float accuracy and
choose topology by float number alone.

**4-bit Q2.1 collapses everywhere — never reaches the 30/30 step3 gate.**
The collapse is *resolution-driven, not size-driven*:

| Resolution | 4-bit best (across hidden) | 4-bit worst |
|---|---|---|
| 28×28 (input=784) | 24.05 % (h=8) | 11.81 % (h=64) |
| 14×14 (input=196) | 64.99 % (h=16) | 37.89 % (h=128) |
| 7×7  (input=49)  | 84.12 % (h=64) | 69.52 % (h=16) |

With Q2.1's coarse 0.5 weight resolution, every weight under magnitude
0.25 quantizes to zero, sparsifying the network. Bigger input dimensions
have more dead weights per layer-1 row, which destroys layer-1 outputs
faster than smaller hidden dimensions reduce capacity. So a 49→64 net
(3,850 params) survives 4-bit better than a 784→128 net (101,770 params)
— the inverse of the usual "bigger network is more robust" intuition.

Practical implication: **without quantization-aware training, 4-bit is
not usable on this MLP at any size. Stick to 8-bit for any ASIC variant
you intend to actually run.** The 4-bit results are still valuable as a
"how far does post-training quantization break before QAT is required"
data point.

### Surprise: tiny networks still classify MNIST

The 49 → 8 → 10 net has only **490 parameters**, runs in ~16 KB of
hardware register state, and still reaches **86.80 % at 8-bit fixed**
(vs 87.04 % float). That's 8.7× better than random guessing on a
10-class task. MNIST is mostly linearly separable — even logistic
regression hits ~91–92 % — so the hidden layer mostly buys you the last
~5 %. Useful for the smallest hardwired-MLP corner of the experiment.

## Reproduce result

1. Clone:

    ```sh
    git clone <repo>
    cd ece6745-project3-mnist-data-preparation
    ```

2. Run defaults (28×28, h=128, 8-bit):

    ```sh
    ./run
    ```

3. Run a specific variant (e.g., 14×14, h=64, 8-bit):

    ```sh
    ./run --precision=8 --resolution=14 --hidden=64
    ```

4. Run a full sweep across all 15 topologies × {8-bit, 4-bit}:

    ```sh
    for res in 28 14 7; do
      for hid in 128 64 32 16 8; do
        for prec in 8 4; do
          ./run --precision=$prec --resolution=$res --hidden=$hid
        done
      done
    done
    ```

5. Run only step3 with a different precision, reusing existing step1/2
   output (avoids retraining):

    ```sh
    ./step3/run 4
    ```

Default seed at the training step is 42, so the result is reproducible.
On first run, torchvision downloads MNIST into `./data/` (cached for
later runs).

## Outputs

### `step1/output/` (overwritten each run)
- `weights.json` — float32 W1, b1, W2, b2 + normalization mean/std + dim
  metadata (`RESOLUTION`, `INPUT_DIM`, `HIDDEN_DIM`, `OUTPUT_DIM`).
- `test_data.json` — the 30 exported (already-normalized) inputs, labels,
  PyTorch predictions on those 30, plus the full-10k PyTorch accuracy and
  the same dim metadata.
- `summary.txt` — human-readable summary (now includes resolution and
  derived shape rather than the hardcoded `784 → 128 → 10`).

### `step2/output/` (overwritten each run)
- `test_vectors_float.json` — the 30 vectors with NumPy-float logits and
  predictions; verifies bit-for-bit reproduction of PyTorch.
- `summary.txt`.

### `step3/output/` (overwritten each run)
- `weights_fixed.json` — quantized W1, b1, W2, b2 with metadata
  (`FRAC_BITS`, `WEIGHT_BITS`, `ACC_BITS`, `RESOLUTION`, `INPUT_DIM`,
  `HIDDEN_DIM`, `OUTPUT_DIM`).
- `test_vectors.json` — 30 quantized input vectors with expected
  fixed-point logits, labels, and the float prediction for each;
  also full-10k accuracy fields (`full_float_accuracy`,
  `full_fixed_accuracy`, `full_matches_float`, `full_match_rate`).
- `summary.txt`.

### `results/<INPUT_DIM>x<HIDDEN>_<precision>bits/` (preserved per variant)
After each successful `./run`, the top-level script copies step3's
artifacts into a per-variant directory and runs `gen_headers_for_app.py`
to produce the matching C headers. Contents:

- `weights_fixed.json`, `test_vectors.json`, `summary.txt` — copied from
  `step3/output/`.
- `weights_<bits>.h` — auto-generated C header with weights and dim
  `#define`s (`PROJ3_FRAC_BITS`, `PROJ3_RESOLUTION`, `PROJ3_INPUT_DIM`,
  `PROJ3_HIDDEN_DIM`, `PROJ3_OUTPUT_DIM`).
- `test_vectors_<bits>.h` — auto-generated C header with the 30 golden
  vectors.

Re-running the *same* variant overwrites these files; running a
*different* variant adds a new sibling directory. So the `results/`
tree accumulates one directory per (resolution, hidden, precision) combo
you've trained.

## STATUS semantics

Each step prints `STATUS = PASS | FAIL`. The criteria are:

- **step1**: full-test accuracy ≥ 95% (sanity check that the model
  actually learned). **Tuned for the default 784→128→10**; smaller
  variants (lower resolution and/or hidden) naturally land below 95%
  without any training problem. Treat step1 FAIL as "model is small";
  the variant is still usable.
- **step2**: NumPy float reproduces PyTorch exactly on every exported
  sample (matches_pytorch == yes).
- **step3**: fixed-point preserves every float decision on the 30 exported
  samples (matches_float == 30/30). The full-10k flip count is reported
  but not gated on. In the current sweep all 8-bit variants pass;
  every 4-bit variant FAILs because Q2.1 quantization is too coarse for
  post-training quantization (see [Sweep results](#sweep-results-float-vs-fixed-point-accuracy-on-full-10k)).

## Generate C headers

The top-level `./run` invokes this automatically against the per-variant
results directory, so you usually don't need to run it by hand. If you
want to regenerate without retraining:

```sh
python3 results/gen_headers_for_app.py --dir results/784x128_8bits
```

The bit width is read from `WEIGHT_BITS` in `weights_fixed.json`. Headers
are written into the same directory by default; use `--out-dir` to
redirect. The dim macros (`PROJ3_INPUT_DIM`, `PROJ3_HIDDEN_DIM`, etc.)
are emitted as `#define`s inside the weights header so the C side picks
up dims directly.
