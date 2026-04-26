# ece6745 mnist data preparation

This repo prepares data for an ECE 6745 hardware accelerator that classifies
MNIST handwritten digits. It mirrors the iris repo's pipeline — three steps
run by `./run`:

- **Step 1:** Train a `784 → 128 → 10` neural net in PyTorch and save the
  weights (float32). Architecture matches the canonical TF/Keras MNIST
  tutorial baseline (~97% test accuracy on the full 10k test set).
- **Step 2:** Redo the forward pass in NumPy floats (float32) to confirm it
  matches PyTorch on the exported subset.
- **Step 3:** Convert everything (float32) to signed fixed-point integers
  and check predictions still match — both on the 30-sample exported subset
  and on the full 10,000-sample MNIST test set.

The final fixed-point weights and a 30-sample golden test vector subset
become the reference for the C and Verilog versions built later.

## Network shape

`784 → 128 (ReLU) → 10`. Drop-in for the parameterized `MLPHardwiredXcelRTL`
(`INPUT_DIM=784, HIDDEN_DIM=128, OUTPUT_DIM=10`); no RTL changes needed
beyond the dim parameters.

Total params: 784·128 + 128 + 128·10 + 10 = **101,770**.

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

## Test vector export

The full 10,000-sample test set is too large to embed in a C/RTL
testbench. Step 1 exports the **first 30** test images (in dataset order,
deterministic across runs) as the hardware golden vectors. Step 3 also
runs fixed-point inference on **all 10,000 samples** and reports accuracy
there — that's where you see the real 8-bit vs 16-bit accuracy gap.

## Precision (step 3)

Step 3 supports two fixed-point formats:

| Value | Format | Range | Frac bits | Storage | Accumulator |
|---|---|---|---|---|---|
| **`8` (default)**  | Q3.4 signed | ±7.9375    | 4 | int8  | int32 |
| `16`               | Q7.8 signed | ±127.996   | 8 | int16 | int32 |

Accumulator headroom: layer 1 sums up to 784 int8/int16 products, layer 2
sums up to 128. Both fit comfortably in int32 with trained weight
magnitudes; step 3 also asserts this in the vectorized full-10k pass.

## Accuracy on this baseline

Reference numbers from a clean `./run` (seed = 42, 6 epochs, Adam(0.001),
batch 128):

| Metric | Float32 | **8-bit Q3.4** | **16-bit Q7.8** |
|---|---|---|---|
| Full-10k accuracy        | 97.69% | **97.39%** | **97.72%** |
| Matches float on 10k     | —      | 9899/10000 (101 flips) | 9995/10000 (5 flips) |
| 30-sample export acc     | 96.67% | 96.67%     | 96.67% |
| 30-sample matches float  | —      | 30/30      | 30/30 |

**Reading this:**
- 16-bit Q7.8 is essentially lossless — only 5 of 10,000 predictions flip
  versus float, and the small accuracy bump (97.69→97.72%) is just chance
  among those 5 flips.
- 8-bit Q3.4 costs about **0.3%** accuracy: ~101 of 10,000 decisions flip.
  That's the price of dropping pixel/weight resolution from ~256 levels to
  ~16 levels.
- The 30-sample exported subset is too small (resolution 1/30 ≈ 3.3%) to
  expose the 0.3% gap — both precisions match all 30 float decisions. To
  see the real difference, look at the full-10k numbers.

## Reproduce result

1. Clone:

    ```sh
    git clone <repo>
    cd ece6745-project3-mnist-data-preparation
    ```

2. Run all three steps (8-bit default):

    ```sh
    ./run
    ```

3. Or with 16-bit precision at step 3:

    ```sh
    ./run 16
    # or
    PRECISION=16 ./run
    ```

4. Run only step 3 in the other precision, keeping step 1/2 output:

    ```sh
    ./step3/run 16
    ```

Default seed at the training step is 42, so the result is reproducible.
On first run, torchvision downloads MNIST into `./data/` (cached for
later runs).

## Outputs

### `step1/output/`
- `weights.json` — float32 W1, b1, W2, b2 + normalization mean/std.
- `test_data.json` — the 30 exported (already-normalized) inputs, labels,
  PyTorch predictions on those 30, plus the full-10k PyTorch accuracy.
- `summary.txt` — human-readable summary.

### `step2/output/`
- `test_vectors_float.json` — the 30 vectors with NumPy-float logits and
  predictions; verifies bit-for-bit reproduction of PyTorch.
- `summary.txt`.

### `step3/output/`
- `weights_fixed.json` — quantized W1, b1, W2, b2 with metadata
  (`FRAC_BITS`, `WEIGHT_BITS`, `ACC_BITS`, `INPUT_DIM`, `HIDDEN_DIM`,
  `OUTPUT_DIM`).
- `test_vectors.json` — 30 quantized input vectors with expected
  fixed-point logits, labels, and the float prediction for each;
  also full-10k accuracy fields (`full_float_accuracy`,
  `full_fixed_accuracy`, `full_matches_float`, `full_match_rate`).
- `summary.txt`.

Switching precision overwrites the files in `step3/output/`. If you want
both preserved, rename or copy the directory between runs.

## STATUS semantics

Each step prints `STATUS = PASS | FAIL`. The criteria are:

- **step1**: full-test accuracy ≥ 95% (sanity check that the model
  actually learned).
- **step2**: NumPy float reproduces PyTorch exactly on every exported
  sample (matches_pytorch == yes).
- **step3**: fixed-point preserves every float decision on the 30 exported
  samples (matches_float == 30/30). The full-10k flip count is reported
  but not gated on (small accuracy drift is normal under quantization).

## Generate C headers

Convert the JSON outputs into C headers (`weights_<bits>.h` and
`test_vectors_<bits>.h`) for the C/Verilog testbenches:

```sh
cd results
python3 gen_headers_for_app.py --dir 8bits
```

The bit width is read from `WEIGHT_BITS` in `weights_fixed.json`. Headers
are written into the same directory by default; use `--out-dir` to
redirect.
