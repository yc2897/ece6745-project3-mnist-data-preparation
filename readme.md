# ece6745 mnist data preparation

This repo prepares data for an ECE 6745 hardware accelerator that classifies
MNIST handwritten digits. It mirrors the iris repo's pipeline — three steps
run by `./run`:

- **Step 1:** Train a `784 → 128 → 10` neural net in PyTorch and save the
  weights (float32). Architecture matches the canonical TF/Keras MNIST
  tutorial baseline (~97% test accuracy).
- **Step 2:** Redo the forward pass in NumPy floats (float32) to confirm it
  matches PyTorch.
- **Step 3:** Convert everything (float32) to signed fixed-point integers
  and check predictions still match.

The final fixed-point weights and a 30-sample golden test vector subset
become the reference for the C and Verilog versions built later. The full
10,000-sample MNIST test accuracy is also reported in step 1 for context.

## Network shape

`784 → 128 (ReLU) → 10`. This is a drop-in for the parameterized
`MLPHardwiredXcelRTL` (`INPUT_DIM=784, HIDDEN_DIM=128, OUTPUT_DIM=10`); no
RTL changes needed beyond the dim parameters.

Total params: 784·128 + 128 + 128·10 + 10 = **101,770**.

## Preprocessing

Standard PyTorch MNIST normalization: `(pixel/255 − 0.1307) / 0.3081`,
where 0.1307 / 0.3081 are the global MNIST mean/std. This is conceptually
the same as iris's StandardScaler — zero-mean, ~unit-variance — and is
preferred over plain `/255` because it spreads pixel values across more of
the fixed-point range, giving much better resolution under 8-bit Q3.4
quantization.

## Precision (step 3)

Step 3 supports two fixed-point formats:

| Value | Format | Range | Frac bits | Storage | Accumulator |
|---|---|---|---|---|---|
| **`8` (default)**  | Q3.4 signed | ±7.9375    | 4 | int8  | int32 |
| `16`               | Q7.8 signed | ±127.996   | 8 | int16 | int32 |

Accumulator headroom: layer 1 sums up to 784 int8/int16 products, layer 2
sums 128. Both fit comfortably in int32.

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

## Outputs (step3/output/)

- `weights_fixed.json` — quantized W1, b1, W2, b2 with metadata:
  `FRAC_BITS`, `WEIGHT_BITS`, `ACC_BITS`, `INPUT_DIM`, `HIDDEN_DIM`,
  `OUTPUT_DIM`.
- `test_vectors.json` — 30 input vectors with their expected fixed-point
  logits and labels.
- `summary.txt` — human-readable summary printed to screen.

Switching precision overwrites these files, so if you want both preserved,
rename the `step3/output/` directory between runs.

## Generate C headers

Convert the JSON outputs into C headers (`weights_<bits>.h` and
`test_vectors_<bits>.h`) for the C/Verilog testbenches:

```sh
cd results
python3 gen_headers_for_app.py --dir 8bits
```

The bit width is read from `WEIGHT_BITS` in `weights_fixed.json`. Headers
are written into the same directory by default; use `--out-dir` to redirect.
