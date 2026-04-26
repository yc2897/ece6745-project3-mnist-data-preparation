import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
import sys

# --------------------------------------------------------------------------
# Output directory (passed as argument, defaults to cwd)
# --------------------------------------------------------------------------

output_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility: fix all random seeds
# --------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Number of test vectors to export as golden reference for hardware.
# --------------------------------------------------------------------------
# Full MNIST test set has 10,000 samples — way too many to embed in a C/RTL
# testbench. We measure full-set accuracy for the summary, but only export
# a subset (mirrors iris's 30-sample test export).

NUM_EXPORT_VECTORS = 30

# --------------------------------------------------------------------------
# Load and preprocess MNIST dataset
# --------------------------------------------------------------------------
# Standard PyTorch MNIST normalization: (pixel/255 - 0.1307) / 0.3081
# Mean/std are the global dataset statistics. This is conceptually the same
# as iris's StandardScaler — zero-mean, ~unit-variance — and is preferred
# over plain /255 here because it spreads pixel values across more of the
# fixed-point range (much better resolution under 8-bit Q3.4 quantization).

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# Repo-local dataset cache; reused across runs once downloaded.
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

transform = transforms.Compose([
    transforms.ToTensor(),                          # (1, 28, 28) in [0, 1]
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    transforms.Lambda(lambda x: x.view(-1)),        # flatten to (784,)
])

train_set = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(data_root, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

# --------------------------------------------------------------------------
# Define model: 784 -> 128 (ReLU) -> 10
# --------------------------------------------------------------------------
#
# Matches the canonical TF/Keras MNIST tutorial (which reaches ~97%).
# Drop-in shape for MLPHardwiredXcelRTL with INPUT_DIM=784, HIDDEN_DIM=128,
# OUTPUT_DIM=10.
#
# Why no softmax?
# ---------------
# Softmax is monotonic — it preserves order, so argmax(logits) ==
# argmax(softmax(logits)). For classification we only need argmax, and in
# hardware softmax is expensive (exp + division). nn.CrossEntropyLoss
# applies log_softmax internally, so we feed it raw logits during training.
# In hardware: emit raw logits and do argmax with comparators.

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

num_epochs = 6  # matches TF tutorial; ~97% test acc

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in train_loader:
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    # Eval on full test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}  "
          f"train_loss={running_loss/n_batches:.4f}  "
          f"test_acc={correct/total:.4f}")

# --------------------------------------------------------------------------
# Final evaluation on full test set
# --------------------------------------------------------------------------

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in test_loader:
        all_preds.append(model(x).argmax(dim=1))
        all_labels.append(y)
preds_full = torch.cat(all_preds)
labels_full = torch.cat(all_labels)
acc_full = (preds_full == labels_full).float().mean().item()
print(f"\nFinal test accuracy (full 10000): {acc_full:.4f}")

# --------------------------------------------------------------------------
# Build the exported golden test vector subset
# --------------------------------------------------------------------------
# Take the first NUM_EXPORT_VECTORS test images in dataset order. Save the
# already-normalized inputs so step2/step3 can replay them exactly.

X_export = torch.stack([test_set[i][0] for i in range(NUM_EXPORT_VECTORS)])  # (N, 784)
y_export = torch.tensor([test_set[i][1] for i in range(NUM_EXPORT_VECTORS)], dtype=torch.long)

with torch.no_grad():
    logits_export = model(X_export)
    preds_export  = logits_export.argmax(dim=1)
num_correct_export = int((preds_export == y_export).sum().item())
acc_export = num_correct_export / NUM_EXPORT_VECTORS

print(f"\nExported subset accuracy ({NUM_EXPORT_VECTORS} samples): "
      f"{num_correct_export}/{NUM_EXPORT_VECTORS} = {acc_export:.4f}")

# Print per-sample results for verification
print("\nPer-sample predictions (exported subset):")
for i in range(NUM_EXPORT_VECTORS):
    pred  = int(preds_export[i].item())
    label = int(y_export[i].item())
    status = "OK" if pred == label else "MISS"
    print(f"  sample {i:2d}: pred={pred} true={label} {status}")

# --------------------------------------------------------------------------
# Export weights, normalization params, and test data for next steps
# --------------------------------------------------------------------------

weights = {
    "W1": model[0].weight.detach().numpy().tolist(),  # (128, 784)
    "b1": model[0].bias.detach().numpy().tolist(),     # (128,)
    "W2": model[2].weight.detach().numpy().tolist(),   # (10, 128)
    "b2": model[2].bias.detach().numpy().tolist(),     # (10,)
    "norm_mean": [MNIST_MEAN],                          # global scalar mean
    "norm_std":  [MNIST_STD],                           # global scalar std
}

with open(os.path.join(output_dir, "weights.json"), "w") as f:
    json.dump(weights, f)

# Save test data for golden reference (only the exported subset)
test_data = {
    "X_test_raw": X_export.numpy().tolist(),     # already normalized, flat 784
    "y_test": y_export.numpy().tolist(),
    "pytorch_preds": preds_export.numpy().tolist(),
    "pytorch_num_correct": num_correct_export,
    "pytorch_num_tests": NUM_EXPORT_VECTORS,
    "pytorch_accuracy": acc_export,
    "pytorch_full_accuracy": acc_full,
    "pytorch_full_num_tests": int(labels_full.numel()),
}

with open(os.path.join(output_dir, "test_data.json"), "w") as f:
    json.dump(test_data, f)

print(f"\nWeights saved to {os.path.join(output_dir, 'weights.json')}")
print(f"Test data saved to {os.path.join(output_dir, 'test_data.json')}")

# Print weight shapes for reference
print("\nWeight shapes:")
print(f"  W1: {model[0].weight.shape}  (128x784)")
print(f"  b1: {model[0].bias.shape}    (128)")
print(f"  W2: {model[2].weight.shape}  (10x128)")
print(f"  b2: {model[2].bias.shape}    (10)")
