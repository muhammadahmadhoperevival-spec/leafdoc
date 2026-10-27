import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_loaders

EPOCHS = 8
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


if __name__ == "__main__":
    loaders, classes = get_loaders()
    model = SmallCNN(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    for ep in range(EPOCHS):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders["train"], loss_fn, opt)
        vl, va = run_epoch(model, loaders["val"], loss_fn)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        if va > best:
            best = va
            torch.save(model.state_dict(), "cnn_scratch.pth")
        print(f"Epoch {ep+1}/{EPOCHS} | train acc {ta:.3f} | val acc {va:.3f} | {time.time()-t0:.0f}s")

    # Test accuracy with best weights
    model.load_state_dict(torch.load("cnn_scratch.pth"))
    _, test_acc = run_epoch(model, loaders["test"], loss_fn)
    print(f"SCRATCH CNN TEST ACCURACY: {test_acc:.4f}")
    json.dump({"test_acc": test_acc, "history": hist}, open("cnn_scratch_results.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.savefig("curves_scratch.png", dpi=120)