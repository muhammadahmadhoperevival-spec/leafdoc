import json
import time

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn

from dataset import get_loaders
from train_cnn import run_epoch, device

MODEL_NAME = "resnet18"
FREEZE_EPOCHS = 2
FINETUNE_EPOCHS = 6
PATIENCE = 3

if __name__ == "__main__":
    loaders, classes = get_loaders()
    json.dump(classes, open("classes.json", "w"))

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, bad = 0.0, 0

    def log(ep, tl, ta, vl, va, t0, phase):
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        print(f"[{phase}] Epoch {ep} | train acc {ta:.3f} | val acc {va:.3f} | {time.time()-t0:.0f}s")

    # Phase 1: backbone freeze
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.get_classifier().parameters(), lr=1e-3)
    for ep in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders["train"], loss_fn, opt)
        vl, va = run_epoch(model, loaders["val"], loss_fn)
        log(ep, tl, ta, vl, va, t0, "freeze")
        if va > best:
            best = va
            torch.save(model.state_dict(), "tl_model.pth")

    # Phase 2: unfreezing everything
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    for ep in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders["train"], loss_fn, opt)
        vl, va = run_epoch(model, loaders["val"], loss_fn)
        sched.step()
        log(FREEZE_EPOCHS + ep, tl, ta, vl, va, t0, "finetune")
        if va > best:
            best, bad = va, 0
            torch.save(model.state_dict(), "tl_model.pth")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load("tl_model.pth"))
    _, test_acc = run_epoch(model, loaders["test"], loss_fn)
    print(f"TRANSFER MODEL TEST ACCURACY: {test_acc:.4f}")
    json.dump({"model": MODEL_NAME, "test_acc": test_acc, "history": hist},
              open("tl_results.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.savefig("curves_transfer.png", dpi=120)