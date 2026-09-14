"""
train_synthetic.py
------------------
Train YOLOv11 baseline detector on PURELY synthetic data.
This establishes the upper bound for synthetic-only performance
and the baseline for the sim-to-real gap measurement.

Usage:
    python scripts/train_synthetic.py

Output:
    runs/synthetic_baseline/synthetic_only/weights/best.pt
"""

from ultralytics import YOLO
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYNTHETIC_DATA = "data/synthetic/data.yaml"
OUTPUT_DIR     = "runs/synthetic_baseline"
EPOCHS         = 100
BATCH_SIZE     = 8
IMG_SIZE       = 640
DEVICE         = 0  # GPU; set to 'cpu' if no GPU

# ── TRAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Training YOLOv11 on SYNTHETIC data only")
    print("=" * 60)

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=SYNTHETIC_DATA,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=OUTPUT_DIR,
        name="synthetic_only",
        patience=20,
        save=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Synthetic mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Model saved to: {OUTPUT_DIR}/synthetic_only/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
