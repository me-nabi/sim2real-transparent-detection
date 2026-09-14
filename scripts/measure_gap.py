"""
measure_gap.py
--------------
Measure the sim-to-real gap by evaluating a synthetically-trained
model on real images.

Usage:
    python scripts/measure_gap.py \
        --model runs/synthetic_baseline/synthetic_only/weights/best.pt \
        --synthetic_data data/synthetic/data.yaml \
        --real_data data/real_combined/data.yaml
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="runs/synthetic_baseline/synthetic_only/weights/best.pt")
    parser.add_argument("--synthetic_data", default="data/synthetic/data.yaml")
    parser.add_argument("--real_data",      default="data/real_combined/data.yaml")
    args = parser.parse_args()

    model = YOLO(args.model)

    print("=" * 60)
    print("MEASURING SIM-TO-REAL GAP")
    print("=" * 60)

    # Evaluate on synthetic test set
    print("\n[1] Evaluating on SYNTHETIC test set...")
    syn_results = model.val(data=args.synthetic_data, split="test")
    syn_map = syn_results.box.map50
    print(f"    Synthetic mAP50: {syn_map:.3f}")

    # Evaluate on real test set
    print("\n[2] Evaluating on REAL test set...")
    real_results = model.val(data=args.real_data, split="test")
    real_map = real_results.box.map50
    print(f"    Real mAP50:      {real_map:.3f}")

    # Report gap
    gap = syn_map - real_map
    gap_pct = (gap / syn_map) * 100 if syn_map > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Synthetic mAP50:  {syn_map:.3f} ({syn_map*100:.1f}%)")
    print(f"  Real mAP50:       {real_map:.3f} ({real_map*100:.1f}%)")
    print(f"  Absolute gap:     {gap:.3f}")
    print(f"  Relative gap:     {gap_pct:.1f}%  ← sim-to-real gap")
    print("=" * 60)


if __name__ == "__main__":
    main()
