from ultralytics import YOLO
import os, shutil, random
import matplotlib.pyplot as plt

REAL_DIR   = "/home/nabi/synthetic-defect-detection/data/real_combined"
SYN_DIR    = "/home/nabi/synthetic-defect-detection/data/synthetic"
OUT_BASE   = "/home/nabi/synthetic-defect-detection/data/mixed_v2"
MODEL_PATH = "/home/nabi/Downloads/ml_project_scaffold/runs/detect/runs/synthetic_baseline/synthetic_only/weights/best.pt"
REAL_TEST  = "/home/nabi/synthetic-defect-detection/data/real_combined/data.yaml"

real_imgs = sorted(os.listdir(f"{REAL_DIR}/images"))
print(f"Total real images available: {len(real_imgs)}")

results_log = [(0, 0.014)]  # baseline from previous experiment
print(f"Baseline (0 real images): mAP50 = 0.014")

for n_real in [50, 100, 200, 400, 700, 998]:
    print(f"\n{'='*40}")
    print(f"Training with {n_real} real images + synthetic")

    mixed = f"{OUT_BASE}/real_{n_real}"
    os.makedirs(f"{mixed}/train/images", exist_ok=True)
    os.makedirs(f"{mixed}/train/labels", exist_ok=True)

    # Copy all synthetic train images
    for f in os.listdir(f"{SYN_DIR}/train/images"):
        shutil.copy(f"{SYN_DIR}/train/images/{f}", f"{mixed}/train/images/{f}")
        lbl = f.replace('.png', '.txt')
        if os.path.exists(f"{SYN_DIR}/train/labels/{lbl}"):
            shutil.copy(f"{SYN_DIR}/train/labels/{lbl}", f"{mixed}/train/labels/{lbl}")

    # Add n_real real images
    random.seed(42)
    selected = random.sample(real_imgs, min(n_real, len(real_imgs)))
    for img in selected:
        src_img = f"{REAL_DIR}/images/{img}"
        lbl = img.rsplit('.', 1)[0] + '.txt'
        src_lbl = f"{REAL_DIR}/labels/{lbl}"
        shutil.copy(src_img, f"{mixed}/train/images/real_{img}")
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, f"{mixed}/train/labels/real_{lbl}")

    # Create yaml
    with open(f"{mixed}/data.yaml", 'w') as f:
        f.write(f"""train: {mixed}/train/images
val: {SYN_DIR}/val/images
test: {REAL_DIR}/images
nc: 3
names: ['clean', 'crack', 'bubble']
""")

    # Train
    model = YOLO(MODEL_PATH)
    model.train(
        data=f"{mixed}/data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        project="/home/nabi/synthetic-defect-detection/runs/bridge_v2",
        name=f"real_{n_real}",
        patience=10,
        verbose=False
    )

    # Evaluate on real test
    result = model.val(data=REAL_TEST, split='test', verbose=False)
    map50 = result.box.map50
    results_log.append((n_real, map50))
    print(f"  real_{n_real} → Real mAP50: {map50:.3f}")

# Print results
print("\n" + "="*50)
print("BRIDGE EXPERIMENT V2 — FINAL RESULTS")
print("="*50)
print(f"{'Real Images':>12} | {'mAP50':>8}")
print("-"*25)
for n, m in results_log:
    print(f"{n:>12} | {m:>8.3f}")

# Plot gap curve
ns = [r[0] for r in results_log]
ms = [r[1] for r in results_log]

plt.figure(figsize=(10, 6))
plt.plot(ns, ms, 'bo-', linewidth=2.5, markersize=10)
plt.axhline(y=40.5, color='green', linestyle='--', linewidth=2, label='Synthetic mAP50 (40.5%)')
plt.axhline(y=1.4, color='red', linestyle='--', linewidth=1.5, label='Zero-shot baseline (1.4%)')
for x, y in zip(ns, ms):
    plt.annotate(f'{y*100:.1f}%', (x, y), textcoords="offset points", xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
plt.xlabel('Number of Real Images Added', fontsize=13)
plt.ylabel('mAP@0.5 on Real Test Set', fontsize=13)
plt.title('Sim-to-Real Gap Bridging\nTransparent Glass Defect Detection (998 real images)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/nabi/synthetic-defect-detection/results/gap_curve_v2.png', dpi=150)
print("\nGap curve saved to results/gap_curve_v2.png")
