"""Batch runner: 4 quadrants (Optimizer × Activation) × 3 seeds each."""
import subprocess, os, json, re
from pathlib import Path

QUADRANTS = [
    # (label, model_module, train_script, optimizer, lr, momentum, seeds, save_prefix)
    ("SGD+Tanh (原论文)",  "model_tanh", "train_sgd_tanh", "sgd", 0.01, 0.9, [42, 123, 456], "experiments_q1_sgd_tanh"),
    ("SGD+ReLU(只改激活)", "model",     "train",          "sgd", 0.01, 0.9, [123, 456],      "experiments_q2_sgd_relu"),
    ("Adam+Tanh(只改优化)", "model_tanh", "train_tanh",    "adam",0.001, 0,   [123, 456],      "experiments_q3_adam_tanh"),
    ("Adam+ReLU(现代)",   "model",     "train",          "adam",0.001, 0,   [],                "experiments_seed"),  # already done
]

ROOT = Path(__file__).parent.parent

for label, model_mod, train_script, opt, lr, mom, seeds, prefix in QUADRANTS:
    if not seeds:
        print(f"[SKIP] {label} — 已有 3 轮数据")
        continue
    for seed in seeds:
        save_dir = f"{prefix}_seed{seed}" if prefix.endswith("_") else f"{prefix}{seed}"
        # Check if already done
        if (ROOT / save_dir / "best_model.pth").exists():
            print(f"[SKIP] {label} seed={seed} — 已存在 {save_dir}")
            continue

        cmd = [
            "python", str(ROOT / "src" / f"{train_script}.py"),
            "--epochs", "10",
            "--optimizer", opt,
            "--lr", str(lr),
            "--seed", str(seed),
            "--save-dir", str(ROOT / save_dir),
            "--data-root", str(ROOT / "data"),
        ]
        if mom > 0:
            cmd += ["--momentum", str(mom)]

        print(f"\n{'='*60}")
        print(f"[RUN] {label} | seed={seed} | {opt} lr={lr} | -> {save_dir}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, cwd=str(ROOT / "src"), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[FAIL] {label} seed={seed}")
            print(result.stderr[-500:])
        else:
            # Extract best accuracy
            match = re.search(r'Best test accuracy: ([\d.]+)', result.stdout + result.stderr)
            acc = match.group(1) if match else "?"
            print(f"[DONE] {label} seed={seed} -> Best: {acc}%")

print("\n" + "="*60)
print("全部完成！")
