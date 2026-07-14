"""
LeNet-5 MNIST Reproduction Project - Main Entry

One-command pipeline: Train -> Evaluate -> Visualize

Usage:
 python main.py                          # Full pipeline (10 epochs, Adam)
 python main.py --epochs 5 --optimizer sgd  # Custom
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src module can be imported
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import main as train_main
from evaluate import evaluate_test, per_class_accuracy, confusion_matrix
from visualize import (
   plot_training_curves,
   show_random_predictions,
   show_misclassified,
   plot_confusion_matrix,
)
from dataset import get_dataloaders


def main():
   parser = argparse.ArgumentParser(description="LeNet-5 MNIST Full Pipeline")
   parser.add_argument("--epochs", type=int, default=10)
   parser.add_argument("--batch-size", type=int, default=64)
   parser.add_argument("--lr", type=float, default=0.001)
   parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
   parser.add_argument("--momentum", type=float, default=0.9)
   parser.add_argument("--seed", type=int, default=42)
   parser.add_argument("--skip-train", action="store_true",
                       help="Skip training, use existing model")
   args = parser.parse_args()

   save_dir = Path(__file__).parent / "experiments"
   fig_dir = save_dir / "figures"
   model_path = save_dir / "best_model.pth"

   if not args.skip_train:
       # ---- Step 1: Training ----
       history, model, device = train_main()

       # Plot training curves
       plot_training_curves(history, save_path=str(fig_dir / "training_curves.png"))

       # Save training history as JSON
       with open(save_dir / "logs" / "training_history.json", "w") as f:
           json.dump(history, f, indent=2)
       print(f"Training history saved: {save_dir / 'logs' / 'training_history.json'}")
   else:
       # ---- Skip training: load existing model ----
       import torch
       from model import lenet5

       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = lenet5(num_classes=10).to(device)
       if model_path.exists():
           model.load_state_dict(
               torch.load(model_path, map_location=device, weights_only=True)
           )
           print(f"Loaded model: {model_path}")
       else:
           print(f"ERROR: {model_path} not found. Train first or check path.")
           return

   # ---- Step 2: Evaluation ----
   loaders = get_dataloaders(batch_size=args.batch_size)
   test_loader = loaders["test_loader"]
   results = evaluate_test(model, test_loader, device)

   print("\n" + "=" * 50)
   print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
   print("=" * 50)

   # Per-class accuracy
   per_class = per_class_accuracy(results["predictions"], results["targets"])
   print("\nPer-class accuracy:")
   for digit in range(10):
       info = per_class[digit]
       print(f"  Digit {digit}: {info['accuracy']:.2f}% ({info['correct']}/{info['total']})")

   # Confusion matrix
   cm = confusion_matrix(results["predictions"], results["targets"])
   print(f"\nConfusion matrix (row=true, col=pred):\n{cm}")

   # ---- Step 3: Visualization ----
   # 1. Random predictions
   show_random_predictions(
       model, test_loader, device, num_samples=10,
       save_path=str(fig_dir / "predictions.png"),
   )

   # 2. Misclassified samples
   if results["misclassified"]:
       show_misclassified(
           results["misclassified"], num_samples=10,
           save_path=str(fig_dir / "misclassified.png"),
       )

   # 3. Confusion matrix heatmap
   plot_confusion_matrix(cm, save_path=str(fig_dir / "confusion_matrix.png"))

   print("\nAll results saved to:", fig_dir)
   print("Report template: report/experiment_report.md")


if __name__ == "__main__":
   main()
