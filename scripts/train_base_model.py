#!/usr/bin/env python3
"""Train the base comfort model on the historical weather CSV."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.base_model import train_base_model


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "history_data.csv")
    output_path = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "base_comfort_model.onnx")

    print(f"Training base comfort model from {csv_path}...")
    model, metrics = train_base_model(csv_path, output_path)

    print("\nTraining complete!")
    print(f"  Samples: {metrics['n_train']} train / {metrics['n_test']} test")
    print(f"  Features: {metrics['n_features']}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
