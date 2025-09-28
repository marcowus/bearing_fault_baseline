"""Evaluate CNN baseline with Bayesian + LLM arbitration (HCAA)."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss

from config import Config
from data_loader import get_data_loaders
from hcaa import (
    FeatureExtractor,
    FaultDiagnosticEngine,
    HCAAEnsemble,
    SiliconFlowLLMArbiter,
    calculate_aurc,
    calculate_ece,
)
from model import SimpleCNN


def _prepare_model(config: Config) -> SimpleCNN:
    model = SimpleCNN(num_classes=config.num_classes)
    checkpoint_path = os.path.join(config.save_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def _split_batch(batch):
    if len(batch) == 3:
        spectrograms, raw_signals, labels = batch
    elif len(batch) == 2:
        spectrograms, labels = batch
        raw_signals = None
    else:  # pragma: no cover - defensive
        raise ValueError("Unexpected batch structure")
    return spectrograms, raw_signals, labels


def _collect_predictions(
    model: SimpleCNN,
    dataloader: torch.utils.data.DataLoader,
    config: Config,
    feature_extractor: FeatureExtractor,
    bn_engine: FaultDiagnosticEngine,
    llm_arbiter: SiliconFlowLLMArbiter,
    stage: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    cnn_probs_list = []
    bn_probs_list = []
    llm_probs_list = []

    for batch_idx, batch in enumerate(dataloader):
        spectrograms, raw_signals, batch_labels = _split_batch(batch)
        if raw_signals is None:
            raise RuntimeError(
                "DataLoader must be created with return_signal=True for HCAA evaluation"
            )

        spectrograms = spectrograms.to(config.device)
        with torch.no_grad():
            outputs = model(spectrograms)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        raw_signals_np = raw_signals.detach().cpu().numpy()
        batch_labels_np = batch_labels.detach().cpu().numpy()

        for idx in range(len(batch_labels_np)):
            signal = raw_signals_np[idx]
            label = batch_labels_np[idx]
            features = feature_extractor.extract_all(signal, fs=config.sampling_rate)
            bn_probs = bn_engine.diagnose(features)
            llm_response = llm_arbiter.get_structured_probs(features, probs[idx], bn_probs)
            llm_probs = llm_response.probabilities

            labels.append(label)
            cnn_probs_list.append(probs[idx])
            bn_probs_list.append(bn_probs)
            llm_probs_list.append(llm_probs)

        if (batch_idx + 1) % 10 == 0:
            print(f"[{stage}] Processed {batch_idx + 1} batches")

    return (
        np.array(labels, dtype=np.int64),
        np.array(cnn_probs_list, dtype=np.float64),
        np.array(bn_probs_list, dtype=np.float64),
        np.array(llm_probs_list, dtype=np.float64),
    )


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate CNN + LLM arbitration")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip temperature calibration when reporting HCAA results",
    )
    args = parser.parse_args()

    config = Config()
    feature_extractor = FeatureExtractor()
    bn_engine = FaultDiagnosticEngine()
    llm_arbiter = SiliconFlowLLMArbiter()

    print("Loading dataloaders with raw signal access...")
    _, val_loader, test_loader = get_data_loaders(config, return_signal=True)

    print("Loading CNN baseline model...")
    model = _prepare_model(config)

    print("Collecting validation predictions for ensemble training...")
    val_labels, val_cnn_probs, val_bn_probs, val_llm_probs = _collect_predictions(
        model,
        val_loader,
        config,
        feature_extractor,
        bn_engine,
        llm_arbiter,
        stage="val",
    )

    print("Collecting test predictions...")
    test_labels, test_cnn_probs, test_bn_probs, test_llm_probs = _collect_predictions(
        model,
        test_loader,
        config,
        feature_extractor,
        bn_engine,
        llm_arbiter,
        stage="test",
    )

    ensemble = HCAAEnsemble(["cnn", "llm"])
    ensemble.train({"cnn": val_cnn_probs, "llm": val_llm_probs}, val_labels)

    results: Dict[str, Dict[str, float]] = {}

    def _evaluate_model(name: str, probs: np.ndarray) -> Dict[str, float]:
        preds = np.argmax(probs, axis=1)
        return {
            "Accuracy": accuracy_score(test_labels, preds),
            "NLL": log_loss(test_labels, probs),
            "ECE": calculate_ece(test_labels, probs),
            "AURC": calculate_aurc(test_labels, probs),
        }

    results["CNN-Only"] = _evaluate_model("CNN-Only", test_cnn_probs)
    results["BN-Only"] = _evaluate_model("BN-Only", test_bn_probs)
    results["LLM-Only"] = _evaluate_model("LLM-Only", test_llm_probs)

    combined = (test_cnn_probs + test_bn_probs + test_llm_probs) / 3.0
    combined /= np.sum(combined, axis=1, keepdims=True)
    results["Simple Average"] = _evaluate_model("Simple Average", combined)

    hcaa_uncalibrated = ensemble.predict({"cnn": test_cnn_probs, "llm": test_llm_probs}, calibrated=False)
    results["HCAA (Uncalibrated)"] = _evaluate_model("HCAA (Uncalibrated)", hcaa_uncalibrated)

    if args.no_calibration:
        hcaa_calibrated = hcaa_uncalibrated
    else:
        hcaa_calibrated = ensemble.predict({"cnn": test_cnn_probs, "llm": test_llm_probs}, calibrated=True)
    results["HCAA (Calibrated)"] = _evaluate_model("HCAA (Calibrated)", hcaa_calibrated)

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df[["Accuracy", "NLL", "ECE", "AURC"]]
    print("\n=== Final Performance Comparison ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    output_dir = os.path.join(config.save_dir, "hcaa")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "metrics.csv"))
    print(f"\nSaved metrics to {os.path.join(output_dir, 'metrics.csv')}")


if __name__ == "__main__":
    evaluate()
