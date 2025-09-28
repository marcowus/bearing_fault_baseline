"""Evaluate CNN baseline with Bayesian + LLM arbitration (HCAA)."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from scipy import signal as sp_signal

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


def _has_cwru_dataset(config: Config) -> bool:
    data_path = config.data_path
    if data_path.endswith(".mat"):
        return os.path.exists(data_path)
    if not os.path.isdir(data_path):
        return False
    return any(entry.endswith(".mat") for entry in os.listdir(data_path))


class SyntheticBearingDataset(Dataset):
    """Lightweight synthetic dataset for smoke-testing the HCAA pipeline."""

    def __init__(self, config: Config, samples_per_class: int = 64, seed: int = 0) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        self.spectrograms = []
        self.signals = []
        self.labels = []

        t = np.arange(config.signal_length) / float(config.sampling_rate)

        for label in range(config.num_classes):
            base_freq = 40.0 * (label + 1)
            modulation = 0.2 * (label + 1)
            for _ in range(samples_per_class):
                amplitude = 1.0 + 0.1 * rng.standard_normal()
                signal = amplitude * np.sin(2 * np.pi * base_freq * t)
                signal += 0.3 * np.sin(2 * np.pi * (base_freq + modulation) * t)
                signal += 0.15 * np.sin(2 * np.pi * (base_freq / 2.0) * t)
                signal += 0.05 * rng.standard_normal(t.shape)
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

                spectrogram = self._to_spectrogram(signal)

                self.signals.append(signal.astype(np.float32))
                self.spectrograms.append(spectrogram)
                self.labels.append(label)

    def _to_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        fs = self.config.sampling_rate
        _, _, zxx = sp_signal.stft(
            signal,
            fs=fs,
            nperseg=128,
            noverlap=96,
            window="hann",
        )
        magnitude = 20 * np.log10(np.abs(zxx) + 1e-8)
        magnitude -= magnitude.min()
        magnitude /= magnitude.max() + 1e-12
        return magnitude.astype(np.float32)[None, ...]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        spectrogram = torch.from_numpy(self.spectrograms[idx])
        signal = torch.from_numpy(self.signals[idx])
        label = int(self.labels[idx])
        return spectrogram, signal, label


def _build_synthetic_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SyntheticBearingDataset(config)
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        labels,
        test_size=config.test_ratio,
        random_state=42,
        stratify=labels,
    )
    val_ratio = config.val_ratio / (config.train_ratio + config.val_ratio)
    train_idx, val_idx, _, _ = train_test_split(
        train_idx,
        y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def _prepare_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if _has_cwru_dataset(config):
        return get_data_loaders(config, return_signal=True)

    print(
        "CWRU dataset not found at configured path. "
        "Falling back to a synthetic dataset for smoke testing."
    )
    return _build_synthetic_loaders(config)


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

            bn_sum = float(np.sum(bn_probs))
            llm_sum = float(np.sum(llm_probs))
            if bn_sum > 0:
                bn_probs = bn_probs / bn_sum
            else:
                bn_probs = np.full_like(bn_probs, 1.0 / len(bn_probs))
            if llm_sum > 0:
                llm_probs = llm_probs / llm_sum
            else:
                llm_probs = np.full_like(llm_probs, 1.0 / len(llm_probs))

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

    print("Preparing dataloaders with raw signal access...")
    _, val_loader, test_loader = _prepare_dataloaders(config)

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
        probs = np.asarray(probs, dtype=np.float64)
        denom = np.sum(probs, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        probs = probs / denom
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
