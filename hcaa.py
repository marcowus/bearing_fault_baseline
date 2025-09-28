"""HCAA utilities for fusing CNN predictions with Bayesian heuristics and LLM arbitration."""
from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from sklearn.metrics import log_loss

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore


FAULT_CLASSES: List[str] = [
    "normal",
    "ball_fault",
    "inner_race_fault",
    "outer_race_fault",
]

CLASS_TO_INDEX: Mapping[str, int] = {name: idx for idx, name in enumerate(FAULT_CLASSES)}
INDEX_TO_CLASS: Mapping[int, str] = {idx: name for name, idx in CLASS_TO_INDEX.items()}


class FeatureExtractor:
    """Compute descriptive statistics from a raw vibration signal."""

    @staticmethod
    def time_domain_features(signal: np.ndarray) -> Dict[str, float]:
        signal = np.asarray(signal, dtype=float)
        rms = math.sqrt(float(np.mean(signal ** 2)))
        peak = float(np.max(np.abs(signal)))
        eps = 1e-10
        return {
            "rms": rms,
            "crest_factor": peak / (rms + eps),
            "kurtosis": float(kurtosis(signal, fisher=False)),
            "skewness": float(skew(signal)),
            "peak_to_peak": float(np.ptp(signal)),
        }

    @staticmethod
    def frequency_domain_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
        signal = np.asarray(signal, dtype=float)
        n = signal.size
        spectrum = np.fft.rfft(signal * np.hanning(n))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power) + 1e-12)

        low_band = power[freqs < 1000.0]
        mid_band = power[(freqs >= 1000.0) & (freqs < 3000.0)]
        high_band = power[freqs >= 3000.0]

        envelope = np.abs(hilbert(signal))
        envelope_kurtosis = float(kurtosis(envelope, fisher=False))

        return {
            "low_freq_ratio": float(np.sum(low_band) / total_power),
            "mid_freq_ratio": float(np.sum(mid_band) / total_power),
            "high_freq_ratio": float(np.sum(high_band) / total_power),
            "spectral_centroid": float(np.sum(freqs * power) / (total_power + 1e-12)),
            "envelope_kurtosis": envelope_kurtosis,
        }

    @staticmethod
    def extract_all(signal: np.ndarray, fs: float) -> Dict[str, Dict[str, float]]:
        return {
            "time": FeatureExtractor.time_domain_features(signal),
            "frequency": FeatureExtractor.frequency_domain_features(signal, fs),
        }

    @staticmethod
    def summary(features: Mapping[str, Mapping[str, float]]) -> str:
        """Return a JSON-formatted summary for prompts/caching."""
        return json.dumps(features, indent=2, sort_keys=True)


class FaultDiagnosticEngine:
    """Rule-based Bayesian diagnostic engine derived from handcrafted features."""

    def __init__(self) -> None:
        base_prob = 1.0 / len(FAULT_CLASSES)
        self.priors: Dict[str, float] = {fault: base_prob for fault in FAULT_CLASSES}
        self.likelihoods: Dict[str, Dict[str, float]] = {
            "impact_signature": {
                "normal": 0.05,
                "ball_fault": 0.85,
                "inner_race_fault": 0.70,
                "outer_race_fault": 0.80,
            },
            "high_frequency_energy": {
                "normal": 0.10,
                "ball_fault": 0.55,
                "inner_race_fault": 0.90,
                "outer_race_fault": 0.45,
            },
            "low_frequency_dominance": {
                "normal": 0.35,
                "ball_fault": 0.25,
                "inner_race_fault": 0.20,
                "outer_race_fault": 0.75,
            },
            "balanced_spectrum": {
                "normal": 0.65,
                "ball_fault": 0.30,
                "inner_race_fault": 0.40,
                "outer_race_fault": 0.35,
            },
            "stable_rms": {
                "normal": 0.85,
                "ball_fault": 0.25,
                "inner_race_fault": 0.15,
                "outer_race_fault": 0.35,
            },
        }

    def _extract_evidence(self, features: Mapping[str, Mapping[str, float]]) -> Dict[str, bool]:
        time_feats = features["time"]
        freq_feats = features["frequency"]

        evidence = {
            "impact_signature": time_feats["kurtosis"] > 4.5 or time_feats["crest_factor"] > 3.5,
            "high_frequency_energy": freq_feats["high_freq_ratio"] > 0.35,
            "low_frequency_dominance": freq_feats["low_freq_ratio"] > 0.45,
            "balanced_spectrum": 0.25 < freq_feats["mid_freq_ratio"] < 0.5,
            "stable_rms": time_feats["rms"] < 0.8,
        }
        return evidence

    def diagnose(self, features: Mapping[str, Mapping[str, float]]) -> np.ndarray:
        evidence = self._extract_evidence(features)
        posteriors: Dict[str, float] = {}

        for fault, prior in self.priors.items():
            posterior = prior
            for evidence_type, present in evidence.items():
                likelihood = self.likelihoods[evidence_type][fault]
                if not present:
                    likelihood = 1.0 - likelihood
                posterior *= likelihood
            posteriors[fault] = posterior

        # heuristic boosts for clearer separations
        time_feats = features["time"]
        freq_feats = features["frequency"]
        if time_feats["kurtosis"] > 6.0 and time_feats["crest_factor"] > 4.0:
            posteriors["ball_fault"] *= 2.5
        if freq_feats["high_freq_ratio"] > 0.45:
            posteriors["inner_race_fault"] *= 2.0
        if freq_feats["low_freq_ratio"] > 0.55:
            posteriors["outer_race_fault"] *= 2.0

        total = sum(posteriors.values()) + 1e-12
        normalized = np.array([posteriors[name] / total for name in FAULT_CLASSES], dtype=np.float64)
        return normalized


@dataclass
class LLMResponse:
    probabilities: np.ndarray
    from_cache: bool = False
    error: Optional[str] = None


class SiliconFlowLLMArbiter:
    """Query SiliconFlow hosted models for structured probability outputs."""

    def __init__(self, cache_dir: str = "results/llm_cache", model: str = "deepseek-ai/DeepSeek-V3") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = model
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.client = None
        if self.api_key and OpenAI is not None:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _cache_path(self, signature: str) -> str:
        digest = hashlib.md5(signature.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def _fallback(self, cnn_probs: np.ndarray, bn_probs: np.ndarray) -> np.ndarray:
        combined = (cnn_probs + bn_probs) / 2.0
        combined /= np.sum(combined)
        return combined

    def build_prompt(
        self,
        features: Mapping[str, Mapping[str, float]],
        cnn_probs: np.ndarray,
        bn_probs: np.ndarray,
    ) -> str:
        feature_summary = FeatureExtractor.summary(features)
        cnn_top = FAULT_CLASSES[int(np.argmax(cnn_probs))].replace("_", " ")
        bn_top = FAULT_CLASSES[int(np.argmax(bn_probs))].replace("_", " ")
        prompt = f"""
You are acting as a cognitive arbitrator for rotating machinery diagnostics. Combine quantitative features with the existing models to produce calibrated fault probabilities.\n\nTop-1 prediction from CNN: {cnn_top}\nTop-1 diagnosis from Bayesian rules: {bn_top}\n\nQuantitative features:\n{feature_summary}\n\nReturn a JSON object with a single key \"fault_probs\" whose value is a dictionary mapping the following fault classes to probabilities that sum to 1.0: {json.dumps(FAULT_CLASSES)}.\nOnly output the JSON object. Provide numbers with up to 4 decimal places.
"""
        return prompt.strip()

    def get_structured_probs(
        self,
        features: Mapping[str, Mapping[str, float]],
        cnn_probs: np.ndarray,
        bn_probs: np.ndarray,
    ) -> LLMResponse:
        signature = json.dumps({
            "features": features,
            "cnn": cnn_probs.tolist(),
            "bn": bn_probs.tolist(),
        }, sort_keys=True)
        cache_path = self._cache_path(signature)

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = np.array(json.load(handle), dtype=np.float64)
            return LLMResponse(probabilities=cached, from_cache=True)

        if self.client is None:
            probs = self._fallback(cnn_probs, bn_probs)
            return LLMResponse(probabilities=probs, from_cache=False, error="LLM client unavailable")

        prompt = self.build_prompt(features, cnn_probs, bn_probs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            probs_dict = data.get("fault_probs", {})
            probs = np.array([float(probs_dict.get(name, 0.0)) for name in FAULT_CLASSES], dtype=np.float64)
            if probs.sum() <= 0:
                probs = self._fallback(cnn_probs, bn_probs)
            else:
                probs /= probs.sum()
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(probs.tolist(), handle)
            return LLMResponse(probabilities=probs, from_cache=False)
        except Exception as exc:  # pragma: no cover - network failure path
            probs = self._fallback(cnn_probs, bn_probs)
            return LLMResponse(probabilities=probs, from_cache=False, error=str(exc))


class HCAAEnsemble:
    """Product-of-experts ensemble with temperature scaling."""

    def __init__(self, expert_names: Iterable[str]) -> None:
        self.expert_names = list(expert_names)
        self.weights: Dict[str, float] = {name: 1.0 for name in self.expert_names}
        self.temperature: float = 1.0

    def _poe(self, expert_probs: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
        fused_log = None
        for name in self.expert_names:
            probs = expert_probs[name]
            weight = weights[name]
            log_component = np.log(probs + 1e-9) * weight
            fused_log = log_component if fused_log is None else fused_log + log_component
        fused = np.exp(fused_log)
        fused /= np.sum(fused, axis=1, keepdims=True)
        return fused

    @staticmethod
    def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(probs + 1e-9) / temperature
        scaled = np.exp(logits)
        scaled /= np.sum(scaled, axis=1, keepdims=True)
        return scaled

    def train(self, expert_probs: Mapping[str, np.ndarray], labels: np.ndarray) -> None:
        from scipy.optimize import minimize  # local import to avoid eager dependency

        def objective(params: np.ndarray) -> float:
            weights = {name: params[idx] for idx, name in enumerate(self.expert_names)}
            temperature = params[-1]
            fused = self._poe(expert_probs, weights)
            calibrated = self._apply_temperature(fused, temperature)
            return float(log_loss(labels, calibrated))

        initial = np.array([1.0] * len(self.expert_names) + [1.0], dtype=float)
        bounds = [(0.1, 5.0)] * len(self.expert_names) + [(0.2, 5.0)]
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        for idx, name in enumerate(self.expert_names):
            self.weights[name] = float(result.x[idx])
        self.temperature = float(result.x[-1])

    def predict(self, expert_probs: Mapping[str, np.ndarray], calibrated: bool = True) -> np.ndarray:
        fused = self._poe(expert_probs, self.weights)
        if calibrated:
            return self._apply_temperature(fused, self.temperature)
        return fused


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        mask = (confidences > bin_boundaries[idx]) & (confidences <= bin_boundaries[idx + 1])
        if not np.any(mask):
            continue
        prop = float(np.mean(mask))
        accuracy = float(np.mean(accuracies[mask]))
        confidence = float(np.mean(confidences[mask]))
        ece += abs(confidence - accuracy) * prop
    return float(ece)


def calculate_aurc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    confidences = np.max(y_prob, axis=1)
    order = np.argsort(-confidences)
    sorted_true = y_true[order]
    sorted_pred = np.argmax(y_prob[order], axis=1)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    coverage = np.arange(1, len(y_true) + 1) / len(y_true)
    risk = cumulative_errors / np.arange(1, len(y_true) + 1)
    aurc = np.trapz(1.0 - risk, coverage)
    return float(aurc)

