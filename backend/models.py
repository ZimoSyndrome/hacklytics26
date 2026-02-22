from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class DistributionPoint:
    quarter: str        # e.g. "4Q Ahead"
    probability: float  # fraud likelihood 0.0–1.0


@dataclass
class FeatureImportance:
    name: str           # e.g. "slope_F0semitoneFrom27.5Hz_sma3nz_amean"
    importance: float   # gain importance (normalized 0–1)
    label: str = ""     # human-readable label for display


@dataclass
class ModelInfo:
    horizon: int        # quarters ahead
    modality: str       # "text" | "audio" | "late_fusion"
    pr_auc: float       # PR-AUC on test set
    pr_ci_lower: float
    pr_ci_upper: float
    brier: float


@dataclass
class AnalysisResult:
    overallRiskScore: int                        # 0–100 (derived from distribution)
    riskLevel: str                               # "critical" | "high" | "elevated" | "moderate" | "low"
    distribution: List[DistributionPoint]
    topFeatures: List[FeatureImportance] = field(default_factory=list)
    modelInfo: ModelInfo | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def score_to_risk_level(score: int) -> str:
    if score >= 80:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 50:
        return "elevated"
    if score >= 30:
        return "moderate"
    return "low"


# Human-readable labels for common audio feature name patterns
_AUDIO_LABEL_MAP = {
    "F0semitone": "Fundamental Pitch",
    "loudness": "Vocal Loudness",
    "jitter": "Pitch Jitter",
    "shimmer": "Amplitude Shimmer",
    "HNR": "Harmonics-to-Noise Ratio",
    "alphaRatio": "Spectral Tilt",
    "hammarberg": "Hammarberg Index",
    "spectralFlux": "Spectral Flux",
    "mfcc": "MFCC Coefficient",
    "F1frequency": "1st Formant Frequency",
    "F2frequency": "2nd Formant Frequency",
    "F3frequency": "3rd Formant Frequency",
    "voicing": "Voicing Probability",
    "logRelF0": "Log Relative Pitch",
    "slopeV0": "Pitch Slope (Voiced)",
    "Loudness_sma3": "Loudness",
    "speaking_rate": "Speaking Rate",
    "pause": "Pause Duration",
    "energy": "Speech Energy",
}

_STAT_LABEL_MAP = {
    "mean": "avg",
    "std": "variability",
    "slope": "trend",
    "max": "peak",
    "min": "valley",
    "first": "onset",
    "last": "closing",
}


def make_feature_label(name: str) -> str:
    """Convert a raw feature name like 'slope_F0semitoneFrom27.5Hz_sma3nz_amean'
    into a human-readable label like 'Pitch Trend'."""
    parts = name.split("_")
    stat = parts[0] if parts else ""
    rest = "_".join(parts[1:]) if len(parts) > 1 else name

    feat_label = ""
    for key, label in _AUDIO_LABEL_MAP.items():
        if key.lower() in rest.lower():
            feat_label = label
            break

    if not feat_label:
        # Fall back to a cleaned version of the raw name
        feat_label = rest.replace("_sma3nz", "").replace("_sma3", "").replace("_amean", "")
        feat_label = feat_label[:40]

    stat_label = _STAT_LABEL_MAP.get(stat, stat)
    if stat_label:
        return f"{feat_label} ({stat_label})"
    return feat_label
