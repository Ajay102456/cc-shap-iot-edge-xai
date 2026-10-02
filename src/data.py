"""
Data loading for the IoT-edge XAI benchmark.

Primary path: load a real CICIoT2023 (or Edge-IIoTset) CSV export placed in
data/raw/. CICIoT2023's official release (~8-13GB) is a directory of
per-class subdirectories, one per attack/benign category, each holding
one or more CSVs with NO label column (the label is the directory name):

    data/raw/ciciot2023/CSV/Benign_Final/*.csv
    data/raw/ciciot2023/CSV/DDoS-SYN_Flood/*.csv
    data/raw/ciciot2023/CSV/Mirai-greeth_flood/*.csv
    ... (34 category directories total)

That's the layout `load_ciciot2023()` expects by default. A flat directory
of per-class CSVs that already contain their own label column
(data/raw/ciciot2023/*.csv), or a single pre-merged
data/raw/ciciot2023_merged.csv, are also supported as a fallback for a
hand-prepared variant.

Fallback path (no download available yet): `make_synthetic_iot_traffic()`
generates a labeled tabular dataset with the same *shape* of problem
(imbalanced multiclass network-flow classification, mixed numeric feature
scales, some redundant/correlated features) so the full pipeline
(train -> explain -> resource-cap -> fidelity -> adversarial) can be
developed and unit-tested end to end before the real dataset is downloaded.

IMPORTANT: synthetic data is for pipeline validation only. Any numbers
reported in the plan's Phase 2/3 deliverables must come from the real
CICIoT2023 / Edge-IIoTset loaders, not the synthetic generator.
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    source: str  # "ciciot2023" | "edge-iiotset" | "synthetic"


# The actual CICIoT2023 release (as distributed via the official UNB
# download, e.g. under CSV/) has NO label column in the per-flow CSVs at
# all: each of the 34 attack/benign categories is its own subdirectory
# (data/raw/ciciot2023/CSV/<ClassName>/*.csv), and the label is implied by
# the directory name. A CSV_README.pdf/README_CSV.pdf sits alongside them
# and must be skipped.
#
# 34-way raw classification is far harder and not what most CICIoT2023
# papers report accuracy against -- the dataset's own documentation and
# nearly all follow-up XAI/IDS papers group into ~8 categories (Benign +
# 7 attack families). This mapping implements that standard grouping so
# reported accuracy is comparable to the literature (~95-99% in the
# 8-class setting). Pass label_grouping="raw" to keep all 34 fine-grained
# labels instead.
_CICIOT_GROUP_MAP = {
    "Benign_Final": "Benign",
    "DictionaryBruteForce": "BruteForce",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    "BrowserHijacking": "Web",
    "CommandInjection": "Web",
    "SqlInjection": "Web",
    "Uploading_Attack": "Web",
    "XSS": "Web",
    "Backdoor_Malware": "Web",
    "VulnerabilityScan": "Recon",
}


def _group_ciciot_label(raw_class_dir_name: str, label_grouping: str) -> str:
    if label_grouping == "raw":
        return raw_class_dir_name
    if raw_class_dir_name in _CICIOT_GROUP_MAP:
        return _CICIOT_GROUP_MAP[raw_class_dir_name]
    if raw_class_dir_name.startswith("DDoS-"):
        return "DDoS"
    if raw_class_dir_name.startswith("DoS-"):
        return "DoS"
    if raw_class_dir_name.startswith("Mirai-"):
        return "Mirai"
    if raw_class_dir_name.startswith("Recon-"):
        return "Recon"
    return raw_class_dir_name  # unmapped -- kept as-is rather than silently dropped


def _find_ciciot_class_dirs() -> dict[str, list[str]]:
    """Returns {class_dir_name: [csv_paths]} for the real per-class-folder
    CICIoT2023 release layout under data/raw/ciciot2023/CSV/.
    """
    csv_root = os.path.join(RAW_DIR, "ciciot2023", "CSV")
    if not os.path.isdir(csv_root):
        return {}
    result = {}
    for entry in sorted(os.listdir(csv_root)):
        class_dir = os.path.join(csv_root, entry)
        if not os.path.isdir(class_dir):
            continue  # skips README_CSV.pdf etc.
        csvs = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
        if csvs:
            result[entry] = csvs
    return result


def _find_ciciot_files() -> list[str]:
    """Legacy fallback: a single merged CSV, or a flat directory of
    per-class CSVs each already containing their own label column.
    """
    merged = os.path.join(RAW_DIR, "ciciot2023_merged.csv")
    if os.path.exists(merged):
        return [merged]
    return sorted(glob.glob(os.path.join(RAW_DIR, "ciciot2023", "*.csv")))


def load_ciciot2023(
    n_per_class: int = 5000,
    label_grouping: str = "8class",
    random_state: int = 42,
) -> Dataset:
    """Load and stratified-subsample CICIoT2023 from data/raw/.

    Download instructions (not automated here -- large, license/attribution
    required): https://www.unb.ca/cic/datasets/iotdataset-2023.html

    Supports two layouts, tried in this order:
      1. data/raw/ciciot2023/CSV/<ClassName>/*.csv -- the real official
         release layout (no label column; label = directory name).
      2. data/raw/ciciot2023/*.csv (flat, each file already has a label
         column) or a single data/raw/ciciot2023_merged.csv -- for a
         hand-merged/pre-labeled variant.

    label_grouping: "8class" (default, matches most published CICIoT2023
    results: Benign/DDoS/DoS/Mirai/Recon/Spoofing/Web/BruteForce) or "raw"
    (keep all 34 fine-grained categories).

    n_per_class caps the FINAL grouped label (e.g. all 12 "DDoS-*" raw
    subfolders combined count as one "DDoS" class for the 8class grouping),
    not each raw subfolder independently -- capping per raw subfolder would
    let multi-folder groups like DDoS/Web/Recon end up with many times
    n_per_class rows while single-folder groups like Benign/BruteForce stay
    at n_per_class, producing a badly imbalanced "n_per_class-class"
    dataset that doesn't mean what the flag says.

    Reads up to ~3x n_per_class rows per RAW subfolder first (via nrows
    caps, not the whole multi-GB file), groups those partial reads by
    final label, then subsamples each group down to exactly n_per_class --
    avoids loading the full ~8GB corpus into memory just to keep a few
    thousand rows per class.
    """
    class_dirs = _find_ciciot_class_dirs()
    if class_dirs:
        # group raw subfolders by their FINAL label first, so the budget
        # (and the final n_per_class cap) applies once per grouped label,
        # not once per raw subfolder that happens to share that label.
        label_to_dirs: dict[str, list[str]] = {}
        for class_name, csv_paths in class_dirs.items():
            label = _group_ciciot_label(class_name, label_grouping)
            label_to_dirs.setdefault(label, []).extend(csv_paths)

        frames = []
        for label, csv_paths in label_to_dirs.items():
            budget = max(n_per_class * 3, n_per_class) if n_per_class else None
            per_file_cap = (budget // len(csv_paths) + 1) if budget else None

            class_frames = []
            rows_so_far = 0
            for f in csv_paths:
                if budget and rows_so_far >= budget:
                    break
                df = pd.read_csv(f, nrows=per_file_cap)
                class_frames.append(df)
                rows_so_far += len(df)
            class_df = pd.concat(class_frames, ignore_index=True)

            if n_per_class and len(class_df) > n_per_class:
                class_df = class_df.sample(n=n_per_class, random_state=random_state)
            class_df["label"] = label
            frames.append(class_df)

        full = pd.concat(frames, ignore_index=True)
        return _finalize(full, source="ciciot2023", random_state=random_state)

    files = _find_ciciot_files()
    if not files:
        raise FileNotFoundError(
            "No CICIoT2023 files found under data/raw/. Expected either "
            "data/raw/ciciot2023/CSV/<ClassName>/*.csv (the official release "
            "layout) or data/raw/ciciot2023/*.csv with a label column. "
            "Download from https://www.unb.ca/cic/datasets/iotdataset-2023.html, "
            "or use make_synthetic_iot_traffic() for pipeline development."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "label" not in df.columns:
            label_col = [c for c in df.columns if c.lower() in ("label", "type", "class")]
            if not label_col:
                raise ValueError(f"Could not find a label column in {f}")
            df = df.rename(columns={label_col[0]: "label"})
        if n_per_class and len(df) > n_per_class:
            df = df.sample(n=n_per_class, random_state=random_state)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    return _finalize(full, source="ciciot2023", random_state=random_state)


def _find_edge_iiotset_csv() -> str | None:
    """Locate the official ML-ready CSV
    ('Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv') wherever it
    landed under data/raw/ -- the official release nests it several
    directories deep with spaces in the names, so this searches rather
    than assuming an exact path.
    """
    candidates = glob.glob(
        os.path.join(RAW_DIR, "**", "ML-EdgeIIoT-dataset.csv"), recursive=True
    )
    return candidates[0] if candidates else None


def load_edge_iiotset(n_per_class: int | None = 1000, random_state: int = 42) -> Dataset:
    """Load Edge-IIoTset's official ML-ready CSV with a de-duplication-based
    leakage fix applied.

    Background: Edge-IIoTset (and CICIoT2023) are both documented in the
    literature as having a "serialization artifact" where a non-trivial
    fraction of rows are exact or near-exact duplicates of other rows --
    if a random train/test split isn't aware of this, duplicate rows can
    end up on both sides, letting the model "memorize" a row it has
    technically already seen, inflating test accuracy and every downstream
    fidelity/robustness number computed from it. Measured directly on this
    release (2026-10-02): 821/157,800 rows (0.5%) share identical feature
    values with at least one other row. This loader drops exact duplicates
    (on feature columns, keeping the first occurrence) BEFORE the
    stratified per-class subsample and train/test split, so no duplicate
    pair can land on both sides of the split. This is the general,
    conservative version of the leakage fix documented for this class of
    dataset (see arXiv:2608.15761 for the specific precision-agriculture
    variant of this problem on Edge-IIoTset) -- applied here directly
    against the measured duplication in this release, not a blind
    reproduction of that paper's exact pipeline.

    Label column: "Attack_type" (15 classes: Normal + 14 attack types).
    Two classes are naturally small in the raw release (MITM: 1,214,
    Fingerprinting: 1,001) -- n_per_class defaults to 1,000 to stay within
    what every class can actually supply; raising it above ~1,000 will
    silently cap those two classes at whatever they have available (see
    _finalize's per-class sampling, which only downsamples, never upsamples).
    """
    path = _find_edge_iiotset_csv()
    if path is None:
        raise FileNotFoundError(
            "Could not find ML-EdgeIIoT-dataset.csv under data/raw/. "
            "Expected the official release's "
            "'Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'."
        )
    df = pd.read_csv(path, low_memory=False)
    if "Attack_type" not in df.columns:
        raise ValueError(f"Expected an 'Attack_type' label column in {path}")
    df = df.rename(columns={"Attack_type": "label"})
    df = df.drop(columns=["Attack_label"], errors="ignore")  # redundant binary version of label

    feature_cols = [c for c in df.columns if c != "label"]
    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        warnings.warn(
            f"Edge-IIoTset leakage fix: dropped {n_dropped}/{n_before} exact-duplicate "
            f"rows (by feature values) before splitting, so no duplicate pair can land "
            f"on both sides of train/test."
        )

    if n_per_class:
        # manual per-group sampling (not groupby().apply()) -- pandas 3.x's
        # apply() excludes the grouping column from the result by default,
        # which silently dropped "label" here.
        sampled = []
        for _, group in df.groupby("label"):
            sampled.append(group.sample(n=min(len(group), n_per_class), random_state=random_state))
        df = pd.concat(sampled, ignore_index=True)

    return _finalize(df, source="edge-iiotset", random_state=random_state)


def make_synthetic_iot_traffic(
    n_samples: int = 20000,
    n_features: int = 24,
    n_classes: int = 5,
    random_state: int = 42,
) -> Dataset:
    """Pipeline-development stand-in with IoT-traffic-like structure:
    imbalanced classes, mixed feature scales, correlated feature blocks.
    """
    rng = np.random.default_rng(random_state)

    # class imbalance typical of IDS datasets (benign dominates)
    class_weights = np.array([0.55, 0.2, 0.12, 0.08, 0.05])[:n_classes]
    class_weights = class_weights / class_weights.sum()
    y = rng.choice(n_classes, size=n_samples, p=class_weights)

    # latent structure per class + correlated feature blocks + noise
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    X = centers[y] + rng.normal(0, 1.0, size=(n_samples, n_features))

    # simulate mixed scales (e.g., packet counts vs. normalized ratios)
    scale = rng.uniform(1, 5000, size=n_features)
    X = X * scale

    # inject a correlated redundant feature block (common in flow features)
    X[:, -1] = X[:, 0] * 0.8 + rng.normal(0, 50, size=n_samples)

    feature_names = [f"flow_feat_{i}" for i in range(n_features)]
    class_names = ["Benign", "DDoS", "DoS", "Recon", "Mirai"][:n_classes]

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = [class_names[i] for i in y]
    return _finalize(df, source="synthetic", random_state=random_state)


def _finalize(df: pd.DataFrame, source: str, random_state: int) -> Dataset:
    y_raw_full = df["label"].astype(str)
    X_df = df.drop(columns=["label"])
    X_df = X_df.select_dtypes(include=[np.number])  # drop non-numeric/IP/time cols

    # CICIoT2023-specific data-quality issue: rate-style features (e.g.
    # "Rate" = packet_count / duration) are literal +/-inf when duration
    # rounds to 0 in the original capture -- NOT NaN, so a plain dropna()
    # (which ran BEFORE this fix, and before the numeric-only column
    # selection) never caught them, and StandardScaler crashed on them
    # downstream with "Input X contains infinity". Replace inf with NaN
    # here, on the already-numeric frame, then drop those rows.
    n_before = len(X_df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    keep_mask = X_df.notna().all(axis=1)
    n_inf_or_nan = n_before - int(keep_mask.sum())
    if n_inf_or_nan > 0:
        warnings.warn(
            f"Dropped {n_inf_or_nan}/{n_before} rows with inf/NaN feature values "
            f"(known CICIoT2023 issue: rate-style features divide by a duration "
            f"that rounds to 0)."
        )
    X_df = X_df.loc[keep_mask]
    y_raw = y_raw_full.loc[keep_mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_df.columns),
        class_names=list(le.classes_),
        source=source,
    )


def load_best_available(n_per_class: int = 5000, random_state: int = 42) -> Dataset:
    """Try CICIoT2023, fall back to synthetic with a loud warning."""
    try:
        return load_ciciot2023(n_per_class=n_per_class, random_state=random_state)
    except FileNotFoundError as e:
        warnings.warn(
            f"{e}\nFalling back to SYNTHETIC data for pipeline development. "
            "Results are NOT valid experiment results until real data is used."
        )
        return make_synthetic_iot_traffic(random_state=random_state)
