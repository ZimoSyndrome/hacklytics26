"""
classifier_suite.py
--------------------
Runs a suite of scikit-learn classifiers of varying complexity on binary
classification data (X, y), with handling for class imbalance.

Assumptions:
  - X is a NumPy array or pandas DataFrame of features
  - y is a 1-D array of binary labels (0/1 or similar)
  - y is imbalanced

Imbalance strategy:
  - class_weight='balanced' where supported (adjusts loss/criterion)
  - SMOTE oversampling (via imbalanced-learn) as an alternative pipeline step
  - Primary evaluation metrics are ROC-AUC and Average Precision (PR-AUC),
    which are more informative than accuracy for imbalanced data
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    make_scorer, roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import pandas as pd

# ── Optional: SMOTE (requires imbalanced-learn: pip install imbalanced-learn) ──
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Note: imbalanced-learn not installed. Skipping SMOTE pipeline.\n"
          "Install with: pip install imbalanced-learn\n")

# ── Optional: XGBoost ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =============================================================================
# 1.  Define classifiers
# =============================================================================

def build_classifiers(use_smote=False):
    """
    Returns an ordered dict of (name -> sklearn Pipeline).
    Pipelines include StandardScaler for models that need it.
    class_weight='balanced' is set where supported to handle imbalance.
    """
    PipelineCls = ImbPipeline if (use_smote and HAS_SMOTE) else Pipeline
    smote_step  = [("smote", SMOTE(random_state=42))] if (use_smote and HAS_SMOTE) else []

    classifiers = {
        # ── Baseline ────────────────────────────────────────────────────────
        "Dummy (stratified)": Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=42))
        ]),

        # ── Simple / interpretable ──────────────────────────────────────────
        "Logistic Regression": PipelineCls(smote_step + [
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42))
        ]),

        "Decision Tree": PipelineCls(smote_step + [
            ("clf", DecisionTreeClassifier(
                class_weight="balanced", max_depth=5, random_state=42))
        ]),

        "K-Nearest Neighbours": PipelineCls(smote_step + [
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5))
            # KNN has no class_weight; SMOTE or prior resampling helps here
        ]),

        # ── Moderate complexity ─────────────────────────────────────────────
        "SVM (RBF kernel)": PipelineCls(smote_step + [
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", class_weight="balanced",
                          probability=True, random_state=42))
        ]),

        "Random Forest": PipelineCls(smote_step + [
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1))
        ]),

        "Extra Trees": PipelineCls(smote_step + [
            ("clf", ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1))
        ]),

        # ── High complexity ─────────────────────────────────────────────────
        "Gradient Boosting": PipelineCls(smote_step + [
            # GBM doesn't support class_weight directly;
            # use sample_weight via SMOTE or scale_pos_weight (XGB)
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42))
        ]),

        "MLP Neural Network": PipelineCls(smote_step + [
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, random_state=42))
        ]),
    }

    if HAS_XGB:
        classifiers["XGBoost"] = PipelineCls(smote_step + [
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                eval_metric="logloss",
                # scale_pos_weight handles imbalance natively in XGBoost
                # set to neg/pos ratio; cross_val won't know it automatically,
                # so we leave it at 1 and rely on SMOTE or manual setting
                use_label_encoder=False, random_state=42, n_jobs=-1))
        ])

    return classifiers


# =============================================================================
# 2.  Scoring metrics (suited for imbalanced data)
# =============================================================================

import sklearn
from packaging.version import Version

if Version(sklearn.__version__) >= Version("1.4"):
    scoring = {
        "ROC-AUC":    make_scorer(roc_auc_score,          response_method="predict_proba"),
        "PR-AUC":     make_scorer(average_precision_score, response_method="predict_proba"),
        "F1 (macro)": make_scorer(f1_score, average="macro", zero_division=0),
        "MCC":        make_scorer(matthews_corrcoef),
    }
else:
    scoring = {
        "ROC-AUC":    make_scorer(roc_auc_score,           needs_proba=True),
        "PR-AUC":     make_scorer(average_precision_score, needs_proba=True),
        "F1 (macro)": make_scorer(f1_score, average="macro", zero_division=0),
        "MCC":        make_scorer(matthews_corrcoef),
    }


# =============================================================================
# 3.  Run cross-validated evaluation
# =============================================================================

def run_suite(X, y, n_splits=5, use_smote=False, random_state=42):
    """
    Evaluates all classifiers using stratified k-fold cross-validation.

    Parameters
    ----------
    X            : array-like, shape (n_samples, n_features)
    y            : array-like, shape (n_samples,), binary labels
    n_splits     : number of CV folds (default 5)
    use_smote    : if True and imbalanced-learn is available, apply SMOTE
                   inside each fold
    random_state : random seed

    Returns
    -------
    results_df : pd.DataFrame with mean ± std for each metric per classifier
    raw        : dict of raw cross_validate outputs
    """
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
    clfs   = build_classifiers(use_smote=use_smote)
    raw    = {}
    rows   = []

    print(f"{'Classifier':<28} {'ROC-AUC':>12} {'PR-AUC':>12} "
          f"{'F1-macro':>12} {'MCC':>10}")
    print("-" * 78)

    for name, pipeline in clfs.items():
        try:
            cv_results = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
                error_score="raise"
            )
            raw[name] = cv_results

            means = {m: cv_results[f"test_{m}"].mean() for m in scoring}
            stds  = {m: cv_results[f"test_{m}"].std()  for m in scoring}

            print(f"{name:<28} "
                  f"{means['ROC-AUC']:>6.3f}±{stds['ROC-AUC']:.3f}  "
                  f"{means['PR-AUC']:>6.3f}±{stds['PR-AUC']:.3f}  "
                  f"{means['F1 (macro)']:>6.3f}±{stds['F1 (macro)']:.3f}  "
                  f"{means['MCC']:>6.3f}±{stds['MCC']:.3f}")

            row = {"Classifier": name}
            for m in scoring:
                row[f"{m} mean"] = round(means[m], 4)
                row[f"{m} std"]  = round(stds[m],  4)
            rows.append(row)

        except Exception as e:
            import traceback
            print(f"{name:<28}  ERROR: {e}")
            traceback.print_exc()

    print()
    results_df = pd.DataFrame(rows).set_index("Classifier")
    return results_df, raw


# =============================================================================
# 4.  Feature importance helper (for tree-based models after full fit)
# =============================================================================

def print_feature_importances(pipeline, feature_names, top_n=15):
    """Print top-N feature importances for tree-based models."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        # handle imblearn pipelines where step name may differ
        clf = pipeline.steps[-1][1]
    if not hasattr(clf, "feature_importances_"):
        print("Model does not expose feature_importances_.")
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} feature importances:")
    for rank, i in enumerate(idx, 1):
        name = feature_names[i] if feature_names is not None else f"feature_{i}"
        print(f"  {rank:>2}. {name:<30} {importances[i]:.4f}")


# =============================================================================
# 5.  Entry point — replace X and y with your data
# =============================================================================

if __name__ == "__main__":

    # # ── Replace this block with your actual data ───────────────────────────
    # from sklearn.datasets import make_classification
    # X, y = make_classification(
    #     n_samples=2000, n_features=20, n_informative=8,
    #     weights=[0.85, 0.15],   # ~85/15 imbalance
    #     random_state=42
    # )
    load_path = 'data/processed/audio_call_matrix.csv'
    audio = pd.read_csv(load_path)
    n_audio_features = audio.shape[1] - 3 # X_unscaled removes 3 audio features a few lines down
    load_path = 'data/processed/text_call_matrix.csv'
    text = pd.read_csv(load_path)
    n_text_features = text.shape[1] - 3 # X_unscaled removes 3 textual features a few lines down
    X_all = pd.concat([text,audio],axis=1)
    X_unscaled = X_all.select_dtypes(include="number") # only include numerical data
    n_features = X_unscaled

    text_features_vec = np.arange(n_text_features)

    scaler = StandardScaler()
    scaler.fit(X_unscaled)
    X = pd.DataFrame(scaler.transform(X_unscaled))
    y_all = pd.read_csv('data/y.csv')
    y_all = y_all.drop(columns=["Unnamed: 0"])
    y_just_ones = y_all[y_all.sum(axis=1) < 2]
    X = X[y_all.sum(axis=1) < 2]
    print(f'X.shape = {X.shape}')
    y = y_just_ones.sum(axis=1)

    X = np.array(X)
    y = np.array(y).ravel()

    print(f'y.shape = {y.shape}')

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    # ───────────────────────────────────────────────────────────────────────

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")
    print()

    # Run without SMOTE (class_weight='balanced' handles imbalance)
    print("=" * 78)
    print("RESULTS  (class_weight='balanced', no SMOTE)")
    print("=" * 78)
    results, raw = run_suite(X, y, n_splits=5, use_smote=False)

    # Optionally run again with SMOTE
    if HAS_SMOTE:
        print("=" * 78)
        print("RESULTS  (SMOTE oversampling inside each fold)")
        print("=" * 78)
        results_smote, raw_smote = run_suite(X, y, n_splits=5, use_smote=True)

    # Show full results table
    print("\nFull results table (mean scores):")
    mean_cols = [c for c in results.columns if "mean" in c]
    print(results[mean_cols].sort_values("ROC-AUC mean", ascending=False).to_string())

    # Feature importances from a Random Forest fit on all data
    print("\n--- Feature importances from Random Forest (fit on full data) ---")
    from sklearn.ensemble import RandomForestClassifier as RFC
    rf = RFC(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y)
    # Wrap in a dummy pipeline-like object for the helper
    class _Wrap:
        steps = [("clf", rf)]
        def named_steps(self): pass
    _Wrap.named_steps = {"clf": rf}
    print_feature_importances(_Wrap, feature_names, top_n=10)
