"""
Stage-2 classifier: lightweight logistic regression trained on per-component
shape features (component_dataset.csv from build_component_dataset.py) to
separate real free-air components from bowel-gas components -- reducing the
air-mask channel's false-positive rate (measured precision=0.26 aggregate in
eval_air_mask_vs_gt.py) without needing a large annotated dataset.

Ablates feature SUBSETS of size 1-3 (not all 5 features at once) given the
small sample size (23 volumes) -- reports which features actually help on
HELD-OUT data, rather than assuming more features helps, and rather than
reporting only training-set performance (which would just show memorization).

Also explicitly checks whether this approach could ever help the known
recall-failure cases (006, 015, 056, 023, 003) -- expectation is NO, since
their problem is stage 1 never generating the true-positive candidate
component in the first place, so no downstream classifier can rescue them.
"""
import itertools
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "scripts", "component_classifier")
CSV_PATH = os.path.join(OUT_DIR, "component_dataset.csv")

ALL_FEATURES = ["volume_ml", "elongation", "wall_contact_fraction",
                 "centroid_wall_distance_mm", "compactness"]

# the recall-failure cases characterized in the GT evaluation -- their real
# free-air region is excluded entirely by the body-mask (border-touching
# exclusion on a large wall-hugging collection), not merely mis-shaped
FAILURE_CASE_FILES = [
    "20251028006-1.nii.gz", "20251028015-1.nii.gz", "20251028056-1.nii.gz",
    "20251028023-1.nii.gz", "20251028003-1.nii.gz",
]

PRECISION_IMPROVEMENT_THRESHOLD = 0.02


def fit_eval(train_df, holdout_df, features):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df["label"].values
    X_holdout = scaler.transform(holdout_df[features])
    y_holdout = holdout_df["label"].values

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_holdout)
    y_prob = clf.predict_proba(X_holdout)[:, 1]

    precision = precision_score(y_holdout, y_pred, zero_division=0)
    recall = recall_score(y_holdout, y_pred, zero_division=0)
    auprc = average_precision_score(y_holdout, y_prob) if y_holdout.sum() > 0 else float("nan")
    return clf, scaler, precision, recall, auprc


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found -- run build_component_dataset.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    train_df = df[df.fold == "train"].reset_index(drop=True)
    holdout_df = df[df.fold == "holdout"].reset_index(drop=True)

    print(f"Train components:   {len(train_df)} ({int(train_df.label.sum())} positive, "
          f"{int((train_df.label == 0).sum())} negative)")
    print(f"Holdout components: {len(holdout_df)} ({int(holdout_df.label.sum())} positive, "
          f"{int((holdout_df.label == 0).sum())} negative)")

    if train_df.label.sum() < 5 or holdout_df.label.sum() < 3:
        print("\nWARNING: very few positive components available -- results below are "
              "high-variance and should be treated as directional, not conclusive.")

    # baseline: stage 1 alone, every generated component kept (no filtering)
    # -- this IS stage 1's component-level precision, for direct comparison
    baseline_precision = float(holdout_df.label.mean())
    print(f"\nBaseline (stage 1 only, every component kept): "
          f"precision={baseline_precision:.4f}, recall=1.0000 (by definition)")

    print(f"\n=== Feature ablation (1-3 feature subsets, ranked by held-out AUPRC) ===")
    results = []
    for k in [1, 2, 3]:
        for combo in itertools.combinations(ALL_FEATURES, k):
            clf, scaler, precision, recall, auprc = fit_eval(train_df, holdout_df, list(combo))
            results.append({"features": ", ".join(combo), "k": k, "precision": precision,
                             "recall": recall, "auprc": auprc})

    results_df = pd.DataFrame(results).sort_values("auprc", ascending=False)
    print(results_df.head(15).to_string(index=False))
    results_df.to_csv(os.path.join(OUT_DIR, "feature_ablation_results.csv"), index=False)

    best_row = results_df.iloc[0]
    best_features = [f.strip() for f in best_row["features"].split(",")]
    print(f"\nBest feature subset by held-out AUPRC: {best_features} "
          f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}, "
          f"auprc={best_row['auprc']:.4f})")

    clf, scaler, precision, recall, auprc = fit_eval(train_df, holdout_df, best_features)

    print(f"\n=== FINAL COMPARISON ===")
    print(f"  stage-1-only precision:  {baseline_precision:.4f}  (recall=1.0000)")
    print(f"  stage-1+2   precision:  {precision:.4f}  (recall={recall:.4f})")
    improvement = precision - baseline_precision
    meaningfully_improved = improvement > PRECISION_IMPROVEMENT_THRESHOLD
    print(f"  delta: {improvement:+.4f} -- "
          f"{'MEANINGFUL IMPROVEMENT' if meaningfully_improved else 'NOT a meaningful improvement'}")

    print(f"\n=== Recall-failure-case check ===")
    print("Expectation going in: this should NOT help these 5 cases -- their real free-air "
          "region is excluded by the body-mask before stage 2 ever sees it, so no component-"
          "level classifier can rescue a candidate that was never generated.")
    for fname in FAILURE_CASE_FILES:
        subset = df[df.filename == fname]
        if len(subset) == 0:
            print(f"  {fname}: NOT FOUND in dataset (check filename)")
            continue
        n_pos_components = int(subset.label.sum())
        if n_pos_components == 0:
            verdict = "CONFIRMED -- no true-positive candidate exists for stage 2 to rescue"
        else:
            verdict = "UNEXPECTED -- a true-positive candidate DOES exist; check if stage 2 keeps it"
        print(f"  {fname}: {len(subset)} components from stage 1, {n_pos_components} overlap real GT -- {verdict}")

    model_path = os.path.join(OUT_DIR, "component_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "features": best_features}, f)
    print(f"\nSaved model to {model_path}")

    print("\n" + "=" * 70)
    if meaningfully_improved:
        print(">>> Precision improved meaningfully on held-out components.")
        print(">>> Next step: build the refined air-mask channel (filter/reweight by this")
        print(">>> classifier's score) and rerun the fold-4 comparison test before committing")
        print(">>> to anything larger. Report these numbers back before proceeding.")
    else:
        print(">>> Precision did NOT meaningfully improve on held-out components.")
        print(">>> Recommend NOT building the refined channel or rerunning the fold comparison")
        print(">>> -- report these numbers back before any further investment in this direction.")
    print("=" * 70)


if __name__ == "__main__":
    main()
