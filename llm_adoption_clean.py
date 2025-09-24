
import os, sys, argparse, warnings, random, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------- Config & CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM Adoption (Finance) — Clean End-to-End Pipeline")
    p.add_argument("--data",   type=str, default="synthetic_llm_adoption.csv",
                   help="Path to synthetic dataset CSV")
    p.add_argument("--schema", type=str, default="LLM_Adoption_Final_Variable_Schema.xlsx",
                   help="Path to Excel schema (optional). If missing, schema check is skipped.")
    p.add_argument("--seed",   type=int, default=42, help="Random seed")
    p.add_argument("--search_n", type=int, default=1200,
                   help="Number of training rows used for compact CV search")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    return p.parse_args()

# ------------------ Globals (consistent with Chapters) ------------------

TARGET   = "LLM_Adoption_Status"
ID_COLS  = ["Institution_ID"]

CATEGORICAL_NOMINAL = ["Institution_Type","Country","Vendor_Strategy","Deployment_Mode"]
BINARY_COLS         = ["Has_Dedicated_AI_Team","Data_Lake_Exists","Governance_Framework_Exists",
                       "AI_Governance_Committee","Data_Privacy_Compliant"]
ORDINAL_1_5         = ["Governance_Maturity","Model_Risk_Management_Maturity",
                       "Risk_Culture_Score","Cybersecurity_Level","Leadership_Commitment"]
NUMERIC_LOG         = ["Employee_Count","Annual_Revenue_M","AI_Budget_Allocated",
                       "Training_Investment_Per_Employee"]
NUMERIC_OTHER_BASE  = ["IT_Spend_Percent_Revenue","AI_Budget_Percent_Revenue","Budget_Growth_YoY",
                       "Technical_Readiness_Score","API_Integration_Score","Legacy_System_Score",
                       "Data_Maturity_Index","Existing_AI_Projects","Use_Case_Breadth",
                       "Compliance_Incidents_12M"]

# ------------------ Utility: reproducibility & folders ------------------

def setup_environment(seed: int):
    random.seed(seed); np.random.seed(seed)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/ETHICS_NOTE.txt","w") as f:
        f.write("This research uses ONLY synthetic, non-personal, non-sensitive data. "
                "No human subjects or personal data were used.\n")
    print("[INFO] Seeds set and output folders ready.")

# ------------------ Load & Schema Guard ------------------

def load_data(data_path: str, schema_path: str|None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset at: {data_path}")
    df = pd.read_csv(data_path).copy()

    # Optional schema validation
    if schema_path and os.path.exists(schema_path):
        try:
            df_schema = pd.read_excel(schema_path, sheet_name=0)
            # Try common header names
            for colname in ["Variable Name", "Variable", "Name", "Variable_Name"]:
                if colname in df_schema.columns:
                    schema_vars = df_schema[colname].astype(str).str.strip().tolist()
                    break
            else:
                print("[WARN] Could not locate a column containing variable names in schema; skipping strict check.")
                schema_vars = None
        except Exception as e:
            print("[WARN] Could not read schema workbook:", e)
            schema_vars = None
    else:
        if schema_path and not os.path.exists(schema_path):
            print(f"[WARN] Schema workbook not found at {schema_path}. Skipping schema check.")
        schema_vars = None

    if schema_vars is not None:
        expected = set([v for v in schema_vars if v != "TRxGov"])  # derived
        present  = set(df.columns)
        missing  = sorted(list(expected - present))
        extra    = sorted(list(present - expected - {"TRxGov"}))
        print("[INFO] Schema check — missing:", missing, "| extra:", extra)
        if missing:
            raise ValueError("Dataset is missing expected variables from schema.")

    return df

# ------------------ Preprocessing ------------------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def make_preprocessor(include_interaction: bool):
    """Builds the ColumnTransformer as described in Chapter 3.
       - OneHotEncoder for nominal categoricals (drop first)
       - log1p+scale for positive skewed magnitudes
       - scale-only for other continuous
       - passthrough for binaries & ordinals
       - optional TRxGov interaction
    """
    num_other = NUMERIC_OTHER_BASE + (["TRxGov"] if include_interaction else [])
    # version-safe OHE
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    log_scale  = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scale", StandardScaler())
    ])
    scale_only = Pipeline([("scale", StandardScaler())])

    ct = ColumnTransformer([
        ("log_num",   log_scale,   NUMERIC_LOG),
        ("num_other", scale_only,  num_other + ORDINAL_1_5),
        ("ohe_nom",   ohe,         CATEGORICAL_NOMINAL),
        ("binary",    "passthrough", BINARY_COLS),
    ], remainder="drop")

    # If supported, keep pandas output so columns are named
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass

    return ct

# Robust feature-name extraction (no external helpers)
from sklearn.preprocessing import OneHotEncoder as _OHE

def get_feature_names_from_ct(prep: ColumnTransformer, X_transformed):
    """Return feature names in the exact order emitted by a fitted ColumnTransformer.
       Works if X_transformed is DataFrame OR numpy array.
    """
    # If we already have a DataFrame, use its columns
    if hasattr(X_transformed, "columns"):
        return np.array(X_transformed.columns, dtype=object)

    # Otherwise, reconstruct names by walking the fitted CT
    names = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        # Pipeline?
        if hasattr(trans, "named_steps"):
            # search for OneHotEncoder
            found_ohe = None
            for step_name, step_obj in trans.named_steps.items():
                if isinstance(step_obj, _OHE):
                    found_ohe = step_obj
                    break
            if found_ohe is not None:
                ohe_names = found_ohe.get_feature_names_out(cols)
                names.extend(list(ohe_names))
            else:
                names.extend(list(cols))
        elif isinstance(trans, _OHE):
            ohe_names = trans.get_feature_names_out(cols)
            names.extend(list(ohe_names))
        else:
            names.extend(list(cols))
    return np.array(names, dtype=object)

# ------------------ Train/Test Split & EDA ------------------

from sklearn.model_selection import train_test_split

def split_and_eda(df: pd.DataFrame, seed: int):
    # Create interaction once (used by preprocessor when requested)
    if "TRxGov" not in df.columns:
        df["TRxGov"] = df["Technical_Readiness_Score"] * df["Governance_Maturity"]

    dfm = df.drop(columns=ID_COLS, errors="ignore")
    X = dfm.drop(columns=[TARGET])
    y = dfm[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    # --- TRAIN-ONLY EDA ---
    # Class balance
    ax = y_train.value_counts().sort_index().plot(kind="bar")
    ax.set_xlabel("LLM_Adoption_Status (0=No, 1=Yes)")
    ax.set_ylabel("Count"); ax.set_title("Target Class Balance (TRAIN only)")
    plt.tight_layout(); plt.savefig("figures/Fig3-1_ClassBalance.png", dpi=180); plt.close()

    # Numeric correlations with target (train only)
    train_num = pd.concat([X_train.select_dtypes(include="number"), y_train], axis=1)
    corr_mat = train_num.corr().round(3)
    corr_mat.to_csv("artifacts/correlation_matrix_train.csv", index=True)

    return X_train, X_test, y_train, y_test

# ------------------ Models & Search ------------------

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_models(X_train, y_train, seed: int, search_n: int, n_jobs: int):
    preprocessor = make_preprocessor(include_interaction=True)
    pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression())])

    param_grid = [
        # Logistic (transparent baseline)
        {"clf": [LogisticRegression(solver="saga", penalty="elasticnet",
                                    max_iter=4000, random_state=seed)],
         "clf__l1_ratio": [0.3, 0.5],
         "clf__C": [0.1, 0.5, 1.0],
         "clf__class_weight": [None, "balanced"]},

        # Decision Tree (white-box)
        {"clf": [DecisionTreeClassifier(random_state=seed)],
         "clf__max_depth": [4, 6, 8],
         "clf__min_samples_leaf": [20, 50, 100]},

        # Random Forest (compact)
        {"clf": [RandomForestClassifier(random_state=seed, n_jobs=n_jobs)],
         "clf__n_estimators": [200, 400],
         "clf__max_depth": [None, 12],
         "clf__min_samples_leaf": [10, 50],
         "clf__class_weight": [None, "balanced_subsample"]},
    ]

    # Subsample for faster CV
    SEARCH_N = min(search_n, len(X_train))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SEARCH_N, random_state=seed)
    idx, _ = next(sss.split(X_train, y_train))
    X_search, y_search = X_train.iloc[idx], y_train.iloc[idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc",
                        cv=cv, n_jobs=n_jobs, refit=True, verbose=1)
    grid.fit(X_search, y_search)

    best = grid.best_estimator_
    print("[INFO] Best params:", grid.best_params_, "| CV AUC:", round(grid.best_score_, 3))

    # Fit best on full TRAIN
    best.fit(X_train, y_train)
    dump(best, "artifacts/best_model_pipeline.joblib")
    print("[INFO] Saved: artifacts/best_model_pipeline.joblib")

    return best

# ------------------ Evaluation & Thresholding ------------------

from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve, brier_score_loss)
from sklearn.calibration import calibration_curve

def pick_threshold_cv(pipe, X_tr, y_tr, seed: int, n_splits: int = 3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    thrs = []
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        pipe.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        p = pipe.predict_proba(X_tr.iloc[va_idx])[:,1]
        prec, rec, thr = precision_recall_curve(y_tr.iloc[va_idx], p)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        j = np.argmax(f1)
        thrs.append(thr[max(j-1, 0)])
    return float(np.median(thrs))

def report_metrics(name, y_true, p_score, y_hat):
    out = {
        "Where": name,
        "ROC_AUC": roc_auc_score(y_true, p_score),
        "PR_AUC": average_precision_score(y_true, p_score),
        "Accuracy": accuracy_score(y_true, y_hat),
        "Precision": precision_score(y_true, y_hat),
        "Recall": recall_score(y_true, y_hat),
        "F1": f1_score(y_true, y_hat)
    }
    print(f"\n=== {name} ===")
    print("ROC-AUC:", round(out["ROC_AUC"],3),
          "PR-AUC:", round(out["PR_AUC"],3),
          "Acc:", round(out["Accuracy"],3),
          "Prec:", round(out["Precision"],3),
          "Rec:", round(out["Recall"],3),
          "F1:", round(out["F1"],3))
    print("Confusion:\n", confusion_matrix(y_true, y_hat))
    return out

def evaluate(best, X_train, y_train, X_test, y_test, seed: int):
    proba_te = best.predict_proba(X_test)[:,1]
    pred_05  = (proba_te >= 0.5).astype(int)
    rows = [report_metrics("TEST @ 0.5", y_test, proba_te, pred_05)]

    thr_star = pick_threshold_cv(best, X_train, y_train, seed=seed, n_splits=3)
    print("[INFO] Chosen threshold (train-CV):", round(thr_star, 3))
    pred_cv  = (proba_te >= thr_star).astype(int)
    rows.append(report_metrics(f"TEST @ CV-threshold={thr_star:.3f}", y_test, proba_te, pred_cv))

    # PR curve
    prec, rec, thr = precision_recall_curve(y_test, proba_te)
    plt.figure(); plt.step(rec, prec, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (TEST)")
    plt.grid(alpha=.2); plt.savefig("figures/Fig4-PRCurve.png", dpi=180); plt.close()

    # Calibration
    bs = brier_score_loss(y_test, proba_te)
    prob_true, prob_pred = calibration_curve(y_test, proba_te, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration (TEST) — Brier={bs:.4f}")
    plt.grid(alpha=.2); plt.savefig("figures/Fig4-Calibration.png", dpi=180); plt.close()

    # Export metrics
    pd.DataFrame(rows).to_csv("artifacts/model_metrics_best.csv", index=False)
    print("[INFO] Saved: artifacts/model_metrics_best.csv")

# ------------------ Moderation Test (TR×Gov) ------------------

def moderation_delta_auc(X_train, y_train, X_test, y_test, seed: int):
    from sklearn.linear_model import LogisticRegression
    pre_with = make_preprocessor(include_interaction=True)
    pre_no   = make_preprocessor(include_interaction=False)
    lr_with = Pipeline([("prep", pre_with), ("clf", LogisticRegression(max_iter=4000, solver="liblinear", random_state=seed))])
    lr_no   = Pipeline([("prep", pre_no),   ("clf", LogisticRegression(max_iter=4000, solver="liblinear", random_state=seed))])
    lr_with.fit(X_train, y_train); p_with = lr_with.predict_proba(X_test)[:,1]
    lr_no.fit(X_train, y_train);   p_no   = lr_no.predict_proba(X_test)[:,1]
    delta_auc = roc_auc_score(y_test, p_with) - roc_auc_score(y_test, p_no)
    print(f"[INFO] Interaction gain in ROC-AUC (WITH vs NO TRxGov): {delta_auc:.003f}")
    return float(delta_auc)

# ------------------ SHAP Explanations ------------------

def run_shap(best, X_train, X_test):
    try:
        import shap
    except Exception as e:
        print("[WARN] SHAP not installed; skipping SHAP explanations. Install with: pip install shap")
        return

    prep = best.named_steps["prep"]
    clf  = best.named_steps["clf"]

    # Background and test (transformed)
    X_bg_arr = prep.transform(X_train.sample(n=min(2000, len(X_train)), random_state=42))
    X_te_arr = prep.transform(X_test.sample(n=min(2000, len(X_test)),  random_state=123))

    # Feature names in correct order
    feat_names = get_feature_names_from_ct(prep, X_te_arr)

    # Ensure pandas DataFrames so SHAP labels features in plots
    if not hasattr(X_bg_arr, "columns"):
        X_bg = pd.DataFrame(X_bg_arr, columns=feat_names)
        X_te = pd.DataFrame(X_te_arr, columns=feat_names)
    else:
        X_bg, X_te = X_bg_arr, X_te_arr

    # Model-agnostic SHAP
    explainer = shap.Explainer(clf, X_bg)
    ex = explainer(X_te)  # shap.Explanation

    vals = ex.values  # (n_samples, n_features) OR (n_samples, n_outputs, n_features)
    if isinstance(vals, np.ndarray) and vals.ndim == 3:
        out_idx = 1 if vals.shape[1] == 2 else 0
        vals = vals[:, out_idx, :]

    # Global ranking table
    mean_abs = np.abs(vals).mean(axis=0)
    rank = (pd.Series(mean_abs, index=feat_names)
              .sort_values(ascending=False)
              .rename("mean_abs_shap"))
    rank.to_csv("artifacts/shap_global_ranking.csv", index=True)
    print("[INFO] Saved: artifacts/shap_global_ranking.csv")

    # Summary plot
    shap.summary_plot(vals, X_te, feature_names=feat_names, show=False)
    plt.title("SHAP Summary — Best Model")
    plt.savefig("figures/Fig4-SHAP-Summary.png", dpi=180, bbox_inches="tight")
    plt.close()

# ------------------ Main ------------------

def main():
    args = parse_args()
    setup_environment(args.seed)

    # Versions (optional print)
    import sklearn
    print(f"[INFO] Versions — sklearn: {sklearn.__version__} | pandas: {pd.__version__} | numpy: {np.__version__}")

    df = load_data(args.data, args.schema)

    X_train, X_test, y_train, y_test = split_and_eda(df, seed=args.seed)

    best = train_models(X_train, y_train, seed=args.seed, search_n=args.search_n, n_jobs=args.n_jobs)

    evaluate(best, X_train, y_train, X_test, y_test, seed=args.seed)

    moderation_delta_auc(X_train, y_train, X_test, y_test, seed=args.seed)

    run_shap(best, X_train, X_test)

    print("\n[DONE] Outputs saved:")
    print("  - figures/: Fig3-1_ClassBalance.png, Fig4-PRCurve.png, Fig4-Calibration.png, Fig4-SHAP-Summary.png")
    print("  - artifacts/: correlation_matrix_train.csv, model_metrics_best.csv, shap_global_ranking.csv, best_model_pipeline.joblib, ETHICS_NOTE.txt")

if __name__ == "__main__":
    main()