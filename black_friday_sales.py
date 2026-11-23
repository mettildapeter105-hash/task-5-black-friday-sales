"""
Black Friday Sales Prediction (scikit-learn / XGBoost)

Predicts the purchase amount of a Black Friday transaction from customer, city and
product information. This is a regression problem; the metric is RMSE.

Pipeline
    1. load and inspect the data
    2. explore it (plots of purchase amount by customer group)
    3. clean + encode the features
    4. feature engineering that captures customer and product effects
         - how often each customer / product appears            (counts)
         - the typical spend of each customer / product         (target encoding,
           computed out-of-fold on the training set so nothing leaks)
         - how many product categories a purchase has
    5. split 80/20, compare several models, report RMSE / MAE / R2
    6. save plots: EDA, model comparison, predicted vs actual, feature importance

Run:
    python black_friday_sales.py
    python black_friday_sales.py --sample 100000      # quick run on a subset
    python black_friday_sales.py --no-xgboost
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

SEED = 42
TARGET = "Purchase"
AGE_ORDER = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path, sample=None):
    df = pd.read_csv(path)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=SEED).reset_index(drop=True)
    return df


def inspect(df):
    print("Shape:", df.shape)
    print("\nFirst rows:")
    print(df.head().to_string())
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())
    print(f"\nUnique customers: {df['User_ID'].nunique():,}   Unique products: {df['Product_ID'].nunique():,}")
    print("\nPurchase amount summary:")
    print(df[TARGET].describe().round(1).to_string())


def plot_eda(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(df[TARGET], bins=50, color="#3d6fb6")
    axes[0, 0].set_title("Distribution of purchase amount")
    axes[0, 0].set_xlabel("Purchase")

    groups = [
        ("Gender", axes[0, 1], None),
        ("Age", axes[0, 2], list(AGE_ORDER)),
        ("City_Category", axes[1, 0], None),
        ("Stay_In_Current_City_Years", axes[1, 1], ["0", "1", "2", "3", "4+"]),
        ("Marital_Status", axes[1, 2], None),
    ]
    for col, ax, order in groups:
        means = df.groupby(col)[TARGET].mean()
        if order:
            means = means.reindex(order)
        ax.bar(means.index.astype(str), means.values, color="#d1495b")
        ax.set_title(f"Average purchase by {col}")
        ax.set_ylim(means.min() * 0.9, means.max() * 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def encode_features(df):
    """Clean and encode the raw columns (no use of the target here)."""
    out = pd.DataFrame(index=df.index)
    out["Gender"] = (df["Gender"] == "M").astype(int)
    out["Age"] = df["Age"].map(AGE_ORDER)
    out["Occupation"] = df["Occupation"]
    for city in ("A", "B", "C"):
        out[f"City_{city}"] = (df["City_Category"] == city).astype(int)
    out["Stay_Years"] = df["Stay_In_Current_City_Years"].replace("4+", "4").astype(int)
    out["Marital_Status"] = df["Marital_Status"]
    out["Product_Category_1"] = df["Product_Category_1"]
    # A missing category 2 / 3 simply means the product has no second / third category
    out["Product_Category_2"] = df["Product_Category_2"].fillna(0)
    out["Product_Category_3"] = df["Product_Category_3"].fillna(0)
    out["n_categories"] = df[["Product_Category_1", "Product_Category_2", "Product_Category_3"]].notna().sum(axis=1)
    return out


def add_count_features(df, features):
    """How often each customer / product appears (uses no target information)."""
    features["user_count"] = df.groupby("User_ID")["Product_ID"].transform("count")
    features["product_count"] = df.groupby("Product_ID")["User_ID"].transform("count")
    return features


def smoothed_means(keys, target, prior, m):
    """Mean of the target per key, pulled toward the global mean for rare keys."""
    stats = target.groupby(keys).agg(["sum", "count"])
    return (stats["sum"] + prior * m) / (stats["count"] + m)


def target_encode(train_keys, train_target, test_keys, m=10, n_splits=5):
    """
    Target (mean) encoding without leakage.
    Training rows are encoded out-of-fold; test rows use statistics from the whole training set.
    """
    prior = train_target.mean()
    train_enc = pd.Series(prior, index=train_keys.index, dtype=float)
    for fit_idx, enc_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(train_keys):
        means = smoothed_means(train_keys.iloc[fit_idx], train_target.iloc[fit_idx], prior, m)
        train_enc.iloc[enc_idx] = train_keys.iloc[enc_idx].map(means).fillna(prior).values
    full_means = smoothed_means(train_keys, train_target, prior, m)
    test_enc = test_keys.map(full_means).fillna(prior)
    return train_enc, test_enc


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models(use_xgboost=True):
    models = {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=20, max_samples=0.3,
            n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED),
    }
    if use_xgboost:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
        except ImportError:
            print("\n(xgboost is not installed - skipping the XGBoost model. "
                  "Install it with: pip install xgboost)")
    return models


def evaluate(models, X_train, y_train, X_test, y_test):
    rows, preds = [], {}
    print("\nTraining and evaluating models on the 20% test set:")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred
        rows.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
            "Seconds": time.time() - start,
        })
        r = rows[-1]
        print(f"  {name:<26} RMSE={r['RMSE']:8.1f}  MAE={r['MAE']:8.1f}  R2={r['R2']:.4f}  ({r['Seconds']:.0f}s)")
    return pd.DataFrame(rows), preds


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_model_comparison(results, path):
    results = results.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(results["Model"], results["RMSE"], color="#3d6fb6")
    for i, v in enumerate(results["RMSE"]):
        ax.text(v + 20, i, f"{v:,.0f}", va="center")
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title("Model comparison")
    ax.set_xlim(0, results["RMSE"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_predicted_vs_actual(y_test, pred, name, path):
    idx = np.random.RandomState(SEED).choice(len(y_test), size=min(4000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test.values[idx], pred[idx], s=6, alpha=0.35, color="#3d6fb6")
    lim = [0, max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "k--", label="perfect prediction")
    ax.set_xlabel("Actual purchase")
    ax.set_ylabel("Predicted purchase")
    ax.set_title(f"Predicted vs actual - {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, name, path):
    idx = np.random.RandomState(SEED).choice(len(X_test), size=min(20000, len(X_test)), replace=False)
    result = permutation_importance(
        model, X_test.iloc[idx], y_test.iloc[idx], n_repeats=3, random_state=SEED,
        scoring="neg_root_mean_squared_error", n_jobs=1)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values, color="#d1495b")
    ax.set_xlabel("Increase in RMSE when the feature is shuffled")
    ax.set_title(f"Feature importance - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return imp.sort_values(ascending=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def find_data(arg, here):
    candidates = [arg] if arg else []
    candidates += [
        os.path.join(here, "Data", "BlackFridaySales.csv"),
        os.path.join(here, "BlackFridaySales.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise SystemExit("Dataset not found. Put BlackFridaySales.csv next to this script "
                     "(or in a 'Data' folder), or pass its path with --data.")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Black Friday sales prediction")
    parser.add_argument("--data", default=None, help="path to BlackFridaySales.csv")
    parser.add_argument("--out-dir", default=os.path.join(here, "outputs"))
    parser.add_argument("--sample", type=int, default=None, help="use only N random rows (quick test)")
    parser.add_argument("--no-xgboost", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(find_data(args.data, here), args.sample)
    inspect(df)
    plot_eda(df, os.path.join(args.out_dir, "eda_plots.png"))

    # Features that do not use the target
    X = encode_features(df)
    X = add_count_features(df, X)
    y = df[TARGET]

    # 80 / 20 split, then target-based features are built from the training part only
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=SEED)
    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    for key, col in (("User_ID", "user_avg_purchase"), ("Product_ID", "product_avg_purchase")):
        X_train[col], X_test[col] = target_encode(df.loc[train_idx, key], y_train, df.loc[test_idx, key])
    print(f"\nFeatures used ({X_train.shape[1]}): {', '.join(X_train.columns)}")
    print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    models = build_models(use_xgboost=not args.no_xgboost)
    results, preds = evaluate(models, X_train, y_train, X_test, y_test)
    print("\nResults:")
    print(results.round({"RMSE": 1, "MAE": 1, "R2": 4, "Seconds": 0}).to_string(index=False))
    results.to_csv(os.path.join(args.out_dir, "model_results.csv"), index=False)

    best_name = results.loc[results["RMSE"].idxmin(), "Model"]
    best_rmse = results["RMSE"].min()
    base_rmse = results.loc[results["Model"] == "Baseline (predict mean)", "RMSE"].iloc[0]
    print(f"\nBest model: {best_name}  (RMSE {best_rmse:,.1f}, "
          f"{(1 - best_rmse / base_rmse) * 100:.1f}% lower than the baseline's {base_rmse:,.1f})")

    plot_model_comparison(results, os.path.join(args.out_dir, "model_comparison.png"))
    plot_predicted_vs_actual(y_test, preds[best_name], best_name, os.path.join(args.out_dir, "predicted_vs_actual.png"))
    importance = plot_feature_importance(models[best_name], X_test, y_test, best_name,
                                         os.path.join(args.out_dir, "feature_importance.png"))
    print("\nMost important features (increase in RMSE when shuffled):")
    print(importance.head(6).round(1).to_string())
    print(f"\nSaved plots and results to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
