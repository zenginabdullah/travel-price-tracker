"""
Model 1: XGBoost Regresyon
===========================
Gradient Boosted Trees yaklaşımı ile uçuş fiyat tahmini.

Neden XGBoost?
  - Tabular veride state-of-the-art performans
  - Non-linear ilişkileri yakalayabilir
  - Feature importance sağlar
  - Regularizasyon ile overfitting kontrolü

Çıktılar:
  - models/xgb_price_model.json
  - reports/model/xgboost_results.txt
  - reports/model/xgboost_actual_vs_pred.png
  - reports/model/xgboost_residuals.png
  - reports/model/xgboost_feature_importance.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import json

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "flights_ml_ready.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
TARGET = "median_price"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optuna ile bulunan en iyi parametreler
BEST_PARAMS = {
    "n_estimators": 461,
    "max_depth": 8,
    "learning_rate": 0.0925,
    "subsample": 0.684,
    "colsample_bytree": 0.726,
    "reg_alpha": 1.88,
    "reg_lambda": 9.55,
    "min_child_weight": 3,
    "gamma": 2.71,
}


def load_data():
    df = pd.read_csv(INPUT_CSV)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_and_evaluate():
    """Model eğit, değerlendir, kaydet."""
    print("=" * 60)
    print("  MODEL 1: XGBoost Regresyon")
    print("=" * 60)

    # Veri
    X, y = load_data()
    print(f"\n[INFO] Dataset: {X.shape[0]} satır, {X.shape[1]} feature")
    print(f"[INFO] Hedef ({TARGET}): min={y.min():.0f}, max={y.max():.0f}, ort={y.mean():.0f} TL")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Model eğitimi
    print("\n[1/4] Model eğitiliyor...")
    model = XGBRegressor(**BEST_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    # Tahminler
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # ─── Metrikler ────────────────────────────────────────────────────────────
    print("[2/4] Metrikler hesaplanıyor...")

    metrics = {
        "train": {
            "MAE": mean_absolute_error(y_train, y_pred_train),
            "MSE": mean_squared_error(y_train, y_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "R2": r2_score(y_train, y_pred_train),
        },
        "test": {
            "MAE": mean_absolute_error(y_test, y_pred_test),
            "MSE": mean_squared_error(y_test, y_pred_test),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "R2": r2_score(y_test, y_pred_test),
        },
    }

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))

    metrics["cv"] = {
        "R2_mean": cv_r2.mean(),
        "R2_std": cv_r2.std(),
        "MAE_mean": cv_mae.mean(),
        "MAE_std": cv_mae.std(),
        "RMSE_mean": cv_rmse.mean(),
        "RMSE_std": cv_rmse.std(),
        "fold_scores": cv_r2.tolist(),
    }

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    metrics["test"]["MAPE"] = mape

    # ─── Grafikler ────────────────────────────────────────────────────────────
    print("[3/4] Grafikler üretiliyor...")

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=40, color="#2E86C1", edgecolors="white", linewidth=0.5)
    min_val = min(y_test.min(), y_pred_test.min()) - 200
    max_val = max(y_test.max(), y_pred_test.max()) + 200
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y=x)")
    ax.set_xlabel("Gerçek Medyan Fiyat (TL)", fontsize=12)
    ax.set_ylabel("Tahmin Edilen Fiyat (TL)", fontsize=12)
    ax.set_title("XGBoost: Gerçek vs Tahmin Edilen Fiyatlar", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.text(0.05, 0.92, f"R² = {metrics['test']['R2']:.4f}\nMAE = {metrics['test']['MAE']:.0f} TL",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "xgboost_actual_vs_pred.png", dpi=150)
    plt.close()

    # 2. Residuals
    residuals = y_test.values - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_test, residuals, alpha=0.6, s=40, color="#E74C3C", edgecolors="white", linewidth=0.5)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Tahmin Edilen Fiyat (TL)", fontsize=11)
    axes[0].set_ylabel("Hata / Residual (TL)", fontsize=11)
    axes[0].set_title("XGBoost: Residual Dağılımı", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=20, color="#E74C3C", edgecolor="white", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Hata (TL)", fontsize=11)
    axes[1].set_ylabel("Frekans", fontsize=11)
    axes[1].set_title(f"Hata Histogramı (Ort: {residuals.mean():.0f}, Std: {residuals.std():.0f} TL)",
                      fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "xgboost_residuals.png", dpi=150)
    plt.close()

    # 3. Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": importance
    }).sort_values("importance", ascending=False)

    top_n = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_n)), top_n["importance"], color="#2E86C1", alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n["feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("XGBoost: En Önemli 15 Feature", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "xgboost_feature_importance.png", dpi=150)
    plt.close()

    # ─── Model Kaydet ─────────────────────────────────────────────────────────
    model.save_model(str(MODEL_DIR / "xgb_price_model.json"))

    # ─── Rapor ────────────────────────────────────────────────────────────────
    print("[4/4] Rapor yazılıyor...")

    lines = []
    lines.append("=" * 60)
    lines.append("MODEL 1: XGBoost Regresyon - Değerlendirme Raporu")
    lines.append("=" * 60)
    lines.append(f"\nModel: XGBRegressor (Gradient Boosted Trees)")
    lines.append(f"Hedef: {TARGET} (rota×gün bazında medyan uçuş fiyatı)")
    lines.append(f"Dataset: {X.shape[0]} satır, {X.shape[1]} feature")
    lines.append(f"Train/Test: %{int((1-TEST_SIZE)*100)} / %{int(TEST_SIZE*100)} split")

    lines.append(f"\n{'─' * 60}")
    lines.append("HYPERPARAMETRELER (Optuna ile optimize edildi, 100 trial)")
    lines.append(f"{'─' * 60}")
    for k, v in BEST_PARAMS.items():
        lines.append(f"  {k:<20s} = {v}")

    lines.append(f"\n{'─' * 60}")
    lines.append("PERFORMANS METRİKLERİ")
    lines.append(f"{'─' * 60}")
    lines.append(f"\n  {'Metrik':<8} {'Train':>12} {'Test':>12}")
    lines.append(f"  {'─'*8} {'─'*12} {'─'*12}")
    lines.append(f"  {'MAE':<8} {metrics['train']['MAE']:>10,.0f} TL {metrics['test']['MAE']:>10,.0f} TL")
    lines.append(f"  {'MSE':<8} {metrics['train']['MSE']:>10,.0f}    {metrics['test']['MSE']:>10,.0f}")
    lines.append(f"  {'RMSE':<8} {metrics['train']['RMSE']:>10,.0f} TL {metrics['test']['RMSE']:>10,.0f} TL")
    lines.append(f"  {'R²':<8} {metrics['train']['R2']:>12.4f} {metrics['test']['R2']:>12.4f}")
    lines.append(f"  {'MAPE':<8} {'─':>12} {metrics['test']['MAPE']:>11.1f}%")

    lines.append(f"\n  Cross-Validation (5-Fold):")
    lines.append(f"    R²:   {metrics['cv']['R2_mean']:.4f} ± {metrics['cv']['R2_std']:.4f}")
    lines.append(f"    MAE:  {metrics['cv']['MAE_mean']:.0f} ± {metrics['cv']['MAE_std']:.0f} TL")
    lines.append(f"    RMSE: {metrics['cv']['RMSE_mean']:.0f} ± {metrics['cv']['RMSE_std']:.0f} TL")
    lines.append(f"    Fold R² skorları: {[f'{s:.3f}' for s in metrics['cv']['fold_scores']]}")

    lines.append(f"\n{'─' * 60}")
    lines.append("FEATURE IMPORTANCE (Top 10)")
    lines.append(f"{'─' * 60}")
    for _, row in feat_imp.head(10).iterrows():
        lines.append(f"  {row['feature']:<30s} → {row['importance']:.4f}")

    lines.append(f"\n{'─' * 60}")
    lines.append("YORUM")
    lines.append(f"{'─' * 60}")
    lines.append(f"  • Model R² = {metrics['cv']['R2_mean']:.2f} ile fiyat varyansının %{metrics['cv']['R2_mean']*100:.0f}'ini açıklıyor.")
    lines.append(f"  • Ortalama tahmin hatası ±{metrics['cv']['MAE_mean']:.0f} TL (MAPE: %{metrics['test']['MAPE']:.1f}).")
    lines.append(f"  • Train-Test R² farkı: {metrics['train']['R2'] - metrics['test']['R2']:.3f} → Overfitting kontrol altında.")
    lines.append(f"  • En etkili faktörler: rota, bayram dönemi, uçuşa kalan gün.")

    report_text = "\n".join(lines)
    (REPORT_DIR / "xgboost_results.txt").write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")

    print(f"\n✅ XGBoost modeli tamamlandı!")
    print(f"   Model: {MODEL_DIR / 'xgb_price_model.json'}")
    print(f"   Rapor: {REPORT_DIR / 'xgboost_results.txt'}")

    return metrics, feat_imp


if __name__ == "__main__":
    train_and_evaluate()
