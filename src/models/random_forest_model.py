"""
Model 2: Random Forest Regresyon
=================================
Ensemble (topluluk) öğrenme yaklaşımı ile uçuş fiyat tahmini.

Neden Random Forest?
  - XGBoost'a karşılaştırma modeli olarak
  - Overfitting'e karşı doğal dayanıklılık (bagging)
  - Yorumlanabilirlik (feature importance)
  - Hyperparameter'lara daha az hassas

XGBoost ile farkı:
  - XGBoost: Sıralı ağaçlar, her ağaç öncekinin hatasını düzeltir (boosting)
  - Random Forest: Paralel ağaçlar, sonuçları ortalar (bagging)

Çıktılar:
  - models/rf_price_model.joblib
  - reports/model/random_forest_results.txt
  - reports/model/rf_actual_vs_pred.png
  - reports/model/rf_residuals.png
  - reports/model/rf_feature_importance.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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


def load_data():
    df = pd.read_csv(INPUT_CSV)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def tune_random_forest(X_train, y_train):
    """GridSearchCV ile Random Forest hyperparameter tuning."""
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
    }

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring="r2",
        n_jobs=-1, verbose=0, refit=True
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_evaluate():
    """Model eğit, değerlendir, kaydet."""
    print("=" * 60)
    print("  MODEL 2: Random Forest Regresyon")
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

    # Model eğitimi (GridSearch ile tuning)
    print("\n[1/4] GridSearchCV ile hyperparameter tuning...")
    print("      (Bu biraz sürebilir...)")
    model, best_params, best_cv_score = tune_random_forest(X_train, y_train)
    print(f"      Best CV R²: {best_cv_score:.4f}")
    print(f"      Best params: {best_params}")

    # Tahminler
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # ─── Metrikler ────────────────────────────────────────────────────────────
    print("\n[2/4] Metrikler hesaplanıyor...")

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

    # Cross-validation (full dataset)
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

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    metrics["test"]["MAPE"] = mape

    # ─── Grafikler ────────────────────────────────────────────────────────────
    print("[3/4] Grafikler üretiliyor...")

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=40, color="#27AE60", edgecolors="white", linewidth=0.5)
    min_val = min(y_test.min(), y_pred_test.min()) - 200
    max_val = max(y_test.max(), y_pred_test.max()) + 200
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y=x)")
    ax.set_xlabel("Gerçek Medyan Fiyat (TL)", fontsize=12)
    ax.set_ylabel("Tahmin Edilen Fiyat (TL)", fontsize=12)
    ax.set_title("Random Forest: Gerçek vs Tahmin Edilen Fiyatlar", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.text(0.05, 0.92, f"R² = {metrics['test']['R2']:.4f}\nMAE = {metrics['test']['MAE']:.0f} TL",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rf_actual_vs_pred.png", dpi=150)
    plt.close()

    # 2. Residuals
    residuals = y_test.values - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_test, residuals, alpha=0.6, s=40, color="#8E44AD", edgecolors="white", linewidth=0.5)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Tahmin Edilen Fiyat (TL)", fontsize=11)
    axes[0].set_ylabel("Hata / Residual (TL)", fontsize=11)
    axes[0].set_title("Random Forest: Residual Dağılımı", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=20, color="#8E44AD", edgecolor="white", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Hata (TL)", fontsize=11)
    axes[1].set_ylabel("Frekans", fontsize=11)
    axes[1].set_title(f"Hata Histogramı (Ort: {residuals.mean():.0f}, Std: {residuals.std():.0f} TL)",
                      fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rf_residuals.png", dpi=150)
    plt.close()

    # 3. Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": importance
    }).sort_values("importance", ascending=False)

    top_n = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_n)), top_n["importance"], color="#27AE60", alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n["feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Impurity Reduction)", fontsize=11)
    ax.set_title("Random Forest: En Önemli 15 Feature", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()

    # ─── Model Kaydet ─────────────────────────────────────────────────────────
    joblib.dump(model, str(MODEL_DIR / "rf_price_model.joblib"))

    # ─── Rapor ────────────────────────────────────────────────────────────────
    print("[4/4] Rapor yazılıyor...")

    lines = []
    lines.append("=" * 60)
    lines.append("MODEL 2: Random Forest Regresyon - Değerlendirme Raporu")
    lines.append("=" * 60)
    lines.append(f"\nModel: RandomForestRegressor (Bagging Ensemble)")
    lines.append(f"Hedef: {TARGET} (rota×gün bazında medyan uçuş fiyatı)")
    lines.append(f"Dataset: {X.shape[0]} satır, {X.shape[1]} feature")
    lines.append(f"Train/Test: %{int((1-TEST_SIZE)*100)} / %{int(TEST_SIZE*100)} split")
    lines.append(f"Tuning: GridSearchCV (5-fold CV)")

    lines.append(f"\n{'─' * 60}")
    lines.append("HYPERPARAMETRELER (GridSearchCV ile optimize edildi)")
    lines.append(f"{'─' * 60}")
    for k, v in best_params.items():
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
    overfit = metrics['train']['R2'] - metrics['test']['R2']
    if overfit > 0.15:
        lines.append(f"  • ⚠️ Train-Test R² farkı: {overfit:.3f} → Hafif overfitting eğilimi.")
    else:
        lines.append(f"  • Train-Test R² farkı: {overfit:.3f} → Overfitting kontrol altında.")

    report_text = "\n".join(lines)
    (REPORT_DIR / "random_forest_results.txt").write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")

    print(f"\n✅ Random Forest modeli tamamlandı!")
    print(f"   Model: {MODEL_DIR / 'rf_price_model.joblib'}")
    print(f"   Rapor: {REPORT_DIR / 'random_forest_results.txt'}")

    return metrics, feat_imp


if __name__ == "__main__":
    train_and_evaluate()
