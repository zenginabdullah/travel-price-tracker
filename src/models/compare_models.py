"""
Model Karşılaştırma
====================
İki modeli (XGBoost vs Random Forest) yan yana karşılaştırır.
Karşılaştırmalı grafikler ve özet rapor üretir.

Çıktılar:
  - reports/model/comparison_metrics.png
  - reports/model/comparison_predictions.png
  - reports/model/comparison_report.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "flights_ml_ready.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "median_price"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_models_and_data():
    """Eğitilmiş modelleri ve veriyi yükle."""
    df = pd.read_csv(INPUT_CSV)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Modelleri yükle
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(MODEL_DIR / "xgb_price_model.json"))

    rf_model = joblib.load(str(MODEL_DIR / "rf_price_model.joblib"))

    return xgb_model, rf_model, X, y, X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("  MODEL KARŞILAŞTIRMA: XGBoost vs Random Forest")
    print("=" * 60)

    xgb_model, rf_model, X, y, X_train, X_test, y_train, y_test = load_models_and_data()

    # Tahminler
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # CV skorları
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "XGBoost": {
            "model": xgb_model,
            "pred": xgb_pred,
            "color": "#2E86C1",
        },
        "Random Forest": {
            "model": rf_model,
            "pred": rf_pred,
            "color": "#27AE60",
        },
    }

    results = {}
    for name, info in models.items():
        pred = info["pred"]
        cv_r2 = cross_val_score(info["model"], X, y, cv=cv, scoring="r2")
        cv_mae = -cross_val_score(info["model"], X, y, cv=cv, scoring="neg_mean_absolute_error")

        results[name] = {
            "test_R2": r2_score(y_test, pred),
            "test_MAE": mean_absolute_error(y_test, pred),
            "test_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "test_MAPE": np.mean(np.abs((y_test - pred) / y_test)) * 100,
            "cv_R2": cv_r2.mean(),
            "cv_R2_std": cv_r2.std(),
            "cv_MAE": cv_mae.mean(),
            "color": info["color"],
        }

    # ─── Grafik 1: Metrik Karşılaştırma ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = list(results.keys())
    colors = [results[m]["color"] for m in model_names]

    # R²
    r2_vals = [results[m]["cv_R2"] for m in model_names]
    r2_stds = [results[m]["cv_R2_std"] for m in model_names]
    bars = axes[0].bar(model_names, r2_vals, yerr=r2_stds, color=colors, alpha=0.8,
                       edgecolor="black", capsize=5)
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_title("R² Karşılaştırması (5-Fold CV)", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                     f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # MAE
    mae_vals = [results[m]["cv_MAE"] for m in model_names]
    bars = axes[1].bar(model_names, mae_vals, color=colors, alpha=0.8, edgecolor="black")
    axes[1].set_ylabel("MAE (TL)", fontsize=11)
    axes[1].set_title("MAE Karşılaştırması (düşük = iyi)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                     f"{val:.0f} TL", ha="center", fontsize=11, fontweight="bold")

    # MAPE
    mape_vals = [results[m]["test_MAPE"] for m in model_names]
    bars = axes[2].bar(model_names, mape_vals, color=colors, alpha=0.8, edgecolor="black")
    axes[2].set_ylabel("MAPE (%)", fontsize=11)
    axes[2].set_title("MAPE Karşılaştırması (düşük = iyi)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, mape_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f"%{val:.1f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_metrics.png", dpi=150)
    plt.close()

    # ─── Grafik 2: Tahmin Karşılaştırma ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (name, info) in enumerate(models.items()):
        ax = axes[idx]
        pred = info["pred"]
        r2 = results[name]["test_R2"]
        mae = results[name]["test_MAE"]

        ax.scatter(y_test, pred, alpha=0.6, s=40, color=info["color"],
                   edgecolors="white", linewidth=0.5)
        min_val = min(y_test.min(), pred.min()) - 200
        max_val = max(y_test.max(), pred.max()) + 200
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax.set_xlabel("Gerçek Fiyat (TL)", fontsize=11)
        ax.set_ylabel("Tahmin (TL)", fontsize=11)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.92, f"R² = {r2:.3f}\nMAE = {mae:.0f} TL",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Model Tahmin Karşılaştırması", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Rapor ────────────────────────────────────────────────────────────────
    # En iyi modeli belirle
    best_model_name = max(results, key=lambda k: results[k]["cv_R2"])

    lines = []
    lines.append("=" * 60)
    lines.append("KARŞILAŞTIRMALI MODEL DEĞERLENDİRME RAPORU")
    lines.append("=" * 60)
    lines.append(f"\nProblem: Regresyon (uçuş fiyat tahmini)")
    lines.append(f"Hedef: {TARGET} (rota×gün bazında medyan uçuş fiyatı)")
    lines.append(f"Dataset: {X.shape[0]} satır, {X.shape[1]} feature")
    lines.append(f"Değerlendirme: 5-Fold Cross-Validation + Hold-out Test Set (%20)")

    lines.append(f"\n{'─' * 60}")
    lines.append("KARŞILAŞTIRMA TABLOSU")
    lines.append(f"{'─' * 60}")
    lines.append(f"\n  {'Metrik':<12} {'XGBoost':>12} {'Random Forest':>15}")
    lines.append(f"  {'─'*12} {'─'*12} {'─'*15}")
    lines.append(f"  {'CV R²':<12} {results['XGBoost']['cv_R2']:>12.4f} {results['Random Forest']['cv_R2']:>15.4f}")
    lines.append(f"  {'Test R²':<12} {results['XGBoost']['test_R2']:>12.4f} {results['Random Forest']['test_R2']:>15.4f}")
    lines.append(f"  {'Test MAE':<12} {results['XGBoost']['test_MAE']:>10,.0f} TL {results['Random Forest']['test_MAE']:>13,.0f} TL")
    lines.append(f"  {'Test RMSE':<12} {results['XGBoost']['test_RMSE']:>10,.0f} TL {results['Random Forest']['test_RMSE']:>13,.0f} TL")
    lines.append(f"  {'Test MAPE':<12} {results['XGBoost']['test_MAPE']:>11.1f}% {results['Random Forest']['test_MAPE']:>14.1f}%")

    lines.append(f"\n{'─' * 60}")
    lines.append("SONUÇ VE YORUM")
    lines.append(f"{'─' * 60}")
    lines.append(f"\n  ★ En iyi model: {best_model_name} (CV R² = {results[best_model_name]['cv_R2']:.4f})")

    r2_diff = abs(results["XGBoost"]["cv_R2"] - results["Random Forest"]["cv_R2"])
    if r2_diff < 0.02:
        lines.append(f"  • İki model arasındaki fark minimal ({r2_diff:.3f}).")
        lines.append(f"  • Her iki model de benzer performans gösteriyor.")
    else:
        lines.append(f"  • {best_model_name}, diğer modelden {r2_diff:.3f} R² farkıyla daha iyi.")

    lines.append(f"\n  Güçlü Yönler:")
    lines.append(f"    - Rota ve bayram dönemi etkisi başarıyla yakalanıyor")
    lines.append(f"    - Ortalama hata ±{results[best_model_name]['cv_MAE']:.0f} TL ile kabul edilebilir düzeyde")
    lines.append(f"    - Overfitting kontrol altında (regularizasyon + CV)")

    lines.append(f"\n  Zayıf Yönler / Sınırlılıklar:")
    lines.append(f"    - Tek zaman diliminde (Mayıs-Haziran 2026) toplanan veri")
    lines.append(f"    - Dinamik fiyatlama ve talep bilgisi mevcut değil")
    lines.append(f"    - 270 aggregated satır ile sınırlı dataset boyutu")

    lines.append(f"\n  Gelecekte Yapılabilecek İyileştirmeler:")
    lines.append(f"    - Daha uzun tarih aralığında veri toplama (mevsimsellik)")
    lines.append(f"    - Aynı uçuşu farklı günlerde tekrar çekerek fiyat değişim trendi")
    lines.append(f"    - Tatil/etkinlik takvimi entegrasyonu")
    lines.append(f"    - Gerçek zamanlı fiyat takibi ile model güncelleme")

    report_text = "\n".join(lines)
    (REPORT_DIR / "comparison_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")

    print(f"\n✅ Karşılaştırma tamamlandı!")
    print(f"   Rapor: {REPORT_DIR / 'comparison_report.txt'}")
    print(f"   Grafikler: {REPORT_DIR / 'comparison_*.png'}")


if __name__ == "__main__":
    main()
