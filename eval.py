import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from agent import analyze_text
warnings.filterwarnings("ignore")

# ── 1. Загрузка и подготовка датасета ────────────────────────────────────────
print("Загружаю датасет...")
df = pd.read_csv("data/financial_moderation_dataset_10.5k_with_normals.csv")

LABEL_MAP = {
    "pass":          {"true_risk": "LOW",    "true_binary": 0},
    "flag-low":      {"true_risk": "LOW",    "true_binary": 0},
    "flag-medium":   {"true_risk": "MEDIUM", "true_binary": 1},
    "block-high":    {"true_risk": "HIGH",   "true_binary": 1},
    "block-extreme": {"true_risk": "HIGH",   "true_binary": 1},
}

df["true_risk"]   = df["label"].map(lambda x: LABEL_MAP[x]["true_risk"])
df["true_binary"] = df["label"].map(lambda x: LABEL_MAP[x]["true_binary"])

# Берём по 10 из каждой категории без groupby
parts = []
for lbl in LABEL_MAP.keys():
    parts.append(df[df["label"] == lbl].sample(10, random_state=42))
sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Выборка: {len(sample)} текстов")
print(sample["label"].value_counts().to_string())

# ── 3. Прогон через агента ────────────────────────────────────────────────────
results = []

for i, row in sample.iterrows():
    text = str(row["text"])[:1500]
    print(f"[{i+1}/{len(sample)}] {row['label']:15s} | {text[:60]}...")

    try:
        result    = analyze_text(text)
        risk      = result.get("risk_level", "LOW")
        predicted = 1 if risk in ("HIGH", "MEDIUM") else 0

        results.append({
            "text":        text[:120],
            "label":       row["label"],
            "true_risk":   row["true_risk"],
            "true_binary": row["true_binary"],
            "pred_risk":   risk,
            "pred_binary": predicted,
            "risk_score":  result.get("risk_score", 0),
            "flags_count": len(result.get("flags", [])),
            "summary":     result.get("summary", ""),
        })

    except Exception as e:
        print(f"  Ошибка: {e}")
        results.append({
            "text": text[:120], "label": row["label"],
            "true_risk": row["true_risk"], "true_binary": row["true_binary"],
            "pred_risk": "ERROR", "pred_binary": 0,
            "risk_score": 0, "flags_count": 0, "summary": str(e),
        })

results_df = pd.DataFrame(results)
results_df.to_csv("data/eval_results_v2.csv", index=False)

# ── 4. Метрики (бинарные) ─────────────────────────────────────────────────────
y_true = results_df["true_binary"]
y_pred = results_df["pred_binary"]

print("\n" + "="*55)
print("БИНАРНЫЕ МЕТРИКИ (0=безопасно, 1=мошенничество)")
print("="*55)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
print()
print(classification_report(
    y_true, y_pred,
    target_names=["Безопасно (0)", "Мошенничество (1)"],
    zero_division=0
))

# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Безопасно", "Мошенничество"],
            yticklabels=["Безопасно", "Мошенничество"])
plt.title("Confusion Matrix — Financial Scam Detector", fontsize=13)
plt.ylabel("Истинный класс")
plt.xlabel("Предсказанный класс")
plt.tight_layout()
plt.savefig("data/confusion_matrix_v2.png", dpi=150)
print("Confusion matrix сохранена: data/confusion_matrix_v2.png")

# ── 6. Детальный breakdown по оригинальным меткам ────────────────────────────
print("ДЕТАЛЬНЫЙ BREAKDOWN ПО МЕТКАМ ДАТАСЕТА")
breakdown = results_df.groupby("label").agg(
    count=("pred_binary", "count"),
    predicted_scam=("pred_binary", "sum"),
    avg_score=("risk_score", "mean"),
    avg_flags=("flags_count", "mean"),
).round(2)
breakdown["detection_rate_%"] = (breakdown["predicted_scam"] / breakdown["count"] * 100).round(1)
print(breakdown.to_string())

# ── 7. Анализ ошибок ──────────────────────────────────────────────────────────
fp = results_df[(results_df.true_binary == 0) & (results_df.pred_binary == 1)]
fn = results_df[(results_df.true_binary == 1) & (results_df.pred_binary == 0)]

print(f"\nЛожноположительные (FP): {len(fp)}")
print(f"Ложноотрицательные (FN): {len(fn)}")

if len(fp):
    print("\nПримеры FP (легальный → детектирован как мошенничество):")
    for _, r in fp.head(3).iterrows():
        print(f"  [{r.label}] score={r.risk_score} | {r.text[:80]}...")

if len(fn):
    print("\nПримеры FN (мошенничество → пропущено):")
    for _, r in fn.head(3).iterrows():
        print(f"  [{r.label}] score={r.risk_score} | {r.text[:80]}...")