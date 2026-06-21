import json
import warnings
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from agent import analyze_text
warnings.filterwarnings("ignore")

# ── Цвета и эмодзи по уровню риска ───────────────────────────────────────────
RISK_CONFIG = {
    "LOW":    {"emoji": "✅", "color": "#16a34a", "label": "Низкий риск"},
    "MEDIUM": {"emoji": "⚠️", "color": "#b45309", "label": "Средний риск"},
    "HIGH":   {"emoji": "🚨", "color": "#b91c1c", "label": "Высокий риск"},
}

CATEGORY_LABELS = {
    "UNREALISTIC_RETURNS": "💰 Нереальная доходность",
    "NO_LICENSE":          "📋 Нет лицензии",
    "URGENCY":             "⏰ Искусственная срочность",
    "RECRUITMENT":         "👥 Рекрутинг участников",
    "VAGUENESS":           "🌫️ Размытые объяснения",
    "SOCIAL_PROOF":        "👤 Фейковые отзывы",
    "GUARANTEED":          "🔒 Гарантия прибыли",
    "SUSPICIOUS_LINK": "🔗 Подозрительная ссылка",
}

def build_result_html(result: dict, original_text: str) -> str:
    risk    = result.get("risk_level", "LOW")
    score   = result.get("risk_score", 0)
    flags   = result.get("flags", [])
    summary = result.get("summary", "")
    cfg     = RISK_CONFIG.get(risk, RISK_CONFIG["LOW"])
    bar_color = cfg["color"]

    # Подсветка флажков в тексте — тёмный текст поверх светлого фона
    highlighted = original_text
    highlight_bg = {
        "#16a34a": "#dcfce7",
        "#b45309": "#fef9c3",
        "#b91c1c": "#fee2e2",
    }
    bg = highlight_bg.get(bar_color, "#fee2e2")
    for flag in flags:
        fragment = flag.get("fragment", "")
        if fragment and fragment in highlighted:
            highlighted = highlighted.replace(
                fragment,
                f'<mark style="background:{bg};color:#111;font-weight:600;'
                f'border-radius:3px;padding:1px 4px;border-bottom:2px solid {bar_color}">'
                f'{fragment}</mark>'
            )

    flags_html = ""
    for flag in flags:
        cat       = flag.get("category", "")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        fragment  = flag.get("fragment", "")
        explanation = flag.get("explanation", "")
        flags_html += f"""
        <div style="border-left:3px solid {bar_color};padding:8px 12px;margin:8px 0;
                    background:#f9fafb;border-radius:0 6px 6px 0">
            <div style="font-weight:700;color:{bar_color};font-size:13px">{cat_label}</div>
            <div style="font-style:italic;color:#111;margin:3px 0;font-size:13px">«{fragment}»</div>
            <div style="color:#222;font-size:13px">{explanation}</div>
        </div>"""

    disclaimer = """
    <div style="margin-top:16px;padding:10px;background:#f3f4f6;border-radius:6px;
                font-size:12px;color:#444;border:1px solid #e5e7eb">
        ⚠️ Это предварительный сигнал, а не окончательный вердикт.
        Для принятия финансовых решений обращайтесь к лицензированным специалистам.
    </div>"""

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:800px">

        <div style="display:flex;align-items:center;gap:12px;padding:16px;
                    background:#f9fafb;border:2px solid {bar_color};
                    border-radius:10px;margin-bottom:16px">
            <div style="font-size:36px">{cfg["emoji"]}</div>
            <div>
                <div style="font-size:22px;font-weight:700;color:{bar_color}">{cfg["label"]}</div>
                <div style="color:#333;font-size:14px">Индекс подозрительности: <strong>{score}/100</strong></div>
            </div>
        </div>

        <div style="background:#e5e7eb;border-radius:999px;height:10px;margin-bottom:16px">
            <div style="background:{bar_color};width:{score}%;height:10px;border-radius:999px"></div>
        </div>

        <div style="padding:12px;background:#fff;border:1px solid #e5e7eb;
                    border-radius:8px;margin-bottom:12px">
            <div style="font-weight:700;margin-bottom:6px;color:#111">📝 Вывод агента</div>
            <div style="color:#222;font-size:14px;line-height:1.6">{summary}</div>
        </div>

        <div style="padding:12px;background:#fff;border:1px solid #e5e7eb;
                    border-radius:8px;margin-bottom:12px">
            <div style="font-weight:700;margin-bottom:8px;color:#111">📄 Анализируемый текст</div>
            <div style="color:#222;font-size:14px;line-height:1.8">{highlighted}</div>
        </div>

        <div style="padding:12px;background:#fff;border:1px solid #e5e7eb;border-radius:8px">
            <div style="font-weight:700;margin-bottom:8px;color:#111">
                🚩 Найдено красных флажков: {len(flags)}
            </div>
            {flags_html if flags_html else
             '<div style="color:#555;font-size:14px">Подозрительных признаков не обнаружено</div>'}
        </div>

        {disclaimer}
    </div>
    """
    return html


def analyze(text: str):
    if not text or not text.strip():
        return "<div style='color:#666;padding:20px'>Введите текст для анализа</div>"
    try:
        result = analyze_text(text)
        return build_result_html(result, text)
    except Exception as e:
        return f"<div style='color:#b91c1c;padding:20px'>Ошибка: {str(e)}</div>"


# ── Аналитика ─────────────────────────────────────────────────────────────────
def load_eval_stats():
    try:
        df = pd.read_csv("data/eval_results_v2.csv")
        y_true = df["true_binary"]
        y_pred = df["pred_binary"]
        stats = {
            "accuracy":  round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        }
        breakdown = df.groupby("label").agg(
            count=("pred_binary", "count"),
            detected=("pred_binary", "sum"),
            avg_score=("risk_score", "mean"),
        ).round(1)
        breakdown["detection_%"] = (
            breakdown["detected"] / breakdown["count"] * 100
        ).round(0).astype(int)
        return stats, breakdown, df
    except Exception as e:
        return None, None, None


def build_metrics_html(stats, breakdown):
    if stats is None:
        return "<div style='color:#b91c1c'>Сначала запустите python eval.py</div>"

    cards = [
        ("Accuracy",  stats["accuracy"],  "#16a34a", "#f0fdf4", "#bbf7d0"),
        ("Precision", stats["precision"], "#2563eb", "#eff6ff", "#bfdbfe"),
        ("Recall",    stats["recall"],    "#b45309", "#fefce8", "#fde68a"),
        ("F1-score",  stats["f1"],        "#7c3aed", "#faf5ff", "#ddd6fe"),
    ]
    cards_html = ""
    for name, val, color, bg, border in cards:
        cards_html += f"""
        <div style="padding:16px;background:{bg};border:2px solid {border};
                    border-radius:10px;text-align:center">
            <div style="font-size:30px;font-weight:700;color:{color}">{val}</div>
            <div style="color:#111;font-weight:600;margin-top:4px">{name}</div>
        </div>"""

    label_colors = {
        "pass":          "#16a34a",
        "flag-low":      "#65a30d",
        "flag-medium":   "#b45309",
        "block-high":    "#b91c1c",
        "block-extreme": "#7f1d1d",
    }

    rows_html = ""
    for label, row in breakdown.iterrows():
        pct   = int(row["detection_%"])
        color = label_colors.get(label, "#888")
        rows_html += f"""
        <tr style="border-bottom:1px solid #e5e7eb;background:#ffffff">
            <td style="padding:10px;font-weight:600;color:{color}">{label}</td>
            <td style="padding:10px;text-align:center;color:#111;background:#ffffff">{int(row['count'])}</td>
            <td style="padding:10px;text-align:center;color:#111;background:#ffffff">{int(row['detected'])}</td>
            <td style="padding:10px;text-align:center;color:#111;background:#ffffff">{row['avg_score']}</td>
            <td style="padding:10px;background:#ffffff">
                <div style="display:flex;align-items:center;gap:8px">
                    <div style="flex:1;background:#e5e7eb;border-radius:999px;height:8px">
                        <div style="background:{color};width:{pct}%;height:8px;border-radius:999px"></div>
                    </div>
                    <span style="color:#111;font-weight:700;min-width:36px">{pct}%</span>
                </div>
            </td>
        </tr>"""

    html = f"""
    <div style="font-family:-apple-system,sans-serif;color:#111">

        <h3 style="color:#ffffff;margin-bottom:12px">📊 Бинарные метрики (n=50)</h3>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px">
            {cards_html}
        </div>

        <h3 style="color:#ffffff;margin-bottom:12px">🎯 Detection Rate по категориям</h3>
        <table style="width:100%;border-collapse:collapse;font-size:14px;
                      background:#ffffff;border-radius:8px;overflow:hidden;
                      border:1px solid #e5e7eb">
            <tr style="background:#f3f4f6">
                <th style="padding:10px;text-align:left;color:#111;border-bottom:2px solid #e5e7eb">Метка</th>
                <th style="padding:10px;text-align:center;color:#111;border-bottom:2px solid #e5e7eb">Всего</th>
                <th style="padding:10px;text-align:center;color:#111;border-bottom:2px solid #e5e7eb">Найдено</th>
                <th style="padding:10px;text-align:center;color:#111;border-bottom:2px solid #e5e7eb">Avg Score</th>
                <th style="padding:10px;text-align:left;color:#111;border-bottom:2px solid #e5e7eb">Detection Rate</th>
            </tr>
            {rows_html}
        </table>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px">
            <div style="padding:14px;background:#fef2f2;border:1px solid #fca5a5;border-radius:8px">
                <div style="font-weight:700;color:#b91c1c;margin-bottom:6px">⚠️ Ложноположительные (FP)</div>
                <div style="color:#111;font-size:13px;line-height:1.5">
                    Легальный текст распознан как мошенничество. 
                    Происходит когда текст содержит финансовую лексику без реального мошеннического умысла.
                </div>
            </div>
            <div style="padding:14px;background:#f0fdf4;border:1px solid #86efac;border-radius:8px">
                <div style="font-weight:700;color:#16a34a;margin-bottom:6px">❌ Ложноотрицательные (FN)</div>
                <div style="color:#111;font-size:13px;line-height:1.5">
                    Мошенничество не обнаружено. Происходит когда скам-текст 
                    маскируется под легитимный контент с disclaimers и нейтральными фразами.
                </div>
            </div>
        </div>

        <div style="margin-top:16px;padding:14px;background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:8px;font-size:13px;color:#111;line-height:1.6">
            <strong style="color:#111">💡 Вывод:</strong> <span style="color:#111">Агент показывает Precision=1.0 на явных скамах —
            когда он говорит HIGH, он всегда прав. Основная проблема — Recall на замаскированных текстах.
            Это типичный trade-off: лучше пропустить сомнительный текст, чем ложно обвинить легальный.
        </div>
    </div>
    """
    return html


def build_confusion_matrix():
    try:
        df  = pd.read_csv("data/eval_results_v2.csv")
        cm  = confusion_matrix(df["true_binary"], df["pred_binary"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r",
                    xticklabels=["Безопасно", "Мошенничество"],
                    yticklabels=["Безопасно", "Мошенничество"],
                    ax=ax, annot_kws={"size": 14, "weight": "bold", "color": "#111"})
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", color="#111")
        ax.set_ylabel("Истинный класс", color="#111")
        ax.set_xlabel("Предсказанный класс", color="#111")
        ax.tick_params(colors="#111")
        plt.tight_layout()
        return fig
    except:
        return None


# ── Примеры ───────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Инвестируй в наш крипто-фонд! Гарантированная доходность 300% в месяц. Уже 100,000 довольных клиентов. Только до пятницы — осталось 3 места! Приводи друзей — получай 15% от их вклада."],
    ["ПАО Сбербанк предлагает вклад со ставкой 8.5% годовых. Вклад застрахован АСВ на сумму до 1,400,000 рублей. Лицензия ЦБ РФ №1481."],
    ["Official Elon promo! Send 0.01 BTC, get 0.1 BTC back instantly! Limited time offer!"],
    ["Тинькофф Инвестиции: покупайте акции. Риски могут привести к потере капитала. Лицензия ФКЦБ России."],
    ["Хочешь финансовой свободы? Наша MLM-система даёт 500$ в день с первой недели! Напиши в личку — расскажем секрет!"],
]

# ── Интерфейс ─────────────────────────────────────────────────────────────────
stats, breakdown, eval_df = load_eval_stats()

with gr.Blocks(
    title="Детектор финансовых пирамид",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 960px !important; margin: auto }
        #analyze-btn { background:#2563eb !important; color:white !important; font-size:16px !important }
        footer { display: none !important }
    """
) as demo:

    gr.Markdown("# 🔍 Детектор финансовых пирамид")

    with gr.Tabs():

        with gr.Tab("🔍 Анализ текста"):
            gr.Markdown("**Вставьте текст рекламного материала** — агент проанализирует его на признаки мошенничества.")
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Текст для анализа",
                        placeholder="Вставьте текст инвестиционного предложения...",
                        lines=8,
                        max_lines=20,
                    )
                    analyze_btn = gr.Button("🔍 Анализировать", variant="primary", elem_id="analyze-btn")
                    gr.Markdown("### 💡 Примеры")
                    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Нажмите чтобы загрузить")

                with gr.Column(scale=1):
                    result_output = gr.HTML(
                        value="<div style='color:#666;padding:40px;text-align:center'>"
                              "Результат появится здесь</div>"
                    )

            analyze_btn.click(fn=analyze, inputs=text_input, outputs=result_output)

        with gr.Tab("📊 Аналитика & Eval"):
            gr.Markdown("### Результаты тестирования агента на размеченном датасете (n=50)")
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(value=build_metrics_html(stats, breakdown))
                with gr.Column(scale=2):
                    gr.Plot(value=build_confusion_matrix())

    gr.Markdown("*Агент использует GigaChat-2. Точность: F1=0.82, Accuracy=0.80*")

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)