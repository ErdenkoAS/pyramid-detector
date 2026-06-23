# 🔍 Детектор финансовых пирамид

LLM-агент на основе GigaChat-2 для выявления признаков мошенничества в инвестиционных текстах.

## Описание

Агент анализирует текст рекламных материалов и выявляет «красные флажки»:
- 💰 Нереалистичная доходность
- 📋 Отсутствие лицензий
- ⏰ Искусственная срочность
- 👥 Рекрутинг участников
- 🔒 Гарантия прибыли
- 👤 Фейковые отзывы
- 🔗 Подозрительные ссылки

Проект относит текст к одной из категорий риска: **LOW / MEDIUM / HIGH**.

## Результаты тестирования (n=50)

| Метрика   | Значение |
|-----------|----------|
| Accuracy  | 0.800    |
| Precision | 0.885    |
| Recall    | 0.767    |
| F1-score  | 0.821    |

Датасет: [Financial Scam Posts](https://www.kaggle.com/datasets/rmg9725/financial-scam-posts) (10.5k записей, 5 категорий).

## Стек

- **LLM**: GigaChat-2 (Sber) через LangChain
- **UI**: Gradio
- **Eval**: scikit-learn, pandas, matplotlib

## Установка

```bash
conda create -n pyramid_detector python=3.11 -y
conda activate pyramid_detector
pip install -r requirements.txt
```

Создай файл `config.py` и добавь свой GigaChat токен:

```python
GIGACHAT_AUTH = "Basic ВАШ_КЛЮЧ"
```

## Запуск

```bash
# Веб-интерфейс
python app.py

# Eval на датасете
python eval.py

# Тест агента
python agent.py
```

## Структура проекта

```
pyramid_detector/
├── agent.py          # LLM-агент и промпт
├── app.py            # Gradio веб-интерфейс
├── eval.py           # Оценка качества
├── config.py         # Подключение к LLM (не коммитить!)
├── requirements.txt
├── README.md
└── data/
    ├── financial_moderation_dataset_10.5k_with_normals.csv
    ├── eval_results_v2.csv
    └── confusion_matrix_v2.png
```


## ⚠️ Дисклеймер

Это предварительный сигнал, а не окончательный вердикт.
Для принятия финансовых решений обращайтесь к лицензированным специалистам.

---
*Проект выполнен в рамках курса «Прикладной ИИ», ВШЭ, 2026*
