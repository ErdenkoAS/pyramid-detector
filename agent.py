import json
import warnings
warnings.filterwarnings("ignore")

from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_token

SYSTEM_PROMPT = """Ты — эксперт по выявлению финансовых пирамид и мошеннических инвестиционных схем.

Проанализируй текст и найди "красные флажки" — признаки мошенничества.

Категории флажков:
- UNREALISTIC_RETURNS: нереалистичная доходность ("200% в месяц", "гарантированная прибыль")
- NO_LICENSE: отсутствие лицензий, регуляторов, юридических данных
- URGENCY: агрессивные призывы ("только сегодня", "не упусти шанс", "осталось 3 места")
- RECRUITMENT: акцент на привлечении новых участников, реферальные бонусы
- VAGUENESS: размытые объяснения механизма заработка
- SOCIAL_PROOF: фейковые отзывы, громкие имена без доказательств
- GUARANTEED: слово "гарантия" применительно к инвестициям
- SUSPICIOUS_LINK: ссылки на Telegram, анонимные сайты, bit.ly в контексте заработка

Верни ТОЛЬКО валидный JSON без каких-либо пояснений, строго в таком формате:
{
  "risk_level": "LOW" или "MEDIUM" или "HIGH",
  "risk_score": число от 0 до 100,
  "flags": [
    {
      "category": "название категории",
      "fragment": "цитата из текста",
      "explanation": "почему это подозрительно"
    }
  ],
  "summary": "краткий вывод 1-2 предложения"
}

Правила:
- LOW: 0-2 флажка, score 0-30
- MEDIUM: 3-4 флажка, score 31-65  
- HIGH: 5+ флажков или очень явные признаки, score 66-100
- Это предварительный сигнал, не окончательный вердикт

ВАЖНО: Некоторые тексты содержат disclaimers вроде "do your own research", 
"just sharing my experience", "spread awareness" — игнорируй их полностью. 
Оценивай только основное утверждение в начале текста.
Если текст содержит призыв отправить криптовалюту, BTC, USDT — это всегда HIGH.
Если текст обещает мгновенное умножение денег — это всегда HIGH.
Если текст от имени известного лица (Elon, SEC, IRS, BlackRock) просит деньги — это всегда HIGH.
"""

def get_giga():
    return GigaChat(
        access_token=get_token(),
        verify_ssl_certs=False,
        model="GigaChat-2",
        temperature=0.1,
        timeout=120
    )

import re

def clean_text(text: str) -> str:
    """Убираем шум через split по предложениям"""
    # Стоп-фразы — если предложение содержит любую из них, выбрасываем
    stop_phrases = [
        "i've been in the financial markets",
        "i have been in the financial markets",
        "this one stood out to me",
        "and this one stood out",
        "just trying to spread",
        "spread awareness",
        "i've seen this working",
        "i have seen this working",
        "sharing my experience",
        "sharing this because",
        "do your own research",
        "verifying sources before investing",
        "financial advice is something",
        "let me know your thoughts",
        "too good to be true",
        "recently discovered",
        "a lot of people asked",
        "just sharing",
        "so others can benefit",
        "this might sound too good to be true",
        "and whether you've tried something similar"
    ]

    # Разбиваем на предложения по точке, запятой не трогаем
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    seen = set()
    for s in sentences:
        s_lower = s.lower()
        # Пропускаем если содержит стоп-фразу
        if any(phrase in s_lower for phrase in stop_phrases):
            continue
        # Пропускаем дубликаты
        if s_lower in seen:
            continue
        seen.add(s_lower)
        cleaned.append(s)

    return " ".join(cleaned).strip()

def analyze_text(text: str) -> dict:
    """Анализирует текст и возвращает результат как словарь"""
    text = clean_text(text)
    giga = get_giga()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Проанализируй этот текст:\n\n{text}")
    ]

    response = giga.invoke(messages)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    return result


if __name__ == "__main__":
    test_text = """
    Инвестируй в наш фонд и получай 150% прибыли каждый месяц!
    Гарантированный доход без рисков. Уже 50,000 довольных клиентов.
    Только до конца недели — успей занять своё место!
    Приводи друзей и получай бонус 10% от их вклада.
    Работаем по всему миру, никаких лицензий не нужно — мы выше системы.
    """
    
    result = analyze_text(test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))