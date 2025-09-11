import requests
import json
import pandas as pd
import os

API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = ""
MODEL_NAME = "mistral-small-latest"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

INSTRUCTION = """
Ты — помощник по разметке резюме (CV). Извлеки структурированную информацию из текста резюме.

Верни ответ строго в формате JSON. Не добавляй пояснений, комментариев или текста вокруг.

Если какое-либо поле отсутствует в тексте — верни его как null. Если поле предполагает список (например, skills или languages), но ничего не указано — верни пустой список.

Извлеки следующие поля:
- full_name
- position
- salary_from
- salary_to
- currency
- city
- country
- email
- phone
- age
- gender
- citizenship
- relocation
- travel_ready
- work_schedule
- employment
- education_level
- education: [{institution, faculty, degree, graduation_year}]
- experience_years
- experience: [{company, position, start_date, end_date, description}]
- languages: [{language, level}]
- skills
- certifications
- achievements
- summary
- portfolio_links
- linkedin
- github
- other_links
"""

def prepare_prompt(cv_text: str) -> list:
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"Текст резюме:\n{cv_text}"}
    ]

def query_mistral(prompt: list):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={
            "model": MODEL_NAME,
            "messages": prompt,
            "temperature": 0.2,
            "max_tokens": 2500
        },
        timeout=120
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()

    if content.lower() == "null":
        return None

    return json.loads(content)

def process_single_resume(input_path, output_path, text_column="CV_Text"):
    # Определим формат
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Поддерживаются только CSV и Excel файлы")

    if text_column not in df.columns:
        raise ValueError(f"В файле нет колонки '{text_column}'")

    cv_text = df[text_column].iloc[0]  # берём первую строку
    prompt = prepare_prompt(cv_text)
    result = query_mistral(prompt)

    if result is None:
        print("Модель вернула null")
        result = {}

    # Сохраняем результат
    result_df = pd.DataFrame([result])
    if output_path.endswith(".xlsx"):
        result_df.to_excel(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False)

    print(f"✅ Разметка завершена. Результат сохранён в: {output_path}")