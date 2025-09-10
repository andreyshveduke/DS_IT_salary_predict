import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

def prepare_input(user_input: dict, ohe, mlb):
    """
    user_input = {
        'position': str,
        'experience': str,
        'schedule': str,
        'employment': str,
        'vacancy': str,
        'level': str,
        'region': str,
        'skills': List[str]
    }
    """
    # Категориальные признаки — строго в порядке, как при обучении
    X_cat = ohe.transform([[
        user_input['position'],
        user_input['experience'],
        user_input['schedule'],
        user_input['employment'],
        user_input['vacancy'],
        user_input['level'],
        user_input['region']
    ]])

    # Skills
    all_skills_model = mlb.classes_  # важно сохранить порядок
    X_skills = [1 if skill in user_input['skills'] else 0 for skill in all_skills_model]
    X_skills = np.array([X_skills])

    # Объединение
    X_final = np.hstack([X_cat, X_skills])
    return X_final

# Загрузка моделей и энкодеров
rf_model_from = joblib.load('stage_3/joblib/rf_model_from.joblib')
rf_model_to = joblib.load('stage_3/joblib/rf_model_to.joblib')
ohe = joblib.load('stage_3/joblib/ohe_encoder.joblib')
mlb = joblib.load('stage_3/joblib/mlb_encoder.joblib')

# Пример запроса
user_input = {
    "position": "Тестировщик",
    "experience": "1–3 года",
    "schedule": "Полный день",
    "employment": "Полная занятость",
    "vacancy": "QA Engineer",
    "level": "Unknown",
    "region": "Москва",
    "skills": ["JAVA", "LINUX", "Python"]
}

X_pred = prepare_input(user_input, ohe, mlb)

salary_from = rf_model_from.predict(X_pred)[0]
salary_to = rf_model_to.predict(X_pred)[0]

print(f"Ожидаемая зарплата: {salary_from:.0f} — {salary_to:.0f} ₽")
