# =============================
# Пример вызова для backend
# =============================


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from catboost import CatBoostRegressor

def prepare_input(user_input: dict, ohe, mlb):
    """
    Подготовка входных данных для модели.

    user_input = {
    'position': str,
    'experience': str, # мапится в experience_ord
    'schedule': str,
    'employment': str,
    'level': str, # мапится в level_ord
    'region': str,
    'skills': List[str]
    }
    """
    # Маппинг категорий experience
    experience_order = {"Нет опыта": 0, "1–3 года": 1, "3–6 лет": 2, "Более 6 лет": 3} # , "Unknown": -1
    level_order = {"Trainee": 0, "Junior": 1, "Middle": 2, "Senior": 3, "Lead": 4} # , "Chief": 5, "Unknown": -1


    X_cat = ohe.transform([[user_input['position'], user_input['schedule'], user_input['employment'], user_input['region']]])


    # Порядковые признаки
    X_ord = np.array([[experience_order.get(user_input['experience'], -1),
    level_order.get(user_input['level'], -1)]])


    # Skills
    all_skills_model = mlb.classes_
    X_skills = [1 if skill in user_input['skills'] else 0 for skill in all_skills_model]
    X_skills = np.array([X_skills])


    # Объединение
    X_final = np.hstack([X_cat, X_ord, X_skills])
    return X_final


# Загрузка моделей и энкодеров
model_from = joblib.load('stage_6/joblib/model_from.joblib')
model_to = joblib.load('stage_6/joblib/model_to.joblib')
ohe = joblib.load('stage_6/joblib/ohe_encoder.joblib')
mlb = joblib.load('stage_6/joblib/mlb_encoder.joblib')


# Пример запроса
user_input = {
"position": "Системный администратор",
"experience": "1–3 года",
"schedule": "Полный день",
"employment": "Полная занятость",
"level": "Junior",
"region": "Москва",
"skills": ["MPLS", "Microsoft Windows", "Zabbix"]
}


X_pred = prepare_input(user_input, ohe, mlb)


salary_from = model_from.predict(X_pred)[0]
salary_to = model_to.predict(X_pred)[0]


print(f"Ожидаемая зарплата: {salary_from:.0f} — {salary_to:.0f} ₽")


# Примечание:
# - Списки категорий для фронта берутся из filter_options.json
# - Маппинг experience и level встроен в prepare_input