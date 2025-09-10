from pprint import pprint

import requests
import pandas as pd


def get_crb_rates():
    try:
        response = requests.get('https://www.cbr-xml-daily.ru/daily_json.js')
        if response.status_code == 200:
            data = response.json()
            return data['Valute']
        else:
            print('Ошибка получения данных')
            return None
    except Exception as e:
        print(f'Ошибка: {e}')
        return None

exchange_rates = get_crb_rates()

def convert_to_rub(row):
    if row['currency'] == 'BYR':
        row['currency'] = 'BYN'

    if row['currency'] =='RUB' or row['currency'] == 'RUR' or pd.isna(row['currency']):
        row['currency'] = 'RUB'
        return row

    currency = row['currency'].upper()
    if currency not in exchange_rates:
        print('Неверная валюта')
        return row

    rate = exchange_rates[currency]['Value'] / exchange_rates[currency]['Nominal']

    if not pd.isna(row['salary from']):
        row['salary from'] = round(row['salary from'] * rate)
    if not pd.isna(row['salary to']):
        row['salary to'] = round(row['salary to'] * rate)

    row['currency'] = 'RUB'
    return row

