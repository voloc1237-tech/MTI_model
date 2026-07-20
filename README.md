# MTI Model: Magneto-Tectonic Inertia Hypothesis

[![Daily Forecast](https://github.com/YOUR_USERNAME/mti-model/actions/workflows/daily_forecast.yml/badge.svg)](https://github.com/YOUR_USERNAME/mti-model/actions/workflows/daily_forecast.yml)
[![Tests](https://github.com/YOUR_USERNAME/mti-model/actions/workflows/retrospective_test.yml/badge.svg)](https://github.com/YOUR_USERNAME/mti-model/actions/workflows/retrospective_test.yml)

## Описание

Формальная реализация МТИ-модели (Magneto-Tectonic Inertia) для независимой проверки гипотезы о связи солнечной активности и крупных землетрясений (M≥7.5).

## Ключевые особенности

- **Предиктор**: Первая производная SSN (dSSN/dt), не абсолютное значение
- **Триггер**: Спад SSN при SSN > 80
- **Лаг-окно**: 1-12 месяцев после активации триггера
- **Байесовская оценка**: Интеграция региональных коэффициентов
- **Глобальное покрытие**: 30 сейсмических регионов

## Быстрый старт

```bash
# Клонирование
git clone https://github.com/YOUR_USERNAME/mti-model.git
cd mti-model

# Установка
pip install -r requirements.txt

# Текущий прогноз
python main.py forecast

# Ретроспективный тест (1950-2024)
python main.py retrospective

# Тестовый сценарий (февраль 2026)
python main.py test
