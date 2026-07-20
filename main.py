#!/usr/bin/env python3
"""
MTI Model: Точка входа с автокалибровкой при первом запуске
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Добавляем папку src в путь поиска модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем компоненты модели
from validation import ensure_calibrated, MTIValidator
from mti_model import MTIModel, format_forecast_text, RiskLevel, GlobalForecast
from data_loader import MTIDataLoader
from visualization import create_forecast_plot


def run_with_calibration(force_calibrate: bool = False):
    """
    Проверка и выполнение калибровки модели
    
    При первом запуске:
    1. Загружает исторические данные
    2. Подбирает оптимальные параметры
    3. Сравнивает с теоретическими
    4. Сохраняет калибровку
    
    Parameters:
        force_calibrate: Принудительная перекалибровка даже если уже была
        
    Returns:
        (путь_к_конфигу, была_ли_калибровка)
    """
    
    # Проверяем/выполняем калибровку
    effective_config = ensure_calibrated(
        config_path='config/config.json',
        force=force_calibrate
    )
    
    print(f"\n📋 Используется конфиг: {effective_config}")
    
    # Проверяем, создавался ли калиброванный конфиг
    is_calibrated = os.path.exists('config/config_calibrated.json')
    
    return effective_config, is_calibrated


class MTITelegramNotifier:
    """
    Класс для отправки уведомлений в Telegram
    
    Использует переменные окружения:
    - TELEGRAM_BOT_TOKEN: токен бота от @BotFather
    - TELEGRAM_CHAT_ID: ID чата для отправки
    
    Если переменные не заданы — отправка пропускается
    """
    
    def __init__(self):
        # Получаем настройки из переменных окружения
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    def is_configured(self) -> bool:
        """Проверка, настроен ли Telegram"""
        return bool(self.bot_token and self.chat_id)
    
    def send_forecast(self, text: str, forecast: GlobalForecast) -> None:
        """
        Отправка прогноза в Telegram
        
        Parameters:
            text: Полный текстовый отчёт (для архива)
            forecast: Объект прогноза для формирования короткого сообщения
        """
        # Проверяем, заданы ли настройки
        if not self.is_configured():
            print("\n⚠️ Telegram не настроен (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
            return  # Выходим, не отправляем ничего
        
        print(f"\n📤 Отправка в Telegram (chat: {self.chat_id})...")
        
        try:
            import requests
            
            # Формируем короткое сообщение для Telegram
            compact_text = self._format_telegram_message(forecast)
            
            # Telegram ограничивает длину сообщения 4096 символами
            # Разбиваем на части если нужно
            max_len = 4000
            parts = [compact_text[i:i+max_len] for i in range(0, len(compact_text), max_len)]
            
            # Отправляем каждую часть отдельно
            for i, part in enumerate(parts):
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": part,
                    "parse_mode": "Markdown",  # Разметка для жирного текста
                    "disable_web_page_preview": True  # Без превью ссылок
                }
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()  # Проверка на ошибки HTTP
                print(f"  Часть {i+1}/{len(parts)} отправлена")
            
            # Отправляем график (картинку)
            viz_files = [f for f in os.listdir('outputs') 
                        if f.startswith('mti_forecast_') and f.endswith('.png')]
            if viz_files:
                # Берём самый свежий файл по времени создания
                latest_viz = max(viz_files, key=lambda x: os.path.getctime(os.path.join('outputs', x)))
                viz_path = os.path.join('outputs', latest_viz)
                
                photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                with open(viz_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {
                        'chat_id': self.chat_id,
                        'caption': f'📊 MTI Forecast {datetime.now().strftime("%d.%m.%Y")}'
                    }
                    response = requests.post(photo_url, files=files, data=data, timeout=30)
                    response.raise_for_status()
                print("  График отправлен")
            
            print("✅ Telegram: отправлено успешно")
            
        except Exception as e:
            # Ловим любые ошибки, чтобы не сломать весь процесс
            print(f"❌ Ошибка отправки в Telegram: {e}")
            import traceback
            traceback.print_exc()  # Подробности ошибки в лог
    
    def _format_telegram_message(self, forecast: GlobalForecast) -> str:
        """
        Форматирование компактного сообщения для Telegram
        
        В отличие от полного отчёта — только ключевая информация:
        - Уровень риска
        - Вероятность
        - Топ-10 регионов
        - Окно риска
        
        Parameters:
            forecast: Глобальный прогноз
            
        Returns:
            Текст сообщения в Markdown
        """
        # Словарь иконок для уровней риска
        risk_icons = {
            RiskLevel.CRITICAL: "🔴 КРИТИЧЕСКИЙ",
            RiskLevel.HIGH: "🟠 ВЫСОКИЙ",
            RiskLevel.ELEVATED: "🟡 ПОВЫШЕННЫЙ",
            RiskLevel.NORMAL: "🟢 НОРМА"
        }
        
        # Формируем сообщение построчно
        lines = [
            f"🌍 *MTI Model: Глобальный прогноз*",
            f"",
            f"📅 {datetime.now().strftime('%d.%m.%Y')}",
            f"",
            f"Уровень риска: *{risk_icons.get(forecast.dominant_risk_level, 'Н/Д')}*",
            f"Макс. вероятность: *{forecast.max_probability:.1%}*",
        ]
        
        # Добавляем окно риска если есть
        if forecast.risk_window_start and forecast.risk_window_end:
            lines.extend([
                f"",
                f"⏰ *Окно риска:*",
                f"  {forecast.risk_window_start.strftime('%d.%m.%Y')} – {forecast.risk_window_end.strftime('%d.%m.%Y')}"
            ])
        
        # Добавляем регионы с высоким риском
        if forecast.active_regions:
            lines.extend([
                f"",
                f"🔥 *Регионы с высоким риском ({len(forecast.active_regions)}):*",
                f""
            ])
            
            # Только топ-10, чтобы не превысить лимит
            for i, region in enumerate(forecast.active_regions[:10], 1):
                icon = "🔴" if region.risk_level == RiskLevel.CRITICAL else "🟠"
                lines.append(
                    f"{icon} *{region.name}*: {region.probability:.1%}"
                )
                # Добавляем окно риска региона если есть
                if region.risk_window_start:
                    lines.append(
                        f"   📅 {region.risk_window_start.strftime('%m.%Y')}–{region.risk_window_end.strftime('%m.%Y')}"
                    )
        
        # Футер с ссылкой на репозиторий
        lines.extend([
            f"",
            f"📊 [Полный отчёт](https://github.com/{os.getenv('GITHUB_REPOSITORY', '')})",
            f"",
            f"_МТИ-модель | Автоматический прогноз_"
        ])
        
        return "\n".join(lines)


def run_forecast(ssn=None, dssn_dt=None, date=None, 
                 output_dir='outputs',
                 force_calibrate: bool = False):
    """
    Основная функция прогнозирования
    
    Порядок работы:
    1. Калибровка (при первом запуске)
    2. Загрузка текущих данных SSN
    3. Расчёт производной dSSN/dt
    4. Определение фазы солнечного цикла
    5. Активация триггера (если спад при SSN>80)
    6. Расчёт риска по регионам
    7. Байесовская оценка
    8. Формирование отчёта
    9. Отправка в Telegram
    
    Parameters:
        ssn: Ручное задание SSN (None = авто)
        dssn_dt: Ручное задание производной (None = авто)
        date: Дата прогноза (None = сегодня)
        output_dir: Папка для результатов
        force_calibrate: Принудительная перекалибровка
    """
    
    # === ШАГ 1: Калибровка ===
    config_path, is_calibrated = run_with_calibration(force_calibrate)
    
    # Создаём папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # === ШАГ 2: Создание модели ===
    model = MTIModel(config_path)
    loader = MTIDataLoader('data')
    
    # === ШАГ 3: Получение данных ===
    if ssn is None or dssn_dt is None:
        print("\n📡 Загрузка текущих солнечных данных...")
        loader.fetch_ssn()
        loaded_date, loaded_ssn, loaded_dssn = loader.get_current_ssn()
        ssn = ssn or loaded_ssn
        dssn_dt = dssn_dt or loaded_dssn
        date = date or loaded_date
    
    date = date or datetime.now()
    
    # Маркер калибровки для отображения
    calib_marker = " [КАЛИБРОВАННАЯ]" if is_calibrated else " [ТЕОРЕТИЧЕСКАЯ]"
    
    # === ШАГ 4: Вывод заголовка ===
    print(f"\n{'='*70}")
    print(f"🌍 МТИ-МОДЕЛЬ{calib_marker}: Глобальный прогноз")
    print(f"{'='*70}")
    print(f"Дата: {date.strftime('%Y-%m-%d')}")
    print(f"SSN: {ssn:.1f}, dSSN/dt: {dssn_dt:+.1f}")
    print(f"{'='*70}")
    
    # === ШАГ 5: Расчёт прогноза ===
    forecast = model.forecast_global(date, ssn, dssn_dt)
    
    # === ШАГ 6: Формирование текстового отчёта ===
    text = format_forecast_text(forecast, date, ssn, dssn_dt)
    
    # Добавляем информацию о калибровке
    if is_calibrated:
        text += f"\n\n[Модель откалибрована: {config_path}]"
    else:
        text += "\n\n[Модель использует теоретические параметры]"
    
    print(text)
    
    # === ШАГ 7: Сохранение файлов ===
    
    # Текстовый отчёт
    report_file = os.path.join(output_dir, f'mti_report_{date.strftime("%Y%m%d")}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n💾 Отчёт: {report_file}")
    
    # Визуализация
    import pandas as pd
    ssn_df = loader.ssn_data
    quake_df = loader.quake_data
    
    # Загружаем землетрясения если ещё не загружены
    if quake_df is None:
        try:
            quake_df = loader.fetch_earthquakes(min_magnitude=7.5)
        except:
            quake_df = pd.DataFrame()  # Пустой DataFrame при ошибке
    
    viz_path = os.path.join(output_dir, f'mti_forecast_{date.strftime("%Y%m%d")}.png')
    create_forecast_plot(model, forecast, date, ssn, dssn_dt, ssn_df, quake_df, viz_path)
    print(f"💾 Визуализация: {viz_path}")
    
    # JSON для машинной обработки
    json_output = {
        'date': date.isoformat(),
        'ssn': ssn,
        'dssn_dt': dssn_dt,
        'risk_level': forecast.dominant_risk_level.value,
        'max_probability': forecast.max_probability,
        'is_calibrated': is_calibrated,
        'config_used': config_path,
        'active_regions': [
            {
                'name': r.name,
                'probability': r.probability,
                'risk_level': r.risk_level.value,
                'window_start': r.risk_window_start.isoformat() if r.risk_window_start else None,
                'window_end': r.risk_window_end.isoformat() if r.risk_window_end else None,
            }
            for r in forecast.active_regions
        ]
    }
    
    json_file = os.path.join(output_dir, f'mti_forecast_{date.strftime("%Y%m%d")}.json')
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"💾 JSON: {json_file}")
    
    # === ШАГ 8: Отправка в Telegram ===
    notifier = MTITelegramNotifier()  # Создаём отправщик
    notifier.send_forecast(text, forecast)  # Отправляем
    
    return forecast


def run_retrospective(start_year=1950, end_year=2024, 
                      output_dir='outputs',
                      force_calibrate: bool = False):
    """
    Ретроспективное тестирование модели
    
    Проверяет модель на исторических данных:
    - Загружает SSN за 1950-2024
    - Загружает землетрясения M≥7.5
    - Для каждого месяца проверяет триггер
    - Считает метрики: precision, recall, F1
    - Проверяет статистическую значимость
    
    Parameters:
        start_year: Начало периода теста
        end_year: Конец периода теста
        output_dir: Папка для результатов
        force_calibrate: Перекалибровка перед тестом
    """
    
    # Калибровка если нужна
    config_path, is_calibrated = run_with_calibration(force_calibrate)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = MTIModel(config_path)
    loader = MTIDataLoader('data')
    
    calib_marker = " [КАЛИБРОВАННАЯ]" if is_calibrated else " [ТЕОРЕТИЧЕСКАЯ]"
    
    print(f"\n{'='*70}")
    print(f"📊 МТИ-МОДЕЛЬ{calib_marker}: Ретроспективный тест")
    print(f"{'='*70}")
    print(f"Период: {start_year}-{end_year}")
    
    # Загрузка данных
    ssn_df = loader.fetch_ssn()
    quake_df = loader.fetch_earthquakes(
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 12, 31),
        min_magnitude=model.model_cfg['min_magnitude']
    )
    
    # Запуск теста
    results = model.retrospective_test(
        ssn_df, quake_df,
        datetime(start_year, 1, 1),
        datetime(end_year, 12, 31)
    )
    
    # === Вывод результатов ===
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ:")
    print(f"  Месяцев проанализировано: {results['total_months']}")
    print(f"  True Positives  (триггер сработал, событие было):     {results['true_positives']}")
    print(f"  False Positives (триггер сработал, события не было):  {results['false_positives']}")
    print(f"  True Negatives  (триггер не сработал, события не было): {results['true_negatives']}")
    print(f"  False Negatives (триггер не сработал, событие было): {results['false_negatives']}")
    print(f"  Precision (точность):   {results['precision']:.3f} ({results['precision']*100:.1f}%)")
    print(f"  Recall    (полнота):    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
    print(f"  F1-score:               {results['f1_score']:.3f}")
    
    # Статистическая значимость
    total_events = results['true_positives'] + results['false_negatives']
    if total_events > 0:
        p_value = model.binomial_test(results['true_positives'], total_events)
        print(f"  P-value (биномиальный тест): {p_value:.6f}")
        print(f"  Статистически значимо: {'✅ ДА' if p_value < 0.05 else '❌ НЕТ'} (α=0.05)")
    
    # === Проверка порогов модели ===
    print(f"\n{'='*70}")
    print("ПРОВЕРКА ПОРОГОВ МОДЕЛИ:")
    checks = [
        ('Precision ≥ 60%', results['precision'] >= 0.60),
        ('Recall ≥ 70%', results['recall'] >= 0.70),
        ('F1 ≥ 65%', results['f1_score'] >= 0.65),
        ('P-value < 0.05', p_value < 0.05 if total_events > 0 else False),
    ]
    for name, passed in checks:
        status = '✅ ПРОЙДЕН' if passed else '❌ НЕ ПРОЙДЕН'
        print(f"  {status} {name}")
    
    # Сохранение результатов
    results['start_year'] = start_year
    results['end_year'] = end_year
    results['is_calibrated'] = is_calibrated
    results['config_used'] = config_path
    
    results_file = os.path.join(output_dir, f'retроспектива_{start_year}_{end_year}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Результаты сохранены: {results_file}")
    
    return results


def reset_calibration():
    """
    Сброс калибровки
    
    Удаляет все файлы калибровки, чтобы при следующем запуске
    выполнилась первичная калибровка заново
    
    Полезно для:
    - Тестирования процесса калибровки
    - Смены периода калибровки
    - Исправления ошибок
    """
    files_to_remove = [
        'data/.calibration_done',           # Флаг выполнения
        'config/config_calibrated.json',     # Калиброванный конфиг
        'config/config_calibrated_report.json'  # Отчёт о калибровке
    ]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ Удалено: {f}")
    
    print("✅ Калибровка сброшена. При следующем запуске будет выполнена заново.")


def main():
    """
    Главная функция — обработка аргументов командной строки
    
    Поддерживаемые команды:
    - forecast: Прогноз на текущую дату
    - retrospective: Ретроспективный тест
    - test: Тестовый сценарий (февраль 2026)
    - validate: Только калибровка
    - reset-calibration: Сброс калибровки
    """
    parser = argparse.ArgumentParser(description='MTI Seismic Model with Calibration')
    
    # Подкоманды
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # === Прогноз ===
    forecast_parser = subparsers.add_parser('forecast', help='Запустить прогноз')
    forecast_parser.add_argument('--ssn', type=float, help='Ручное задание SSN')
    forecast_parser.add_argument('--dssn', type=float, help='Ручное задание dSSN/dt')
    forecast_parser.add_argument('--date', help='Дата прогноза (YYYY-MM-DD)')
    forecast_parser.add_argument('--output-dir', default='outputs', help='Папка для результатов')
    forecast_parser.add_argument('--force-calibrate', action='store_true',
                                help='Принудительная перекалибровка перед прогнозом')
    
    # === Ретроспектива ===
    retro_parser = subparsers.add_parser('retrospective', help='Ретроспективный тест')
    retro_parser.add_argument('--start-year', type=int, default=1950, help='Начало периода')
    retro_parser.add_argument('--end-year', type=int, default=2024, help='Конец периода')
    retro_parser.add_argument('--output-dir', default='outputs', help='Папка для результатов')
    retro_parser.add_argument('--force-calibrate', action='store_true',
                             help='Перекалибровка перед тестом')
    
    # === Тест ===
    test_parser = subparsers.add_parser('test', help='Тестовый сценарий (февраль 2026)')
    test_parser.add_argument('--output-dir', default='outputs', help='Папка для результатов')
    test_parser.add_argument('--force-calibrate', action='store_true')
    
    # === Валидация ===
    validate_parser = subparsers.add_parser('validate', help='Только калибровка без прогноза')
    validate_parser.add_argument('--force', action='store_true', help='Принудительная калибровка')
    
    # === Сброс ===
    subparsers.add_parser('reset-calibration', help='Сбросить калибровку')
    
    # Разбор аргументов
    args = parser.parse_args()
    
    # === Выполнение команды ===
    if args.command == 'forecast':
        # Прогноз
        date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else None
        run_forecast(args.ssn, args.dssn, date, args.output_dir, args.force_calibrate)
    
    elif args.command == 'retrospective':
        # Ретроспектива
        run_retrospective(args.start_year, args.end_year, args.output_dir, args.force_calibrate)
    
    elif args.command == 'test':
        # Тестовый сценарий
        print("🧪 Тестовый режим: Февраль 2026")
        print("   SSN=133, dSSN/dt=-23 (крутой спад)")
        run_forecast(133.0, -23.0, datetime(2026, 2, 1), args.output_dir, args.force_calibrate)
    
    elif args.command == 'validate':
        # Только калибровка
        ensure_calibrated(force=args.force)
    
    elif args.command == 'reset-calibration':
        # Сброс
        reset_calibration()
    
    else:
        # Нет команды — показать справку
        parser.print_help()


# Точка входа
if __name__ == '__main__':
    main()
