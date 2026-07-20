"""
Валидационный модуль МТИ-модели
Однократная калибровка при первом запуске + проверка соответствия теории
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mti_model import MTIModel


@dataclass
class CalibrationResult:
    """Результат калибровки параметра"""
    param_name: str
    theoretical_value: float
    optimal_value: float
    deviation_percent: float
    f1_theoretical: float
    f1_optimal: float
    recommendation: str  # 'ACCEPT', 'CALIBRATE', 'REVIEW'


@dataclass
class ValidationReport:
    """Полный отчёт валидации"""
    calibration_date: str
    theoretical_params: Dict
    optimal_params: Dict
    results: List[CalibrationResult]
    overall_deviation: float
    final_recommendation: str
    is_calibrated: bool


class MTIValidator:
    """
    Валидатор и калибровщик МТИ-модели
    
    При первом запуске:
    1. Загружает теоретические параметры из config
    2. Подбирает оптимальные на исторических данных
    3. Сравнивает отклонения
    4. Если отклонение < 15% — принимает теоретические
    5. Если 15-30% — калибрует с предупреждением
    6. Если > 30% — требует ручного ревью
    """
    
    # Пороги отклонения
    DEVIATION_ACCEPT = 0.15      # 15% — принять теорию
    DEVIATION_CALIBRATE = 0.30   # 30% — калибровать автоматически
    # > 30% — требовать ручного решения
    
    CALIBRATION_FILE = 'data/.calibration_done'
    CALIBRATION_CONFIG = 'config/config_calibrated.json'
    
    def __init__(self, base_config_path: str = 'config/config.json'):
        self.base_config_path = base_config_path
        self.model = MTIModel(base_config_path)
        
        # Параметры для калибровки
        self.calibratable_params = [
            'ssn_threshold',
            'decline_threshold', 
            'peak_plateau_threshold',
            'prior_probability'
        ]
        
        # Диапазоны поиска
        self.param_ranges = {
            'ssn_threshold': range(50, 120, 5),
            'decline_threshold': range(-15, -1, 1),
            'peak_plateau_threshold': range(2, 12, 1),
            'prior_probability': np.arange(0.30, 0.80, 0.02)
        }
    
    def is_calibrated(self) -> bool:
        """Проверка, была ли уже выполнена калибровка"""
        return os.path.exists(self.CALIBRATION_FILE)
    
    def get_effective_config_path(self) -> str:
        """Возвращает путь к актуальному конфигу"""
        if os.path.exists(self.CALIBRATION_CONFIG):
            return self.CALIBRATION_CONFIG
        return self.base_config_path
    
    def run_retrospective_for_params(self, 
                                      ssn_df: pd.DataFrame,
                                      quake_df: pd.DataFrame,
                                      test_params: Dict) -> Dict:
        """
        Запуск ретроспективного теста с заданными параметрами
        
        Returns:
            {'precision': float, 'recall': float, 'f1_score': float, ...}
        """
        # Временно подменяем параметры
        original_config = self.model.model_cfg.copy()
        
        for key, value in test_params.items():
            if key in self.model.model_cfg:
                self.model.model_cfg[key] = value
        
        try:
            from data_loader import MTIDataLoader
            loader = MTIDataLoader()
            
            start_date = pd.Timestamp('1950-01-01')
            end_date = pd.Timestamp('2024-12-31')
            
            results = self.model.retrospective_test(ssn_df, quake_df, start_date, end_date)
            return results
            
        finally:
            # Восстанавливаем оригинальные параметры
            self.model.model_cfg = original_config
    
    def find_optimal_param(self,
                          ssn_df: pd.DataFrame,
                          quake_df: pd.DataFrame,
                          param_name: str,
                          param_range) -> Tuple[float, float, Dict]:
        """
        Поиск оптимального значения одного параметра
        
        Returns:
            (optimal_value, best_f1, all_results)
        """
        print(f"  Калибровка {param_name}...")
        
        theoretical_value = self.model.model_cfg.get(param_name)
        best_f1 = 0
        optimal_value = theoretical_value
        all_results = {}
        
        for test_value in param_range:
            test_params = {param_name: test_value}
            results = self.run_retrospective_for_params(ssn_df, quake_df, test_params)
            
            f1 = results.get('f1_score', 0)
            all_results[test_value] = {
                'f1': f1,
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0)
            }
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_value = test_value
        
        return optimal_value, best_f1, all_results
    
    def validate_and_calibrate(self,
                                ssn_df: pd.DataFrame,
                                quake_df: pd.DataFrame,
                                force: bool = False) -> ValidationReport:
        """
        Основной метод: валидация и однократная калибровка
        
        Parameters:
            ssn_df: данные SSN
            quake_df: данные землетрясений
            force: принудительная перекалибровка даже если уже была
        
        Returns:
            ValidationReport
        """
        
        # Проверяем, не калибровали ли уже
        if self.is_calibrated() and not force:
            print("✅ Калибровка уже выполнена ранее. Используется сохранённый конфиг.")
            print(f"   Конфиг: {self.CALIBRATION_CONFIG}")
            
            # Загружаем отчёт
            report_path = self.CALIBRATION_CONFIG.replace('.json', '_report.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    data = json.load(f)
                return ValidationReport(**data)
            
            return None
        
        print("=" * 70)
        print("🔬 МТИ-МОДЕЛЬ: ПЕРВИЧНАЯ ВАЛИДАЦИЯ И КАЛИБРОВКА")
        print("=" * 70)
        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Период данных: {ssn_df['Date'].min().year}-{ssn_df['Date'].max().year}")
        print(f"Землетрясений M≥7.5: {len(quake_df)}")
        print()
        
        # Базовый тест с теоретическими параметрами
        print("📊 Шаг 1: Тест с теоретическими параметрами...")
        theoretical_results = self.run_retrospective_for_params(
            ssn_df, quake_df, {}
        )
        f1_theoretical = theoretical_results['f1_score']
        print(f"   F1 (теория): {f1_theoretical:.3f}")
        print(f"   Precision:   {theoretical_results['precision']:.3f}")
        print(f"   Recall:      {theoretical_results['recall']:.3f}")
        print()
        
        # Калибровка каждого параметра
        calibration_results = []
        optimal_params = self.model.model_cfg.copy()
        
        print("🔧 Шаг 2: Поиск оптимальных параметров...")
        
        for param_name in self.calibratable_params:
            param_range = self.param_ranges.get(param_name)
            if param_range is None:
                continue
            
            theoretical = self.model.model_cfg.get(param_name)
            optimal, f1_opt, _ = self.find_optimal_param(
                ssn_df, quake_df, param_name, param_range
            )
            
            # Расчёт отклонения
            if theoretical != 0:
                deviation = abs(optimal - theoretical) / abs(theoretical)
            else:
                deviation = abs(optimal - theoretical)
            
            # Рекомендация
            if deviation <= self.DEVIATION_ACCEPT:
                recommendation = 'ACCEPT'
                final_value = theoretical  # Принимаем теорию
            elif deviation <= self.DEVIATION_CALIBRATE:
                recommendation = 'CALIBRATE'
                final_value = optimal      # Калибруем
            else:
                recommendation = 'REVIEW'
                final_value = theoretical  # По умолчанию теория, но требуем внимания
            
            result = CalibrationResult(
                param_name=param_name,
                theoretical_value=theoretical,
                optimal_value=optimal,
                deviation_percent=deviation * 100,
                f1_theoretical=f1_theoretical,
                f1_optimal=f1_opt,
                recommendation=recommendation
            )
            calibration_results.append(result)
            
            # Применяем финальное значение
            optimal_params[param_name] = final_value
            
            print(f"   {param_name}:")
            print(f"      Теория: {theoretical} | Оптимум: {optimal} | Отклонение: {deviation*100:.1f}%")
            print(f"      Решение: {recommendation} | Используем: {final_value}")
            print()
        
        # Финальный тест с выбранными параметрами
        print("📊 Шаг 3: Финальная проверка...")
        final_results = self.run_retrospective_for_params(
            ssn_df, quake_df, 
            {k: optimal_params[k] for k in self.calibratable_params}
        )
        f1_final = final_results['f1_score']
        print(f"   F1 (финал): {f1_final:.3f}")
        print(f"   Precision:  {final_results['precision']:.3f}")
        print(f"   Recall:     {final_results['recall']:.3f}")
        print()
        
        # Расчёт общего отклонения
        deviations = [r.deviation_percent / 100 for r in calibration_results]
        overall_deviation = np.mean(deviations)
        
        # Финальная рекомендация
        if overall_deviation <= self.DEVIATION_ACCEPT:
            final_recommendation = "ТЕОРЕТИЧЕСКИЕ ПАРАМЕТРЫ ПРИНЯТЫ"
        elif overall_deviation <= self.DEVIATION_CALIBRATE:
            final_recommendation = "ПРОИЗВЕДЕНА АВТОМАТИЧЕСКАЯ КАЛИБРОВКА"
        else:
            final_recommendation = "ТРЕБУЕТСЯ РУЧНОЙ АНАЛИЗ"
        
        # Создание отчёта
        report = ValidationReport(
            calibration_date=datetime.now().isoformat(),
            theoretical_params={
                k: self.model.model_cfg[k] 
                for k in self.calibratable_params
            },
            optimal_params={
                k: optimal_params[k] 
                for k in self.calibratable_params
            },
            results=calibration_results,
            overall_deviation=overall_deviation,
            final_recommendation=final_recommendation,
            is_calibrated=True
        )
        
        # Сохранение
        self._save_calibration(report, optimal_params)
        
        # Вывод отчёта
        self._print_report(report)
        
        return report
    
    def _save_calibration(self, report: ValidationReport, optimal_params: Dict):
        """Сохранение результатов калибровки"""
        
        # 1. Флаг калибровки
        with open(self.CALIBRATION_FILE, 'w') as f:
            f.write(f"calibrated: {report.calibration_date}\n")
            f.write(f"recommendation: {report.final_recommendation}\n")
        
        # 2. Калиброванный конфиг
        with open(self.base_config_path, 'r') as f:
            full_config = json.load(f)
        
        for key in self.calibratable_params:
            if key in full_config.get('model', {}):
                full_config['model'][key] = optimal_params[key]
        
        with open(self.CALIBRATION_CONFIG, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        # 3. Отчёт
        report_dict = {
            'calibration_date': report.calibration_date,
            'theoretical_params': report.theoretical_params,
            'optimal_params': report.optimal_params,
            'results': [
                {
                    'param_name': r.param_name,
                    'theoretical_value': r.theoretical_value,
                    'optimal_value': r.optimal_value,
                    'deviation_percent': r.deviation_percent,
                    'recommendation': r.recommendation
                }
                for r in report.results
            ],
            'overall_deviation': report.overall_deviation,
            'final_recommendation': report.final_recommendation,
            'is_calibrated': True
        }
        
        report_path = self.CALIBRATION_CONFIG.replace('.json', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"💾 Калибровка сохранена:")
        print(f"   Конфиг: {self.CALIBRATION_CONFIG}")
        print(f"   Отчёт:  {report_path}")
        print(f"   Флаг:   {self.CALIBRATION_FILE}")
    
    def _print_report(self, report: ValidationReport):
        """Красивый вывод отчёта"""
        
        print("=" * 70)
        print("📋 ОТЧЁТ О ВАЛИДАЦИИ")
        print("=" * 70)
        
        print(f"\n{'Параметр':<25} {'Теория':>10} {'Оптимум':>10} {'Отклонение':>12} {'Решение':>12}")
        print("-" * 70)
        
        for r in report.results:
            status_icon = "✅" if r.recommendation == 'ACCEPT' else \
                         "⚠️" if r.recommendation == 'CALIBRATE' else "🔴"
            print(f"{r.param_name:<25} {r.theoretical_value:>10.2f} "
                  f"{r.optimal_value:>10.2f} {r.deviation_percent:>11.1f}% "
                  f"{status_icon} {r.recommendation:>10}")
        
        print("-" * 70)
        print(f"\nСреднее отклонение: {report.overall_deviation*100:.1f}%")
        print(f"\n🎯 ФИНАЛЬНОЕ РЕШЕНИЕ: {report.final_recommendation}")
        
        if report.final_recommendation == "ТЕОРЕТИЧЕСКИЕ ПАРАМЕТРЫ ПРИНЯТЫ":
            print("   Параметры из физической модели подтверждены данными.")
        elif report.final_recommendation == "ПРОИЗВЕДЕНА АВТОМАТИЧЕСКАЯ КАЛИБРОВКА":
            print("   Небольшие корректировки для улучшения точности.")
        else:
            print("   ⚠️ Существенные расхождения! Требуется экспертная оценка.")
        
        print("=" * 70)


def ensure_calibrated(config_path: str = 'config/config.json',
                      force: bool = False) -> str:
    """
    Утилита: гарантировать калибровку перед запуском
    
    Returns:
        Путь к актуальному конфигу (базовому или калиброванному)
    """
    validator = MTIValidator(config_path)
    
    if validator.is_calibrated() and not force:
        return validator.get_effective_config_path()
    
    # Нужна калибровка — загружаем данные
    print("🔧 Требуется первичная калибровка модели...")
    print("   Загрузка данных...")
    
    from data_loader import MTIDataLoader
    loader = MTIDataLoader()
    
    ssn_df = loader.fetch_ssn(use_cache=True)
    quake_df = loader.fetch_earthquakes(
        start_date=pd.Timestamp('1950-01-01'),
        end_date=pd.Timestamp('2024-12-31'),
        min_magnitude=7.5,
        use_cache=True
    )
    
    # Запуск валидации
    report = validator.validate_and_calibrate(ssn_df, quake_df, force=force)
    
    return validator.get_effective_config_path()


# Для прямого запуска
if __name__ == '__main__':
    ensure_calibrated()
