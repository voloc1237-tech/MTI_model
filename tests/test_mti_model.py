"""
Unit-тесты для МТИ-модели
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mti_model import MTIModel, SolarPhase, RiskLevel, TriggerResult


class TestMTIModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = MTIModel('config/config.json')
    
    def test_determine_phase_minimum(self):
        """Тест: солнечный минимум"""
        self.assertEqual(self.model.determine_phase(10, -2), SolarPhase.MINIMUM)
        self.assertEqual(self.model.determine_phase(15, 10), SolarPhase.MINIMUM)
    
    def test_determine_phase_accumulation(self):
        """Тест: фаза накопления"""
        self.assertEqual(self.model.determine_phase(50, 10), SolarPhase.ACCUMULATION)
        self.assertEqual(self.model.determine_phase(100, 20), SolarPhase.ACCUMULATION)
    
    def test_determine_phase_peak(self):
        """Тест: пик/плато"""
        self.assertEqual(self.model.determine_phase(100, 2), SolarPhase.PEAK)
        self.assertEqual(self.model.determine_phase(120, 0), SolarPhase.PEAK)
    
    def test_determine_phase_decline(self):
        """Тест: фаза спада"""
        self.assertEqual(self.model.determine_phase(100, -10), SolarPhase.DECLINE)
        self.assertEqual(self.model.determine_phase(90, -20), SolarPhase.DECLINE)
    
    def test_trigger_active_decline(self):
        """Тест: активация триггера при спаде"""
        trigger = self.model.calculate_trigger(100, -15)
        self.assertTrue(trigger.active)
        self.assertEqual(trigger.phase, SolarPhase.DECLINE)
        self.assertGreater(trigger.decline_rate, 0)
    
    def test_trigger_steep_decline(self):
        """Тест: крутой спад (decline_rate > 0.1)"""
        trigger = self.model.calculate_trigger(100, -15)  # decline_rate = 0.15
        self.assertEqual(trigger.lag_min, 1)
        self.assertEqual(trigger.lag_max, 3)
        self.assertEqual(trigger.risk_multiplier, 2.5)
    
    def test_trigger_moderate_decline(self):
        """Тест: умеренный спад"""
        trigger = self.model.calculate_trigger(100, -8)  # decline_rate = 0.08
        self.assertEqual(trigger.lag_min, 3)
        self.assertEqual(trigger.lag_max, 6)
        self.assertEqual(trigger.risk_multiplier, 2.0)
    
    def test_trigger_peak_quiescence(self):
        """Тест: парадокс затишья на пике"""
        trigger = self.model.calculate_trigger(100, 2)
        self.assertFalse(trigger.active)
        self.assertEqual(trigger.phase, SolarPhase.PEAK)
        self.assertEqual(trigger.risk_multiplier, 0.5)
    
    def test_trigger_below_threshold(self):
        """Тест: SSN ниже порога — триггер неактивен"""
        trigger = self.model.calculate_trigger(50, -10)
        self.assertFalse(trigger.active)
    
    def test_bayesian_calculation(self):
        """Тест: байесовская оценка"""
        P = self.model.bayesian_risk_assessment(1.0, 1.6, 2.5)
        self.assertGreater(P, 0)
        self.assertLessEqual(P, 1.0)
    
    def test_interpret_risk_critical(self):
        """Тест: интерпретация критического риска"""
        self.assertEqual(self.model.interpret_risk(0.8), RiskLevel.CRITICAL)
    
    def test_interpret_risk_normal(self):
        """Тест: интерпретация нормального риска"""
        self.assertEqual(self.model.interpret_risk(0.1), RiskLevel.NORMAL)
    
    def test_calculate_dSSN_dt(self):
        """Тест: расчёт производной"""
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5, freq='MS'),
            'SSN': [50, 60, 55, 70, 80]
        })
        result = self.model.calculate_dSSN_dt(df, smoothing=1)
        self.assertIn('dSSN_dt', result.columns)
        self.assertEqual(result['dSSN_dt'].iloc[1], 10)  # 60-50
    
    def test_risk_window_calculation(self):
        """Тест: расчёт окна риска"""
        start = datetime(2024, 1, 1)
        w_start, w_end = self.model.calculate_risk_window(start, 1, 3)
        self.assertEqual(w_start.month, 2)  # Jan + 1 month
        self.assertEqual(w_end.month, 4)    # Jan + 3 months


class TestRegionalCoefficients(unittest.TestCase):
    
    def test_chile_coefficient(self):
        """Тест: Чили имеет высокий коэффициент"""
        model = MTIModel('config/config.json')
        self.assertEqual(model.regions['Chile_Nazca']['coefficient'], 1.9)
    
    def test_himalaya_coefficient(self):
        """Тест: Гималаи имеют низкий коэффициент"""
        model = MTIModel('config/config.json')
        self.assertEqual(model.regions['Himalayas_Nepal']['coefficient'], 0.5)


if __name__ == '__main__':
    unittest.main()
