#!/usr/bin/env python3
"""
МТИ-модель (Magneto-Tectonic Inertia Model)
"""


import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from enum import Enum

import numpy as np
import pandas as pd


class SolarPhase(Enum):
    MINIMUM = "minimum"
    ACCUMULATION = "accumulation"
    PEAK = "peak"
    DECLINE = "decline"
    TRANSITION = "transition"


class RiskLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TriggerResult:
    active: bool
    phase: SolarPhase
    decline_rate: float = 0.0
    lag_min: int = 0
    lag_max: int = 0
    risk_multiplier: float = 1.0
    note: str = ""


@dataclass
class RegionalForecast:
    name: str
    coefficient: float
    risk_score: float
    probability: float
    risk_level: RiskLevel
    risk_window_start: Optional[datetime] = None
    risk_window_end: Optional[datetime] = None
    description: str = ""


@dataclass
class GlobalForecast:
    max_probability: float
    dominant_risk_level: RiskLevel
    active_regions: List[RegionalForecast]
    risk_window_start: Optional[datetime] = None
    risk_window_end: Optional[datetime] = None
    interpretation: str = ""


class MTIModel:
    
    
    def __init__(self, config_path: str = 'config/config.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model_cfg = self.config['model']
        self.lag_cfg = self.config['lag_windows']
        self.risk_thresholds = self.config['risk_thresholds']
        self.regions = self.config['regions']
        self.global_cfg = self.config.get('global_forecast', {})
    
    def calculate_dSSN_dt(self, ssn_series: pd.DataFrame, 
                          smoothing: Optional[int] = None) -> pd.DataFrame:
        """Расчёт первой производной числа Вольфа"""
        df = ssn_series.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        smooth_window = smoothing or self.model_cfg['smoothing_months']
        if smooth_window > 1:
            df['SSN_smooth'] = df['SSN'].rolling(
                window=smooth_window, 
                center=True, 
                min_periods=1
            ).mean()
        else:
            df['SSN_smooth'] = df['SSN']
        
        df['dSSN_dt'] = df['SSN_smooth'].diff()
        
        return df[['Date', 'SSN', 'SSN_smooth', 'dSSN_dt']].copy()
    
    def determine_phase(self, ssn: float, dssn_dt: float) -> SolarPhase:
        """Определение фазы солнечного цикла"""
        threshold_peak = self.model_cfg['peak_plateau_threshold']
        
        if ssn < 20:
            return SolarPhase.MINIMUM
        elif dssn_dt > threshold_peak:
            return SolarPhase.ACCUMULATION
        elif abs(dssn_dt) <= threshold_peak and ssn > self.model_cfg['ssn_threshold']:
            return SolarPhase.PEAK
        elif dssn_dt < -threshold_peak:
            return SolarPhase.DECLINE
        else:
            return SolarPhase.TRANSITION
    
    def calculate_trigger(self, ssn: float, dssn_dt: float) -> TriggerResult:
        """Расчёт триггерного коэффициента"""
        ssn_threshold = self.model_cfg['ssn_threshold']
        threshold_peak = self.model_cfg['peak_plateau_threshold']
        
        if ssn > ssn_threshold and dssn_dt < 0:
            decline_rate = abs(dssn_dt) / ssn if ssn > 0 else 0
            
            steep = self.lag_cfg['steep']
            moderate = self.lag_cfg['moderate']
            slow = self.lag_cfg['slow']
            
            if decline_rate >= steep['decline_rate_min']:
                return TriggerResult(
                    active=True,
                    phase=SolarPhase.DECLINE,
                    decline_rate=decline_rate,
                    lag_min=steep['min_months'],
                    lag_max=steep['max_months'],
                    risk_multiplier=steep['multiplier']
                )
            elif decline_rate >= moderate['decline_rate_min']:
                return TriggerResult(
                    active=True,
                    phase=SolarPhase.DECLINE,
                    decline_rate=decline_rate,
                    lag_min=moderate['min_months'],
                    lag_max=moderate['max_months'],
                    risk_multiplier=moderate['multiplier']
                )
            else:
                return TriggerResult(
                    active=True,
                    phase=SolarPhase.DECLINE,
                    decline_rate=decline_rate,
                    lag_min=slow['min_months'],
                    lag_max=slow['max_months'],
                    risk_multiplier=slow['multiplier']
                )
        
        elif ssn > ssn_threshold and abs(dssn_dt) <= threshold_peak:
            return TriggerResult(
                active=False,
                phase=SolarPhase.PEAK,
                risk_multiplier=0.5,
                note="Paradoxical quiescence expected"
            )
        
        else:
            return TriggerResult(
                active=False,
                phase=SolarPhase.ACCUMULATION if dssn_dt > 0 else SolarPhase.MINIMUM,
                risk_multiplier=1.0
            )
    
    def calculate_risk_window(self, 
                              trigger_date: datetime,
                              lag_min: int,
                              lag_max: int) -> Tuple[datetime, datetime]:
        """Определение временного окна повышенного риска"""
        window_start = trigger_date + pd.DateOffset(months=lag_min)
        window_end = trigger_date + pd.DateOffset(months=lag_max)
        
        return window_start, window_end
    
    def apply_regional_coefficient(self, 
                                    base_risk: float, 
                                    region_code: str) -> float:
        """Применение регионального коэффициента чувствительности"""
        coeff = self.regions.get(region_code, {}).get('coefficient', 0.5)
        return base_risk * coeff
    
    def bayesian_risk_assessment(self,
                                  P_trigger: float,
                                  region_coeff: float,
                                  lag_coeff: float,
                                  prior: Optional[float] = None) -> float:
        """Байесовская оценка совокупного риска"""
        prior = prior or self.model_cfg['prior_probability']
        norm_const = self.model_cfg['normalization_constant']
        
        normalized_region = min(region_coeff / 2.0, 1.0)
        normalized_lag = min(lag_coeff / 2.5, 1.0)
        
        likelihood = P_trigger * normalized_region * normalized_lag
        P_integrated = (likelihood * prior) / norm_const if norm_const > 0 else 0
        
        return min(max(P_integrated, 0.0), 1.0)
    
    def interpret_risk(self, P: float) -> RiskLevel:
        """Интерпретация интегрированной вероятности"""
        if P >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL
        elif P >= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        elif P >= self.risk_thresholds['elevated']:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL
    
    def forecast_region(self,
                        region_code: str,
                        current_date: datetime,
                        ssn: float,
                        dssn_dt: float) -> Optional[RegionalForecast]:
        """Полный конвейер МТИ-модели для одного региона"""
        region_data = self.regions.get(region_code)
        if not region_data:
            return None
        
        trigger = self.calculate_trigger(ssn, dssn_dt)
        
        window_start = None
        window_end = None
        
        if trigger.active:
            window_start, window_end = self.calculate_risk_window(
                current_date,
                trigger.lag_min,
                trigger.lag_max
            )
            P_trigger = 1.0
        else:
            P_trigger = 0.0
        
        regional_risk = self.apply_regional_coefficient(
            trigger.risk_multiplier,
            region_code
        )
        
        P_integrated = self.bayesian_risk_assessment(
            P_trigger,
            region_data['coefficient'],
            trigger.risk_multiplier
        )
        
        risk_level = self.interpret_risk(P_integrated)
        
        return RegionalForecast(
            name=region_code,
            coefficient=region_data['coefficient'],
            risk_score=regional_risk,
            probability=P_integrated,
            risk_level=risk_level,
            risk_window_start=window_start,
            risk_window_end=window_end,
            description=region_data.get('description', '')
        )
    
    def forecast_global(self,
                        current_date: datetime,
                        ssn: float,
                        dssn_dt: float) -> GlobalForecast:
        """Глобальный прогноз по всем регионам"""
        all_forecasts = []
        for region_code in self.regions.keys():
            forecast = self.forecast_region(region_code, current_date, ssn, dssn_dt)
            if forecast:
                all_forecasts.append(forecast)
        
        active_regions = [f for f in all_forecasts 
                         if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        
        if active_regions:
            max_prob = max(f.probability for f in active_regions)
            
            if any(f.risk_level == RiskLevel.CRITICAL for f in active_regions):
                dominant_level = RiskLevel.CRITICAL
            else:
                dominant_level = RiskLevel.HIGH
            
            all_starts = [f.risk_window_start for f in active_regions 
                         if f.risk_window_start]
            all_ends = [f.risk_window_end for f in active_regions 
                       if f.risk_window_end]
            
            global_start = min(all_starts) if all_starts else None
            global_end = max(all_ends) if all_ends else None
            
            if dominant_level == RiskLevel.CRITICAL:
                interpretation = (
                    f"КРИТИЧЕСКИЙ глобальный риск. "
                    f"{len(active_regions)} регион(ов) в зоне повышенной угрозы. "
                    f"Максимальная вероятность: {max_prob:.1%}"
                )
            else:
                interpretation = (
                    f"ВЫСОКИЙ глобальный риск. "
                    f"{len(active_regions)} регион(ов) в зоне повышенной угрозы. "
                    f"Максимальная вероятность: {max_prob:.1%}"
                )
        else:
            max_prob = max((f.probability for f in all_forecasts), default=0)
            dominant_level = RiskLevel.NORMAL
            global_start = None
            global_end = None
            interpretation = "Глобальный сейсмический риск в норме."
        
        return GlobalForecast(
            max_probability=max_prob,
            dominant_risk_level=dominant_level,
            active_regions=active_regions,
            risk_window_start=global_start,
            risk_window_end=global_end,
            interpretation=interpretation
        )
    
    def retrospective_test(self,
                           ssn_df: pd.DataFrame,
                           quake_df: pd.DataFrame,
                           start_date: datetime,
                           end_date: datetime) -> Dict:
        """Ретроспективная проверка модели"""
        ssn_processed = self.calculate_dSSN_dt(ssn_df)
        
        mask = (ssn_processed['Date'] >= start_date) & (ssn_processed['Date'] <= end_date)
        test_period = ssn_processed[mask].copy()
        
        tp = fp = tn = fn = 0
        
        for _, row in test_period.iterrows():
            current_date = row['Date']
            ssn = row['SSN_smooth'] if 'SSN_smooth' in row else row['SSN']
            dssn_dt = row['dSSN_dt']
            
            trigger = self.calculate_trigger(ssn, dssn_dt)
            
            if trigger.active:
                window_start, window_end = self.calculate_risk_window(
                    current_date,
                    trigger.lag_min,
                    trigger.lag_max
                )
                
                events_in_window = quake_df[
                    (quake_df['Date'] >= window_start) &
                    (quake_df['Date'] <= window_end) &
                    (quake_df['Magnitude'] >= self.model_cfg['min_magnitude'])
                ]
                
                has_event = len(events_in_window) > 0
                
                if has_event:
                    tp += 1
                else:
                    fp += 1
            else:
                check_start = current_date
                check_end = current_date + pd.DateOffset(months=12)
                
                events_near = quake_df[
                    (quake_df['Date'] >= check_start) &
                    (quake_df['Date'] <= check_end) &
                    (quake_df['Magnitude'] >= self.model_cfg['min_magnitude'])
                ]
                
                has_event = len(events_near) > 0
                
                if has_event:
                    fn += 1
                else:
                    tn += 1
        
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'total_months': total
        }
    
    def binomial_test(self, 
                      observed_tp: int, 
                      total_events: int,
                      p_prior: Optional[float] = None) -> float:
        """Биномиальный тест на значимость"""
        try:
            from scipy.stats import binomtest
            p_prior = p_prior or self.model_cfg['prior_probability']
            result = binomtest(observed_tp, total_events, p_prior, alternative='greater')
            return result.pvalue
        except ImportError:
            from math import comb
            p_prior = p_prior or self.model_cfg['prior_probability']
            p_value = sum(
                comb(total_events, k) * (p_prior ** k) * ((1 - p_prior) ** (total_events - k))
                for k in range(observed_tp, total_events + 1)
            )
            return p_value


def format_forecast_text(forecast: GlobalForecast, 
                         current_date: datetime,
                         ssn: float,
                         dssn_dt: float) -> str:
    """Форматирование прогноза в текст"""
    phase_name = {
        SolarPhase.MINIMUM: "Солнечный минимум",
        SolarPhase.ACCUMULATION: "Фаза накопления",
        SolarPhase.PEAK: "Пик/плато (парадокс затишья)",
        SolarPhase.DECLINE: "Фаза спада",
        SolarPhase.TRANSITION: "Переходная фаза"
    }
    
    # Определяем фазу для заголовка
    from mti_model import MTIModel
    temp_model = MTIModel()
    phase = temp_model.determine_phase(ssn, dssn_dt)
    
    lines = [
        f"{'='*60}",
        f"MTI-MODEL: Global Seismic Forecast",
        f"{'='*60}",
        f"",
        f"Forecast Date: {current_date.strftime('%Y-%m-%d')}",
        f"Current SSN: {ssn:.1f}",
        f"dSSN/dt: {dssn_dt:+.1f}",
        f"Solar Cycle Phase: {phase.value}",
        f"",
        f"{'-'*60}",
        f"GLOBAL FORECAST",
        f"{'-'*60}",
        f"Risk Level: {forecast.dominant_risk_level.value}",
        f"Max Probability: {forecast.max_probability:.1%}",
    ]
    
    if forecast.risk_window_start and forecast.risk_window_end:
        lines.extend([
            f"",
            f"Risk Window:",
            f"  From: {forecast.risk_window_start.strftime('%Y-%m-%d')}",
            f"  To:   {forecast.risk_window_end.strftime('%Y-%m-%d')}",
        ])
    
    lines.extend([
        f"",
        f"Interpretation: {forecast.interpretation}",
        f"",
    ])
    
    if forecast.active_regions:
        lines.extend([
            f"{'-'*60}",
            f"HIGH-RISK REGIONS ({len(forecast.active_regions)})",
            f"{'-'*60}",
        ])
        
        for i, region in enumerate(forecast.active_regions, 1):
            icon = "🔴" if region.risk_level == RiskLevel.CRITICAL else "🟠"
            lines.extend([
                f"",
                f"{icon} {i}. {region.name}",
                f"   Sensitivity Coefficient: {region.coefficient:.1f}",
                f"   P(M≥7.5): {region.probability:.1%}",
                f"   Risk Score: {region.risk_score:.2f}",
                f"   Window: {region.risk_window_start.strftime('%Y-%m') if region.risk_window_start else 'N/A'} – "
                f"{region.risk_window_end.strftime('%Y-%m') if region.risk_window_end else 'N/A'}",
                f"   {region.description}",
            ])
    else:
        lines.extend([
            f"{'-'*60}",
            f"No regions with critical or high risk",
            f"{'-'*60}",
        ])
    
    lines.extend([
        f"",
        f"{'='*60}",
        f"Model Limitations:",
        f"• Predicts only M≥7.5 events",
        f"• Spatial accuracy: 500-1000 km",
        f"• Temporal accuracy: 1-12 months",
        f"• Not applicable at SSN < 40",
        f"{'='*60}",
    ])
    
    return "\n".join(lines)


﻿
