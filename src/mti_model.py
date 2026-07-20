#!/usr/bin/env python3
"""
МТИ-модель (Magneto-Tectonic Inertia Model)
Исправленная версия: использует все поля config.json,
включая пространственную фильтрацию по lat/lon и фильтр глубины.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from enum import Enum

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


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
    coefficient: float    risk_score: float
    probability: float
    risk_level: RiskLevel
    risk_window_start: Optional[datetime] = None
    risk_window_end: Optional[datetime] = None
    description: str = ""
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class GlobalForecast:
    max_probability: float
    dominant_risk_level: RiskLevel
    active_regions: List[RegionalForecast]
    risk_window_start: Optional[datetime] = None
    risk_window_end: Optional[datetime] = None
    interpretation: str = ""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками на сфере (км)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class MTIModel:
    """МТИ-модель: расчёт сейсмического риска на основе солнечной активности."""

    # Радиус поиска землетрясений вокруг центра региона (км)
    REGION_RADIUS_KM = 1000.0

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_dir = Path(__file__).parent.resolve()
            config_path = str(base_dir / 'config' / 'config.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.model_cfg = self.config['model']
        self.lag_cfg = self.config['lag_windows']
        self.risk_thresholds = self.config['risk_thresholds']
        self.regions = self.config['regions']
        self.global_cfg = self.config.get('global_forecast', {})
        # Критические пороги из конфига
        self.ssn_threshold = self.model_cfg['ssn_threshold']
        self.peak_plateau_threshold = self.model_cfg['peak_plateau_threshold']
        self.decline_threshold = self.model_cfg.get('decline_threshold', -self.peak_plateau_threshold)
        self.max_depth_km = self.model_cfg.get('max_depth_km', 70.0)
        self.min_magnitude = self.model_cfg.get('min_magnitude', 7.5)
        self.min_regions_for_global_alert = self.global_cfg.get('min_regions_for_global_alert', 3)

    # ------------------------------------------------------------------
    # Обработка рядов SSN
    # ------------------------------------------------------------------
    def calculate_dSSN_dt(self, ssn_series: pd.DataFrame,
                          smoothing: Optional[int] = None) -> pd.DataFrame:
        df = ssn_series.copy()
        df = df.sort_values('Date').reset_index(drop=True)

        smooth_window = smoothing or self.model_cfg['smoothing_months']
        if smooth_window > 1:
            df['SSN_smooth'] = df['SSN'].rolling(
                window=smooth_window, center=True, min_periods=1
            ).mean()
        else:
            df['SSN_smooth'] = df['SSN']

        df['dSSN_dt'] = df['SSN_smooth'].diff()
        return df[['Date', 'SSN', 'SSN_smooth', 'dSSN_dt']].copy()

    # ------------------------------------------------------------------
    # Фазы и триггер
    # ------------------------------------------------------------------
    def determine_phase(self, ssn: float, dssn_dt: float) -> SolarPhase:
        if ssn < 20:
            return SolarPhase.MINIMUM
        if dssn_dt > self.peak_plateau_threshold:
            return SolarPhase.ACCUMULATION
        if dssn_dt < self.decline_threshold:
            return SolarPhase.DECLINE
        if abs(dssn_dt) <= self.peak_plateau_threshold and ssn > self.ssn_threshold:
            return SolarPhase.PEAK
        return SolarPhase.TRANSITION

    def calculate_trigger(self, ssn: float, dssn_dt: float) -> TriggerResult:
        # Спад: SSN высокий и падает быстрее decline_threshold
        if ssn > self.ssn_threshold and dssn_dt < self.decline_threshold:
            decline_rate = abs(dssn_dt) / ssn if ssn > 0 else 0

            steep = self.lag_cfg['steep']
            moderate = self.lag_cfg['moderate']
            slow = self.lag_cfg['slow']
            if decline_rate >= steep['decline_rate_min']:
                cfg = steep
            elif decline_rate >= moderate['decline_rate_min']:
                cfg = moderate
            else:
                cfg = slow

            return TriggerResult(
                active=True,
                phase=SolarPhase.DECLINE,
                decline_rate=decline_rate,
                lag_min=cfg['min_months'],
                lag_max=cfg['max_months'],
                risk_multiplier=cfg['multiplier']
            )

        # Пик/плато
        if ssn > self.ssn_threshold and abs(dssn_dt) <= self.peak_plateau_threshold:
            return TriggerResult(
                active=False,
                phase=SolarPhase.PEAK,
                risk_multiplier=0.5,
                note="Paradoxical quiescence expected"
            )

        # Накопление
        if ssn > self.ssn_threshold and dssn_dt > 0:
            return TriggerResult(
                active=False,
                phase=SolarPhase.ACCUMULATION,
                risk_multiplier=1.0
            )

        # Минимум / переход
        return TriggerResult(
            active=False,
            phase=self.determine_phase(ssn, dssn_dt),
            risk_multiplier=1.0
        )

    # ------------------------------------------------------------------
    # Окна, коэффициенты, Байес
    # ------------------------------------------------------------------
    def calculate_risk_window(self, trigger_date: datetime,
                              lag_min: int, lag_max: int) -> Tuple[datetime, datetime]:
        return (trigger_date + relativedelta(months=lag_min),
                trigger_date + relativedelta(months=lag_max))

    def apply_regional_coefficient(self, base_risk: float, region_code: str) -> float:
        coeff = self.regions.get(region_code, {}).get('coefficient', 0.5)        return base_risk * coeff

    def bayesian_risk_assessment(self, P_trigger: float, region_coeff: float,
                                 lag_coeff: float,
                                 prior: Optional[float] = None) -> float:
        prior = prior or self.model_cfg['prior_probability']
        norm_const = self.model_cfg['normalization_constant']

        if norm_const <= 0:
            return 0.0

        normalized_region = min(region_coeff / 2.0, 1.0)
        normalized_lag = min(lag_coeff / 2.5, 1.0)

        likelihood = P_trigger * normalized_region * normalized_lag
        P_integrated = (likelihood * prior) / norm_const
        return min(max(P_integrated, 0.0), 1.0)

    def interpret_risk(self, P: float) -> RiskLevel:
        if P >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL
        if P >= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        if P >= self.risk_thresholds['elevated']:
            return RiskLevel.ELEVATED
        return RiskLevel.NORMAL

    # ------------------------------------------------------------------
    # Прогноз
    # ------------------------------------------------------------------
    def forecast_region(self, region_code: str, current_date: datetime,
                        ssn: float, dssn_dt: float) -> Optional[RegionalForecast]:
        region_data = self.regions.get(region_code)
        if not region_data:
            return None

        trigger = self.calculate_trigger(ssn, dssn_dt)

        window_start = window_end = None
        if trigger.active:
            window_start, window_end = self.calculate_risk_window(
                current_date, trigger.lag_min, trigger.lag_max)
            P_trigger = 1.0
        else:
            P_trigger = 0.0

        regional_risk = self.apply_regional_coefficient(trigger.risk_multiplier, region_code)
        P_integrated = self.bayesian_risk_assessment(
            P_trigger, region_data['coefficient'], trigger.risk_multiplier)
        risk_level = self.interpret_risk(P_integrated)
        return RegionalForecast(
            name=region_code,
            coefficient=region_data['coefficient'],
            risk_score=regional_risk,
            probability=P_integrated,
            risk_level=risk_level,
            risk_window_start=window_start,
            risk_window_end=window_end,
            description=region_data.get('description', ''),
            lat=region_data.get('lat', 0.0),
            lon=region_data.get('lon', 0.0),
        )

    def forecast_global(self, current_date: datetime,
                        ssn: float, dssn_dt: float) -> GlobalForecast:
        all_forecasts: List[RegionalForecast] = []
        for region_code in self.regions.keys():
            f = self.forecast_region(region_code, current_date, ssn, dssn_dt)
            if f:
                all_forecasts.append(f)

        active_regions = [f for f in all_forecasts
                          if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]

        # Используем min_regions_for_global_alert из конфига
        if len(active_regions) >= self.min_regions_for_global_alert:
            max_prob = max(f.probability for f in active_regions)
            dominant_level = (RiskLevel.CRITICAL
                              if any(f.risk_level == RiskLevel.CRITICAL for f in active_regions)
                              else RiskLevel.HIGH)

            starts = [f.risk_window_start for f in active_regions if f.risk_window_start]
            ends = [f.risk_window_end for f in active_regions if f.risk_window_end]
            global_start = min(starts) if starts else None
            global_end = max(ends) if ends else None

            interpretation = (
                f"{'КРИТИЧЕСКИЙ' if dominant_level == RiskLevel.CRITICAL else 'ВЫСОКИЙ'} "
                f"глобальный риск. {len(active_regions)} регион(ов) в зоне угрозы. "
                f"Максимальная вероятность: {max_prob:.1%}"
            )
        else:
            max_prob = max((f.probability for f in all_forecasts), default=0)
            dominant_level = RiskLevel.NORMAL
            global_start = global_end = None
            interpretation = (
                f"Глобальный сейсмический риск в норме "
                f"(активных регионов: {len(active_regions)}, "
                f"требуется: {self.min_regions_for_global_alert})."            )

        return GlobalForecast(
            max_probability=max_prob,
            dominant_risk_level=dominant_level,
            active_regions=active_regions,
            risk_window_start=global_start,
            risk_window_end=global_end,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Пространственная фильтрация землетрясений
    # ------------------------------------------------------------------
    def _filter_quakes_near_region(self, quake_df: pd.DataFrame,
                                   lat: float, lon: float,
                                   date_start, date_end) -> pd.DataFrame:
        """Отбирает землетрясения в радиусе REGION_RADIUS_KM от центра региона."""
        mask_time = (quake_df['Date'] >= date_start) & (quake_df['Date'] <= date_end)
        mask_mag = quake_df['Magnitude'] >= self.min_magnitude

        # Фильтр по глубине, если есть столбец
        if 'Depth' in quake_df.columns:
            mask_depth = quake_df['Depth'] <= self.max_depth_km
        else:
            mask_depth = pd.Series(True, index=quake_df.index)

        candidates = quake_df[mask_time & mask_mag & mask_depth].copy()
        if candidates.empty:
            return candidates

        # Фильтр по расстоянию (векторизованный)
        lat2 = np.radians(candidates['Latitude'].values)
        lon2 = np.radians(candidates['Longitude'].values)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distances = 2 * 6371.0 * np.arcsin(np.sqrt(a))

        return candidates[distances <= self.REGION_RADIUS_KM]

    # ------------------------------------------------------------------
    # Ретроспективный тест
    # ------------------------------------------------------------------
    def retrospective_test(self, ssn_df: pd.DataFrame, quake_df: pd.DataFrame,
                           start_date: datetime, end_date: datetime) -> Dict:
        ssn_processed = self.calculate_dSSN_dt(ssn_df)
        quake_df = quake_df.copy()
        for col in ('Date',):
            if not pd.api.types.is_datetime64_any_dtype(quake_df[col]):
                quake_df[col] = pd.to_datetime(quake_df[col])
        if not pd.api.types.is_datetime64_any_dtype(ssn_processed['Date']):
            ssn_processed['Date'] = pd.to_datetime(ssn_processed['Date'])

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        mask = (ssn_processed['Date'] >= start_ts) & (ssn_processed['Date'] <= end_ts)
        test_period = ssn_processed[mask].copy()

        tp = fp = tn = fn = 0
        covered_until = start_ts - pd.Timedelta(days=1)

        for row in test_period.itertuples():
            current_ts = row.Date
            if current_ts <= covered_until:
                continue

            current_date = current_ts.to_pydatetime() if isinstance(current_ts, pd.Timestamp) else current_ts
            ssn = row.SSN_smooth if not pd.isna(row.SSN_smooth) else row.SSN
            dssn_dt = row.dSSN_dt if not pd.isna(row.dSSN_dt) else 0.0

            trigger = self.calculate_trigger(ssn, dssn_dt)

            if trigger.active:
                window_start, window_end = self.calculate_risk_window(
                    current_date, trigger.lag_min, trigger.lag_max)

                # Проверяем, есть ли ХОТЯ БЫ ОДИН регион, в радиусе которого
                # произошло целевое землетрясение в окне риска
                hit_any_region = False
                for region_code, region_data in self.regions.items():
                    regional_quakes = self._filter_quakes_near_region(
                        quake_df,
                        region_data['lat'], region_data['lon'],
                        pd.Timestamp(window_start), pd.Timestamp(window_end)
                    )
                    if not regional_quakes.empty:
                        hit_any_region = True
                        break

                if hit_any_region:
                    tp += 1
                else:
                    fp += 1
                covered_until = pd.Timestamp(window_end)            else:
                check_start = pd.Timestamp(current_date)
                check_end = min(check_start + pd.DateOffset(months=12), end_ts)

                hit_any_region = False
                for region_code, region_data in self.regions.items():
                    regional_quakes = self._filter_quakes_near_region(
                        quake_df,
                        region_data['lat'], region_data['lon'],
                        check_start, check_end
                    )
                    if not regional_quakes.empty:
                        hit_any_region = True
                        break

                if hit_any_region:
                    fn += 1
                else:
                    tn += 1
                covered_until = check_start + pd.DateOffset(months=1)

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            'true_positives': tp, 'false_positives': fp,
            'true_negatives': tn, 'false_negatives': fn,
            'precision': precision, 'recall': recall,
            'f1_score': f1, 'accuracy': accuracy,
            'total_months': total
        }

    def binomial_test(self, observed_tp: int, total_events: int,
                      p_prior: Optional[float] = None) -> float:
        p_prior = p_prior or self.model_cfg['prior_probability']
        try:
            from scipy.stats import binomtest
            return binomtest(observed_tp, total_events, p_prior, alternative='greater').pvalue
        except ImportError:
            pass
        try:
            from scipy.stats import binom_test
            return binom_test(observed_tp, total_events, p_prior, alternative='greater')
        except ImportError:
            pass
        if total_events <= 0 or observed_tp <= 0:
            return 1.0        return sum(
            math.comb(total_events, k) * (p_prior ** k) * ((1 - p_prior) ** (total_events - k))
            for k in range(observed_tp, total_events + 1)
        )


# ======================================================================
# Форматирование вывода
# ======================================================================
def format_forecast_text(forecast: GlobalForecast, current_date: datetime,
                         ssn: float, dssn_dt: float,
                         phase: Optional[SolarPhase] = None,
                         model: Optional[MTIModel] = None) -> str:
    phase_name = {
        SolarPhase.MINIMUM: "Солнечный минимум",
        SolarPhase.ACCUMULATION: "Фаза накопления",
        SolarPhase.PEAK: "Пик/плато (парадокс затишья)",
        SolarPhase.DECLINE: "Фаза спада",
        SolarPhase.TRANSITION: "Переходная фаза",
    }

    if phase is None:
        phase = model.determine_phase(ssn, dssn_dt) if model else SolarPhase.TRANSITION

    lines = [
        "=" * 60,
        "MTI-MODEL: Global Seismic Forecast",
        "=" * 60, "",
        f"Forecast Date: {current_date.strftime('%Y-%m-%d')}",
        f"Current SSN: {ssn:.1f}",
        f"dSSN/dt: {dssn_dt:+.1f}",
        f"Solar Cycle Phase: {phase_name.get(phase, phase.value)}", "",
        "-" * 60, "GLOBAL FORECAST", "-" * 60,
        f"Risk Level: {forecast.dominant_risk_level.value}",
        f"Max Probability: {forecast.max_probability:.1%}",
    ]

    if forecast.risk_window_start and forecast.risk_window_end:
        lines += ["", "Risk Window:",
                  f"  From: {forecast.risk_window_start.strftime('%Y-%m-%d')}",
                  f"  To:   {forecast.risk_window_end.strftime('%Y-%m-%d')}"]

    lines += ["", f"Interpretation: {forecast.interpretation}", ""]

    if forecast.active_regions:
        lines += ["-" * 60,
                  f"HIGH-RISK REGIONS ({len(forecast.active_regions)})",
                  "-" * 60]
        for i, r in enumerate(forecast.active_regions, 1):
            icon = "🔴" if r.risk_level == RiskLevel.CRITICAL else "🟠"            ws = r.risk_window_start.strftime('%Y-%m') if r.risk_window_start else 'N/A'
            we = r.risk_window_end.strftime('%Y-%m') if r.risk_window_end else 'N/A'
            lines += ["",
                      f"{icon} {i}. {r.name} ({r.lat:.1f}°, {r.lon:.1f}°)",
                      f"   Sensitivity: {r.coefficient:.1f}",
                      f"   P(M≥7.5): {r.probability:.1%}",
                      f"   Risk Score: {r.risk_score:.2f}",
                      f"   Window: {ws} – {we}",
                      f"   {r.description}"]
    else:
        lines += ["-" * 60, "No regions with critical or high risk", "-" * 60]

    lines += ["", "=" * 60,
              "Model Limitations:",
              "• Predicts only M≥7.5 events, depth ≤ 70 km",
              "• Spatial accuracy: ~1000 km around region center",
              "• Temporal accuracy: 1-12 months",
              "=" * 60]

    return "\n".join(lines)
