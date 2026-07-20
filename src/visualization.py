"""
Визуализация МТИ-прогнозов
"""

import os
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

from mti_model import MTIModel, RiskLevel, GlobalForecast


def create_forecast_plot(model: MTIModel,
                         forecast: GlobalForecast,
                         date: datetime,
                         ssn: float,
                         dssn_dt: float,
                         ssn_df: pd.DataFrame,
                         quake_df: pd.DataFrame,
                         output_path: str = 'outputs/mti_forecast.png'):
    """Создание визуализации прогноза"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(
        f'MTI Model: Global Seismic Forecast\n{date.strftime("%Y-%m-%d")}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Панель 1: SSN история
    ax1 = fig.add_subplot(4, 1, 1)
    recent_ssn = ssn_df[ssn_df['Date'] >= date - pd.DateOffset(years=20)].copy()
    
    ax1.plot(recent_ssn['Date'], recent_ssn['SSN'], 
            color='orange', linewidth=1.5, alpha=0.7, label='SSN')
    
    recent_ssn['SSN_smooth'] = recent_ssn['SSN'].rolling(
        window=3, center=True, min_periods=1
    ).mean()
    ax1.plot(recent_ssn['Date'], recent_ssn['SSN_smooth'], 
            color='darkorange', linewidth=2.5, label='SSN (smoothed)')
    
    ax1.scatter([date], [ssn], color='red', s=150, zorder=5, marker='*',
               label=f'Current: SSN={ssn:.0f}')
    ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    
    if forecast.active_regions:
        ax1.annotate('TRIGGER ACTIVE\nDecline Phase', 
                    xy=(date, ssn), xytext=(date - pd.DateOffset(years=3), ssn + 50),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax1.set_title('Solar Activity (SSN) — Last 20 Years', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wolf Number')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Панель 2: dSSN/dt
    ax2 = fig.add_subplot(4, 1, 2)
    recent_ssn['dSSN_dt'] = recent_ssn['SSN_smooth'].diff()
    
    ax2.fill_between(recent_ssn['Date'], 0, recent_ssn['dSSN_dt'], 
                    where=(recent_ssn['dSSN_dt'] > 5), 
                    color='green', alpha=0.3, label='Accumulation')
    ax2.fill_between(recent_ssn['Date'], 0, recent_ssn['dSSN_dt'], 
                    where=(recent_ssn['dSSN_dt'] < -5), 
                    color='red', alpha=0.3, label='Decline (trigger)')
    ax2.fill_between(recent_ssn['Date'], -5, 5, 
                    color='yellow', alpha=0.2, label='Peak/plateau')
    
    ax2.plot(recent_ssn['Date'], recent_ssn['dSSN_dt'], color='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    ax2.scatter([date], [dssn_dt], color='red', s=150, zorder=5, marker='*')
    
    ax2.set_title('dSSN/dt (Key MTI Predictor)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ΔSSN/month')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Панель 3: Earthquakes M≥7.5
    ax3 = fig.add_subplot(4, 1, 3)
    
    if quake_df is not None and not quake_df.empty:
        recent_quakes = quake_df[quake_df['Date'] >= date - pd.DateOffset(years=20)]
        sizes = (recent_quakes['Magnitude'] - 7) ** 3 * 10
        colors = plt.cm.Reds((recent_quakes['Magnitude'] - 7.5) / 1.5)
        
        ax3.scatter(recent_quakes['Date'], recent_quakes['Magnitude'],
                   s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax3.axhline(y=7.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=8.0, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=9.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Earthquakes M≥7.5 — History', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Magnitude')
    ax3.set_ylim([7.3, 9.5])
    ax3.grid(True, alpha=0.3)
    
    # Панель 4: Regional risk
    ax4 = fig.add_subplot(4, 1, 4)
    
    all_regions = []
    for region_code in model.regions.keys():
        rf = model.forecast_region(region_code, date, ssn, dssn_dt)
        if rf:
            all_regions.append(rf)
    
    all_regions.sort(key=lambda x: x.probability, reverse=True)
    top_n = 15
    display = all_regions[:top_n]
    display.reverse()
    
    names = [r.name for r in display]
    probs = [r.probability for r in display]
    
    bar_colors = []
    for r in display:
        if r.risk_level == RiskLevel.CRITICAL:
            bar_colors.append('darkred')
        elif r.risk_level == RiskLevel.HIGH:
            bar_colors.append('red')
        elif r.risk_level == RiskLevel.ELEVATED:
            bar_colors.append('orange')
        else:
            bar_colors.append('green')
    
    bars = ax4.barh(range(len(names)), probs, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=9)
    
    ax4.axvline(x=model.risk_thresholds['critical'], color='darkred', linestyle='--', linewidth=2)
    ax4.axvline(x=model.risk_thresholds['high'], color='red', linestyle='--', linewidth=1.5)
    ax4.axvline(x=model.risk_thresholds['elevated'], color='orange', linestyle='--', linewidth=1)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=8, fontweight='bold')
    
    ax4.set_title(f'Regional Risk (top-{top_n})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('P(M≥7.5)')
    ax4.set_xlim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path
