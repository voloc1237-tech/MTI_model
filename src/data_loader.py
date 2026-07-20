"""
Загрузчик данных для МТИ-модели
"""

import io
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class MTIDataLoader:
    """Загрузка и предобработка данных"""
    
    SSN_URL = "http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"
    USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.ssn_data: Optional[pd.DataFrame] = None
        self.quake_data: Optional[pd.DataFrame] = None
    
    def fetch_ssn(self, use_cache: bool = True) -> pd.DataFrame:
        """Загрузка SSN из SILSO"""
        cache_path = os.path.join(self.data_dir, 'ssn_cache.csv')
        
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['Date'])
            self.ssn_data = df
            return df
        
        response = requests.get(self.SSN_URL, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(
            io.StringIO(response.text),
            sep=';',
            header=None,
            comment='#',
            names=['Year', 'Month', 'DecimalYear', 'SSN', 'StdDev', 'Observations', 'Definitive']
        )
        
        df['Date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + 
            df['Month'].astype(str).str.zfill(2) + '-01'
        )
        
        df = df[['Date', 'SSN']].copy()
        df['SSN'] = pd.to_numeric(df['SSN'], errors='coerce')
        df = df.dropna()
        
        self.ssn_data = df
        df.to_csv(cache_path, index=False)
        
        return df
    
    def fetch_earthquakes(self,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          min_magnitude: float = 7.5,
                          max_depth_km: float = 70,
                          use_cache: bool = True) -> pd.DataFrame:
        """Загрузка землетрясений из USGS"""
        cache_path = os.path.join(self.data_dir, 'quakes_cache.csv')
        
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['Date'])
            self.quake_data = df
            return df
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=100*365))
        
        params = {
            'format': 'geojson',
            'starttime': start_date.strftime('%Y-%m-%d'),
            'endtime': end_date.strftime('%Y-%m-%d'),
            'minmagnitude': min_magnitude,
            'maxdepth': max_depth_km,
            'orderby': 'time-asc',
            'limit': 20000
        }
        
        response = requests.get(self.USGS_API, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        quakes = []
        for feature in data.get('features', []):
            props = feature['properties']
            geom = feature['geometry']
            
            quakes.append({
                'Date': pd.to_datetime(props['time'], unit='ms'),
                'Magnitude': props['mag'],
                'Latitude': geom['coordinates'][1],
                'Longitude': geom['coordinates'][0],
                'Depth': geom['coordinates'][2],
                'Place': props.get('place', '')
            })
        
        df = pd.DataFrame(quakes)
        
        if not df.empty:
            df = df.sort_values('Date').reset_index(drop=True)
            self.quake_data = df
            df.to_csv(cache_path, index=False)
        
        return df
    
    def get_current_ssn(self) -> tuple:
        """Получение текущего SSN и производной"""
        if self.ssn_data is None:
            self.fetch_ssn()
        
        df = self.ssn_data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        
        ssn = current['SSN']
        dssn_dt = ssn - prev['SSN']
        
        return current['Date'], ssn, dssn_dt
