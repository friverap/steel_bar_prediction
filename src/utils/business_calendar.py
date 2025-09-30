#!/usr/bin/env python3
"""
Business Calendar - DeAcero Steel Price Predictor
Manejo de días hábiles, feriados mexicanos y cálculo de próximos días de trading

Este módulo maneja:
1. Identificación de días hábiles vs fines de semana/feriados
2. Cálculo del próximo día hábil para predicciones
3. Feriados mexicanos oficiales
4. Alineación temporal para predicciones
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BusinessCalendar:
    """
    Calendario de días hábiles para el mercado mexicano
    """
    
    def __init__(self):
        # Feriados mexicanos fijos
        self.fixed_holidays = {
            (1, 1): "Año Nuevo",
            (2, 5): "Día de la Constitución", 
            (3, 21): "Natalicio de Benito Juárez",
            (5, 1): "Día del Trabajo",
            (9, 16): "Día de la Independencia",
            (11, 20): "Día de la Revolución Mexicana",
            (12, 25): "Navidad"
        }
        
        # Cache de días hábiles calculados
        self._business_days_cache = {}
        
    def is_business_day(self, date_input: date) -> bool:
        """
        Verificar si una fecha es día hábil
        
        Args:
            date_input: Fecha a verificar
            
        Returns:
            True si es día hábil, False si es fin de semana o feriado
        """
        # Verificar fin de semana
        if date_input.weekday() >= 5:  # Sábado=5, Domingo=6
            return False
        
        # Verificar feriados fijos
        if (date_input.month, date_input.day) in self.fixed_holidays:
            return False
        
        # Verificar feriados variables (Semana Santa, etc.)
        if self._is_variable_holiday(date_input):
            return False
        
        return True
    
    def get_next_business_day(self, from_date: Optional[date] = None) -> date:
        """
        Obtener el próximo día hábil
        
        Args:
            from_date: Fecha de referencia (default: hoy)
            
        Returns:
            Próximo día hábil
        """
        if from_date is None:
            from_date = date.today()
        
        # Buscar próximo día hábil
        current_date = from_date + timedelta(days=1)
        max_iterations = 10  # Evitar bucles infinitos
        
        for _ in range(max_iterations):
            if self.is_business_day(current_date):
                return current_date
            current_date += timedelta(days=1)
        
        # Fallback si no encuentra en 10 días
        logger.warning(f"No se encontró día hábil en 10 días desde {from_date}")
        return current_date
    
    def get_last_business_day(self, before_date: Optional[date] = None) -> date:
        """
        Obtener el último día hábil antes de una fecha
        
        Args:
            before_date: Fecha de referencia (default: hoy)
            
        Returns:
            Último día hábil
        """
        if before_date is None:
            before_date = date.today()
        
        # Buscar último día hábil
        current_date = before_date
        max_iterations = 10
        
        for _ in range(max_iterations):
            if self.is_business_day(current_date):
                return current_date
            current_date -= timedelta(days=1)
        
        logger.warning(f"No se encontró día hábil en 10 días antes de {before_date}")
        return current_date
    
    def business_days_between(self, start_date: date, end_date: date) -> int:
        """
        Contar días hábiles entre dos fechas
        
        Args:
            start_date: Fecha inicial
            end_date: Fecha final
            
        Returns:
            Número de días hábiles
        """
        if start_date > end_date:
            return 0
        
        business_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_business_day(current_date):
                business_days += 1
            current_date += timedelta(days=1)
        
        return business_days
    
    def get_prediction_target_date(self, reference_date: Optional[date] = None) -> Tuple[date, int]:
        """
        Calcular fecha objetivo para predicción y gap en días
        
        Args:
            reference_date: Fecha de referencia (default: hoy)
            
        Returns:
            (target_date, gap_days): Fecha objetivo y días de gap
        """
        if reference_date is None:
            reference_date = date.today()
        
        # Si hoy es día hábil, predecir mañana hábil
        # Si hoy no es hábil, predecir próximo hábil
        if self.is_business_day(reference_date):
            target_date = self.get_next_business_day(reference_date)
        else:
            target_date = self.get_next_business_day(reference_date)
        
        # Calcular gap
        gap_days = (target_date - reference_date).days
        
        return target_date, gap_days
    
    def _is_variable_holiday(self, date_input: date) -> bool:
        """
        Verificar feriados variables (Semana Santa, etc.)
        Implementación básica - se puede extender
        """
        # Para simplicidad, solo verificamos algunos feriados variables conocidos
        year = date_input.year
        
        # Semana Santa 2025 (ejemplo)
        if year == 2025:
            easter_week = [
                date(2025, 4, 14),  # Lunes Santo
                date(2025, 4, 15),  # Martes Santo  
                date(2025, 4, 16),  # Miércoles Santo
                date(2025, 4, 17),  # Jueves Santo
                date(2025, 4, 18),  # Viernes Santo
            ]
            if date_input in easter_week:
                return True
        
        return False
    
    def get_trading_schedule_info(self, reference_date: Optional[date] = None) -> dict:
        """
        Obtener información completa del calendario de trading
        
        Returns:
            Información detallada del calendario
        """
        if reference_date is None:
            reference_date = date.today()
        
        last_business_day = self.get_last_business_day(reference_date)
        next_business_day = self.get_next_business_day(reference_date)
        target_date, gap_days = self.get_prediction_target_date(reference_date)
        
        return {
            'reference_date': reference_date.strftime('%Y-%m-%d'),
            'is_business_day': self.is_business_day(reference_date),
            'last_business_day': last_business_day.strftime('%Y-%m-%d'),
            'next_business_day': next_business_day.strftime('%Y-%m-%d'),
            'prediction_target_date': target_date.strftime('%Y-%m-%d'),
            'gap_days': gap_days,
            'days_since_last_trading': (reference_date - last_business_day).days,
            'days_to_next_trading': (next_business_day - reference_date).days
        }


class PredictionScheduler:
    """
    Scheduler para automatizar predicciones y reentrenamiento
    """
    
    def __init__(self):
        self.calendar = BusinessCalendar()
        self.last_data_date = None
        self.last_retrain_date = None
        
    def should_update_data(self) -> bool:
        """
        Verificar si se deben actualizar los datos
        
        Returns:
            True si se necesita actualización de datos
        """
        today = date.today()
        
        # Si es fin de semana, no actualizar
        if not self.calendar.is_business_day(today):
            return False
        
        # Si es después de las 18:00 de un día hábil, actualizar
        current_time = datetime.now().time()
        market_close_time = datetime.strptime("18:00", "%H:%M").time()
        
        if current_time >= market_close_time:
            # Verificar si ya actualizamos hoy
            if self.last_data_date != today:
                return True
        
        return False
    
    def should_retrain_models(self) -> bool:
        """
        Verificar si se deben reentrenar los modelos
        
        Returns:
            True si se necesita reentrenamiento
        """
        today = date.today()
        
        # Reentrenar solo en días hábiles después del cierre
        if not self.calendar.is_business_day(today):
            return False
        
        current_time = datetime.now().time()
        retrain_time = datetime.strptime("19:00", "%H:%M").time()  # 1 hora después del cierre
        
        if current_time >= retrain_time:
            # Verificar si ya reentrenamos hoy
            if self.last_retrain_date != today:
                return True
        
        return False
    
    def get_prediction_context(self) -> dict:
        """
        Obtener contexto completo para predicción
        
        Returns:
            Contexto temporal para la predicción
        """
        today = date.today()
        schedule_info = self.calendar.get_trading_schedule_info(today)
        
        # Determinar qué datos necesitamos
        last_data_date = self.calendar.get_last_business_day(today)
        target_date = schedule_info['prediction_target_date']
        
        return {
            'current_date': today.strftime('%Y-%m-%d'),
            'last_data_available': last_data_date.strftime('%Y-%m-%d'),
            'prediction_target': target_date,
            'gap_days': schedule_info['gap_days'],
            'trading_schedule': schedule_info,
            'data_lag_explanation': f"Usando datos de {last_data_date.strftime('%Y-%m-%d')} para predecir {target_date}",
            'assumption': "Variables exógenas: Last Available Value (LAV) - práctica estándar en finanzas"
        }


# Función de utilidad global
def get_business_calendar() -> BusinessCalendar:
    """Obtener instancia global del calendario"""
    return BusinessCalendar()


def get_prediction_scheduler() -> PredictionScheduler:
    """Obtener instancia global del scheduler"""
    return PredictionScheduler()
