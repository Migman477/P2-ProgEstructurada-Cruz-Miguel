import datetime
import math
import random
import statistics
import sys

def gestionar_tiempo():
    """Maneja la telemetría del tiempo de entrenamiento usando datetime."""
    print("--- Gestión del Tiempo ---")
    
    # 1. Obtener la fecha y hora exacta del inicio
    tiempo_inicio = datetime.datetime.now()
    
    # 2. Formatear la fecha en formato legible en español (Día/Mes/Año Hora:Minuto:Segundo)
    formato_fecha = "%d/%m/%Y %H:%M:%S"
    inicio_formateado = tiempo_inicio.strftime(formato_fecha)
    print(f"Inicio de la simulación: {inicio_formateado}")
    
    # Simular que el entrenamiento toma cierto tiempo agregando un delta
    delta_simulado = datetime.timedelta(hours=1, minutes=45, seconds=30)
    tiempo_fin = tiempo_inicio + delta_simulado
    fin_formateado = tiempo_fin.strftime(formato_fecha)
    print(f"Fin de la simulación: {fin_formateado}")
    
    # 3. Calcular la diferencia de tiempo simulada
    diferencia_tiempo = tiempo_fin - tiempo_inicio
    print(f"Duración total del proceso: {diferencia_tiempo}\n")
    
    return diferencia_tiempo

