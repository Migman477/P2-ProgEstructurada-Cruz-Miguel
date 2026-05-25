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

def calcular_metricas_error(lista_errores_crudos, epochs_flotante):
    """Realiza cálculos matemáticos de error usando math."""
    print("--- Cálculos Matemáticos ---")
    
    # Validación con if para evitar errores si la lista está vacía (regla de oro)
    if len(lista_errores_crudos) == 0:
        print("No hay errores para calcular métricas.")
        return 0, 0
    
    # 1. Usar redondeo hacia arriba para el cálculo final de los epochs
    epochs_totales = math.ceil(epochs_flotante)
    print(f"Epochs estimados: {epochs_flotante} -> Redondeado a enteros (ceil): {epochs_totales}")
    
    # 2. Aplicar función de potencia para elevar diferencias de error al cuadrado
    errores_cuadrado = []
    for error in lista_errores_crudos:
        error_cuadrado = math.pow(error, 2)
        errores_cuadrado.append(error_cuadrado)
    
    # Calcular la media de los errores al cuadrado
    suma_errores = 0.0
    for error_sq in errores_cuadrado:
        suma_errores += error_sq
    media_error_cuadrado = suma_errores / len(errores_cuadrado)
    
    # 3. Calcular la raíz cuadrada para la métrica RMSE
    rmse = math.sqrt(media_error_cuadrado)
    
    # Uso adicional (absoluto)
    error_maximo_absoluto = math.fabs(lista_errores_crudos[0])
    for error in lista_errores_crudos:
        if math.fabs(error) > error_maximo_absoluto:
            error_maximo_absoluto = math.fabs(error)
            
    print(f"Métrica RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"Error absoluto máximo observado (fabs): {error_maximo_absoluto:.4f}\n")
    
    return rmse, epochs_totales

