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

def simular_entrenamiento(epochs):
    """Simula los valores estocásticos de las iteraciones usando random."""
    print("--- Simulación Estocástica ---")
    
    # Lista de strings de eventos
    eventos_log = [
        "Epoch exitoso",
        "Gradiente inestable",
        "Actualización de pesos",
        "Tasa de aprendizaje ajustada"
    ]
    
    historial_loss = []
    latencias = []
    
    for epoch in range(1, epochs + 1):
        # 1. Generar la fluctuación del error de pérdida (loss) con decimales aleatorios
        loss_actual = random.uniform(0.1, 2.8)
        historial_loss.append(loss_actual)
        
        # Generar latencia aleatoria en milisegundos
        latencia_ms = random.uniform(20.0, 150.0)
        latencias.append(latencia_ms)
        
        # 2. Simular la probabilidad de éxito de una iteración
        probabilidad_exito = random.random()
        estado = "Exitoso"
        # Si la probabilidad es menor a 0.20 (20%), falla la iteración
        if probabilidad_exito < 0.20:
            estado = "Advertencia de Rendimiento"
            
        # 3. Seleccionar de manera aleatoria un evento de log
        evento_seleccionado = random.choice(eventos_log)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {loss_actual:.4f} | Estado: {estado} | Log: '{evento_seleccionado}'")
        
    print()
    return historial_loss, latencias

def analizar_resultados(historial_loss, latencias):
    """Analiza estadísticamente el rendimiento del entrenamiento usando statistics."""
    print("--- Análisis de Rendimiento ---")
    
    # Validaciones if/else para no lanzar excepciones con listas vacías o con 1 solo elemento en stdev
    if len(historial_loss) > 1 and len(latencias) > 0:
        # 1. Calcular la media de los valores de pérdida
        media_loss = statistics.mean(historial_loss)
        
        # 2. Calcular la desviación estándar para medir la estabilidad
        desviacion_loss = statistics.stdev(historial_loss)
        
        # 3. Obtener la mediana de la latencia del proceso
        mediana_latencia = statistics.median(latencias)
        
        print(f"Media de pérdida (Loss): {media_loss:.4f}")
        print(f"Desviación estándar de la pérdida (Estabilidad): {desviacion_loss:.4f}")
        print(f"Mediana de latencia: {mediana_latencia:.2f} ms\n")
        
        return media_loss
    else:
        print("Datos insuficientes para realizar un análisis estadístico válido.\n")
        return 0.0

