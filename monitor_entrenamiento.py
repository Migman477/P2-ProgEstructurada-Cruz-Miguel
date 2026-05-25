import datetime
import math
import random
import statistics
import sys

def gestionar_tiempo():
    """
    Maneja la telemetría del tiempo de la simulación del entrenamiento.
    
    Utiliza la biblioteca datetime para:
    1. Obtener la fecha y hora de inicio.
    2. Formatear la fecha en un formato legible (DD/MM/YYYY HH:MM:SS).
    3. Simular un delta de tiempo y calcular la duración total del proceso.
    
    Retorna:
        datetime.timedelta: La diferencia de tiempo simulada entre inicio y fin.
    """
    print("--- Gestión del Tiempo ---")
    
    # Obtener y formatear fecha de inicio
    tiempo_inicio = datetime.datetime.now()
    formato_fecha = "%d/%m/%Y %H:%M:%S"
    inicio_formateado = tiempo_inicio.strftime(formato_fecha)
    print(f"Inicio de la simulación: {inicio_formateado}")
    
    # Simular avance del tiempo
    delta_simulado = datetime.timedelta(hours=1, minutes=45, seconds=30)
    tiempo_fin = tiempo_inicio + delta_simulado
    fin_formateado = tiempo_fin.strftime(formato_fecha)
    print(f"Fin de la simulación: {fin_formateado}")
    
    # Calcular y mostrar duración total
    diferencia_tiempo = tiempo_fin - tiempo_inicio
    print(f"Duración total del proceso: {diferencia_tiempo}\n")
    
    return diferencia_tiempo

def calcular_metricas_error(lista_errores_crudos, epochs_flotante):
    """
    Realiza cálculos matemáticos sobre los errores simulados.
    
    Utiliza la biblioteca math para:
    1. Redondear hacia arriba (ceil) la cantidad de epochs.
    2. Elevar al cuadrado (pow) los errores.
    3. Obtener la raíz cuadrada (sqrt) para el cálculo de RMSE.
    4. Usar valor absoluto (fabs) para encontrar el error máximo.
    
    Argumentos:
        lista_errores_crudos (list): Lista de valores numéricos representando errores.
        epochs_flotante (float): Valor decimal de epochs estimados.
        
    Retorna:
        tuple: Un par con el valor RMSE (float) y la cantidad de epochs enteros (int).
    """
    print("--- Cálculos Matemáticos ---")
    
    # Validación segura para evitar división por cero en las medias
    if len(lista_errores_crudos) == 0:
        print("Advertencia: No hay errores en la lista para calcular métricas.")
        return 0.0, 0
    
    # Redondeo de epochs
    epochs_totales = math.ceil(epochs_flotante)
    print(f"Epochs estimados: {epochs_flotante} -> Redondeado a enteros: {epochs_totales}")
    
    # Procesar errores al cuadrado
    errores_cuadrado = []
    for error in lista_errores_crudos:
        error_cuadrado = math.pow(error, 2)
        errores_cuadrado.append(error_cuadrado)
    
    # Sumatoria y media manuales para ciclo demostrativo
    suma_errores = 0.0
    for error_sq in errores_cuadrado:
        suma_errores += error_sq
    media_error_cuadrado = suma_errores / len(errores_cuadrado)
    
    # Cálculo final RMSE
    rmse = math.sqrt(media_error_cuadrado)
    
    # Cálculo de valor absoluto máximo
    error_maximo_absoluto = math.fabs(lista_errores_crudos[0])
    for error in lista_errores_crudos:
        valor_absoluto = math.fabs(error)
        if valor_absoluto > error_maximo_absoluto:
            error_maximo_absoluto = valor_absoluto
            
    print(f"Métrica RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"Error absoluto máximo observado: {error_maximo_absoluto:.4f}\n")
    
    return rmse, epochs_totales

def simular_entrenamiento(epochs):
    """
    Simula iterativamente los valores de un proceso de entrenamiento.
    
    Utiliza la biblioteca random para:
    1. Generar fluctuación del loss y latencias (uniform).
    2. Simular probabilidad de éxito (random).
    3. Seleccionar un evento de log aleatorio (choice).
    
    Argumentos:
        epochs (int): Cantidad de iteraciones a simular.
        
    Retorna:
        tuple: (historial_loss, latencias) ambas son listas de floats.
    """
    print("--- Simulación Estocástica ---")
    
    eventos_log = [
        "Epoch exitoso",
        "Gradiente inestable",
        "Actualización de pesos",
        "Tasa de aprendizaje ajustada"
    ]
    
    historial_loss = []
    latencias = []
    
    # Ciclo principal de simulación
    for epoch in range(1, epochs + 1):
        # Generación de métricas aleatorias
        loss_actual = random.uniform(0.1, 2.8)
        latencia_ms = random.uniform(20.0, 150.0)
        
        historial_loss.append(loss_actual)
        latencias.append(latencia_ms)
        
        # Probabilidad y condicionales de estado
        probabilidad_exito = random.random()
        
        if probabilidad_exito < 0.20:
            estado = "Advertencia de Rendimiento"
        else:
            estado = "Exitoso"
            
        # Selección de texto aleatorio
        evento_seleccionado = random.choice(eventos_log)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {loss_actual:.4f} | Estado: {estado} | Log: '{evento_seleccionado}'")
        
    print()
    return historial_loss, latencias

def analizar_resultados(historial_loss, latencias):
    """
    Analiza estadísticamente los resultados generados en la simulación.
    
    Utiliza la biblioteca statistics para:
    1. Calcular la media (mean) del historial de pérdidas.
    2. Calcular la desviación estándar (stdev) para medir estabilidad.
    3. Encontrar la mediana (median) de los tiempos de latencia.
    
    Argumentos:
        historial_loss (list): Lista con los valores de error de cada iteración.
        latencias (list): Lista con los tiempos de respuesta de cada iteración.
        
    Retorna:
        float: El valor de la media de la pérdida, o 0.0 si los datos son insuficientes.
    """
    print("--- Análisis de Rendimiento ---")
    
    # Validación estricta para evitar errores estadísticos con datos nulos o unitarios
    if len(historial_loss) > 1 and len(latencias) > 0:
        media_loss = statistics.mean(historial_loss)
        desviacion_loss = statistics.stdev(historial_loss)
        mediana_latencia = statistics.median(latencias)
        
        print(f"Media de pérdida (Loss): {media_loss:.4f}")
        print(f"Desviación estándar de la pérdida (Estabilidad): {desviacion_loss:.4f}")
        print(f"Mediana de latencia: {mediana_latencia:.2f} ms\n")
        
        return media_loss
    else:
        print("Datos insuficientes para realizar el análisis estadístico válido.\n")
        return 0.0

def interaccion_so(media_loss):
    """
    Toma decisiones a nivel de sistema dependiendo de las métricas.
    
    Utiliza la biblioteca sys para:
    1. Mostrar información del sistema operativo (platform, version, getfilesystemencoding).
    2. Medir tamaño en memoria (getsizeof).
    3. Forzar salida del programa con código de estado (exit).
    
    Argumentos:
        media_loss (float): El valor promedio del error para decidir la salida.
    """
    print("--- Interacción con el Sistema ---")
    
    # Información informativa de la plataforma
    print(f"Plataforma del sistema: {sys.platform}")
    print(f"Versión de Python: {sys.version.split()[0]}")
    print(f"Codificación del sistema de archivos: {sys.getfilesystemencoding()}")
    
    # Tamaño de variable
    tamanio_variable = sys.getsizeof(media_loss)
    print(f"Tamaño en memoria de la variable 'media_loss': {tamanio_variable} bytes")
    
    # Condicional de salida basado en métrica de éxito
    limite_tolerancia_loss = 1.6
    
    if media_loss > limite_tolerancia_loss:
        print(f"ALERTA CRÍTICA: La pérdida media ({media_loss:.4f}) superó el límite de {limite_tolerancia_loss}.")
        print("Forzando salida no exitosa del programa...")
        sys.exit(1)
    else:
        print(f"Entrenamiento completado dentro de los límites estables ({media_loss:.4f} <= {limite_tolerancia_loss}).")
        print("Saliendo exitosamente...")
        sys.exit(0)

def main():
    """
    Función principal que orquesta el flujo de ejecución del monitor.
    Llama secuencialmente a los componentes sin uso de excepciones.
    """
    print("="*60)
    print("MONITOR DE ENTRENAMIENTO DE IA (SIMULACIÓN)")
    print("="*60 + "\n")
    
    # 1. Ejecución de telemetría inicial
    gestionar_tiempo()
    
    # Datos semilla
    errores_de_prueba = [-1.5, 0.8, -0.2, 2.1, -0.9, 0.5]
    estimacion_epochs = 10.4
    
    # 2. Cálculos estáticos
    _, epochs = calcular_metricas_error(errores_de_prueba, estimacion_epochs)
    
    # Flujo limpio: Asegurar que los epochs sean positivos
    if epochs <= 0:
        epochs = 1
        
    # 3. Ciclos de simulación
    historial_loss, latencias = simular_entrenamiento(epochs)
    
    # 4. Análisis de resultados generados
    media_loss = analizar_resultados(historial_loss, latencias)
    
    # 5. Cierre y reporte del sistema
    interaccion_so(media_loss)

if __name__ == "__main__":
    main()