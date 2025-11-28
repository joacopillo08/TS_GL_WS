# Introduccion

El electrocardiograma (ECG) es una señal biomédica de referencia para el análisis de la actividad eléctrica del corazón. En condiciones de registro reales, especialmente durante movimiento o esfuerzo físico, el ECG se ve afectado por fuentes de ruido que degradan su morfología y dificultan la identificación confiable de las ondas características. Entre estas perturbaciones se destacan la deriva de la línea de base —generalmente asociada a la respiración y a desplazamientos corporales lentos— y los ruidos de mayor frecuencia vinculados a la actividad muscular o al desacople entre electrodos y piel. Estas interferencias pueden alterar el nivel isoeléctrico, distorsionar el complejo QRS e incluso generar falsos picos que confundan a los algoritmos de procesamiento.

A diferencia de los filtros lineales tradicionales utilizados en tareas anteriores, la presente práctica se enfoca en métodos no lineales de eliminación de ruido aplicados directamente sobre la señal de ECG. En particular, se abordan tres estrategias ampliamente utilizadas en el ámbito biomédico:
- Filtro de mediana, capaz de suprimir variaciones lentas de la línea de base sin distorsionar la forma de los complejos QRS.
- Corrección de la línea de base mediante spline cúbico, que interpola segmentos estables del ECG para reconstruir un nivel isoeléctrico suave y continuo.
- Filtro adaptado (matched filter) para la detección de complejos QRS, basado en la correlación con una plantilla que modela la forma típica del latido.

Cada método se implementa sobre el registro real de un ECG provista en el archivo *ecg.mat* brindado por el docente y se analiza su desempeño tanto en la eliminación de artefactos como en la preservación de la información fisiológica relevante.



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
from scipy.interpolate import CubicSpline

##################
# Lectura de ECG #
##################
fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

```

## Eliminación de la línea de base mediante filtro de mediana

La señal de ECG suele presentar una deriva lenta del nivel isoeléctrico causada por movimientos respiratorios, desplazamiento de electrodos y variaciones en la impedancia de contacto. Esta contribución aparece principalmente en bajas frecuencias (por debajo de 0.5–1 Hz), por lo que un método eficaz para su corrección consiste en estimar la envolvente lenta de la señal y sustraerla del registro original.
El filtro de mediana es una herramienta no lineal especialmente adecuada para este propósito. A diferencia de un filtro lineal pasabajos, la mediana no promedia sino que selecciona el valor central de una ventana deslizante. Esto le permite eliminar variaciones lentas sin suavizar o distorsionar los picos abruptos del complejo QRS, cuya morfología debe preservarse.

En esta práctica se aplicaron dos filtros de mediana consecutivos. Uno con una ventana de 200 muestras, que atenúa oscilaciones de baja frecuencia; Y otro de 600 muestras, que capta únicamente la componente más lenta asociada a la respiración y desplazamientos del cuerpo.

El resultado de esta doble filtración constituye una estimación efectiva de la línea de base. Restando dicha estimación del ECG original se obtiene una señal corregida, donde el nivel isoeléctrico se mantiene estable y la morfología del complejo QRS queda preservada.


```python

# Ventanas de mediana
mediana_200 = signal.medfilt(ecg_one_lead, 199)
mediana_600 = signal.medfilt(mediana_200, 599)

# Señal sin la línea de base
ecg_mediana = ecg_one_lead - mediana_600

###################################
# Gráfico 1: Señal + línea de base
###################################
plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead[500000:600000], label='ECG original', linewidth=1)
plt.plot(mediana_600[500000:600000], label='Línea de base estimada (mediana 200→600)', linewidth=2)
plt.title('Estimación de la Línea de Base mediante Filtro Mediana')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [a.u.]')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()

###################################
# Gráfico 2: Señal corregida
###################################
plt.figure(figsize=(12,4))
plt.plot(ecg_mediana[500000:600000], label='ECG filtrado (sin línea de base)', linewidth=1)
plt.plot(ecg_one_lead[500000:600000], alpha=0.4, label='ECG original', linewidth=1)
plt.title('ECG luego del filtrado por mediana')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [a.u.]')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()

plt.show()

##########################################
# Regiones de interés sin ruido claro
##########################################

regs_interes = [4000, 5500],[10_000, 11_000]

for ini, fin in regs_interes:
    zoom = np.arange(ini, fin, dtype='uint')
    
    plt.figure(figsize=(12,4))
    plt.plot(zoom, ecg_one_lead[zoom], label='ECG original', linewidth=1.5)
    plt.plot(zoom, ecg_mediana[zoom], label='ECG filtrado (mediana)', linewidth=1)
    plt.title(f'Región limpia: muestras {ini}–{fin}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()


##########################################
# Regiones con ruido
##########################################

regs_interes_ruido = [
    (np.array([5, 5.2]) * 60 * fs_ecg),
    (np.array([15, 15.2]) * 60 * fs_ecg)
]

for ventana in regs_interes_ruido:
    ini, fin = ventana.astype(int)
    zoom = np.arange(ini, fin, dtype='uint')

    plt.figure(figsize=(12,4))
    plt.plot(zoom, ecg_one_lead[zoom], label='ECG original', linewidth=1.5)
    plt.plot(zoom, ecg_mediana[zoom], label='ECG filtrado (mediana)', linewidth=1)
    plt.title(f'Región con ruido: muestras {ini}–{fin}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()
```


    
![png](output_3_0.png)
    



    
![png](output_3_1.png)
    



    
![png](output_3_2.png)
    



    
![png](output_3_3.png)
    



    
![png](output_3_4.png)
    



    
![png](output_3_5.png)
    


## Corrección de la línea de base mediante spline cúbico

Además del filtrado por mediana, otra estrategia frecuente para eliminar la deriva lenta del ECG consiste en reconstruir explícitamente la línea de base mediante interpolación. La idea es seleccionar puntos de la señal donde se sabe que el ECG se encuentra en un estado relativamente estable y cercano al nivel isoeléctrico, y utilizar esos valores como nodos para generar una curva suave que represente la componente de baja frecuencia.
En esta práctica se empleó un spline cúbico, una función compuesta por segmentos polinomiales de tercer grado que garantiza continuidad en la señal, su derivada y su segunda derivada. Esta suavidad es adecuada para modelar la variación lenta de la línea de base sin introducir oscilaciones artificiales.
Los nodos para el spline se eligieron a partir de las detecciones de QRS provistas por el docente. Para cada pico R se seleccionó una muestra adelantada 80 muestras hacia la izquierda. Ese punto se encuentra típicamente en el segmento PQ o en una región de baja pendiente, donde la influencia del complejo ventricular es mínima. Los valores del ECG en esos nodos se utilizaron como referencia para que el spline trace una envolvente suave que represente la evolución lenta del nivel isoeléctrico a lo largo del registro.
Una vez generada esta curva, se sustrae del ECG original, obteniendo así una señal corregida libre de la deriva respiratoria y de artefactos lentos de movimiento. A diferencia del filtro de mediana, el método del spline produce una línea de base extremadamente suave y continua.



```python

#################################
# Eliminación de línea de base  #
# mediante Spline Cúbico        #
#################################

maximos = mat_struct['qrs_detections'].flatten()
muestras = np.arange(N)

# Nodos para el spline: un poco antes del R (donde la señal es más estable)
nodos = maximos - 80
nodos = nodos[nodos > 0]  # evitar índices negativos

# Spline sobre los nodos
spl = CubicSpline(nodos, ecg_one_lead[nodos])
baseline_spline = spl(np.arange(N))

# Señal corregida
ecg_spline = ecg_one_lead - baseline_spline


########################
# Gráfico general
########################

plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead[500000:600000], label='ECG original', alpha=0.7)
plt.plot(ecg_spline[500000:600000], label='ECG filtrado (spline)', linewidth=1)
plt.plot(baseline_spline[500000:600000], label='Línea de base estimada', linewidth=2)
plt.title('ECG con y sin filtrado por Spline')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()


##########################################
# Regiones de interés sin ruido claro
##########################################

regs_interes = [4000, 5500],[10_000, 11_000]

for ini, fin in regs_interes:
    zoom = np.arange(ini, fin, dtype='uint')
    
    plt.figure(figsize=(12,4))
    plt.plot(zoom, ecg_one_lead[zoom], label='ECG original', linewidth=1.5)
    plt.plot(zoom, ecg_spline[zoom], label='ECG filtrado (spline)', linewidth=1)
    plt.title(f'Región limpia: muestras {ini}–{fin}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()


##########################################
# Regiones con ruido
##########################################

regs_interes_ruido = [
    (np.array([5, 5.2]) * 60 * fs_ecg),
    (np.array([15, 15.2]) * 60 * fs_ecg),
]

for ventana in regs_interes_ruido:
    ini, fin = ventana.astype(int)
    zoom = np.arange(ini, fin, dtype='uint')

    plt.figure(figsize=(12,4))
    plt.plot(zoom, ecg_one_lead[zoom], label='ECG original', linewidth=1.5)
    plt.plot(zoom, ecg_spline[zoom], label='ECG filtrado (spline)', linewidth=1)
    plt.title(f'Región con ruido: muestras {ini}–{fin}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()


###########################
# Comparación baseline vs mediana
###########################

plt.figure(figsize=(12,4))
plt.plot(baseline_spline[400000:700000], label='Spline', linewidth=2)
plt.plot(mediana_600[400000:700000], label='Mediana 600', linewidth=2, alpha=0.7)
plt.title('Comparación de Líneas de Base')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()



```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    



    
![png](output_5_4.png)
    



    
![png](output_5_5.png)
    


## Detección de complejos QRS mediante filtro adaptado (Matched Filter)

La identificación precisa de los complejos QRS es una de las tareas centrales en el análisis de señales de ECG. Para dectar estos, se empleó un filtro adaptado (matched filter), una herramienta clásica en procesamiento de señales que maximiza la relación señal-ruido cuando se conoce la forma aproximada del pulso a detectar. El concepto es simple: si se dispone de una plantilla que representa la forma típica del QRS, su correlación con la señal amplifica aquellos segmentos que se asemejan a dicha plantilla y atenúa los que no.

Para construir la plantilla se utilizó la derivada de una gaussiana, ajustada y centrada de modo que su pico coincida con la ubicación temporal esperada del complejo QRS. Esta elección es razonable porque la derivada de una gaussiana presenta un comportamiento impulsivo y simétrico que se asemeja al rápido ascenso y descenso característico del complejo ventricular. La plantilla se recortó para conservar únicamente la región de mayor energía, reduciendo así el aporte de ruido en la convolución.
El filtrado adaptado se implementó como una convolución entre el ECG y la plantilla invertida en el tiempo, tal como establece la definición del matched filter. El resultado es una señal donde los QRS aparecen como picos bien definidos.



```python
maximos = mat_struct['qrs_detections'].flatten()

QRS = 140
t = np.linspace(-1, 1, QRS)
sigma = 0.2
gauss_derivada = -t * np.exp(-t**2 / (2*sigma**2))


#tengo que mover porque el pico de la gaussiana no esta clavada en el medio de mi # de muestras
max_gauss = np.argmax(gauss_derivada)
mover = QRS//2 - max_gauss
gauss_centrada = np.roll(gauss_derivada, mover)

##doy vuelta mi patron para convolcionar 
h = gauss_centrada[::-1]   
h = h[10:110]

y = np.convolve(ecg_one_lead, h, mode='same')


###################################
# Gráfico de la plantilla
###################################

plt.figure(figsize=(10,3))
plt.plot(h, label="Plantilla (Gauss derivada invertida) recortada")
plt.title("Plantilla usada para el filtro adaptado (Matched Filter)")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()
###################################
# Gráfico de la convolución
###################################

plt.figure(figsize=(14,8))

# Panel 2: zoom a 1000 muestras
plt.subplot(2,1,2)
plt.plot(ecg_one_lead[0:1000], label="ECG", alpha=0.6)
plt.plot(y[0:1000], label="Matched Filter", linewidth=1)
plt.title("Zoom del ECG y la respuesta del matched filter (0–1000 muestras)")
plt.xlabel("Muestras")
plt.grid(True, ls=":")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

```


    
![png](output_7_0.png)
    



    
![png](output_7_1.png)
    


Sobre esta señal se aplicó el algoritmo de búsqueda de máximos (find_peaks) utilizando restricciones en la prominencia y en la distancia mínima entre picos para evitar falsas detecciones y asegurar coherencia fisiológica.

Las posiciones detectadas se compararon luego con las anotaciones de referencia provistas por el docente. A partir de esta comparación se construyó una matriz de confusión que contabiliza verdaderos positivos, falsos positivos y falsos negativos, y se calcularon métricas de desempeño como precisión, sensibilidad y F1-score. Este análisis cuantitativo permite evaluar objetivamente la calidad del detector y determinar en qué medida el filtrado adaptado logra identificar latidos verdaderos aun frente a ruidos residuales o a anotaciones incompletas.


```python
peaks, _ = signal.find_peaks(y,distance=QRS, prominence=np.max(y)*0.39)

###################################
# Matriz de Confusión
###################################

def matriz_confusion_qrs(peaks, maximos, tolerancia_ms=80, fs=1000):
    peaks = np.array(peaks)
    maximos = np.array(maximos)

    tol = int(tolerancia_ms * fs / 1000)

    TP = FP = FN = 0
    emp_peaks = np.zeros(len(peaks), dtype=bool)
    emp_ref   = np.zeros(len(maximos), dtype=bool)

    for i, det in enumerate(peaks):
        diffs = np.abs(maximos - det)
        j = np.argmin(diffs)
        if diffs[j] <= tol and not emp_ref[j]:
            TP += 1
            emp_peaks[i] = True
            emp_ref[j] = True

    FP = np.sum(~emp_peaks)
    FN = np.sum(~emp_ref)

    matriz = np.array([
        [TP, FP],
        [FN, 0]
    ])

    return matriz, TP, FP, FN


matriz, TP, FP, FN = matriz_confusion_qrs(peaks, maximos)

# Métricas
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall    = TP / (TP + FN) if TP + FN > 0 else 0
f1        = 2 * (precision*recall) / (precision + recall) if precision + recall > 0 else 0

###################################
# Gráfico final de detecciones sobre ECG
###################################

plt.figure(figsize=(13,4))
plt.plot(ecg_one_lead, label="ECG", alpha=0.7)
plt.plot(peaks, ecg_one_lead[peaks], "ro", label="Detecciones")
plt.title("QRS detectados")
plt.grid(True, ls=':')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

###################################
# Impresión prolija de resultados
###################################

print("\n==== MATRIZ DE CONFUSIÓN ====\n")
print("           Predicho")
print("           Sí     No")
print(f"Real Sí:  [{TP:4d}  {FN:4d}]")
print(f"Real No:  [{FP:4d}     - ]")

print("\n==== MÉTRICAS ====")
print(f"Precisión     : {precision*100:5.2f}%")
print(f"Sensibilidad  : {recall*100:5.2f}%")
print(f"F1-score      : {f1*100:5.2f}%\n")
```


    
![png](output_9_0.png)
    


    
    ==== MATRIZ DE CONFUSIÓN ====
    
               Predicho
               Sí     No
    Real Sí:  [1900     3]
    Real No:  [   2     - ]
    
    ==== MÉTRICAS ====
    Precisión     : 99.89%
    Sensibilidad  : 99.84%
    F1-score      : 99.87%
    
    


```python

```
