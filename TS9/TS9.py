#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:05:59 2025

@author: milenawaichnan
"""

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

# Ventanas de mediana
mediana_200 = signal.medfilt(ecg_one_lead, 199)
mediana_600 = signal.medfilt(mediana_200, 599)

# Señal sin la línea de base
ecg_mediana = ecg_one_lead - mediana_600

###################################
# Gráfico 1: Señal + línea de base
###################################
plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead[400000:700000], label='ECG original', linewidth=1)
plt.plot(mediana_600[400000:700000], label='Línea de base estimada (mediana 200→600)', linewidth=2)
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
plt.plot(ecg_mediana[400000:700000], label='ECG filtrado (sin línea de base)', linewidth=1)
plt.plot(ecg_one_lead[400000:700000], alpha=0.4, label='ECG original', linewidth=1)
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

# %%


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
plt.plot(ecg_one_lead[400000:700000], label='ECG original', alpha=0.7)
plt.plot(ecg_spline[400000:700000], label='ECG filtrado (spline)', linewidth=1)
plt.plot(baseline_spline[400000:700000], label='Línea de base estimada', linewidth=2)
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


# %%


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

peaks, _ = signal.find_peaks(y,distance=QRS, prominence=np.max(y)*0.39)

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
plt.legend()

plt.tight_layout()
plt.show()


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

###################################
# Gráfico final de detecciones sobre ECG
###################################

plt.figure(figsize=(13,4))
plt.plot(ecg_one_lead, label="ECG", alpha=0.7)
plt.plot(peaks, ecg_one_lead[peaks], "ro", label="Detecciones")
plt.title("QRS detectados")
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()





