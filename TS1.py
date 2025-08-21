# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:36:27 2025

@author: JGL
"""
import matplotlib.pyplot as plt
import numpy as np

# Una señal sinusoidal de 2KHz.
# Misma señal amplificada y desfazada en π/2.
# Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
# Señal anterior recortada al 75% de su potencia.
# Una señal cuadrada de 4KHz.
# Un pulso rectangular de 10ms.
# En cada caso indique tiempo entre muestras, número de muestras y potencia.

fs = 20000
N = 500
f = 2000
deltaF = fs/N
Ts = 1/(N*deltaF)
def mi_funcion_sen(vmax, dc, f, fase, N, fs):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    return tt, xx

def modulacion(vmax, dc, f, fase, N, fs):
    tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
    tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f/2, fase=0, N = N, fs=fs)
    x2 = xx * x1
    return x2


tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f, fase=np.pi/2, N = N, fs=fs)
x2 = modulacion(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)

x3 = np.clip(xx,-0.75,0.75,out=None)

plt.subplot(2,2,1)
plt.plot(tt, xx)
plt.title("2000 Hz")

plt.subplot(2,2,2)
plt.plot(tt, x1)
plt.title("2000 Hz + desfasaje")

plt.subplot(2,2,3)
plt.plot(tt, x2)
plt.title("modulacion")

plt.subplot(2,2,4)
plt.plot(tt, x3)
plt.title("recortada en el 75% de la amplitud ")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()