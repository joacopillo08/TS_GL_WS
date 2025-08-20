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

fs = 1000 
N = 1000
f = 2000
def mi_funcion_sen(vmax, dc, f, fase, N, fs):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    return tt, xx
    
tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
plt.plot(tt, xx)
plt.show()
