# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:35:21 2025

@author: JGL
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy 
from scipy.signal import lfilter
import TS1viejo



# =============================================================================
# Ejercicio 1
# HECHO --> Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. Considere que las mismas son causales.
# HECHO --> Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de salida para alguna de las señales de entrada consideradas en el punto anterior.
# En cada caso indique la frecuencia de muestreo, el tiempo de simulación y la potencia o energía de la señal de salida.
# =============================================================================
fs = 1000
N = 500
f = 2
Ts = 1/fs
tt = np.arange(N) * Ts          
deltaF = fs/N
Ts = 1/fs
tt = np.arange(len(TS1.xx)) * Ts

def en_diferencias(N,x):
    y = np.zeros(N)
    for n in range (N):
        x0 = x[n]
        x1 = x[n-1] if n-1 >= 0 else 0
        x2 = x[n-2] if n-2 >= 0 else 0
        y1 = y[n-1] if n-1 >= 0 else 0
        y2 = y[n-2] if n-2 >= 0 else 0
        y[n] = 3* 10**(-2)*x0 + 5 * 10**(-2)*x1 +  3 * 10**(-2)*x2 + 1.5*y1-0.5*y2
    return y

def plot_salida(x, nombre):
    y = en_diferencias(N = len(x), x = x)
    tt = np.arange(len(x)) * Ts
    plt.figure()
    plt.plot(tt, y, label="y salida")
    plt.plot(tt, x,color='green', label="x entrada")
    plt.legend()
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(f"Respuesta a {nombre}")
    plt.show()

plot_salida(TS1.xx, "xx")
plot_salida(TS1.x1, "x1")
plot_salida(TS1.x2, "x2")
plot_salida(TS1.x3, "x3")
plot_salida(TS1.x4, "x4")

#tt, TS1.xx = TS1.mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)

delta = np.zeros(N)
delta[0] = 1

h = en_diferencias(N = N, x = delta)
y_conv = np.convolve(TS1.xx, h)[:N] 

plt.figure(1)
plt.plot(tt, y_conv, "--",color='red' , label="y conv")
plt.legend()

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("2 Hz")
plt.show()

#------------------------------------
"""
2) Hallar la respuesta al impulso y la salida correspondiente a una señal de entrada senoidal en los sistemas definidos mediante las siguientes ecuaciones en diferencias:

y[n]=x[n]+3⋅x[n−10]
y[n]=x[n]+3⋅y[n−10]

"""

def en_diferencias2(N,x):
    y2 = np.zeros(N)
    for n in range (N):
        x0 = x[n]
        x10 = x[n-10] if n-10 >= 0 else 0
        y2[n]=x0+3*x10
    return y2

def en_diferencias3(N,x):
    y3 = np.zeros(N)
    for n in range (N):
        x0 = x[n]
        y10 = y3[n-10] if n-10 >= 0 else 0
        y3[n]=x0+3*y10
    return y3

y2 = en_diferencias2(N = len(TS1.xx), x = TS1.xx)
h2 = en_diferencias2(N = N, x = delta)
y_conv2 = np.convolve(TS1.xx, h2)[:N] 

plt.figure()
plt.plot(tt, y_conv2, "o",color='green' , label="y conv")
plt.plot(tt, y2, label="y salida")
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("2 Hz")
plt.show()

y3 = en_diferencias3(N = len(TS1.xx), x = TS1.xx)
h3 = en_diferencias3(N = N, x = delta)
y_conv3 = np.convolve(TS1.xx, h3)[:N] 

plt.figure()
plt.plot(tt, y_conv3, "o",color='green' , label="y conv")
plt.plot(tt, y3, label="y salida")
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("2 Hz")
plt.show()



























