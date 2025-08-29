# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 18:57:47 2025

@author: JGL
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Aug 27 21:08:55 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def mi_funcion_sen (A0, offset, fx, phase, nn, fs):
    tiempo = np.arange(0, nn* Ts, Ts)
    x = A0 * np.sin(2 * np.pi * fx * tiempo)
    return tiempo, x

#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
fs = 1000 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = 1000   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs / N
Ts = 1/fs
#fx = 1000
freqs = np.fft.fftfreq(N, Ts)   # eje de frecuencias en Hz


# grilla temporal

tiempo, x1 = mi_funcion_sen(A0 = 1, offset = 0, fx = 2, phase = 0, nn = N, fs = fs)
tiempo, x2 = mi_funcion_sen(A0 = 1, offset = 0, fx = fs/4 * deltaF, phase = 0, nn = N, fs = fs)
tiempo, x3 = mi_funcion_sen(A0 = 1, offset = 0, fx = ((fs/4) + 1) * deltaF, phase = 0, nn = N, fs = fs)
tiempo, x4 = mi_funcion_sen(A0 = 1, offset = 0, fx = ((fs/4) + 0.1) * deltaF, phase = 0, nn = N, fs = fs)


X1 = fft(x1)
X1abs = np.abs(X1)
X1ang = np.angle(X1)

X2 = fft(x2)
X2abs = np.abs(X2)
X2ang = np.angle(X2)

X3 = fft(x3)
X3abs = np.abs(X3)
X3ang = np.angle(X3)

X4 = fft(x4)
X4abs = np.abs(X4)
X4ang = np.angle(X4)

# Graficar solo hasta N/2
plt.figure()

plt.scatter(freqs[:N//2], X1abs[:N//2])
plt.scatter(freqs[:N//2], X2abs[:N//2])
plt.scatter(freqs[:N//2], X3abs[:N//2])
#plt.scatter(freqs[:N//2], X4abs[:N//2])

plt.title("FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)


plt.plot(freqs, np.log10(X2abs) * 20, 'x', label = 'X2 abs dB')
plt.plot(freqs, np.log10(X3abs) * 20, 'x', label = 'X3 abs dB')
plt.plot(freqs, np.log10(X4abs) * 20, 'x', label = 'X4 abs dB')

plt.legend()
plt.show()

# =============================================================================
# ejercicio nuevo clase 28/08
# =============================================================================
#Punto 1 de la varianza

_, x5 = mi_funcion_sen(A0 = np.sqrt(2), offset = 0, fx = fs/4 * deltaF, phase = 0, nn = N, fs = fs)
X5 = fft(x5)
X5abs = np.abs(X5)
X5ang = np.angle(X5)


var_x5 = np.var(x5)
print("Varianza:", var_x5)

#Punto 2 del modulo cuadrado, uso X2 porque necesito que este en frecuencia 
modulo_cuadrado = np.abs(X5)**2  
plt.figure()
plt.plot(freqs, 10 * np.log10(modulo_cuadrado), 'x', label='X5 |X|^2 dB')



sumaModulo = np.sum(modulo_cuadrado)
sumaCuadrado = np.sum(x5 ** 2)

if sumaModulo == sumaCuadrado:
    print("Se cumple Parseval")
else: 
    print("No se cumple Parseval")



