# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 19:40:40 2025

@author: JGL
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

#Aca uso todo lo mismo que en la ts1. Tambien podria importar el archivo pero se le va a complicar a los profes.
fs = 1000
N = 1000
f = 2000
Ts = 1/fs
deltaF = fs/N
tt = np.arange(N)*Ts
freqs = np.fft.fftfreq(N, Ts)   # eje de frecuencias en Hz

def mi_funcion_sen(f, N, fs, a0=1, fase=0):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    x = a0 * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return tt, x, var_x

tt, x1, var_x = mi_funcion_sen( f = fs/4, N = N, fs  = fs )
tt, x2, var_x = mi_funcion_sen( f = fs/4 + 0.25,  N = N, fs  = fs )
tt, x3, var_x = mi_funcion_sen( f = fs/4 + 0.5,  N = N, fs  = fs )


X1 = fft(x1)
X1abs = np.abs(X1)

X2 = fft(x2)
X2abs = np.abs(X2)

X3 = fft(x3)
X3abs = np.abs(X3)

#Densidad espectral de potencia. Paso a db y ademas elevo al cuadrado. Le multplico por dos por algo de la aporximacion de la fft para que me quede bien la mediciion

#graficos
plt.figure()
plt.title("FFT")
plt.xlabel("Muestras ")
plt.ylabel("Db")
plt.grid(True)
plt.plot(freqs, 10*np.log10(2* X1abs**2) , label = 'dens esp pot fs/4') ##densidad espectral de potencia
plt.plot(freqs, 10*np.log10(2* X2abs**2) , label = 'dens esp pot fs/4 + 0.25') ##densidad espectral de potencia
plt.plot(freqs, 10*np.log10(2* X3abs**2) , label = 'dens esp pot fs/4 + 0.5') ##densidad espectral de potencia
#plt.xlim((0,fs/2))
plt.legend()

pot_tiempo1 = 1/N*np.sum(np.abs(x1)**2)
pot_tiempo2 = 1/N*np.sum(np.abs(x2)**2)
pot_tiempo3 = 1/N*np.sum(np.abs(x3)**2)
pot_frec1 = 1/N**2*np.sum(np.abs(X1)**2)
pot_frec2 = 1/N**2*np.sum(np.abs(X2)**2)
pot_frec3 = 1/N**2*np.sum(np.abs(X3)**2)
    
if pot_tiempo1 == pot_frec1:
    print("Se cumple Parseval para x1 y X1")
else: 
    print("No Se cumple Parseval para x1 y X1")
    
if pot_tiempo2 == pot_frec2:
    print("Se cumple Parseval para x2 y X2")
else: 
    print("No Se cumple Parseval para x2 y X2")

if pot_tiempo3 == pot_frec3:
    print("Se cumple Parseval para x3 y X3")
else: 
    print("No Se cumple Parseval para x3y X3")




