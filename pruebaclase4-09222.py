# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 22:54:05 2025

@author: JGL
"""
#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Sep  4 20:20:58 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy import signal

def mi_funcion_cos (A0, offset, fx, phase, nn, fs):
    tiempo = np.arange(0, N * Ts, Ts)
    x = A0 * np.cos(2 * np.pi * fx * tiempo)
    return tiempo, x

#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
N = 31 #si pongo 31 queda igual al del holton, si pongo otro N se ve mas colapsado
#fs = 500 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
fs = 31
  ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs / N
Ts = 1 / fs
#fx = 1000

tiempo1, x1 = mi_funcion_cos(A0 = 1, offset = 0, fx = 1 , phase = 0, nn = N, fs = fs/2)


#x_BH = x1 * ventana_BH
ventana_rectangular = np.ones(N)

ventana_BH = signal.windows.blackmanharris(N)

ventana_Hamming = signal.windows.hamming(N)

ventana_Hann = signal.windows.hann(N)

ventana_FT = signal.windows.flattop(N)
Nfft = 2048   

k = np.arange(-Nfft//2, Nfft//2)   # índice entero
freq = k * deltaF 

#plt.figure()
A = fft(ventana_BH, Nfft) / (len(ventana_BH)/2.0)
B = fft(ventana_Hamming, Nfft) / (len(ventana_Hamming)/2.0)
C = fft(ventana_Hann, Nfft )/ (len(ventana_Hann)/2.0)
D = fft(ventana_rectangular, Nfft) / (len(ventana_rectangular)/2.0)
E = fft(ventana_FT, Nfft) / (len(ventana_FT)/2.0)
#freq = np.linspace(-0.5, 0.5, 2048)

responseBH = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
responseHamming = 20 * np.log10(np.abs(fftshift(B / abs(B).max())))
responseHann = 20 * np.log10(np.abs(fftshift(C / abs(C).max())))
responseRect = 20 * np.log10(np.abs(fftshift(D / abs(D).max())))
responseFT = 20 * np.log10(np.abs(fftshift(E / abs(E).max())))

rango = np.abs(freq) <= 2048  # solo los múltiplos de Δf entre -50 y 50

# Gráfico
plt.figure(figsize=(10,5))
#plt.plot(freq[rango], responseBH[rango], color="red", label='Blackman Harris')
plt.plot(freq[rango], responseHamming[rango], color="green", label='Hamming')
plt.plot(freq[rango], responseHann[rango], color="orange", label='Hann')
plt.plot(freq[rango], responseRect[rango], color="blue", label='Rectangular')
#plt.plot(freq[rango], responseFT[rango], color="brown", label='Flattop')
plt.legend()
plt.grid()

plt.title("Ventanas en función de Δf")
plt.ylabel("|W_N(ω)| [dB]")
plt.xlabel("Frecuencia [Hz] (múltiplos de Δf)")
plt.ylim(-100, 5)  # límite en dB para ver mejor

plt.show()