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
freqs = np.fft.fftfreq(N, Ts)  

def mi_funcion_sen(f, N, fs, a0=1, fase=0):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    x = a0 * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return tt, x, var_x

tt, x1, var_x = mi_funcion_sen( f = fs/4 * deltaF, N = N, fs  = fs )
tt, x2, var_x = mi_funcion_sen( f = (fs/4 + 0.25) * deltaF ,  N = N, fs  = fs )
tt, x3, var_x = mi_funcion_sen( f = (fs/4 + 0.5) * deltaF,  N = N, fs  = fs )



X1 = fft(x1)
X1abs = 1/N * np.abs(X1)

X2 = fft(x2)
X2abs = 1/N * np.abs(X2)

X3 = fft(x3)
X3abs =  1/N * np.abs(X3)

#Densidad espectral de potencia. Paso a db y ademas elevo al cuadrado. Le multplico por dos por algo de la aporximacion de la fft para que me quede bien la mediciion
#graficos
plt.figure()
plt.title("FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Db")
plt.grid(True)

plt.plot(freqs, X1abs ,"o", label = "") ##densidad espectral de potencia

plt.plot(freqs, 10*np.log10(2* X1abs**2) ,"--", label = 'dens esp pot fs/4') ##densidad espectral de potencia
plt.plot(freqs, 10*np.log10(2* X2abs**2) ,"--", label = 'dens esp pot fs/4 + 0.25') ##densidad espectral de potencia
plt.plot(freqs, 20*np.log10(2*X3abs**2) ,"--", label = 'dens esp pot fs/4 + 0.5') ##densidad espectral de potencia
#plt.xlim((0,fs/2))
plt.legend()


pot_tiempo1 = 1/N*np.sum(np.abs(x1)**2)
pot_tiempo2 = 1/N*np.sum(np.abs(x2)**2)
pot_tiempo3 = 1/N*np.sum(np.abs(x3)**2)
pot_frec1 = 1/N**2*np.sum(np.abs(X1)**2)
pot_frec2 = 1/N**2*np.sum(np.abs(X2)**2)
pot_frec3 = 1/N**2*np.sum(np.abs(X3)**2)
    
if np.isclose(pot_tiempo1, pot_frec1, rtol=1e-10, atol=1e-12):
    print("Se cumple Parseval para x1 y X1")
else: 
    print("No se cumple Parseval para x1 y X1")

if np.isclose(pot_tiempo2, pot_frec2, rtol=1e-10, atol=1e-12):
    print("Se cumple Parseval para x2 y X2")
else: 
    print("No se cumple Parseval para x2 y X2")

if np.isclose(pot_tiempo3, pot_frec3, rtol=1e-10, atol=1e-12):
    print("Se cumple Parseval para x3 y X3")
else: 
    print("No se cumple Parseval para x3 y X3")



# Notar que a cada senoidal se le agrega una pequeña desintonía respecto a  Δ𝑓
# . Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál es el efecto de dicha desintonía en el espectro visualizado.

# b) Verificar la potencia unitaria de cada PSD, puede usar la identidad de Parseval. En base a la teoría estudiada. Discuta la razón por la cual una señal senoidal tiene un espectro tan diferente respecto a otra de muy pocos Hertz de diferencia. 

# c) Repetir el experimento mediante la técnica de zero padding. Dicha técnica consiste en agregar ceros al final de la señal para aumentar Δ𝑓
#  de forma ficticia. Probar agregando un vector de 9*N ceros al final. Discuta los resultados obtenidos.

##Zero padding

zeroPadding1 = np.zeros(10 * N)
zeroPadding2 = np.zeros(10 * N)
zeroPadding3 = np.zeros(10 * N)

zeroPadding1[0:N] = x1 #x1 x1 x1 x1  0 0 0 0 0 0 0 0
zeroPadding2[0:N] = x2 #x1 x1 x1 x1  0 0 0 0 0 0 0 0
zeroPadding3[0:N] = x3 #x1 x1 x1 x1  0 0 0 0 0 0 0 0

fft_zeroPadding1 = fft(zeroPadding1)
fft_zeroPadding2 = fft(zeroPadding2)
fft_zeroPadding3 = fft(zeroPadding3)

fft_zeroPadding1abs = 1/N * np.abs(fft_zeroPadding1)
fft_zeroPadding2abs = 1/N * np.abs(fft_zeroPadding2)
fft_zeroPadding3abs = 1/N * np.abs(fft_zeroPadding3)


freqs1 = np.arange(10 * N) * (deltaF/10)

plt.plot(freqs1, fft_zeroPadding1, 'o',label = 'Zero Padding con fs/4')
plt.plot(250, np.max(fft_zeroPadding1abs), 'o',label = 'Zero Padding con fs/4')


#freq1 = np.abs(fft_zeroPadding) ** 2
plt.figure(figsize=(10,8))
# Subplot 1
plt.subplot(3, 1, 1)
plt.plot(freqs1, np.log10(fft_zeroPadding1abs**2)*10, '--',label = 'Zero Padding con fs/4')
plt.title('Zero padding: fs/4')
plt.ylabel('dB')
plt.grid(True)
plt.legend()
plt.xlim(2200, 2800) # zoom al lóbulo principal

# Subplot 2
plt.subplot(3, 1, 2)
plt.plot(freqs1, np.log10(fft_zeroPadding2abs**2)*10, '--',label = 'Zero Padding con fs/4 + 0.25')
plt.title('Zero padding: fs/4 + 0.25 Δf')
plt.ylabel('dB')
plt.grid(True)
plt.legend()
plt.xlim(2200, 2800) # zoom al lóbulo principal

# Subplot 3
plt.subplot(3, 1, 3)
plt.plot(freqs1, np.log10(fft_zeroPadding3abs**2)*10, '--',label = 'Zero Padding con fs/4 + 0.5')
plt.title('Zero padding: fs/4 + 0.5 Δf')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('dB')
plt.grid(True)
plt.legend()
plt.xlim(2200, 2800) # zoom al lóbulo principal

plt.tight_layout()
plt.show()

Npad = 10*N
eps = 1e-20

# FFT lado único (0..fs/2) y eje de frec correcto
X = np.fft.rfft(zeroPadding2, n=Npad)                 # para el caso fs/4 + 0.25 Δf
fp = np.fft.rfftfreq(Npad, d=1/fs)                    # 0 .. fs/2 (en Hz)

# Magnitud en dB (sin -inf)
Z = 10*np.log10(np.abs(X)**2 + eps)

plt.figure(figsize=(8,4))
plt.plot(fp, Z, '--', label='Zero Padding con fs/4 + 0.25 Δf')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('dB')
plt.grid(True)
plt.legend()
plt.show()












