# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 18:57:47 2025

@author: JGL
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Aug 27 21:08:55 2025

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
tiempo, x3 = mi_funcion_sen(A0 = 1, offset = 0, fx = ((fs/4) + 0.25) * deltaF, phase = 0, nn = N, fs = fs)
tiempo, x4 = mi_funcion_sen(A0 = 1, offset = 0, fx = ((fs/4) + 0.5) * deltaF, phase = 0, nn = N, fs = fs)


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

#plt.scatter(freqs[:N//2], X1abs[:N//2])
#plt.scatter(freqs[:N//2], X2abs[:N//2])
#plt.scatter(freqs[:N//2], X3abs[:N//2])
#plt.scatter(freqs[:N//2], X4abs[:N//2])

plt.title("FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)


#plt.plot(freqs, np.log10(X2abs) * 20, '--', label = 'X2 abs dB')
#plt.plot(freqs, np.log10(X3abs) * 20, 'x', label = 'X3 abs dB')

plt.plot(freqs, np.log10(X1abs) * 20, label = 'X4 abs dB')

plt.legend()
plt.show()

#%%

varianza = np.var(x2)
print(varianza)


modulo_cuadrado = np.abs(X2) ** 2
#moduloDB = 10 * np.log10(modulo_cuadrado)

plt.figure(2)
plt.plot(freqs, np.log10(modulo_cuadrado) * 10, 'x', label = 'X1 abs dB')
plt.legend()


sumaModulo = np.sum(modulo_cuadrado)
sumaCuadrado = np.sum(np.abs(X2) ** 2)

if sumaModulo == sumaCuadrado:
    print("Se cumple Parseval")
else: 
    print("No se cumple Parseval")
    


##Zero padding

#zeroPadding = np.zeros(100 * N)
zeroPadding1 = np.zeros(10 * N)
#zeroPadding1[9000:10000] = x2 #x1 x1 x1 x1  0 0 0 0 0 0 0 0
zeroPadding1[0:N] = x2 #x1 x1 x1 x1  0 0 0 0 0 0 0 0

#fft_zeroPadding = fft(zeroPadding)
fft_zeroPadding1 = fft(zeroPadding1)



#freqs = np.arange(100 * N) * deltaF
freqs1 = np.arange(10 * N) * deltaF

#freq1 = np.abs(fft_zeroPadding) ** 2


plt.figure()
#plt.plot(freqs, np.log10(fft_zeroPadding)*10, '--',label = 'Zero Padding')
plt.plot(freqs1, np.log10(fft_zeroPadding1)*10, '--',label = 'Zero Padding')

plt.xlim(0, 5*N)
plt.legend()


