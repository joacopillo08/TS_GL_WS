#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:46:52 2025

@author: milenawaichnan
"""

import matplotlib.pyplot as plt
import numpy as np
#import scipy as sc
from scipy import signal
from numpy.fft import fft

fs = 1000 # frecuencia de muestreo, aplica para todas las señales
N = fs  # cantidad de muestras, aplica para todas las señales 
#fx = 2000 #frecuencia para las senoidales
Ts = 1/fs
deltaF = 2*np.pi/N
fr = 0

A0 = np.sqrt(2)

omega0 = fs / 4
omega1 = omega0 + (fr * deltaF)



SNRdb = np.random.uniform(3, 10)
sigmaCuad = 10**(-SNRdb/10)
print("varianza = ", sigmaCuad)



n = np.arange(N) * Ts
s = A0 * np.sin(omega1 * n)
varS = np.var(s)
print("Var s: ", varS)

Na = np.random.normal(0, sigmaCuad, N) #ruido
#Na = 3
#print(Na)
varNa = np.var(Na)
print("Var Na: ", varNa)

x = s + Na
varX = np.var(x)
print("Var x: ", varX)


X_fft= fft(x)
Xabs = np.abs(x)
Xang = np.angle(x)
Xcuad = Xabs ** 2

freqs = np.arange(N) * deltaF
plt.figure()
plt.plot(freqs, np.log10(Xcuad) * 10, label = 'X1 abs dB')
plt.legend()
  

