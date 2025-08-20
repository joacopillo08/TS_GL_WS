# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:11:39 2025

@author: JGL
"""

#aca hago la senoideal normal la comunacha
import matplotlib.pyplot as plt
import numpy as np

#n = np.arange(0,2*np.pi, 0.1)
#x = np.sin(n)

fs = 1000 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = 1000   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs/N
Ts = 1/(N*deltaF)

tiempo = np.arange(0, N * Ts, Ts)
fx =1001
A = 1
x = A * np.sin(2 * np.pi * fx * tiempo)
plt.plot(tiempo,x)
plt.show

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    Ts = 1/fs
    tt = np.arange(nn) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)  # señal senoidal
    return tt, xx
    
tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=1001, ph=0, nn=N, fs=fs)
plt.plot(tt, xx)
plt.show()

