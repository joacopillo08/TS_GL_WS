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

fs = 4000 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = 1000   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs/N
Ts = 1/(N*deltaF)

tiempo = np.arange(0, N * Ts, Ts)
fx = 1

x = np.sin(2 * np.pi * fx * tiempo)
plt.plot(tiempo,x)
plt.show


