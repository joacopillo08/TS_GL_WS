
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 17:27:32 2025

@author: JGL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal




fs = 1000


def resp_en_w(b,a,fs = 1000):
    w, h= signal.freqz(b = b, a = a, worN = 1024, fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

    fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

    #w_rad = w * (fs / (2 * np.pi))
    gd = -np.diff(fase) / np.diff(2*np.pi*w) * fs   # τg [muestras]

    z, p, k =  signal.tf2zpk(b, a) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k
    
        # --- Gráficas ---
    #plt.figure(figsize=(12,10))
    
    # Magnitud
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(w, 20*np.log10(abs(h)))
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    
    # Fase
    plt.subplot(2,2,2)
    plt.plot(w, fase)
    plt.title('Fase')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    
    # Retardo de grupo
    plt.subplot(2,2,3)
    plt.plot(w[:-1], gd)
    plt.title('Retardo de Grupo ')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

b1 = ([1,1,1,1])
a1 = ([1])

b2 = ([1,1,1,1,1])
a2 = ([1])

b3 = ([1,-1])
a3 = ([1])

b4 = ([1,0,-1])
a4 = ([1])

funcionA = resp_en_w(b = b1, a = a1)
funcionB = resp_en_w(b = b2, a = a2)
funcionC = resp_en_w(b = b3, a = a3)
funcionD = resp_en_w(b = b4, a = a4)
