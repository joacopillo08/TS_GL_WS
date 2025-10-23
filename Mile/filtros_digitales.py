#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 20:55:05 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#filtro normalizado -> todas las singularidades en el circulo unitario?
#--- Plantilla de diseño ---

fs = 1000
wp = [1, 35] #freq de corte/paso (rad/s)
ws = [0.1, 40] #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

#Aprox de modulo
#f_aprox = 'butter'
#f_aprox = 'cheby1'
#f_aprox = 'cheby2'
#f_aprox = 'ellip'
f_aprox = 'cauer'

#Aprox fase
#f_aprox = 'bessel'

# --- Diseño de filtro analogico ---

mi_sos = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q

# %%

# --- Respuesta en frecuencia ---
w, h= signal.freqz_sos(mi_sos, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

z, p, k = signal.sos2zpk(mi_sos) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.plot(w, np.degrees(fase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
