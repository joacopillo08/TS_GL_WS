# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:27:54 2025

@author: Milena
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#filtro normalizado -> todas las singularidades en el circulo unitario?
#--- Plantilla de diseño ---

wp = 1 #freq de corte/paso (rad/s)
ws = 5 #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

#Aprox de modulo
#f_aprox = 'butter'
f_aprox = 'cheby1'
#f_aprox = 'cheby2'
#f_aprox = 'ellip'
#f_aprox = 'cauer'

#Aprox fase
#f_aprox = 'bessel'

# --- Diseño de filtro analogico ---

b, a = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = True, ftype = f_aprox, output ='ba') #devuelve dos listas de coeficientes, b para P y a para Q

# %%

# --- Respuesta en frecuencia ---
w, h= signal.freqs(b = b, a = a, worN= np.logspace(-1, 2, 1000)) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

gd = -np.diff(fase) / np.diff(w)

z, p, k = signal.tf2zpk(b, a) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(fase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%

sos = signal.tf2sos(b, a, analog = True)





