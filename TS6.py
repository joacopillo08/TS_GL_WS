# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:01:53 2025

@author: Milena
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================
# ===  Ensayo de tres funciones bicuadráticas (T1, T2, T3) ===
# ============================================================

# Diccionario con cada filtro (num, den)
filtros = {
    "T1(s) = (s² + 9)/(s² + √2·s + 1)": ([1, 0, 9], [1, np.sqrt(2), 1]),
    "T2(s) = (s² + 1/9)/(s² + s 1/5 + 1)": ([1, 0, 1/9], [1, 1/5, 1]),
    "T3(s) = (s² + s 1/5 + 1)/(s² + √2 s + 1)": ([1, 1/5, 1], [1, np.sqrt(2), 1])
}

# ============================================================
# ===  Gráficos para cada función transferencia  ============
# ============================================================

for f_nombre, (b, a) in filtros.items():
    
    # --- Respuesta en frecuencia ---
    w, h = signal.freqs(b=b, a=a, worN=np.logspace(-1, 2, 1000))
    fase = np.unwrap(np.angle(h))
    gd = -np.diff(fase) / np.diff(w)
    
    # --- Polos y ceros ---
    z, p, k = signal.tf2zpk(b, a)
    
    # --- Gráficas ---
    plt.figure(figsize=(12, 10))
    plt.suptitle(f_nombre, fontsize=14, fontweight='bold')
    
    # Magnitud
    plt.subplot(2, 2, 1)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [rad/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    
    # Fase
    plt.subplot(2, 2, 2)
    plt.semilogx(w, np.degrees(fase))
    plt.title('Fase')
    plt.xlabel('Pulsación angular [rad/s]')
    plt.ylabel('Fase [°]')
    plt.grid(True, which='both', ls=':')
    
    # Retardo de grupo
    plt.subplot(2, 2, 3)
    plt.semilogx(w[:-1], gd)
    plt.title('Retardo de Grupo')
    plt.xlabel('Pulsación angular [rad/s]')
    plt.ylabel('τg [s]')
    plt.grid(True, which='both', ls=':')
    
    # Diagrama de polos y ceros
    plt.subplot(2, 2, 4)
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.title('Diagrama de Polos y Ceros (plano s)')
    plt.xlabel('σ [rad/s]')
    plt.ylabel('jω [rad/s]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
