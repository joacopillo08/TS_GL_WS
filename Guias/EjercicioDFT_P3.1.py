# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:06:37 2025

@author: Milena
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ===== Parámetros =====
N = 8
n = np.arange(N)

# Señal: x[n] = 4 + 3 sin(pi*n/2)
x = 4 + 3 * np.sin(np.pi * n / 2)
# x =  3 * np.sin(np.pi * n / 2)

# ===== DFT (N puntos) =====
X = np.fft.fft(x, n=N)
k = np.arange(N)
omega_k = 2 * np.pi * k / N  # radianes

# ===== Impresiones útiles =====
print("x[n]:", x)
print("X[k] (complejo):")
for kk in range(N):
    print(f"k={kk:>2d}: {X[kk]: .3f}")

print("\n|X[k]| (magnitud):", np.abs(X))
print("∠X[k] (fase, rad):", np.angle(X))

# ===== Gráficos =====
plt.figure(figsize=(10, 8))

# 1) x[n] en tiempo
plt.subplot(3,1,1)
plt.stem(n, x)#, basefmt=" ")#, use_line_collection=True)
plt.title("Señal en tiempo: x[n] = 4 + 3 sin(π n / 2)")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

# 2) Magnitud |X[k]|
plt.subplot(3,1,2)
plt.stem(k, np.abs(X) )#, basefmt=" ")#, use_line_collection=True)
plt.title("DFT (N=8): Magnitud |X[k]|")
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.grid(True)

# 3) Fase ∠X[k]
plt.subplot(3,1,3)
plt.stem(k, np.angle(X) )#, basefmt=" ")#, use_line_collection=True)
plt.title("DFT (N=8): Fase ∠X[k] (rad)")
plt.xlabel("k")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

# ===== Verificación teórica (opcional) =====
# Deberías ver: X[0]=32, X[2]=-12j, X[6]=+12j, otros = 0 (≈ numéricamente)
