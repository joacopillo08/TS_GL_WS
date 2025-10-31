# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ======== Parámetros ========
L = 100                  # longitud del pulso: U(n) - U(n-L)
N_time = 100            # cantidad de muestras a mostrar en el tiempo
N_dft = 100000         # puntos de la DFT (zero-padding para ver la forma)

# ======== Señal x[n] = U(n) - U(n-L) ========
def pulso_unitario(L, N):
    """
    Devuelve x[n] = U(n) - U(n-L) con longitud de visualización N.
    x[n] vale 1 para n=0,...,L-1 y 0 en otro caso.
    """
    x = np.zeros(N)
    x[:L] = 1.0
    return x

x = pulso_unitario(L, N_time)

# ======== DFT ========
def dft(x, N_fft):
    """
    Calcula la DFT de N_fft puntos de x (con zero-padding si hace falta).
    Devuelve X[k] y las frecuencias discretas omega_k = 2*pi*k/N_fft.
    """
    # zero-padding si N_fft > len(x)
    xzp = np.zeros(N_fft)
    xzp[:min(len(x), N_fft)] = x[:min(len(x), N_fft)]
    X = np.fft.fft(xzp, n=N_fft)
    k = np.arange(N_fft)
    omega_k = 2*np.pi*k/N_fft
    return X, omega_k

X, omega_k = dft(x, N_dft)

# ======== Gráfico 1: deltas en el tiempo ========
plt.figure(figsize=(7,4))
n = np.arange(N_time)
plt.stem(n, x)#, use_line_collection=True)
plt.title("x[n] = U(n) - U(n-{}) (L = {})".format(L, L))
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#%%
# ======== Gráfico 2: |DFT| vs ω_k ========
plt.figure(figsize=(7,4))
plt.scatter(omega_k,np.abs(X))
plt.title("Magnitud DFT |X[k]| del pulso (N = {})".format(N_dft))
plt.xlabel("ω_k = 2πk/N")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()
#%%
# ======== Gráfico 3 (opcional): comparación con la curva continua |1 + 2 cos(ω)| ========
# (Muestreo de DTFT teórica para L=3: |1 + 2 cos(ω)|)
omega_dense = np.linspace(0, 2*np.pi, 2000, endpoint=False)
mag_dtft = np.abs(1 + 2*np.cos(omega_dense))

plt.figure(figsize=(7,4))
plt.plot(omega_dense, mag_dtft, label="|1 + 2 cos(ω)| (DTFT)")
plt.plot(omega_k, np.abs(X), 'o', label="Muestras DFT (N={})".format(N_dft))
plt.title("DTFT vs muestras DFT (L=3)")
plt.xlabel("ω (radianes)")
plt.ylabel("Magnitud")
plt.legend()
plt.grid(True)
plt.show()
