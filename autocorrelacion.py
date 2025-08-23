import numpy as np
import matplotlib.pyplot as plt

fs = 1000   # frecuencia de muestreo
N = 1000    # cantidad de muestras
Ts = 1/fs   # periodo de muestreo

tiempo = np.arange(0, N * Ts, Ts)
fx = 1001   # frecuencia de la señal
A = 1
x = A * np.sin(2 * np.pi * fx * tiempo)

# ============================
# AUTOCORRELACIÓN - Opción 1
# ============================
 # porque el centro está en la posición N-1

Rxx_np = np.correlate(x, x, mode="full")

# ============================
# AUTOCORRELACIÓN - Opción 2 (a mano)
# ============================
Rxx_manual = []
for k in range(-N+1, N-1):   # corrimientos negativos y positivos
    suma = 0
    for n in range(N):
        if 0 <= n+k < N:   # me aseguro de no salir del rango
            suma += x[n] * x[n+k]
    Rxx_manual.append(suma)

# ============================
# GRAFICAR
# ============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Autocorrelación con np.correlate")
plt.plot(Rxx_np)
plt.grid()

plt.subplot(1,2,2)
plt.title("Autocorrelación calculada a mano")
plt.plot(Rxx_manual)
plt.grid()

plt.tight_layout()
plt.show()

