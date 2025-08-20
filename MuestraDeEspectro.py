import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
fs = 1000           # Frecuencia de muestreo
N = 1000            # Número de muestras
Ts = 1/fs           # Periodo de muestreo
tiempo = np.arange(0, N*Ts, Ts)

# Preparo la figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Señal en tiempo
linea_tiempo, = ax1.plot([], [], lw=1)
ax1.set_xlim(0, 0.02)  # primeros 20 ms
ax1.set_ylim(-1.2, 1.2)
ax1.set_title("Señal muestreada")
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("x[n]")

# Espectro en frecuencia
ax2.set_xlim(-fs/2, fs/2)
ax2.set_ylim(0, 0.6)
ax2.set_title("Espectro (FFT)")
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("|X(f)|")

# Frecuencias para FFT
freqs = np.fft.fftfreq(N, Ts)
freqs_shift = np.fft.fftshift(freqs)

# ==============================
# Función de actualización
# ==============================
def actualizar(frame):
    fx = frame  # frecuencia de la señal cambia en cada frame
    x = np.sin(2 * np.pi * fx * tiempo)

    # Señal en tiempo
    linea_tiempo.set_data(tiempo[:200], x[:200])  # muestro 200 muestras

    # FFT
    X = np.fft.fft(x) / N
    Xshift = np.fft.fftshift(X)

    # Actualizo espectro
    ax2.cla()
    ax2.set_xlim(-fs/2, fs/2)
    ax2.set_ylim(0, 0.6)
    ax2.set_title(f"Espectro (fx={fx} Hz)")
    ax2.set_xlabel("Frecuencia [Hz]")
    ax2.set_ylabel("|X(f)|")
    ax2.stem(freqs_shift, np.abs(Xshift))

    return linea_tiempo,

# ==============================
# Animación
# ==============================
ani = FuncAnimation(fig, actualizar, frames=np.arange(0, 2000, 20), 
                    interval=200, blit=False, repeat=True)

plt.tight_layout()
plt.show()


