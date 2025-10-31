import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal.windows as spsw


# === LECTURA DEL AUDIO ===
fs, x = wavfile.read("La440.wav")

# Si viene en estéreo → a mono
if x.ndim == 2:
    x = x.mean(axis=1)

# Normalizo amplitud a [-1, 1]
x = x.astype(np.float64)
x = x / np.max(np.abs(x))

# === PARÁMETROS DEL DELAY ===
delay_time = 0.35        # segundos (ajustá para más/menos eco)
a = 0.8                   # feedback (0 < a < 1) — probá entre 0.3 y 0.6
D = int(delay_time * fs)  # retardo en muestras

# === ECUACIÓN EN DIFERENCIAS RECURSIVA ===
y = np.zeros(len(x) + D)
for n in range(len(x)):
    y[n] += x[n]
    if n >= D:
        # versión suavizada con un leve "low-pass" en el feedback
        y[n] += a * (0.7 * y[n - D] + 0.3 * y[n - D - 1])

# === NORMALIZACIÓN FINAL ===
y = y / np.max(np.abs(y)) * 0.8

# === GUARDAR EL RESULTADO ===
wavfile.write("La440_delay.wav", fs, y.astype(np.float32))

print("✅ Delay aplicado correctamente → 'La440_delay.wav' generado.")
N = len(y)
Ts = 1.0 / fs

# --- Ventanas ---
ventanas = {
    "Rectangular": np.ones(N),
    "Hann": spsw.hann(N, sym=False),
}

# --- Eje de frecuencias (unilateral) ---
freqs = np.fft.rfftfreq(N, d=Ts)

# --- Graficar las ventanas en el tiempo ---
tt = np.arange(N) * Ts
plt.figure(figsize=(9,4))
for nombre, w in ventanas.items():
    plt.plot(tt, w, label=nombre, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("w[n]")
plt.title("Ventanas en el dominio del tiempo")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- FFT por ventana (con corrección CG) y gráficos separados ---
for nombre, w in ventanas.items():
    # Coherent Gain (ganancia a una senoidal “bin-centered”)
    CG = w.mean()  # = sum(w)/N

    xw = y * w
    X = np.fft.rfft(xw) / (N * CG)  # corregimos por N y CG para comparar amplitudes
    mag = np.abs(X)

    f0_est = freqs[np.argmax(mag)]

    plt.figure(figsize=(9,4))
    plt.plot(freqs, mag, label=f"{nombre} (pico ~ {f0_est:.2f} Hz)")
    plt.xlim(0, 1000)  # enfocamos cerca de 440 Hz; cambiá si querés ver más
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)| (u.a.)")
    plt.title(f"Espectro con ventana {nombre}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# --- (Opcional) Respuesta en frecuencia de las ventanas ---
# Muestra lóbulos laterales y ancho del lóbulo principal (normalizado a 0 dB en DC)
zp = 16 * N  # zero-padding para ver bien la forma
freqs_win = np.fft.rfftfreq(zp, d=Ts)
plt.figure(figsize=(9,5))
for nombre, w in ventanas.items():
    W = np.fft.rfft(w, n=zp)
    # Normalizamos a 0 dB en DC (|W[0]| = sum(w))
    W_db = 20*np.log10(np.maximum(np.abs(W) / (np.abs(W[0]) + 1e-12), 1e-12))
    plt.plot(freqs_win, W_db, label=nombre, linewidth=1)
plt.xlim(0, 2000)  # para comparar alrededor de 0–2 kHz
plt.ylim(-120, 3)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|W(f)| [dB] (normalizado a 0 dB)")
plt.title("Respuesta en frecuencia de las ventanas (dB)")
plt.grid(True, which="both", axis="both")
plt.legend()
plt.tight_layout()

plt.show()