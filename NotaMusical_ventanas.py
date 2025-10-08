import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal.windows as spsw

# --- Leer WAV ---
fs, data = wavfile.read("La440.wav")

# A mono si viene estéreo
if data.ndim == 2:
    data = data.mean(axis=1)

# A float y quitar DC
x = data.astype(np.float64)
x = x - x.mean()

N = len(x)
Ts = 1.0 / fs

# --- Ventanas ---
ventanas = {
    "Rectangular": np.ones(N),
    "Hann": spsw.hann(N, sym=False),
    "Hamming": spsw.hamming(N, sym=False),
    "Blackman-Harris": spsw.blackmanharris(N, sym=False),
    "Flattop": spsw.flattop(N, sym=False),
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

    xw = x * w
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
