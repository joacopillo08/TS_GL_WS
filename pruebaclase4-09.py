import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy import signal

# Parámetros
N = 1000
fs = 1000
Ts = 1/fs

# Ventanas a comparar (las del Holton)
w_rect = np.ones(N)
w_hamm = signal.windows.hamming(N, sym=False)
w_hann = signal.windows.hann(N, sym=False)
w_black = signal.windows.blackman(N, sym=False)  # Blackman (no Harris) para coincidir con la lámina

# Mucho zero-padding para DTFT suave
Nfft = 131072

# Eje de frecuencia angular normalizada [-pi, pi]
freq = np.fft.fftfreq(Nfft, d=1/fs)             # [-fs/2, fs/2)
omega = 2*np.pi*freq                             # [-2π*fs/2, 2π*fs/2)
omega = fftshift(omega)
# Como queremos ω normalizada (rad/muestra), dividimos por fs:
omega_n = omega / fs                             # ahora va de [-π, π]

def resp_db(win):
    W = fftshift(fft(win, Nfft))
    mag = np.abs(W)
    mag /= mag.max()                             # 0 dB en el pico
    # Evitar log(0)
    return 20*np.log10(np.maximum(mag, 1e-12))

R_rect  = resp_db(w_rect)
R_hamm  = resp_db(w_hamm)
R_hann  = resp_db(w_hann)
R_black = resp_db(w_black)

# ---- Plot ----
plt.figure(figsize=(10,5))
plt.plot(omega_n, R_rect,  label='Rectangular',  color='blue')
plt.plot(omega_n, R_hamm,  label='Hamming',      color='green')
plt.plot(omega_n, R_hann,  label='Hann',         color='orange')
plt.plot(omega_n, R_black, label='Blackman',     color='red')
plt.xlim(-np.pi, np.pi)
plt.ylim(-80, 3)
plt.grid(True, alpha=0.3)
plt.legend()

plt.title('Spectral data windows')
plt.ylabel(r'$|W_N(\omega)|_{\mathrm{dB}}$')
plt.xlabel(r'$\omega$  (rad/muestra)')

# Líneas de referencia de lóbulos laterales (visuales)
for y in [-13, -32, -43, -58]:
    plt.axhline(y, ls='--', lw=0.8, alpha=0.5)

# Cajas con el ancho del lóbulo principal (entre primeros ceros)
def draw_mainlobe(delta_omega, color):
    x0 = -delta_omega/2
    x1 =  delta_omega/2
    plt.plot([x0, x1, x1, x0, x0], [ -80, -80, 0, 0, -80], color=color, lw=1, alpha=0.6)

draw_mainlobe(4*np.pi/N,  'blue')    # Rectangular
draw_mainlobe(8*np.pi/N,  'green')   # Hamming
draw_mainlobe(8*np.pi/N,  'orange')  # Hann
draw_mainlobe(12*np.pi/N, 'red')     # Blackman

plt.show()


