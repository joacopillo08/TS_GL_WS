import numpy as np
from numpy.fft import fft
import scipy.signal.windows as window
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
np.random.seed(7)

#%% DATOS
N = 1000
fs = N
deltaF = fs/N
Npadding = 10* N
deltaF_Padding = fs/Npadding

Ts = 1/fs
k = np.arange(N)*deltaF
t = np.arange(N)*Ts

#a0 = np.sqrt(2)
a0 = 2
R = 200
fr = np.random.uniform(-2, 2, R)
f1 = (N / 4 + fr) * deltaF
PP = a0
t = t.reshape(-1,1)

pot_senal = (a0**2)/2

matriz_t = np.tile(t, (1,R))
matriz_ff = np.tile(f1, (N,1))

freq = np.fft.fftfreq(Npadding, d = 1/fs)

#%% ARMADO DE MATRIZ DE SENOIDALES Y VENTANAS. 
matriz_x = a0 * np.sin(2 * np.pi * matriz_ff * matriz_t)

flattop = window.flattop(N).reshape((-1,1))
bmh = window.blackmanharris(N).reshape((-1,1))
hamming = window.hamming(N).reshape((-1,1))

# matriz_x: N x R  (solo senoidal, SIN ruido)
P_signal_cols = (1/N) * np.sum(matriz_x**2, axis=0)   # potencia de cada realización
P_signal_mean = P_signal_cols.mean()

## CHEQUEO DE POTENCIA 
print("Potencias señal (min / mean / max):",
      P_signal_cols.min(), P_signal_mean, P_signal_cols.max())

# ¿Está normalizada a 1?
print("¿Potencia ~ 2 en TODAS las columnas?",
      np.allclose(P_signal_cols, 2.0, rtol=1e-3, atol=1e-3))

## FUNCIONES PARA ARMAR LAS TABLAS DE VALORES.
def _fmt(x):
    # Asegura float y formato científico prolijo
    return f"{np.asarray(x, dtype=float): .6e}"

def _print_block(title, rows):
    print(f"\n[{title}]")
    print(f"{'Ventana':<18}{'Sesgo':>16}{'Varianza':>16}{'Desvío':>16}")
    print("-"*66)
    for (vent, bias, var) in rows:
        std = np.sqrt(np.maximum(var, 0.0))  # por estabilidad numérica
        print(f"{vent:<18}{_fmt(bias):>16}{_fmt(var):>16}{_fmt(std):>16}")

#%% ESTE FOR LO HICIMOS PARA ARMAR LA TABLA PARA SNR = 3 Y SNR = 10
for SNRdb in [3, 10]:
    print(f"\n===== RESULTADOS PARA SNR = {SNRdb} dB =====")
    sigma_ruido = np.sqrt(pot_senal * 10**(-SNRdb/10))
    matriz_ruido = np.random.normal(0, sigma_ruido , size=(N, R))
    matriz_xn = matriz_x + matriz_ruido

    xx_vent_flt = matriz_xn * flattop
    xx_vent_bmh = matriz_xn * bmh
    xx_vent_hmg = matriz_xn * hamming

    # FFT 
    matriz_Xn1 = (1/N) * fft(matriz_xn, n = Npadding, axis = 0)
    # matriz_Xn2 = (1/N) * fft(xx_vent_flt, n = Npadding, axis = 0)
    # matriz_Xn3 = (1/N) * fft(xx_vent_bmh, n = Npadding, axis = 0)
    # matriz_Xn4 = (1/N) * fft(xx_vent_hmg, n = Npadding, axis = 0)

    P_sig = np.mean(matriz_x**2)
    P_n   = np.mean(matriz_ruido**2)
    SNR_meas_db = 10*np.log10(P_sig / P_n)
    Ps_cols = (1/N)*np.sum(matriz_x**2, axis=0)
    Pn_cols = (1/N)*np.sum(matriz_ruido**2, axis=0)
    SNR_cols_db = 10*np.log10(Ps_cols / Pn_cols)
    print(f"SNR medido (global): {SNR_meas_db:.2f} dB")
    print(f"SNR medido (mean±std por realización): {SNR_cols_db.mean():.2f} ± {SNR_cols_db.std():.2f} dB")
#%%    ESTIMADORES DE AMPLITUD DE DOS FORMAS
    cg_rect = 1.0
    cg_flt  = flattop.mean()
    cg_bmh  = bmh.mean()
    cg_hmg  = hamming.mean()

    # # Estimador de amplitud (valor del pico en cada columna)
    amp_est1 = 2*np.max(np.abs(matriz_Xn1), axis=0) / cg_rect
    amp_est2 = 2*np.max(np.abs(matriz_Xn2), axis=0) /cg_flt        
    amp_est3 = 2*np.max(np.abs(matriz_Xn3), axis=0) /cg_bmh      
    amp_est4 = 2*np.max(np.abs(matriz_Xn4), axis=0) /cg_hmg    

    sesgo_rect = np.mean(amp_est1) - a0
    sesgo_flat = np.mean(amp_est2) - a0
    sesgo_bmh  = np.mean(amp_est3) - a0
    sesgo_hmg  = np.mean(amp_est4) - a0

    var_rect = np.var(amp_est1, ddof=0)
    var_flat = np.var(amp_est2, ddof=0)
    var_bhm  = np.var(amp_est3, ddof=0)
    var_hmg  = np.var(amp_est4, ddof=0)

    #estimador pero solamente en la feta de N//4
    amp_est1_bin = 2*np.abs(matriz_Xn1[Npadding//4, :]) / cg_rect
    amp_est2_bin = 2*np.abs(matriz_Xn2[Npadding//4, :]) / cg_flt
    amp_est3_bin = 2*np.abs(matriz_Xn3[Npadding//4, :]) / cg_bmh
    amp_est4_bin = 2*np.abs(matriz_Xn4[Npadding//4, :]) / cg_hmg    

    sesgo_rect_bin = np.mean(amp_est1_bin) - a0
    sesgo_flat_bin = np.mean(amp_est2_bin) - a0
    sesgo_bmh_bin  = np.mean(amp_est3_bin) - a0
    sesgo_hmg_bin  = np.mean(amp_est4_bin) - a0

    var_rect_bin = np.var(amp_est1_bin, ddof=0)
    var_flat_bin = np.var(amp_est2_bin, ddof=0)
    var_bhm_bin  = np.var(amp_est3_bin, ddof=0)
    var_hmg_bin  = np.var(amp_est4_bin, ddof=0)

#%% ESTIMADORES DE FRECUENCIA DE 1 MANERA 
    # Parte positiva de cada ventana
    X1p = matriz_Xn1[:Npadding//2+1, :]
    X2p = matriz_Xn2[:Npadding//2+1, :]
    X3p = matriz_Xn3[:Npadding//2+1, :]
    X4p = matriz_Xn4[:Npadding//2+1, :]

    idx1 = np.argmax(np.abs(X1p), axis=0)
    idx2 = np.argmax(np.abs(X2p), axis=0)
    idx3 = np.argmax(np.abs(X3p), axis=0)
    idx4 = np.argmax(np.abs(X4p), axis=0)

    frec_est1 = idx1 * deltaF_Padding #- f1
    frec_est2 = idx2 * deltaF_Padding #- f1
    frec_est3 = idx3 * deltaF_Padding #- f1
    frec_est4 = idx4 * deltaF_Padding #- f1

    # Frecuencia verdadera por realización (vector R,)
    #f1  # (= (N/4 + fr) * deltaF)

    # Errores por realización
    err1 = frec_est1 - f1
    err2 = frec_est2 - f1
    err3 = frec_est3 - f1
    err4 = frec_est4 - f1

    sesgo= np.mean(frec_est1) - np.mean(f1)

    # Sesgo y varianza muestral, SON ESCALARES. 
    sesgo_rect1 = float(err1.mean());  var_rect1 = float(err1.var(ddof=0))
    sesgo_flat2 = float(err2.mean());  var_flat2 = float(err2.var(ddof=0))
    sesgo_bmh3  = float(err3.mean());  var_bhm3  = float(err3.var(ddof=0))
    sesgo_hmg4  = float(err4.mean());  var_hmg4  = float(err4.var(ddof=0))
    
#%%TABLA DE VALORES

    print(f"\n=== TABLA SESGO Y VARIANZA — SNR = {SNRdb} dB ===")
    
    # --- Amplitud — Pico (máximo FFT ×2/CG) ---
    rows_amp_peak = [
        ("Rectangular",     sesgo_rect,     var_rect),
        ("Flattop",         sesgo_flat,     var_flat),
        ("Blackman-Harris", sesgo_bmh,      var_bhm),
        ("Hamming",         sesgo_hmg,      var_hmg),
    ]
    _print_block("AMPLITUD — Pico (×2/CG)", rows_amp_peak)
    
    # --- Amplitud — Bin fijo fs/4 (scalloping) ---
    rows_amp_bin = [
        ("Rectangular",     sesgo_rect_bin,     var_rect_bin),
        ("Flattop",         sesgo_flat_bin,     var_flat_bin),
        ("Blackman-Harris", sesgo_bmh_bin,      var_bhm_bin),
        ("Hamming",         sesgo_hmg_bin,      var_hmg_bin),
    ]
    _print_block("AMPLITUD — Bin fijo fs/4", rows_amp_bin)
    
    # --- Frecuencia — Argmax (f̂ − f_true) ---
    rows_freq = [
        ("Rectangular",     sesgo_rect1,    var_rect1),
        ("Flattop",         sesgo_flat2,    var_flat2),
        ("Blackman-Harris", sesgo_bmh3,     var_bhm3),
        ("Hamming",         sesgo_hmg4,     var_hmg4),
    ]
    _print_block("FRECUENCIA — Argmax (f̂ − f_true)", rows_freq)

#%% GRAFICOS

plt.figure()
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn1)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn2)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn3)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn4)**2) + 3)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.xlim(245,255)
plt.ylim(-15,5)
plt.grid(True)

plt.figure()
plt.hist(amp_est1, bins=16, alpha=0.5, label="Rectangular")
plt.hist(amp_est2, bins=16, alpha=0.5, label="Flattop")
plt.hist(amp_est4, bins=16, alpha=0.5, label="Hamming")
plt.hist(amp_est3, bins=16, alpha=0.5, label="Blackman-Harris")
plt.axvline(a0, color='red', linestyle='--', linewidth=2, label="a0 real")
plt.title("Histogramas de estimadores de amplitud (todas las ventanas) para SNR = 10dB")
plt.xlabel("Amplitud estimada")
plt.ylabel("Frecuencia de ocurrencia")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.hist(amp_est1_bin, bins=16, alpha=0.5, label="Rectangular")
plt.hist(amp_est2_bin, bins=16, alpha=0.5, label="Flattop")
plt.hist(amp_est3_bin, bins=16, alpha=0.5, label="Hamming")
plt.hist(amp_est4_bin, bins=16, alpha=0.5, label="Blackman-Harris")
plt.axvline(a0, color='red', linestyle='--', linewidth=2, label="a0 real")
plt.title("Histogramas de estimadores de amplitud (todas las ventanas) para SNR = 10dB")
plt.xlabel("Amplitud estimada")
plt.ylabel("Frecuencia de ocurrencia")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.hist(frec_est1, bins=16, alpha=0.5, label="Rectangular")
plt.hist(frec_est2, bins=16, alpha=0.5, label="Flattop")
plt.hist(frec_est3, bins=16, alpha=0.5, label="Blackman–Harris")
plt.hist(frec_est4, bins=16, alpha=0.5, label="Hamming")
plt.axvline(fs/4, color='red', ls='--', lw=2, label='f0 teórica (sin fr)')
plt.title("Histogramas de estimadores de frecuencia para SNR = 10dB")
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Frecuencia de ocurrencia")
plt.legend(); plt.grid(True); plt.show()




