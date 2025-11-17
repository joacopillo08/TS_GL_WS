#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 10:59:49 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.signal import firwin2, freqz, firls

from pytc2.sistemas_lineales import plot_plantilla

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)


fs = 1000

wp = (0.8, 35) #freq de corte/paso (rad/s)
ws = (0.1, 40) #freq de stop/detenida (rad/s)

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

# ---------------------------------------------
#   FFT del ECG en un segmento con ruido
# ---------------------------------------------


fs = fs_ecg  # tu frecuencia de muestreo

# Segmento de ruido: 10.000 muestras = 10 segundos
segmento = ecg_one_lead[500000:510000]
N = len(segmento)

# Ventana Hann para evitar fugas espectrales
ventana = np.hanning(N)
segmento_win = segmento * ventana

# FFT
X = np.fft.rfft(segmento_win)         # Solo positivo
freqs = np.fft.rfftfreq(N, 1/fs)

# Magnitud normalizada
Xabs = np.abs(X) / N
Xabs = np.maximum(Xabs, 1e-12)        # Evitar log(0)

# ---------------------------------------------
#   Gráfico del espectro
# ---------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(freqs, 20*np.log10(Xabs))
plt.title("Espectro del ECG (segmento con ruido)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.xlim(0, 200)   # Para ECG no necesitás más
plt.tight_layout()
plt.show()



def filtro_IIR(fs, wp, ws, alpha_p, alpha_s, ftype): 
    
    mi_sos = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = ftype, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q
    
    # Respuesta en frecuencia
    w, h= signal.freqz_sos(mi_sos, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)
    fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo
    w_rad = w / (fs / 2) * np.pi
    gd = -np.diff(fase) / np.diff(w_rad) 
    
    # --- Polos y ceros ---
    z, p, k = signal.sos2zpk(mi_sos) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k
    
    # --- Gráficas ---
    #plt.figure(figsize=(12,10))
    
    # Magnitud
    plt.figure(figsize=(8, 8))
    plt.subplot(3,1,1)
    plt.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-10)), label = ftype)
    plot_plantilla(filter_type = 'bandpass' , fpass = wp, ripple = alpha_p*2 , fstop = ws, attenuation = alpha_s*2, fs = fs)
    plt.title('Respuesta en Magnitud')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Fase
    plt.subplot(3,1,2)
    plt.plot(w, np.degrees(fase))
    plt.title('Fase')
    plt.ylabel('Fase [°]')
    plt.grid(True, which='both', ls=':')
    
    # Retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[:-1], gd)
    plt.title('Retardo de Grupo ')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':')
    
    plt.tight_layout()
    plt.show()
    
    return mi_sos


def filtrar_IIR_ECG(mi_sos, nombre_filtro, ecg=ecg_one_lead, fs=fs_ecg):
    ecg_filt = signal.sosfiltfilt(mi_sos, ecg)

    plt.figure()
    plt.plot(ecg[0:100000], label='ECG crudo', alpha=0.7)
    plt.plot(ecg_filt[0:100000], label=f'Filtrado ({nombre_filtro})', linewidth=1.2)
    plt.title(f'ECG completo - {nombre_filtro}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    #################################
    # Regiones de interés sin ruido #
    #################################
    
    cant_muestras = len(ecg_one_lead)
    
    regs_interes = (
            [4000, 5500], # muestras
            [10e3, 11e3], # muestras
            )
     
    # for ii in regs_interes:
       
    #     # intervalo limitado de 0 a cant_muestras
    #     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
       
    #     plt.figure()
    #     plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
    #     plt.plot(zoom_region, ecg_filt[zoom_region], label=nombre_filtro)
       
    #     plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    #     plt.ylabel('Adimensional')
    #     plt.xlabel('Muestras (#)')
       
    #     axes_hdl = plt.gca()
    #     axes_hdl.legend()
    #     axes_hdl.set_yticks(())
               
    #     plt.show()
     
    # #################################
    # # Regiones de interés con ruido #
    # #################################
     
    # regs_interes = (
    #         np.array([5, 5.2]) *60*fs, # minutos a muestras
    #         np.array([12, 12.4]) *60*fs, # minutos a muestras
    #         np.array([15, 15.2]) *60*fs, # minutos a muestras
    #         )
     
    # for ii in regs_interes:
       
    #     # intervalo limitado de 0 a cant_muestras
    #     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
       
    #     plt.figure()
    #     plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
    #     plt.plot(zoom_region, ecg_filt[zoom_region], label=nombre_filtro)
    #     # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
       
    #     plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    #     plt.ylabel('Adimensional')
    #     plt.xlabel('Muestras (#)')
       
    #     axes_hdl = plt.gca()
    #     axes_hdl.legend()
    #     axes_hdl.set_yticks(())
               
    #     plt.show()

#### Comparo distintos IIR 
IIR = {}
tipos = ['butter', 'ellip']  # 'ellip' = Cauer
for tipo in tipos:
    IIR[tipo] = filtro_IIR(fs = fs, wp = wp, ws = ws, alpha_p = alpha_p, alpha_s = alpha_s, ftype = tipo)
        

# Aplicar y graficar ECG filtrado
for tipo, sos in IIR.items():
    filtrar_IIR_ECG(sos, tipo)   

# ============================================================
#           DFT del segmento ruidoso + FILTRO ECG
# ============================================================

## ============================================================
#        FIR por cuadrados mínimos + FFT antes/después
# ============================================================

# Segmento de ruido: 500000-510000
fs = fs_ecg
segmento = ecg_one_lead[500000:510000]
Nseg = len(segmento)

segmento1 = ecg_one_lead[0:10000]
Nseg1 = len(segmento1)

# Ventana Hann
ventana = np.hanning(Nseg)
seg_win = segmento * ventana

ventana1 = np.hanning(Nseg1)
seg_win1 = segmento1 * ventana1

# FFT sin filtrar
X = np.fft.rfft(seg_win)
freqs = np.fft.rfftfreq(Nseg, 1/fs)
Xabs = np.maximum(np.abs(X)/Nseg, 1e-12)

X1 = np.fft.rfft(seg_win1)
freqs = np.fft.rfftfreq(Nseg1, 1/fs)
Xabs1 = np.maximum(np.abs(X1)/Nseg1, 1e-12)
# -----------------------------
#   FIR por cuadrados mínimos
# -----------------------------
# Bandas del TP8:
#   Paso: 0.8 – 35 Hz
#   Stop: [0–0.14] y [35.7–fs/2]

wp = (0.8, 35)
ws = (0.14, 35.7)

# Frecuencias para firls
frecuencias = np.sort(np.concatenate(((0, fs/2), wp, ws)))

# Desired: stop – pass – stop
deseado = [0, 0, 1, 1, 0, 0]

cant_coef = 2001
peso = [6, 1, 1]   # pesos del TP8

# Diseño del FIR
h_firls = firls(numtaps=cant_coef,
                bands=frecuencias,
                desired=deseado,
                weight=peso,
                fs=fs)

# Filtrado FIR (lineal → usamos filtfilt)
seg_filt = signal.filtfilt(h_firls, 1, segmento)
seg_filt1 = signal.filtfilt(h_firls, 1, segmento1)

# FFT filtrado
seg_filt_win = seg_filt * ventana
Y = np.fft.rfft(seg_filt_win)
Yabs = np.maximum(np.abs(Y)/Nseg, 1e-12)

seg_filt_win1 = seg_filt1 * ventana1
Y1 = np.fft.rfft(seg_filt_win1)
Yabs1 = np.maximum(np.abs(Y1)/Nseg1, 1e-12)

# ============================
#         GRÁFICO
# ============================
plt.figure(figsize=(12,6))
plt.plot(freqs, 20*np.log10(Xabs), label='Antes (segmento ruidoso)', alpha=0.7)
plt.plot(freqs, 20*np.log10(Yabs), label='Después (FIR - firls)', linewidth=1.5)

plt.title("Comparación espectral del segmento ruidoso (FIR cuadrados mínimos)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.xlim(0, 200)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
plt.plot(freqs, 20*np.log10(Xabs1), label='Antes (segmento sin ruido)', alpha=0.7)
plt.plot(freqs, 20*np.log10(Yabs1), label='Después (FIR - firls)', linewidth=1.5)

plt.title("Comparación espectral del segmento ruidoso (FIR cuadrados mínimos)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.xlim(0, 200)
plt.legend()
plt.tight_layout()
plt.show()




