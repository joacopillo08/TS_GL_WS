#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:05:59 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
from scipy.interpolate import CubicSpline
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
ecg_one_lead = ecg_one_lead
N = len(ecg_one_lead)

filtro_med_200 = signal.medfilt(ecg_one_lead, 199)

filtro_med_600 = signal.medfilt(filtro_med_200, 599)

x = ecg_one_lead - filtro_med_600

# plt.figure()

# # Retardo de grupo
# plt.plot(x, label = 'mediana')
# plt.plot(ecg_one_lead, label = 'posta')
# plt.title('ecg mediana')
# plt.xlabel('x')
# plt.ylabel('τg [# muestras]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

maximos = mat_struct['qrs_detections'].flatten()
muestras = np.arange(N)
nodos = maximos - 80

spl = CubicSpline(nodos, ecg_one_lead[nodos])

b = spl(np.arange(N))

ecg_spline = ecg_one_lead - b

plt.figure()
t = np.arange(N) / fs_ecg  # vector de tiempo en segundos
plt.plot(ecg_one_lead, label='ECG crudo')
plt.plot(ecg_spline, label='ECG spline')
plt.ylabel('Amplitud')
plt.title('ECG con y sin filtrado')
plt.legend()
plt.grid(True)
plt.show()


regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_spline[zoom_region], label='spline')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#################################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_spline[zoom_region], label='spline')
    # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

plt.plot(ecg_one_lead)
# plt.plot(b)
# plt.plot(filtro_med_600)


QRS = 140
t = np.linspace(-1, 1, QRS)
sigma = 0.2
gauss_derivada = -t * np.exp(-t**2 / (2*sigma**2))

centro = np.argmax(gauss_derivada)
mover = QRS//2 - centro
gauss_centrada = np.roll(gauss_derivada, mover)

h = gauss_centrada[::-1]   
y = np.convolve(ecg_one_lead, h, mode='same')

max_ecg = np.argmax(ecg_one_lead[0:1000])
max_conv = np.argmax(y[0:1000])

peaks, props = signal.find_peaks(y,distance=QRS, prominence=np.max(y)*0.39)

