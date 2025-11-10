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
ecg_one_lead = ecg_one_lead[4000: 5500]
N = len(ecg_one_lead)

filtro_med_200 = signal.medfilt(ecg_one_lead, 199)

filtro_med_600 = signal.medfilt(filtro_med_200, 599)

x = ecg_one_lead - filtro_med_600

plt.figure()

# Retardo de grupo
plt.plot(x, label = 'mediana')
plt.plot(ecg_one_lead, label = 'posta')
plt.title('ecg mediana')
plt.xlabel('x')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()