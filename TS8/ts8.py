# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:39:28 2025

@author: JGL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.signal import firwin2, freqz, firls

from pytc2.sistemas_lineales import plot_plantilla
# from pytc2.filtros_digitales import fir_design_pm
#filtro normalizado -> todas las singularidades en el circulo unitario?
#--- Plantilla de diseño ---
# %% IIR

fs = 1000
wp = (0.8, 35) #freq de corte/paso (rad/s)
ws = (0.1, 40) #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

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

##################
# Lectura de ECG con ruido
##################

fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)



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

# %% FIR



fs = 1000
wp = (0.95, 35) #freq de corte/paso (rad/s)
ws = (0.14, 35.7) #freq de stop/detenida (rad/s) ###VEEEEEERRRRRR clase del otro dia##########

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 


def filtro_FIR(fs, wp, ws, alpha_p, alpha_s, metodo='firwin2'):
    
    # frecuencias= np.sort(np.concatenate(0, wp,ws, fs/2))
    #frecuencias= np.sort(np.concatenate(((0,fs/2), wp,ws)))
    frecuencias = [0, ws[0], wp[0], wp[1], ws[1], fs/2]

    deseado = [0,0,1,1,0,0]
    cant_coef = 2001
    retardo = (cant_coef-1)//2 
    peso = [12,4,4]

        
    # --- Diseño ---
    if metodo == 'firwin2':
        b = firwin2(numtaps=cant_coef, freq=frecuencias, gain=deseado, window='boxcar', fs=fs)
    elif metodo == 'firls':
        b = firls(numtaps=cant_coef, bands=frecuencias, desired=deseado, fs=fs, weight= peso)
    else:
        raise ValueError("Método inválido. Usá 'firwin2' o 'firls'.")
    
    
    # --- Respuesta en frecuencia ---
    w, h= freqz(b, worN = np.logspace(-2, 2, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)    
    fase = np.unwrap(np.angle(h)) 
    w_rad = 2*np.pi*w/fs
    gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]
    
    # --- Polos y ceros ---
    
    z, p, k = signal.sos2zpk(signal.tf2sos(b,a= 1)) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k
    
    # --- Gráficas ---
    plt.figure(figsize=(8,8))
    
    # Magnitud
    
    plt.subplot(3,1,1)
    plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-10)), label = metodo)
    plot_plantilla(filter_type = 'bandpass' , fpass = (0.8, 35), ripple = alpha_p*2 , fstop = (0.1, 40), attenuation = alpha_s*2, fs = fs)
    plt.title('Respuesta en Magnitud')
    plt.ylabel('|H(z)| [dB]')
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
    return b, retardo

def filtrar_FIR_ECG(b, nombre_filtro, ecg, fs, retardo):

    ecg_filt = signal.lfilter(b = b, a = 1, x = ecg)

    plt.figure()
    plt.plot(ecg, label='ECG crudo', alpha=0.7)
    plt.plot(ecg_filt, label=f'Filtrado ({nombre_filtro})', linewidth=1.2)
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
    #     plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label=nombre_filtro)

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
    #     plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label='FIR Window')
       
    #     plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    #     plt.ylabel('Adimensional')
    #     plt.xlabel('Muestras (#)')
       
    #     axes_hdl = plt.gca()
    #     axes_hdl.legend()
    #     axes_hdl.set_yticks(())
               
    #     plt.show()
    
# --- Loop para comparar FIRs ---
FIR = {}
metodos = ['firwin2', 'firls']
for metodo in metodos:
    FIR[metodo] = filtro_FIR(fs=fs, wp=wp, ws=ws, alpha_p=alpha_p, alpha_s=alpha_s, metodo=metodo)

# Aplicar y graficar ECG filtrado
for metodo, (b, retardo) in FIR.items():
    filtrar_FIR_ECG(b, metodo, ecg_one_lead, fs_ecg, retardo)
