# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:36:27 2025

@author: JGL
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy 
from scipy import signal

# Una señal sinusoidal de 2KHz.
# Misma señal amplificada y desfazada en π/2.
# Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
# Señal anterior recortada al 75% de su amplitud.
# Una señal cuadrada de 4kHz.
# Un pulso rectangular de 10ms.
# En cada caso indique tiempo entre muestras, número de muestras y potencia.

fs = 20000
N = 500
f = 2000

deltaF = fs/N
Ts = 1/(N*deltaF)
def mi_funcion_sen(vmax, dc, f, fase, N, fs):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    return tt, xx

#Para hacer modulacion hay que multplicar una señal contra otra señal.
def modulacion(vmax, dc, f, fase, N, fs):
    tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
    tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f/2, fase=0, N = N, fs=fs)
    x2 = xx * x1
    return x2


tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f, fase=np.pi/2, N = N, fs=fs)
x2 = modulacion(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)

#Forma de hacer el clipeo para recortar la amplitud de la señal. 

x3 = np.clip(xx,-0.75,0.75,out=None)

#x5 = sp.square(2 * np.pi * 2*f * tt) #multiplico f por dos porque me pide 4kHz. Esta es con la funcion de scipy
x4 = np.sign(np.sin(2 * np.pi * 2*f * tt)) ##esta es haciendolo con numpy y viendo el signo de la senoidal

# aca hago el del puslo. Como Npulso = Tpulso . fs. Voy a tener 200 muestras
# como mi N lo tengo fijo en 500. voy a tener 300 muestras que estan en 0. Si yo aumento N por ejemplo, siempre voy a tener fijas 200muestras que valen 1 y las N-200=0.
#Si yo cambio fs, me cambia el Npulso entonces ahi ya se modifican las muestras que valen 1. 
T_pulso = 0.01    # 10 ms
N_pulso = int(T_pulso * fs)

pulso = np.zeros(N)
pulso[:N_pulso] = 1  # primeras 200 muestras valen 1

## aca empiezan los graficos.
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(tt, xx)
plt.title("2000 Hz")

plt.subplot(2,2,2)
plt.plot(tt, x1)
plt.title("2000 Hz + desfasaje")

plt.subplot(2,2,3)
plt.plot(tt, x2)
plt.title("modulacion")

plt.subplot(2,2,4)
plt.plot(tt, x3)
plt.title("recortada en el 75% de la amplitud ")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()



plt.figure(2)
plt.subplot(2,1,1)
plt.plot(tt, x4)
plt.title("Funcion cuadrada")

plt.subplot(2,1,2)
plt.scatter(tt, pulso)
plt.title("Pulso rectangular de 10 ms")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()

##2) ortogonalidad

def fun_ortogonalidad(x, y):
    numerador = np.sum(x*y) ##aca ya sabe que el vector tiene N elementos, entonces suma desde n=0 hasta N-1.
    denominador = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)) # aca normalizo cada señal
    return numerador/denominador


print("señal principal vs x1 (desfasada pi/2):", fun_ortogonalidad(xx, x1))
print("señal principal vs x2 (modulada):", fun_ortogonalidad(xx, x2))
print("señal principal vs x3 (clipeada en amplitud):", fun_ortogonalidad(xx, x3))
print("señal principal vs x4 (cuadrada de 4kHz):", fun_ortogonalidad(xx, x4))
print("señal principal vs pulso:", fun_ortogonalidad(xx, pulso))

#3)  3) Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás.
    
Rxx = np.correlate(xx, xx, mode="full")
Rx1 = np.correlate(xx, x1, mode="full")
Rx2 = np.correlate(xx, x2, mode="full")
Rx3 = np.correlate(xx, x3, mode="full")
Rx4 = np.correlate(xx, x4, mode="full")
Rxpulso = np.correlate(xx, pulso, mode="full")

plt.figure(3)
plt.subplot(2,2,1)
plt.plot(Rxx)
plt.title("autocorrelacion")

plt.subplot(2,2,2)
plt.plot(Rx1)
plt.title("x vs x1")

plt.subplot(2,2,3)
plt.plot(Rx2)
plt.title("x vs x2")


plt.subplot(2,2,4)
plt.plot(Rx3)
plt.title("x vs x3")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()

plt.figure(4)

plt.subplot(2,1,1)
plt.plot(Rx4)
plt.title("x vs x4")

plt.subplot(2,1,2)
plt.plot(Rxpulso)
plt.title("x vs pulso")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()



