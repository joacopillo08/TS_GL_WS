import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sympy as sp

k = np.sqrt(10)
# Qn = 60     # ajustable
# Qp = 1.0
f_aprox = 'butter'
# coeficientes analógicos normalizados
b = [1,0,4]## s^2+2^2  RADIO CEROS = 2          
a = [1,2*np.sqrt(2),4]  ## s^2+sqrt(2).2+2^2   Q = 1/sqrt(2), w0 = 2 *np.sqrt(2) 

# Barrido de pulsaciones
w = np.logspace(-2, 2, 1000)  # 0.01 a 100 rad/s aprox

# Respuesta en frecuencia analógica
w, h = signal.freqs(b, a, w)
# plt.figure()


# plt.semilogx(w, 20*np.log10(abs(h)))
# plt.title("Magnitud analógica |H(jω)|")
# plt.xlabel("ω [rad/s]")
# plt.ylabel("Magnitud [dB]")
# plt.grid(True, which="both", ls=":")
# # plt.axvline(2, color='r', linestyle='--')  # líneas guía
# plt.show()

z, p, k = signal.tf2zpk(b,a) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# Diagrama de polos y ceros
# plt.figure()
plt.figure(figsize=(10,10))

axes_hdl = plt.gca()
plt.plot(np.real(p), np.imag(p), 'x', markersize=10)
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none')
    
unit_circle = patches.Circle((0, 0), radius=8, fill=False, color='gray', ls='dotted', lw=2)
unit_circle1 = patches.Circle((0, 0), radius=4, fill=False, color='gray', ls='dotted', lw=2)

axes_hdl.add_patch(unit_circle)
axes_hdl.add_patch(unit_circle1)

plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

# gd = -np.diff(fase) / np.diff(fase) #retardo de grupo [rad/rad]


# plt.plot(w, fase, label = f_aprox)
# plt.title('Fase')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')

# # Retardo de grupo
# plt.subplot(2,2,3)
# plt.plot(w[:-1], gd, label = f_aprox)
# plt.title('Retardo de Grupo ')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('τg [# muestras]')
# plt.grid(True, which='both', ls=':')





# %% punto b
fs = 1
num, den = signal.bilinear(b, a, fs)

wz, hz= signal.freqz(num, den, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)
zz, pz, kz = signal.tf2zpk(num,den) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k


# # plt.figure()
# plt.semilogx(wz, 20*np.log10(abs(hz)))
# plt.title(" digital |H(e^jω)|")
# plt.xlabel("ω [rad/s]")
# plt.ylabel("Magnitud [dB]")
# plt.grid(True, which="both", ls=":")
# # plt.axvline(2, color='r', linestyle='--')  # líneas guía
# plt.show()


plt.figure(figsize=(10,10))
plt.plot(np.real(pz), np.imag(pz), 'x', markersize=10, label=f'{f_aprox} Polos')
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(zz), np.imag(zz), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')

# Ejes y círculo unitario
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)

# Ajustes visuales
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fasez = np.unwrap(np.angle(hz)) #unwrap hace grafico continuo

gdz = -np.diff(fasez) / np.diff(fasez) #retardo de grupo [rad/rad]

plt.figure()
plt.plot(wz, fasez)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.figure()

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(wz[:-1], gdz)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')



# %%
#me piden que haga un T(z)=K. z^2+1/(Q(z))

numz = [1,0,1] ## s^2+2^2 
denz = [1,2*np.sqrt(2), 4]  ## s^2+2+2^2   Q = 1, w0 = 2



















# %%

z = [(0+2j),(0-2j)]
# p = [(-np.sqrt(2)+np.sqrt(2)),(-np.sqrt(2)-np.sqrt(2))]
p = [(-1.4142135623730951+1.4142135623730951j),(-1.4142135623730951-1.4142135623730951j)]
k = 10
# polos y ceros
# z, p, k = signal.tf2zpk(b_s, a_s)
# b,a = signal.zpk2tf(z, p,k)

print("Ceros (z):", z)
print("Polos (p):", p)
print("Ganancia (k):", k)

# Bode (magnitud y fase)
w = np.logspace(-1, 2.5, 1000)  # eje en rad/s
w, mag, phase = signal.bode((b, a), w=w)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.semilogx(w, mag)
plt.axvline(2, color='r', linestyle='--', label='ω0 = 2 rad/s')
plt.title('Magnitud (dB)')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.title('Fase (grados)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Fase
# plt.plot(w, fase)
# plt.title('Fase')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')


# Diagrama de polos y ceros
plt.plot(np.real(p), np.imag(p), 'x', markersize=10)
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
