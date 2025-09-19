import numpy as np
from numpy.fft import fft
import scipy.signal.windows as window
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

N = 1000
fs = N
deltaF = fs/N
Ts = 1/fs
k = np.arange(N)*deltaF
t = np.arange(N)*Ts


def senoidal_estocastica_omega(N, omega0, A0=2): 
    deltaF = fs/N
    k = np.arange(N)*deltaF
    fr = np.random.uniform(-2,2)   
    DeltaOmega = 2*np.pi/N
    omega1 = omega0 + fr * DeltaOmega
    x = A0*np.sin(omega1*k) 
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x 

def senoidal_estocastica_fs(N, f0, A0=2, fase=0):
    Ts = 1/fs
    deltaF = fs/N
    t = np.arange(N)*Ts
    fr = np.random.uniform(-2, 2)
    f1 = f0 + fr*deltaF
    #f1 = fs + 0.5
    x = A0*np.sin(2*np.pi*f1*t*deltaF + fase)
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x
# ---------- SNR y ruido ----------
SNRdb = np.random.uniform(3, 10)
print(f"{SNRdb:3.5f}")

def ruido_para_snr(N, SNRdb):
    var_n = 10**(-SNRdb/10)
    std_n = np.sqrt(var_n) ##std desviacion estandar
    n = np.random.normal(0, std_n, N)
    return n#, var_n
"""
#aca tengo mis cosas listas
#x, var_x = senoidal_estocastica_omega(N, omega0 = np.pi/2)
x, var_x = senoidal_estocastica_fs(N, f0 = fs/4)
n, var_ruido = ruido_para_snr(N, SNRdb)
xn = x + n ## modelo de señal

#las varianzas
var_xn = np.var(xn)
print(f"{var_xn:3.1f}")
print(f"{var_x:3.1f}")
print(f"{var_ruido:3.1f}")

# las fft
X = fft(x) * 1/N
Xabs = np.abs(X)

R = fft(n) * 1/N
Rabs = np.abs(R)

Xn = fft(xn) * 1/N
Xnabs = np.abs(Xn)

#graficos
plt.figure()
plt.title("FFT")
plt.xlabel("Muestras ")
plt.ylabel("Db")
plt.grid(True)
#plt.plot(k,  np.log10(Xabs) * 20, label = 'X')
#plt.plot(k, 2* np.log10(Rabs) * 20, label = 'Ruido')
#plt.plot(k, np.log10(Xnabs) * 20, label = 'Modelo de señal')
plt.plot(k, 10*np.log10(2* Xnabs**2) , label = 'dens esp pot') ##densidad espectral de potencia
plt.xlim((0,fs/2))
plt.legend()

"""

N = 1000
fs = N
deltaF = fs/N
Ts = 1/fs
k = np.arange(N)*deltaF
t = np.arange(N)*Ts

a0 = np.sqrt(2)
R = 200
fr = np.random.uniform(-2, 2, R)
f1 = (N / 4 + fr) 
PP = a0
t = t.reshape(-1,1)

ruido = ruido_para_snr(N, SNRdb)

matriz_t = np.tile(t, (1,R))
matriz_ff = np.tile(f1, (N,1))
#matriz_ruido = np.tile(ruido.reshape(-1,1), (1,R))
matriz_ruido = np.random.normal(0, np.sqrt(10**(-SNRdb/10)), size=(N, R))

matriz_x = a0 * np.sin(2 * np.pi * matriz_ff * deltaF * matriz_t)
matriz_xn = matriz_x + matriz_ruido

flattop = window.flattop(N).reshape((-1,1))
xx_vent_flt = matriz_xn * flattop

bmh = window.blackmanharris(N).reshape((-1,1))
xx_vent_bmh = matriz_xn * bmh

hamming = window.hamming(N).reshape((-1,1))
xx_vent_hmg = matriz_xn * flattop


# FFT bilateral con 10*N puntos (como en tu foto)
matriz_Xn1 = (1/N) * fft(matriz_xn, n = 10 * N, axis = 0)
matriz_Xn2 = (1/N) * fft(xx_vent_flt, n = 10 * N, axis = 0)
matriz_Xn3 = (1/N) * fft(xx_vent_bmh, n = 10 * N, axis = 0)
matriz_Xn4 = (1/N) * fft(xx_vent_hmg, n = 10 * N, axis = 0)

# Eje de frecuencias CONSISTENTE con fft(..., n=10*N)
freq = np.fft.fftfreq(10 * N, d = 1/fs)

# Graficar TODAS las realizaciones (R curvas) sin promediar
# (+1e-20 evita -inf en dB si hay ceros; el +3 dB es el ajuste que suele usarse en la cátedra)
plt.figure()
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn1)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn2)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn3)**2) + 3)
plt.plot(freq, 10*np.log10(np.abs(matriz_Xn4)**2) + 3)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.xlim(0,fs/2)
plt.grid(True)




























