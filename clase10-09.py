import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

N = 1000
fs = N
deltaF = fs/N
Ts = 1/fs
k = np.arange(N)*deltaF


def senoidal_estocastica_omega(N, omega0, A0=2): 
    deltaF = fs/N
    k = np.arange(N)*deltaF
    #fr = np.random.uniform(-2,2)   
    #DeltaOmega = 2*np.pi/N
    omega1 = omega0 #+ fr * DeltaOmega
    x = A0*np.sin(omega1*k) 
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x 

def senoidal_estocastica_fs(N, fs, A0=2, fase=0):
    Ts = 1/fs
    deltaF = fs/N
    t = np.arange(N)*Ts
    fr = np.random.uniform(-2, 2)
    f1 = fs + fr*deltaF
    #f1 = fs + 0.5
    x = A0*np.sin(2*np.pi*f1*t*deltaF + fase)
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x
# ---------- SNR y ruido ----------
SNRdb = 47#np.random.uniform(3, 10)
print(f"{SNRdb:3.5f}")

def ruido_para_snr(N, SNRdb):
    var_n = 10**(-SNRdb/10)
    std_n = np.sqrt(var_n) ##std desviacion estandar
    n = np.random.normal(0, std_n, N)
    return n, var_n

#aca tengo mis cosas listas
#x, var_x = senoidal_estocastica_omega(N, omega0 = np.pi/2)
x, var_x = senoidal_estocastica_fs(N, fs = fs/4)
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

