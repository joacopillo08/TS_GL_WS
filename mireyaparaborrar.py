
a0 = np.sqrt(2)
#a0 = 2

SNR = 10
omega0 = N/4
var = 1
media = 0
R = 200

fr = np.random.uniform(-2, 2, R)
#omega1 = omega0 + fr * 2 * np.pi/N

#na = np.random.normal(media, var)

pot_ruido = a0**2 / (2*10**(SNR/10))
#var_ruido = np.var(ruido)

flattop = window.hamming(N).reshape((-1,1))

# mallas NxR
tt_vector = np.arange(N)/fs
tt_columnas  = tt_vector.reshape((-1,1)) # tamaño N vector columna
TT_sen = np.tile(tt_columnas, (1,R)) # N x R, matriz
 
#matriz de frecuencias verdaderas en Hz para cada columna
f_true = ((N/4) + fr) * deltaF               # (R,)   Hz por columna
F_cols = f_true.reshape((1, R))     # (1,R)  Hz

xx_sen = a0 * np.sin (2 * np.pi * F_cols * TT_sen) # (N,R)
ruido = np.random.normal(loc = 0, scale = np.sqrt(pot_ruido), size = (N, R))
xx_sen_ruido = xx_sen + ruido
xx_vent = xx_sen_ruido * flattop

Npadd = 10#10* N
#XX_sen = fft(xx_sen, n = 10*N, axis = 0) / N
# XX_sen_ruido = fft(xx_vent, n = Npadd, axis = 0)#/  (N)
# fp = np.fft.fftfreq(Npadd, d=1/fs)             # [-fs/2, fs/2)

Npad = 10*N
XX_sen_ruido  = np.fft.rfft(xx_vent, n=Npad, axis=0)  * 1 / Npad          # (K,R)
fp   = np.fft.rfftfreq(Npad, d=1/fs)                     # (K,) eje Hz


plt.figure()
plt.grid(True)
plt.plot(fp, 20*np.log10( np.abs(XX_sen_ruido)))
plt.title("FFT")
plt.xlabel("Hz ")
plt.ylabel("Db")
#lt.xlim(0,fs/2)
