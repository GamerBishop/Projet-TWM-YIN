# -*- coding: utf-8 -*-
__author__ = 'hugostrange'

import numpy as np
import sys, os
import utilFunctions as uF
import stft as STFT

# Fonctions a utiliser


def TWM(pfreq, pmag, f0c):
    # Algorithme Two-way mismatch
    # pfreq, pmag: peak frequencies in Hz and magnitudes,
    # f0c: frequencies of f0 candidates
    # returns f0, f0Error: fundamental frequency detected and its error

    p = 0.5
    q = 1.4
    r = 0.5
    rho = 0.33
    Amax = max(pmag)
    maxnpeaks = 10                                   # Nombre max de pics
    harmonic = np.matrix(f0c)

    ErrorPM = np.zeros(harmonic.size)                # Table d'erreurs Predicted-to-Measured
    MaxNPM = min(maxnpeaks, pfreq.size)

    for i in range(0, MaxNPM):                      # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)    # harmonic.T => Transposée de harmonic
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
        harmonic = harmonic+f0c

    ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size) :                    # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP]/f0c[i])
        nharm = (nharm>=1)*nharm + (nharm<1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	f0index = np.argmin(Error)                       # get the smallest error
	f0 = f0c[f0index]                                # f0 with the smallest error

	return f0, Error[f0index]



def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    # Englobe la detection de f0, selectionne les frequences candidates et appelle la fonction TWM
    # pfreq, pmag: peak Frequencies and Magnitudes
    # ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
    # f0t: f0 of previous frame if stable
    # returns f0: fundamental frequency in Hz

    if (minf0 < 0):  # raise exception if minf0 is smaller than 0
        raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")

    if (maxf0 >= 10000):  # raise exception if maxf0 is bigger than 10000Hz
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (pfreq.size < 3) & (f0t == 0):  # return 0 if less than 3 peaks and not previous f0
        return 0

    f0c = np.argwhere((pfreq > minf0) & (pfreq < maxf0))[:, 0]  # use only peaks within given range
    if (f0c.size == 0):  # return 0 if no peaks within range
        return 0
    f0cf = pfreq[f0c]  # frequencies of peak candidates
    f0cm = pmag[f0c]  # magnitude of peak candidates

    if f0t > 0:  # if stable f0 in previous frame
        shortlist = np.argwhere(np.abs(f0cf - f0t) < f0t / 2.0)[:, 0]  # use only peaks close to it
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc] % f0t
        if maxcfd > f0t / 2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd > (f0t / 4)):  # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]  # frequencies of candidates

    if (f0cf.size == 0):  # return 0 if no peak candidates
        return 0

    f0, f0error = TWM(pfreq, pmag, f0cf)  # call the TWM function with peak candidates

    if (f0 > 0) and (f0error < ef0max):  # accept and return f0 if below max error allowed
        return f0
    else:
        return 0

#*************************************************************************
#Trololo
#*************************************************************************

#Lecture du son
(fs, x) = uF.wavread('../sounds/piano.wav')

# Parametres de l'algo
N = 2048
t = -90
minf0 = 40
maxf0 = 300
f0et = 1
maxnpeaksTwm = 4
H = 128
x1 = x[1.5 * fs:1.8 * fs]


# Algo
hammingWindow = np.hamming(1024)
magnitudeSpectra, phaseSpectra = STFT.stftAnal(x, fs, hammingWindow,N,H)
f0 = f0Twm(phaseSpectra, magnitudeSpectra, hammingWindow, minf0, maxf0, f0et)

print 'F0 trouvée :' + f0.__str__()
#
# code trouve
#



# plt.figure(1, figsize=(9, 7))
# mX, pX = STFT.stftAnal(x, fs, w, N, H)
# f0 = HM.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et)
# f0 = UF.cleaningTrack(f0, 5)
# yf0 = SM.sinewaveSynth(f0, .8, H, fs)
# f0[f0 == 0] = np.nan
# maxplotfreq = 800.0
# numFrames = int(mX[:, 0].size)
# frmTime = H * np.arange(numFrames) / float(fs)
# binFreq = fs * np.arange(N * maxplotfreq / fs) / N
# plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N * maxplotfreq / fs + 1]))
# plt.autoscale(tight=True)
#
# plt.plot(frmTime, f0, linewidth=2, color='k')
# plt.autoscale(tight=True)
# plt.title('mX + f0 (piano.wav), TWM')
#
# plt.tight_layout()
# plt.savefig('f0Twm-piano.png')
# UF.wavwrite(yf0, fs, 'f0Twm-piano.wav')
# plt.show()



#Fin du code