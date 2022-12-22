import numpy as np
import commpy as cp
from matplotlib import pyplot as plt
import config


def IDFT(s):
    return np.fft.ifft(s)

def DFT(t):
    return np.fft.fft(t)

def SHIFT(s):
    return np.fft.fftshift(s)


def generate_ofdm_withpilots():
    gsize = config.GUARD_SIZE    
    fftsize = config.FOURIER_SIZE
    P = config.NUMBER_OF_PILOTS

    K = fftsize - 2*gsize - 1 + 1  # -1 of central zero, and +1 of right guard

    # indicies of all subcarriers
    allCarriers = np.arange(K)


    pilotStep = K//P

    # indicies of pilots
    pilotCarriers = allCarriers[::pilotStep]
    

    pilotValue = 2 + 2j

    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])

    P = P + 1

    dataCarriers = np.delete(allCarriers, pilotCarriers)
    
    # bits per symbol
    mu = 2

    # number of payload bits per OFDM symbol
    payLoadBits_per_OFDM = len(dataCarriers)*mu

    # generate random bits
    bits = np.random.randint(low=0, high=2, size=payLoadBits_per_OFDM)

    # modulate
    M = cp.modulation.QAMModem(2**mu)
    modBits = M.modulate(bits)

    # Put complex data to spectrum
    symbol = np.zeros(K, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    symbol[dataCarriers] = modBits

    ofdmSymbol = np.concatenate([np.zeros(gsize, dtype=complex), symbol[:int(K/2)], np.zeros(1, dtype=complex), symbol[int(K/2):], np.zeros(gsize-1, dtype=complex)])

    ofdmSymbolShifted = SHIFT(ofdmSymbol)

    ofdm_time = IDFT(ofdmSymbolShifted)
    return ofdm_time, (pilotCarriers, dataCarriers)


def degenerate_ofdm_withpilots(time_samples, carriersTuple):
    gsize = config.GUARD_SIZE
    fftsize = config.FOURIER_SIZE
    spectrum = DFT(time_samples)
    spectrum = SHIFT(spectrum)
    removedZeros = np.concatenate([spectrum[gsize:int(fftsize/2)], spectrum[int(fftsize/2) + 1: -(gsize - 1)]])

    pilots = removedZeros[carriersTuple[0]]
    datas = removedZeros[carriersTuple[1]]

    return removedZeros, (pilots, datas)


def generate_ofdm_nopilots():
    gsize = config.GUARD_SIZE    
    fftsize = config.FOURIER_SIZE

    K = fftsize - 2*gsize - 1 + 1  # -1 of central zero, and +1 of right guard

    # bits per symbol
    mu = 2

    # number of payload bits per OFDM symbol
    payLoadBits_per_OFDM = K*mu

    # generate random bits
    bits = np.random.randint(low=0, high=2, size=payLoadBits_per_OFDM)

    # modulate
    M = cp.modulation.QAMModem(2**mu)
    modBits = M.modulate(bits)

    ofdmSymbol = np.concatenate([np.zeros(gsize, dtype=complex), modBits[:int(K/2)], np.zeros(1, dtype=complex), modBits[int(K/2):], np.zeros(gsize-1, dtype=complex)])

    ofdmSymbolShifted = SHIFT(ofdmSymbol)
    ofdm_time = IDFT(ofdmSymbolShifted)

    # return time samples
    return ofdm_time


def degenerate_ofdm_nopilots(time_samples):
    gsize = config.GUARD_SIZE
    fftsize = config.FOURIER_SIZE
    spectrum = SHIFT(DFT(time_samples))
    removedZeros = np.concatenate([spectrum[gsize:int(fftsize/2)], spectrum[int(fftsize/2) + 1: -(gsize-1)]])
    print(f'removedZeros length: {len(removedZeros)}')
    return removedZeros


if __name__ == '__main__':
    pp, t = generate_ofdm_withpilots()
    s = SHIFT(DFT(pp))
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.abs(s))
    axes[0].set_title('spectrum')
    axes[0].grid()

    axes[1].scatter(s.real, s.imag)
    axes[1].set_title('constellation')
    axes[1].grid()
    


    g, p = degenerate_ofdm_withpilots(pp, t)

    fig2, axes2 = plt.subplots(1, 2)
    axes2[0].plot(np.abs(g))
    axes2[1].scatter(g.real, g.imag)


    fig3, axes3 = plt.subplots(1, 2)
    axes3[0].plot(np.abs(p[0]))
    axes3[1].scatter(p[0].real, p[0].imag)

    fig4, axes4 = plt.subplots(1, 2)
    axes4[0].plot(np.abs(p[1]))
    axes4[1].scatter(p[1].real, p[1].imag)
    plt.show()   
