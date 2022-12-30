
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp
import devices


# initializing constants
mode        = 'single'
#transmitter = 'BEARD'
transmitter = 'ANGRY'
receiver    = 'FISHER'
fs          = config.SAMPLE_RATE
fftsize     = config.FOURIER_SIZE
scatter_color       = '#006699'
spectrum_color      = '#b35900'
correlation_color   = '#26734d'
spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


# swap transmitter and receiver
#transmitter, receiver = receiver, transmitter

sdrtx, sdrrx = devices.initialize_sdr(single_mode=False, tx='RED_PIMPLE_RX')


# create and modulate random bits
preambula = mod.get_preambula()


# create spectrum and time samples
preambula_spectrum = preambula.get_spectrum()
preambula_time_domain = preambula.get_time_samples()
preambula_time_domain = np.concatenate((preambula_time_domain, preambula_time_domain))
preambula_length = len(preambula_time_domain)



tx_signal = np.array([])

for _ in range(5):
    # data payload
    data_bits = mod.create_bits(config.BIT_SIZE)[:1648]
    

    data_modulated = mod.qpsk_modualte(data_bits)
    if _ == 0:
        data_compare = data_bits

    # create spectrum and time samples
    data_spectrum = mod.PutDataToZeros(config.FOURIER_SIZE, config.GUARD_SIZE, data_modulated)
    data_time_domain = data_spectrum.get_time_samples()
    tx_signal = np.append(tx_signal, data_time_domain)


reference_data = tx_signal
tx_signal = np.concatenate((preambula_time_domain, tx_signal))


assert (preambula_time_domain[:1024] == preambula_time_domain[1024:2048]).all(), 'preambulas part1 and 2 are different!'




# transmission
sdrtx.tx(tx_signal)


fig, axes = plt.subplots(5, 2)
fi2g, axes2 = plt.subplots(5, 2)

for ne in range(1000):

    for ax in axes:
        ax[0].cla(); ax[1].cla()

    for ax in axes2:
        ax[0].cla(); ax[1].cla()
    # receiving
    data_recieved = sdrrx.rx()
    recived_data_length = len(data_recieved)


    # creating spectrum of recieved data
    spectrum_data_recived = np.fft.fftshift(np.fft.fft(data_recieved))


    # first correlation 
    cutted, abs_first_correlation = dp.correlation(preambula_time_domain, data_recieved, 0)

    # received spec, constellation and correlation graphs
    axes[0][0].plot(np.abs(spectrum_data_recived), spectrum_color)
    axes[0][0].set_title('received sig spec')

    axes[1][0].plot(abs_first_correlation, correlation_color)
    axes[1][0].set_title('correlation')


    # cutting off
    cut_data = cutted
    cut_data_spec = spectrum_and_shift(cut_data)


    # correlating part1 and part2 of cutted data
    part1, part2, data = cut_data[:1024], cut_data[1024:2048], cut_data[2048:]

    print(len(data))
    corr2 = np.correlate(part2, part1, 'full')

    abscorr2 = np.abs(corr2)
    maxx = abscorr2.argmax()
    complex_max = corr2[maxx]


    first_OFDM_symbol = spectrum_and_shift(data[:1024])
    axes[3][0].scatter(first_OFDM_symbol.real, first_OFDM_symbol.imag, color=scatter_color, marker='.')
    axes[3][0].set_title('first_OFDM_symbol before freq ')

    

    first_OFDM_symbol = spectrum_and_shift(data[:1024])
    axes[4][0].scatter(first_OFDM_symbol.real, first_OFDM_symbol.imag, color=scatter_color, marker='.')
    axes[4][0].set_title('first_OFDM_symbol after freq ')


    '''
    # variant 1
    ang2 = np.arctan2(complex_max.imag, complex_max.real)

    for i in range(len(data)):
        data[i] = data[i]*np.exp(1j*i*(ang2/fftsize))
    '''


    '''
    # variant2
    dphi = np.angle(complex_max)
    dt = 1/fs
    tau = config.FOURIER_SIZE*dt
    ocen_freq = dphi/(2*np.pi*tau)
    dphi_ocen = (ocen_freq*2*np.pi)/fs  


    for i in range(len(data)):
        data[i] = data[i]*np.exp(1j*i*(-dphi_ocen))
    '''



    eq = utils.equalize(preambula_time_domain[:1024], part1)
    data_eq = utils.remove_spectrum_zeros(data)

    axes[2][0].scatter(eq.real, eq.imag, color=scatter_color, marker='.')
    axes[2][0].set_title('equalizer')


    s = []
    for spectrum in data_eq:
        try:
            q = spectrum*eq
            q_abs = np.abs(q)
            m = np.max(q_abs)
            q_normilized = (q/m)*1.4142
            s.append(q_normilized)
        except ValueError:
            print('-> Value Error occured')


    data_eq_spectrum_shifted = data_eq


    for i in range(5):
        try:
            axes[i][1].scatter(s[i].real, s[i].imag, color=scatter_color,marker='.')
            axes[i][1].set_title(f's[{i}]')
            axes[i][1].grid()
            axes[i][0].grid()
        except IndexError:
            print('Index Error ')


    a1 = data[0:1024]
    a2 = data[1024:2048]
    a3 = data[2048:3072]
    a4 = data[3072:4096]
    a5 = data[4096:5120]
    
    a1 = np.fft.fft(a1)
    a2 = np.fft.fft(a2)
    a3 = np.fft.fft(a3)
    a4 = np.fft.fft(a4)
    a5 = np.fft.fft(a5)

    try:
        axes2[0][0].scatter(a1.real, a1.imag, color=scatter_color,marker='.')
        axes2[0][0].set_title(f's[0]')
        axes2[0][0].grid()

        axes2[1][0].scatter(a2.real, a2.imag, color=scatter_color,marker='.')
        axes2[1][0].set_title(f's[1]')
        axes2[1][0].grid()

        axes2[2][0].scatter(a3.real, a3.imag, color=scatter_color,marker='.')
        axes2[2][0].set_title(f's[2]')
        axes2[2][0].grid()

        axes2[3][0].scatter(a4.real, a4.imag, color=scatter_color,marker='.')
        axes2[3][0].set_title(f's[3]')
        axes2[3][0].grid()


        axes2[4][0].scatter(a5.real, a5.imag, color=scatter_color,marker='.')
        axes2[4][0].set_title(f's[4]')
        axes2[4][0].grid()
        
    except IndexError:
        print('Index Error ')


    demod_data = mod.qpsk_demodulate(s[0])
    sdrrx.rx_destroy_buffer()

    plt.pause(1)



plt.show()
sdrtx.tx_destroy_buffer()

