
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp


# initializing constants
mode        = 'single1'
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
transmitter, receiver = receiver, transmitter


# find connected devices
devices = utils.detect_devices()
if devices is None:
    raise NoDeviceFoundException('No connected devices found')


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


sdrtx = adi.Pluto(devices[transmitter])
if mode != 'single':
    sdrrx = adi.Pluto(devices[receiver])
else:
    sdrrx = sdrtx


# setting up tx
sdrtx.sample_rate = int(config.SAMPLE_RATE)
sdrtx.tx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrtx.tx_lo = int(config.CENTRAL_FREQUENCY)
sdrtx.tx_hardwaregain_chan0 = 0
sdrtx.tx_cyclic_buffer = True


# setting up rx
sdrrx.rx_lo = int(config.CENTRAL_FREQUENCY)
sdrrx.rx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrrx.rx_buffer_size = config.COMPLEX_SAMPLES_NUMBER
sdrrx.gain_control_mode_chan0 = 'manual'
sdrrx.rx_hardwaregain_chan0 = 0.0


# transmission
sdrtx.tx(tx_signal)


fig, axes = plt.subplots(5, 2)

for ne in range(1000):

    for ax in axes:
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



    eq = utils.equalize(preambula_time_domain[1024:2048], part2)
    data_eq = utils.remove_spectrum_zeros(data)

    axes[2][0].scatter(eq.real, eq.imag, color=scatter_color, marker='.')
    axes[2][0].set_title('equalizer')


    s = []
    for spectrum in data_eq:
        q = spectrum*eq
        q_abs = np.abs(q)
        m = np.max(q_abs)
        q_normilized = (q/m)*1.4142
        s.append(q_normilized)


    data_eq_spectrum_shifted = data_eq

    for i in range(5):
            axes[i][1].scatter(s[i].real, s[i].imag, color=scatter_color,marker='.')
            axes[i][1].set_title(f's[{i}]')
            axes[i][1].grid()
            axes[i][0].grid()

    demod_data = mod.qpsk_demodulate(s[0])
    sdrrx.rx_destroy_buffer()

    plt.pause(0.5)



plt.show()
sdrtx.tx_destroy_buffer()

