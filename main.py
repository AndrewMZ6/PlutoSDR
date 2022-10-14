import adi
import numpy as np
from time import sleep, process_time
from matplotlib import pyplot as plt
import mod
import config
import utils

devices = utils.detect_devices()
print(devices)
cent_freq = config.CENTRAL_FREQUENCY
sample_rate = config.SAMPLE_RATE
num_samps = config.COMPLEX_SAMPLES_NUMBER

bits = mod.create_bits(config.BIT_SIZE)
mod_data = mod.qpsk_modualte(bits)

spec = mod.put_data_to_zeros(config.FOURIER_SIZE, config.GUARD_SIZE, mod_data)
spectrum = spec.get_spectrum()
time_sig = spec.get_time_samples()

sdr = adi.Pluto(devices['BEARD'])
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(cent_freq)
sdr.tx_hardwaregain_chan0 = 0
sdr.tx_cyclic_buffer = True
sdr.tx(time_sig)

#sdr2 = adi.Pluto(devices['NOBEARD'])
sdr.rx_lo = int(cent_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0

fig, axs = plt.subplots(1, 2)


for i in range(10):
    
    data = sdr.rx()
    axs[0].cla()
    axs[1].cla()
    cor = np.correlate(data, time_sig, 'full')
    d1 = np.fft.fft(data)
    d1_shifted = np.fft.fftshift(d1)
    dd = np.fft.fftshift(np.fft.fft(data))

    axs[0].set_title('Spectrum')
    axs[1].set_title('Constelation')

    axs[0].plot(np.absolute(d1))
    axs[1].scatter(dd.real, dd.imag)
    plt.grid()
    #print(data[:5])
    plt.pause(1)

plt.show()
