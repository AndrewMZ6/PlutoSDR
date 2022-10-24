from pickle import TRUE
import adi
import numpy as np
from time import sleep, process_time, perf_counter
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException



#def crc(bits):
#    prev = zlib.crc32(bits,prev)


# find connected devices
devices = utils.detect_devices()

if devices is None:
    raise NoDeviceFoundException('No connected devices found')

cent_freq = config.CENTRAL_FREQUENCY
sample_rate = config.SAMPLE_RATE
num_samps = config.COMPLEX_SAMPLES_NUMBER


# create and modulate random bits
bits = mod.create_bits(config.BIT_SIZE)
mod_data = mod.qpsk_modualte(bits)


# create spectrum and time samples
spec = mod.put_data_to_zeros(config.FOURIER_SIZE, config.GUARD_SIZE, mod_data)
time_sig = spec.get_time_samples()
rr = len(time_sig)
print(np.max(time_sig))

'''
fig1, ax1 = plt.subplots()
ax1.plot(np.absolute(spectrum))
ax1.grid()
ax1.set_title('original spectrum')
'''

# setting up tx
sdr = adi.Pluto(devices['BEARD'])
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(cent_freq)
sdr.tx_hardwaregain_chan0 = 0
#sdr.tx_cyclic_buffer = True



#while TRUE
sdr.tx(time_sig)

# setting up rx
sdr.rx_lo = int(cent_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0

fig, axs = plt.subplots(2, 2)

ar = np.array([])
eq = []

start = perf_counter()
iter_num = 100
for i in range(iter_num):
    
    data = sdr.rx()
    #print(f"data size: {len(data)}")

    
    axs[0][0].cla()
    
    axs[0][1].cla()
    axs[1][0].cla()
    axs[1][1].cla()

    
    cor = np.correlate(data, time_sig, 'full')
    #print(f"cor len: {len(cor)}")
    x1 = cor.argmax(axis=0)
    
    #print(f"max index: {x1}")
    if x1 <= 944:
        #print("if works here")
        left_ind = x1 - int(rr/2) +1 
        right_ind = left_ind + rr 
        #print(f"cutting left index: {left_ind}")
        #print(f"cutting right index: {right_ind}")
    else:
        #print("else works here")
        left_ind = x1 - int(rr/2) - rr +1 
        right_ind = left_ind + rr 
        #print(f"cutting left index: {left_ind}")
        #print(f"cutting right index: {right_ind}")

    cut_data = data[left_ind:right_ind]
    cut_data_spec = np.fft.fft(cut_data)
    cut_data_spec = np.fft.fftshift(cut_data_spec)
    zero_index = int(config.FOURIER_SIZE/2)
    cut_spec = cut_data_spec[config.GUARD_SIZE:zero_index]
    cut_spec = np.append(cut_spec,cut_data_spec[zero_index + 1:-config.GUARD_SIZE + 1])
    print(len(cut_spec))
    #
    #ar = np.append(ar, cut_data)
    #print(f"cut len: {len(cut_data)}", end='\n---------------------------\n')
    
    #d1 = np.fft.fft(cut_data)
    #d1_shifted = np.fft.fftshift(d1)
    #dd = np.fft.fftshift(np.fft.fft(cut_data))

    if not len(eq):
        eq = mod_data/cut_spec

    axs[0][0].set_title('Spectrum')
    '''
    axs[0][1].set_title('Constelation')
    axs[1][0].set_title('corr')
    axs[1][1].set_title('shifted')
    '''
    #axs[0][0].plot(np.absolute(d1))
    recov = cut_spec*eq
    axs[0][0].scatter(recov.real, recov.imag)
    axs[0][1].scatter(eq.real, eq.imag)
    axs[1][0].plot(np.absolute(cor))
    axs[1][1].plot(np.absolute(cut_spec*eq))
    rec_bits = mod.qpsk_demodulate(recov)
    print(rec_bits[:10])
    print(bits[:10])
    #print(data[:5])
    #input()
    plt.grid()
    plt.pause(1)
    

stop = perf_counter()
#plt.show()
print(ar)
print(len(ar))
print(f'time: {stop-start}')
print(f'speed: {((config.BIT_SIZE/8)*iter_num)/(stop-start)} kBps')