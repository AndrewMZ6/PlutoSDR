
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import scipy.signal as sigs


mode = 'singl'
fs = config.SAMPLE_RATE

BEARD_FREQ_SHIFT= 2_000_000 # 915e6
NO_BEARD_FREQ_SHIFT = 11000 
scatter_color = '#006699'
spectrum_color = '#b35900'
correlation_color = '#26734d'


# find connected devices
devices = utils.detect_devices()
if devices is None:
    raise NoDeviceFoundException('No connected devices found')


# create and modulate random bits
bits = mod.create_bits(config.BIT_SIZE)
mod_data = mod.qpsk_modualte(bits)


# create spectrum and time samples
spec = mod.put_data_to_zeros(config.FOURIER_SIZE, config.GUARD_SIZE, mod_data)
time_sig = spec.get_time_samples()
time_sig = np.append(time_sig, time_sig)
sig_len = len(time_sig)
print(f"time_sig len: {sig_len}")
spec_tx = np.fft.fftshift(np.fft.fft(time_sig))


# original signal graphs
fig1, axs1 = plt.subplots(1, 2)
axs1[0].plot(np.abs(spec_tx), spectrum_color)
axs1[0].set_title('generated sig')
axs1[1].scatter(spec_tx.real, spec_tx.imag, color=scatter_color)
axs1[1].set_title('constellation')
for i in axs1: i.grid()


sdrtx = adi.Pluto(devices['NOBEARD'])
if mode != 'single':
    # creating sdr instances
    sdrrx = adi.Pluto(devices['BEARD'])
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
sdrtx.tx(time_sig)


# receiving
data_recv = sdrrx.rx()
recv_len = len(data_recv)
print(f"data_recv len = {recv_len}")


# creating spectrum of recieved data
spec_recv = np.fft.fftshift(np.fft.fft(data_recv))


# correlation 
corr = np.correlate(data_recv, time_sig, 'full')
lags = sigs.correlation_lags(len(data_recv), len(time_sig))
abscorr = np.abs(corr)
m = np.max(abscorr)
maxcorr = abscorr.argmax(axis=0)


# received spec, constellation and correlation graphs
fig2, axs = plt.subplots(2, 2)
axs[0][0].plot(np.abs(spec_recv), spectrum_color)
axs[0][0].set_title('received sig spec')
axs[0][1].scatter(spec_recv.real, spec_recv.imag, color=scatter_color, marker='.')
axs[0][1].set_title('received constellation')
axs[1][0].plot(abscorr, correlation_color)
axs[1][0].annotate(f'max={maxcorr:.1f}', xy=(maxcorr, m), xytext=(maxcorr + 1, m + 1000000), arrowprops=dict(facecolor='black', shrink=0.05))
axs[1][0].set_title('correlation')
for i in axs: i[0].grid(); i[1].grid()


# define left and right cutoff indexes
print(f"max index: {maxcorr}")
if maxcorr <= recv_len - sig_len:
    print("if works here")
    left_ind = np.abs(lags[maxcorr])
    right_ind = left_ind + sig_len 
    print(f"cutting left index: {left_ind}")
    print(f"cutting right index: {right_ind}")
else:
    print("else works here")
    left_ind = np.abs(lags[maxcorr]) - sig_len
    right_ind = left_ind + sig_len 
    print(f"cutting left index: {left_ind}")
    print(f"cutting right index: {right_ind}")


# cutting off
cut_data = data_recv[left_ind:right_ind]
cut_data_spec = np.fft.fftshift(np.fft.fft(cut_data))

print(f"data cutted len: {len(cut_data_spec)}")


# cut data graphs
fig3, axs = plt.subplots(2, 2)
axs[0][1].plot(np.abs(cut_data_spec), spectrum_color)
axs[0][1].set_title('absolute cut_data_spec')
axs[1][0].scatter(cut_data_spec.real, cut_data_spec.imag, color=scatter_color, marker='.')
axs[1][0].set_title('cut constellation')
for i in axs: i[0].grid(); i[1].grid()


# correlating part1 and part2 of cutted data
part1 = cut_data[:1024]
part2 = cut_data[1024:]

corr2 = np.correlate(part1, part2, 'full')

abscorr2 = np.abs(corr2)
maxx = abscorr2.argmax()

complex_max = corr2[maxx]


fig4, ax = plt.subplots(2, 2)
ax[0][0].plot(np.abs(corr2), correlation_color)
ax[0][0].set_title('part1 part2 correlation')

part1_spce = np.fft.fftshift(np.fft.fft(part1))
ax[1][1].scatter(part1_spce.real, part1_spce.imag, color=scatter_color, marker='.')
ax[1][1].set_title('part1 before freq ')


fftsize = config.FOURIER_SIZE
ang = np.angle(complex_max)
for i in range(len(part1)):
    part1[i] = part1[i]*np.exp(1j*i*(ang/fftsize))


#eq = part1/time_sig[:1024]
#np.save(r"/media/andrew/PlutoSDR/eq.npy", eq)
eq = np.load(r"/media/andrew/PlutoSDR/eq.npy")
eqed = part1/eq
eqedspec = np.fft.fftshift(np.fft.fft(eqed))


part1 = np.fft.fftshift(np.fft.fft(part1))
ax[0][1].plot(np.abs(part1), spectrum_color)
ax[0][1].set_title('abs part1')
ax[1][0].scatter(part1.real, part1.imag, color=scatter_color,marker='.')
ax[1][0].set_title('part1')
for i in ax: i[0].grid(); i[1].grid()




fig6, ax = plt.subplots()
ax.scatter(eqedspec.real, eqedspec.imag, color=scatter_color,marker='.')
ax.set_title('eqed')


sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()
plt.show()

exit()
eq = []
axs[0][0].scatter(d.real, d.imag)
axs[0][0].set_title('d.real, d.imag')
d = d**2

axs[1][0].scatter(cut_data_spec.real, cut_data_spec.imag)
axs[1][0].set_title('cut data spec real amd imag')

axs[1][1].plot(np.absolute(cut_data_spec))
axs[1][1].set_title('absolute cut data spec')



plt.show()

'''
for j in range(4):
    
    
    for i in range(10):
        #
        data = sdr2.rx()
       
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
        
        axs[0][1].set_title('Constelation')
        axs[1][0].set_title('corr')
        axs[1][1].set_title('shifted')
        
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
    print("Next iteration")
    sdr2.rx_destroy_buffer()
    sdr.tx_destroy_buffer()
    

stop = perf_counter()
#plt.show()
print(ar)
print(len(ar))
print(f'time: {stop-start}')
print(f'speed: {((config.BIT_SIZE/8)*iter_num)/(stop-start)} kBps')
'''

