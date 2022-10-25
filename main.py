
import adi
import numpy as np
from time import sleep, process_time, perf_counter
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException


BEARD_FREQ_SHIFT= 17000 # 915e6
NO_BEARD_FREQ_SHIFT = 11000 


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
#time_sig = np.append(time_sig, time_sig)
rr = len(time_sig)
print(f"time_sig len: {rr}")

sdr = adi.Pluto(devices['BEARD'])
sdr2 = adi.Pluto(devices['NOBEARD'])


# setting up tx
sdr.sample_rate = int(config.SAMPLE_RATE)
sdr.tx_rf_bandwidth = int(config.SAMPLE_RATE)
sdr.tx_lo = int(config.CENTRAL_FREQUENCY)
sdr.tx_hardwaregain_chan0 = 0
sdr.tx_cyclic_buffer = True



# setting up rx
sdr2.rx_lo = int(config.CENTRAL_FREQUENCY)
sdr2.rx_rf_bandwidth = int(config.SAMPLE_RATE)
sdr2.rx_buffer_size = config.COMPLEX_SAMPLES_NUMBER
sdr2.gain_control_mode_chan0 = 'manual'
sdr2.rx_hardwaregain_chan0 = 0.0


# transmission
sdr.tx(time_sig)


# wating for 3 sec
sleep(3)

# receiving
data_recv = sdr2.rx()


fig, axs = plt.subplots(1, 2)
spec = np.fft.fft(data_recv)

spec_max = np.max(np.abs(spec))
spec_normilized = spec/spec_max
#spec_normilized = np.array([i for i in spec_normilized if np.abs(i) > 0.04])
spec_normilized = spec_normilized**4
axs[0].plot(np.abs(spec_normilized))
axs[1].scatter(spec_normilized.real, spec_normilized.imag)
plt.show()
exit()
# creating axes
cor = np.correlate(data_recv, time_sig, 'full')

part1 = data_recv[512:512 + int(rr/2)]
part2 = data_recv[512 + int(rr/2): 512 + rr]


cor = np.correlate(part1, part2, 'full')

print(len(part1), len(part2))
data_recv = data_recv
ddd = np.fft.fftshift(np.abs(np.fft.fft(data_recv)))

f = np.linspace(-config.SAMPLE_RATE/2.0, config.SAMPLE_RATE/2.0, len(ddd))

plt.plot(np.abs(cor))
plt.show()
exit()

d = np.fft.fft(data_recv)
m = np.max(np.absolute(d))
d = d/m

fig, axs = plt.subplots(2, 2)




print(f"cor len: {len(cor)}")
x1 = cor.argmax(axis=0)

print(f"max index: {x1}")
if x1 <= 2944:
    print("if works here")
    left_ind = x1 - int(rr/2) +1 
    right_ind = left_ind + rr 
    print(f"cutting left index: {left_ind}")
    print(f"cutting right index: {right_ind}")
else:
    print("else works here")
    left_ind = x1 - int(rr/2) - rr +1 
    right_ind = left_ind + rr 
    print(f"cutting left index: {left_ind}")
    print(f"cutting right index: {right_ind}")

cut_data = data_recv[left_ind:right_ind][:512]
cut_data_spec = np.fft.fft(cut_data)
cut_data_spec = np.fft.fftshift(cut_data_spec)

print(len(cut_data_spec))




axs[0][0].plot(np.absolute(cor))
axs[0][0].set_title('absolute cor')

axs[0][1].plot(np.absolute(cut_data_spec))
axs[0][1].set_title('absolute cut_data_spec')

axs[1][0].plot(np.abs(d))
axs[1][0].set_title('before **2')

data_recv = data_recv**2
d = np.fft.fft(data_recv)
m = np.max(np.absolute(d))
d = d/m

axs[1][1].plot(np.abs(d))
axs[1][1].set_title('after **2')


plt.show()
sdr.tx_destroy_buffer()
sdr2.rx_destroy_buffer()

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

