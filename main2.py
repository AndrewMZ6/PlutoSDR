import adi
import numpy as np
from time import sleep, process_time, perf_counter
from matplotlib import pyplot as plt
import mod
import config
import commpy as cp
import utils
from devexceptions import NoDeviceFoundException


# find connected devices
devices = utils.detect_devices()

if devices is None:
    raise NoDeviceFoundException('No connected devices found')

# create and modulate random bits
bits = mod.create_bits(1024)
mod_data = mod.qpsk_modualte(bits)
mod_data_n = mod.normalize_for_pluto(mod_data)


rr = len(mod_data_n)
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


sdr.tx(mod_data_n)

#for freq in 
req = recv = sdr2.rx()

# normalization
m = np.max(np.abs(recv))
recv = recv/m
recv = recv


fs = config.SAMPLE_RATE
f = np.linspace(-fs/2.0, fs/2.0, len(recv))
cor = np.correlate(req, mod_data_n, 'full')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.scatter(recv.real, recv.imag)
ax1.grid()
ax2.plot(f, np.abs(np.fft.fftshift(np.fft.fft(recv))))
ax2.grid()
ax3.plot(np.abs(cor))
ax3.grid()
plt.show()


sdr.tx_destroy_buffer()
sdr2.rx_destroy_buffer()

exit()

# normalize recv
mm = np.max(np.abs(recv))
recv_n = recv/mm

recv_n = recv_n


print(type(recv), len(recv))
print(recv[:10])


fs = config.SAMPLE_RATE
spec = np.fft.fftshift(np.fft.fft(recv_n))
f = np.linspace(-fs/2.0, fs/2.0, len())
#plt.scatter(recv_n.real, recv_n.imag)
plt.plot(np.abs(spec))
plt.show()

sdr.tx_destroy_buffer()
sdr2.rx_destroy_buffer()