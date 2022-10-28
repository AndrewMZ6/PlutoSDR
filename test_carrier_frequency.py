import adi
import config
import utils
import mod
import numpy as np
from devexceptions import NoDeviceFoundException


# find connected devices
devices = utils.detect_devices()
if devices is None:
    raise NoDeviceFoundException('No connected devices found')


testing_pluto = 'NOBEARD'

# generate data
#create and modulate random bits
bits = mod.create_bits(config.BIT_SIZE)
mod_data = mod.qpsk_modualte(bits)


# create spectrum and time samples
spec = mod.put_data_to_zeros(config.FOURIER_SIZE, config.GUARD_SIZE, mod_data)
time_sig = spec.get_time_samples()
time_sig = np.append(time_sig, time_sig)




sdrtx = adi.Pluto(devices[testing_pluto])
sdrtx.sample_rate = int(config.SAMPLE_RATE)
sdrtx.tx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrtx.tx_lo = int(config.CENTRAL_FREQUENCY)
sdrtx.tx_hardwaregain_chan0 = 0
sdrtx.tx_cyclic_buffer = True




sdrtx.tx(time_sig)
input()

exit()