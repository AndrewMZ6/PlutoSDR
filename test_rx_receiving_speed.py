import adi
from time import sleep, time
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp


mode = 'single'
receiver    = 'FISHER'
receiver    = 'ANGRY'




# find connected devices
devices = utils.detect_devices()
if devices is None:
    raise NoDeviceFoundException('No connected devices found')



sdrrx = adi.Pluto(devices[receiver])



# setting up rx
sdrrx.rx_lo = int(config.CENTRAL_FREQUENCY)
sdrrx.rx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrrx.rx_buffer_size = config.COMPLEX_SAMPLES_NUMBER
sdrrx.gain_control_mode_chan0 = 'manual'
sdrrx.rx_hardwaregain_chan0 = 0.0

print('sleeping for 2 sec')
sleep(2)

start = time()


n = 10
for i in range(n):
    if not i:
        c = time()

    data = sdrrx.rx()

end = time()
print(f"time of {n} iterations: {end - start:.4f} seconds")
print(f"time of one measured operation {c - start:.7f} seconds")
print(f"time of one calculated operation {(end - start)/n:.4f}")
print(f"received data length: {len(data)}")


speed = config.COMPLEX_SAMPLES_NUMBER/((end - start)/n)
print(f"estimated recieving speed {speed:.2f} complex vectors per second")