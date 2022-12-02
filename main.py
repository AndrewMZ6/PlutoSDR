import devices
import data_gen
from time import sleep
import iio


sdrtx, sdrrx = devices.initialize_sdr(single_mode=True, tx='FISHER')

tx_signal = data_gen.generate_sine(1e4)

sdrtx.tx(tx_signal)
input()
sdrtx.tx_destroy_buffer()

