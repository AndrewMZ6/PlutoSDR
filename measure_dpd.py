import devices
import numpy as np
import mod
import data_gen

sdrtx, sdrrx = devices.initialize_sdr(single_mode=True, tx='ANGRY', swap=False)

tx_constant_data = np.zeros(1000, dtype=complex)
tx_constant_data[:] = np.complex128(1+1j)
print(tx_constant_data[:4])


transmitted_user_bits, tx_constant_data = data_gen.generate_tx_data(frames=5)
m = mod._normalize_for_pluto(tx_constant_data)
print(m[:5])
sdrtx.tx(m)


receivced_data = sdrrx.rx()

print(receivced_data[:20])
print(len(receivced_data))

sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()