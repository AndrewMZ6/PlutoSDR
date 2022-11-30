import devices
import numpy as np
import mod
import data_gen
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

sdrtx, sdrrx = devices.initialize_sdr(single_mode=True, tx='ANGRY', swap=True)

tx_constant_data = np.zeros(1000, dtype=complex)
for i in range(0, tx_constant_data.size, 8):
    tx_constant_data[i:i+8] = np.array([1+1j, 1-1j, -1+1j, -1-1j, 2+2j, 2-2j, -2+2j, -2-2j])
print(tx_constant_data[:16])


#transmitted_user_bits, tx_constant_data = data_gen.generate_tx_data(frames=5)
m = mod._normalize_for_pluto(tx_constant_data)
print(m[:5])
sdrtx.tx(m)

tx_constant = tx_constant_data*300


fig, ax = plt.subplots()
ax.grid()
scat = ax.scatter(tx_constant.real, tx_constant.imag)



def func(frames, scat):

    receivced_data = sdrrx.rx()
    data = np.array([receivced_data.real, receivced_data.imag])
    scat.set_offsets(data.T)

    return scat,

'''
print(receivced_data[:20])
print(len(receivced_data))

sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()

plt.scatter(receivced_data[:1000].real, receivced_data[:1000].imag)
'''


animation = FuncAnimation(fig,
                            func=func,
                            fargs=(scat, ),
                            interval=10,
                            blit=True,
                            repeat=True)





plt.show()