import numpy as np
import matplotlib.pyplot as plt

snrs128 = np.load('snr_array128.npy')
bers128 = np.load('biterrs_array128.npy')

snrs256 = np.load('snr_array256.npy')
bers256 = np.load('biterrs_array256.npy')

snrs512 = np.load('snr_array512.npy')
bers512 = np.load('biterrs_array512.npy')

snrs1024 = np.load('snr_array1024.npy')
bers1024 = np.load('biterrs_array1024.npy')

fig, ax = plt.subplots(2, 1)

ax[0].set_yscale('log')
ax[0].set_title('BER / SNR (dB)')
ax[0].grid()
ax[0].plot(snrs128, bers128)
ax[0].plot(snrs256, bers256)
ax[0].plot(snrs512, bers512)
ax[0].plot(snrs1024, bers1024)
ax[0].legend(['128', '256', '512', '1024'])


ax[1].grid()
ax[1].plot(snrs128, bers128)
ax[1].plot(snrs256, bers256)
ax[1].plot(snrs512, bers512)
ax[1].plot(snrs1024, bers1024)
ax[1].legend(['128', '256', '512', '1024'])

plt.show()