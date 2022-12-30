from matplotlib import pyplot as plt
import numpy as np


x = np.arange(0, np.pi, 0.1)
y = np.sin(x)
y2 = np.cos(x)


fig, ax = plt.subplots(2, 1, figsize=(3, 6))

ax[0].plot(x, y, linestyle='None', marker='.')
ax[1].plot(x, y2)
ax[0].grid(); ax[1].grid()
ax[0].set_xlabel('time')
ax[0].set_ylabel('amplitude')
ax[0].set_title('sub1')


ax[1].set_xlabel('time')
ax[1].set_ylabel('amplitude')
ax[1].set_title('sub2')

plt.show()