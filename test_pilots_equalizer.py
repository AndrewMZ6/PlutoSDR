import mod_utils
import numpy as np
from matplotlib import pyplot as plt
import utils
from matplotlib.animation import FuncAnimation
import time



phi = np.linspace(0, 2, 100)
a = np.exp(1j*phi)
b = np.exp(+1j*0.1)
c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*2)


plt.ion()
plt.grid()
for i in phi:
    c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*i)

    plt.clf()
    plt.scatter(c.real, c.imag)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.5)

plt.ioff()
plt.show()

