import mod_utils
import numpy as np
from matplotlib import pyplot as plt
import utils
from matplotlib.animation import FuncAnimation
import time



phi = np.linspace(0, 10, 1000)
c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*2)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid()
scat = ax1.scatter(c.real, c.imag)

ax2.grid()
ax2.set_ylim(-1, 1)
line, = ax2.plot(np.arange(10), np.arange(10), marker='.')


def update_plot(frame, line):
    ydata = np.random.rand(10)*2 - 1
    line.set_ydata(ydata)

    return line, 


def update_scatter(frames, scat):
    c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*frames)
    data = np.array([c.real, c.imag])
    scat.set_offsets(data.T)

    return scat, 

'''plt.ion()




for i in phi:
    c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*i)

    scat.set_offsets((c.real, c.imag))
    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.5)


plt.ioff()'''


animation = FuncAnimation(
    fig,
    func=update_scatter,
    frames = phi,
    fargs=(scat, ),
    interval=30,
    blit=True,
    repeat=True)


animation2 = FuncAnimation(
    fig,
    func=update_plot,
    fargs=(line, ),
    interval=30,
    blit=True,
    repeat=True)

plt.show()
