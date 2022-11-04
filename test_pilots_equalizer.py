import mod_utils
import numpy as np
from matplotlib import pyplot as plt
import utils
from matplotlib.animation import FuncAnimation
import time
import threading



phi = np.linspace(0, np.pi/2, 100)
c = np.array([1+1j, 1-1j, -1+1j, -1-1j])*np.exp(-1j*2)


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_facecolor(color='#a4a4c1',)
ax1.set_facecolor(color='#d1d1e0'); ax2.set_facecolor(color='#d1d1e0')
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


def ff():
    for i in range(10):
        print(i)
        time.sleep(2)


animation = FuncAnimation(
    fig,
    func=update_scatter,
    frames = phi,
    fargs=(scat, ),
    interval=10,
    blit=True,
    repeat=True)


animation2 = FuncAnimation(
    fig,
    func=update_plot,
    fargs=(line, ),
    interval=10,
    blit=True,
    repeat=True)


t = threading.Thread(target=ff)
t.start()


plt.show()



