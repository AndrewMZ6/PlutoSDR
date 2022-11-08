import mod_utils
import numpy as np
from matplotlib import pyplot as plt
import utils
from matplotlib.animation import FuncAnimation
import time
import threading



rand = np.random.rand

def rand_complex(n):
	real = 2*rand(n) - 1
	imag = 2*rand(n) - 1
	return real + 1j*imag



fig2, axes = plt.subplots(1, 2)
fig2.set_facecolor(color='#a4a4c1')
plot1, = axes[0].plot([], [], marker='o')
scat1 = axes[1].scatter([], [], marker='o')
axes[1].set_ylim(-1.1, 1.1); axes[1].set_xlim(-1.1, 1.1)
axes[0].set_ylim(-0.1, 1.5); axes[0].set_xlim(-0.1, 50.1)
axes[0].grid(); axes[1].grid()
axes[0].set_facecolor(color='#d1d1e0'); axes[1].set_facecolor(color='#d1d1e0')


def update_func(frames, scat1, plot1):

	ar = rand_complex(50)
	d = np.array([ar.real, ar.imag])
	scat1.set_offsets(d.T)

	plot1.set_data((np.arange(len(ar))), np.abs(ar))

	return scat1, plot1


aaanimation = FuncAnimation(fig2, func=update_func, fargs=(scat1, plot1), interval=100, blit=True)



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





plt.show()



