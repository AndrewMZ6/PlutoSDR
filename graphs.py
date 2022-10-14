import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


fig, ax = plt.subplots()
ln, = ax.plot([])


def get_rand(frame):
    data = np.random.randint(low=0, high=2, size=50)
    ln.set_data(data)
    return ln,

ani = FuncAnimation(fig, func=get_rand)
plt.show()