import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sigs


a = 1000
b = 250
r = np.random.randint(-1, high=2, size=a)

t = np.linspace(start=0, stop=10, num=b)
sin = np.sin(t)
cos = np.cos(t)
signal = np.concatenate((sin[:int(b/4)], cos[:int(b/4)], sin[int(b/4):], cos[int(b/4):]))
#signal = sin
print(f"sin length: {len(signal)}")

conc = np.concatenate((r, signal, r[:int(len(r)/4)]))
print(f"conc length {len(conc)}")


convo = np.convolve(signal[::-1], conc)
print(f"convo length: {len(convo)} of type {type(convo[10])}")

corr = np.correlate(conc, signal, 'full')
print(f"corr length: {len(corr)} of type {type(corr[10])}")

fig, (ax1, ax2) = plt.subplots(2, 2)

print(f"corr max is at X coord: {corr.argmax()}")

lags = sigs.correlation_lags(len(signal), len(conc))
print(f"supposed lags is {lags[abs(corr.argmax())]}")

ax1[0].plot(conc, marker='.')
ax1[1].plot(convo, marker='.')
ax1[1].set_title('convolution')
ax2[1].plot(corr, marker='.')
ax2[1].set_title('correlation')
ax2[0].plot(lags, marker='.')
ax2[0].set_title('lags')
for i in (ax1, ax2): i[0].grid(); i[1].grid()
plt.show()