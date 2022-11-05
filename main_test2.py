import mod_utils
import numpy as np
from matplotlib import pyplot as plt
import utils


spectrum = np.zeros(1024)
eq = np.zeros(1024)
left_indexes, right_indexes = mod_utils.define_spectrum_data_indexes(fftsize=1024, guardsize=100)





for i, index in enumerate(left_indexes):
    if not i%10:
        spectrum[index] = 2
    else:
        spectrum[index] = 1

for i, index in enumerate(right_indexes):
    if not i%10:
        spectrum[index] = 2
    else:
        spectrum[index] = 1


for i, index in enumerate(left_indexes):
    if not i%10:
        eq[index] = spectrum[index]
    else:
        eq[index] = 0

    
for i, index in enumerate(right_indexes):
    if not i%10:
        eq[index] = spectrum[index]
    else:
        eq[index] = 0


data = utils.cut_data_from_spectrum(1024, 100, eq)


plt.plot(spectrum)
plt.show()