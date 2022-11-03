import mod_utils
import numpy as np
from matplotlib import pyplot as plt


spectrum = np.zeros(202)
l, r = mod_utils.define_spectrum_data_indexes(fftsize=202, guardsize=101)

spectrum[l] = 5
spectrum[r] = 5.4


plt.plot(spectrum)
plt.show()