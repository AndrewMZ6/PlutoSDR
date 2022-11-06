from matplotlib import pyplot as plt
import numpy as np


# suppose we have ar = pilots_before/pilots_after. Now we need to interpolate
ar = np.array([1+1j, 0, 0, 0, 0, 0.5-0.5j])
ar_with_modules = ar.copy()



module_diff = np.abs(ar[5]) - np.abs(ar[0])
module_step = module_diff/(len(ar) - 1)



for i in range(1, len(ar) - 1):
    ar_with_modules[i] = np.abs(ar_with_modules[i-1]) + module_step


ar_with_modules_one_lined = ar_with_modules.copy()

ar_with_modules_one_lined[1:5] = ar_with_modules_one_lined[1:5]*np.exp(1j*np.angle(ar[0]))
ar_with_modules_one_lined[5] = ar_with_modules_one_lined[5]*np.exp(1j*(np.angle(ar[0]) - np.angle(ar[5])))


angle_diff = np.angle(ar[5]) - np.angle(ar[0])
angle_step = angle_diff/(len(ar) - 1)



ar_with_modules_and_angles = ar_with_modules.copy()

for j in range(1, len(ar) - 1):
    ar_with_modules_and_angles[j] = ar_with_modules_and_angles[j]*np.exp(1j*(np.angle(ar[[0]]) + j*angle_step))


fig, (ax1, ax2) = plt.subplots(2, 2)

a, b = -1.5, 1.5
for ax in (ax1, ax2):
    ax[0].set_xlim(a, b); ax[0].set_ylim(a, b); ax[0].grid()
    ax[1].set_xlim(a, b); ax[1].set_ylim(a, b); ax[1].grid()

ax1[0].scatter(ar.real, ar.imag)
ax1[0].set_title('original array')

ax1[1].scatter(ar_with_modules.real, ar_with_modules.imag)
ax1[1].plot(0, 0, marker='<', markersize=7, color='black')
ax1[1].set_title('interpolate module')


ax2[0].scatter(ar_with_modules_and_angles.real, ar_with_modules_and_angles.imag)
ax2[0].plot(0, 0, marker='<', markersize=7, color='black')
ax2[0].set_title('interpolate angles')


ax2[1].plot(np.abs(ar_with_modules_and_angles), marker='.')
ax2[1].set_ylim(-0.5, 2)
ax2[1].set_xlim(-1, 6)
ax2[1].plot(0, 0, marker='<', markersize=7, color='black')
ax2[1].set_title('abs spectrum')

plt.show()