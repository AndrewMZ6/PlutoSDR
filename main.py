
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp
import devices


# initializing constants
mode        = 'single'
#transmitter = 'BEARD'
transmitter = 'ANGRY'
receiver    = 'FISHER'
fs          = config.SAMPLE_RATE
fftsize     = config.FOURIER_SIZE
scatter_color       = '#006699'
spectrum_color      = '#b35900'
correlation_color   = '#26734d'
spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


# swap transmitter and receiver
#transmitter, receiver = receiver, transmitter

sdrtx, sdrrx = devices.initialize_sdr(single_mode=True, tx='ANGRY', swap=False)


# create and modulate random bits
preambula = mod.get_preambula()


# create spectrum and time samples
preambula_spectrum = preambula.get_spectrum()
preambula_time_domain = preambula.get_time_samples()
preambula_time_domain = np.concatenate((preambula_time_domain, preambula_time_domain))
preambula_length = len(preambula_time_domain)



tx_signal = np.array([])

for _ in range(5):
    # data payload
    data_bits = mod.create_bits(config.BIT_SIZE)[:1648]
    

    data_modulated = mod.qpsk_modualte(data_bits)
    if _ == 0:
        data_compare = data_bits

    # create spectrum and time samples
    data_spectrum = mod.PutDataToZeros(config.FOURIER_SIZE, config.GUARD_SIZE, data_modulated)
    data_time_domain = data_spectrum.get_time_samples()
    tx_signal = np.append(tx_signal, data_time_domain)


reference_data = tx_signal
tx_signal = np.concatenate((preambula_time_domain, tx_signal))


assert (preambula_time_domain[:1024] == preambula_time_domain[1024:2048]).all(), 'preambulas part1 and 2 are different!'




# transmission
sdrtx.tx(tx_signal)

mm = np.array([2+2j, 2-2j, -2+2j, -2-2j])
fig, axes = plt.subplots()
axes.grid()
scat = axes.scatter(mm.real, mm.imag)


c = 0
def func(frames, scat):
    global c
    try:
        data_recieved = sdrrx.rx()


        # first correlation 
        cutted, abs_first_correlation = dp.correlation(preambula_time_domain, data_recieved, 0)


        # cutting off
        cut_data = cutted


        # correlating part1 and part2 of cutted data
        part1, part2, data = cut_data[:1024], cut_data[1024:2048], cut_data[2048:]


        eq = utils.equalize(preambula_time_domain[1024:2048], part2)
        data_eq = utils.remove_spectrum_zeros(data)


        q = data_eq[0]*eq
        q_abs = np.abs(q)
        m = np.max(q_abs)
        q_normilized = (q/m)*1.4142

    

        data = np.array([q_normilized.real, q_normilized.imag])
        scat.set_offsets(data.T)

    except ValueError:
        c += 1
        print(f'-> ValueError exception happened: {c}')

    return scat, 



animation = FuncAnimation(fig,
                            func=func,
                            fargs=(scat, ),
                            interval=10,
                            blit=True,
                            repeat=True)


plt.show()
sdrtx.tx_destroy_buffer()