
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp


# initializing constants
mode        = 'single1'
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
transmitter, receiver = receiver, transmitter


# find connected devices
devices = utils.detect_devices()
if devices is None:
    raise NoDeviceFoundException('No connected devices found')


# create and modulate random bits
preambula = mod.get_preambula()


# create spectrum and time samples
preambula_spectrum = preambula.get_spectrum()
preambula_time_domain = preambula.get_time_samples()
preambula_time_domain = np.concatenate((preambula_time_domain, preambula_time_domain))
preambula_length = len(preambula_time_domain)
print(f"preambula length: {preambula_length}")


fig12, ax12 = plt.subplots()
ax12.plot(np.abs(np.correlate(preambula_time_domain[:1024], preambula_time_domain[1024:], 'full')))

tx_signal = np.array([])

for _ in range(5):
    # data payload
    data_bits = mod.create_bits(config.BIT_SIZE)[:1648]
    

    data_modulated = mod.qpsk_modualte(data_bits)
    if _ == 0:
        data_compare = data_bits

    # create spectrum and time samples
    data_spectrum = mod.put_data_to_zeros(config.FOURIER_SIZE, config.GUARD_SIZE, data_modulated)
    data_time_domain = data_spectrum.get_time_samples()
    tx_signal = np.append(tx_signal, data_time_domain)


reference_data = tx_signal
tx_signal = np.concatenate((preambula_time_domain, tx_signal))
sig_len = len(tx_signal)
print(f'transmitted signal length: {sig_len}')


#assert (tx_signal[fftsize*2:] == data_time_domain).all(), 'txsignal is wrong!'
assert (preambula_time_domain[:1024] == preambula_time_domain[1024:2048]).all(), 'preambulas part1 and 2 are different!'
print(f"tx signal:{tx_signal[:5]}")

spec = data_spectrum.get_spectrum()
nums = utils.remove_spectrum_zeros(data_time_domain)
# original signal graphs
fig1, axs1 = plt.subplots(2, 2)
axs1[0][0].plot(np.abs(spec), spectrum_color)
axs1[0][0].set_title('generated sig')
axs1[0][1].scatter(spec.real, spec.imag, color=scatter_color)
axs1[0][1].set_title('constellation')
axs1[1][0].plot(np.abs(nums))
for i in axs1: i[0].grid(); i[1].grid()



sdrtx = adi.Pluto(devices[transmitter])
if mode != 'single':
    sdrrx = adi.Pluto(devices[receiver])
else:
    sdrrx = sdrtx


# setting up tx
sdrtx.sample_rate = int(config.SAMPLE_RATE)
sdrtx.tx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrtx.tx_lo = int(config.CENTRAL_FREQUENCY)
sdrtx.tx_hardwaregain_chan0 = 0
sdrtx.tx_cyclic_buffer = True


# setting up rx
sdrrx.rx_lo = int(config.CENTRAL_FREQUENCY)
sdrrx.rx_rf_bandwidth = int(config.SAMPLE_RATE)
sdrrx.rx_buffer_size = config.COMPLEX_SAMPLES_NUMBER
sdrrx.gain_control_mode_chan0 = 'manual'
sdrrx.rx_hardwaregain_chan0 = 0.0


# transmission
sdrtx.tx(tx_signal)


# receiving
data_recieved = sdrrx.rx()
recived_data_length = len(data_recieved)
print(f"recieved data length = {recived_data_length}")
print(f"received data: {data_recieved[1000:1010]}")

# creating spectrum of recieved data
spectrum_data_recived = np.fft.fftshift(np.fft.fft(data_recieved))


# first correlation 
cutted, abs_first_correlation = dp.correlation(preambula_time_domain, data_recieved, 0)

# received spec, constellation and correlation graphs
fig2, axs = plt.subplots(2, 2)
axs[0][0].plot(np.abs(spectrum_data_recived), spectrum_color)
axs[0][0].set_title('received sig spec')
axs[0][1].scatter(spectrum_data_recived.real, spectrum_data_recived.imag, color=scatter_color, marker='.')
axs[0][1].set_title('received constellation')
axs[1][0].plot(abs_first_correlation, correlation_color)
#axs[1][0].annotate(f'max={x_coord_max_first_correlation:.1f}', xy=(x_coord_max_first_correlation, m), xytext=(maxcorr + 1, m + 1000000), arrowprops=dict(facecolor='black', shrink=0.05))
axs[1][0].set_title('correlation')
for i in axs: i[0].grid(); i[1].grid()





# cutting off
cut_data = cutted
cut_data_spec = spectrum_and_shift(cut_data)


fig7, ax7 = plt.subplots()
ax7.plot(np.abs(np.correlate(cut_data, tx_signal, 'full')))
ax7.set_title('cut data and tx_signal correlation')

print(f"data cutted len: {len(cut_data_spec)}")


# cut data graphs
fig3, axs = plt.subplots(2, 2)
axs[0][1].plot(np.abs(cut_data_spec), spectrum_color)
axs[0][1].set_title('absolute cut_data_spec')
axs[1][0].scatter(cut_data_spec.real, cut_data_spec.imag, color=scatter_color, marker='.')
axs[1][0].set_title('cut constellation')
for i in axs: i[0].grid(); i[1].grid()



# correlating part1 and part2 of cutted data
part1, part2, data = cut_data[:1024], cut_data[1024:2048], cut_data[2048:]
print(f"cut data length: {len(data)}")

conc = np.concatenate((part1, part2))
print(f"pre: {len(preambula_time_domain)} conc: {len(conc)}")
fig11, ax11 = plt.subplots()
corr3 = np.correlate(preambula_time_domain, np.concatenate((part1, part2)), 'full')
ax11.plot(np.abs(corr3))


# part1 and data graphs
fig8, ax8 = plt.subplots(2, 2)
ax8[0][0].plot(np.abs(spectrum_and_shift(part1)), spectrum_color)
ax8[0][0].set_title('part1')
ax8[0][1].scatter(spectrum_and_shift(part1).real, spectrum_and_shift(part1).imag, color=scatter_color, marker='.')
ax8[0][1].set_title('part1 constellation')
ax8[1][0].plot(np.abs(spectrum_and_shift(data)), spectrum_color)
ax8[1][0].set_title('data')
ax8[1][1].scatter(spectrum_and_shift(data).real, spectrum_and_shift(data).imag, color=scatter_color, marker='.')
ax8[1][1].set_title('data constellation')
for i in ax8: i[0].grid(); i[1].grid()


corr2 = np.correlate(part2, part1, 'full')

abscorr2 = np.abs(corr2)
maxx = abscorr2.argmax()

complex_max = corr2[maxx]


fig4, ax4 = plt.subplots(2, 2)
ax4[0][0].plot(np.abs(corr2), correlation_color)
ax4[0][0].set_title('part1 part2 correlation')

part1_spce = spectrum_and_shift(part1)
ax4[1][1].scatter(part1_spce.real, part1_spce.imag, color=scatter_color, marker='.')
ax4[1][1].set_title('part1 before freq ')


import math
print(f"complex max = {complex_max}")
# variant 1
ang = math.atan2(complex_max.imag, complex_max.real)
ang2 = np.arctan2(complex_max.imag, complex_max.real)
#print(f"angle math: {ang}, angle numpy: {ang2}")

''''
COR_RE = 0
COR_IM = 0
for i in range(len(part2)):
    COR_RE += part1.real[i]*part2.real[i]+part1.imag[i]*part2.imag[i]
    COR_IM += part1.imag[i]*part2.real[i]-part1.real[i]*part2.imag[i]


angle = math.atan2(COR_IM,COR_RE)
'''
#print(angle, ang, ang2)
#data2 = np.ndarray((1024,),float)
data2=[0]*len(data)
for i in range(len(data)):
    data2[i] = data[i]*np.exp(-1j*i*(0/fftsize))




'''
# variant2
dphi = np.angle(complex_max)
dt = 1/fs
tau = config.FOURIER_SIZE*dt
ocen_freq = dphi/(2*np.pi*tau)
dphi_ocen = (ocen_freq*2*np.pi)/fs  


for i in range(len(data)):
    data[i] = data[i]*np.exp(1j*i*(-dphi_ocen))
'''



eq = utils.equalize(preambula_time_domain[1024:2048], part2)
data_eq = utils.remove_spectrum_zeros(data2)

s = []
for spectrum in data_eq:
    q = spectrum*eq
    q_abs = np.abs(q)
    m = np.max(q_abs)
    q_normilized = (q/m)*1.4142
    s.append(q_normilized)
#np.save(r"/media/andrew/PlutoSDR/eq.npy", eq)
#eq = np.load(r"/media/andrew/PlutoSDR/eq.npy")
#eqed = part1/eq
#eqedspec = np.fft.fftshift(np.fft.fft(eqed))



data_eq_spectrum_shifted = data_eq
ax4[0][1].plot(np.abs(s[0]), spectrum_color)
ax4[0][1].set_title('data_eq_spectrum_shifted')
ax4[1][0].scatter(s[0].real, s[0].imag, color=scatter_color,marker='.')
ax4[1][0].set_title('data_eq_spectrum_shifted constellation')
for i in ax4: i[0].grid(); i[1].grid()

fig9, ax9 = plt.subplots(1, 2)
ax9[0].plot(np.abs(data_eq))
ax9[1].plot(np.abs(spectrum_and_shift(data2)))


fig10, ax10 = plt.subplots(5, 2)
for i in range(5):
    for j in range(2):
        if j == 0:
            ax10[i][j].plot(np.abs(s[i]), spectrum_color)
            ax10[i][j].set_title(f's[{i}]')
        else:
            ax10[i][j].scatter(s[i].real, s[i].imag, color=scatter_color,marker='.')
            ax10[i][j].set_title(f's[{i}]')
        ax10[i][j].grid()


demod_data = mod.qpsk_demodulate(s[0])
print(f"demod_data len = {len(demod_data)}")
print(demod_data[:10])
print(data_compare[:10])

plt.show()
sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()
exit()

fig6, ax = plt.subplots()
ax.scatter(eqedspec.real, eqedspec.imag, color=scatter_color,marker='.')
ax.set_title('eqed')



eq = []
axs[0][0].scatter(d.real, d.imag)
axs[0][0].set_title('d.real, d.imag')
d = d**2

axs[1][0].scatter(cut_data_spec.real, cut_data_spec.imag)
axs[1][0].set_title('cut data spec real amd imag')

axs[1][1].plot(np.absolute(cut_data_spec))
axs[1][1].set_title('absolute cut data spec')



plt.show()

'''
for j in range(4):
    
    
    for i in range(10):
        #
        data = sdr2.rx()
       
        #print(f"data size: {len(data)}")

        
        axs[0][0].cla()
        
        axs[0][1].cla()
        axs[1][0].cla()
        axs[1][1].cla()

        
        cor = np.correlate(data, time_sig, 'full')
        #print(f"cor len: {len(cor)}")
        x1 = cor.argmax(axis=0)
        
        #print(f"max index: {x1}")
        if x1 <= 944:
            #print("if works here")
            left_ind = x1 - int(rr/2) +1 
            right_ind = left_ind + rr 
            #print(f"cutting left index: {left_ind}")
            #print(f"cutting right index: {right_ind}")
        else:
            #print("else works here")
            left_ind = x1 - int(rr/2) - rr +1 
            right_ind = left_ind + rr 
            #print(f"cutting left index: {left_ind}")
            #print(f"cutting right index: {right_ind}")

        cut_data = data[left_ind:right_ind]
        cut_data_spec = np.fft.fft(cut_data)
        cut_data_spec = np.fft.fftshift(cut_data_spec)
        zero_index = int(config.FOURIER_SIZE/2)
        cut_spec = cut_data_spec[config.GUARD_SIZE:zero_index]
        cut_spec = np.append(cut_spec,cut_data_spec[zero_index + 1:-config.GUARD_SIZE + 1])
        print(len(cut_spec))
        #
        #ar = np.append(ar, cut_data)
        #print(f"cut len: {len(cut_data)}", end='\n---------------------------\n')
        
        #d1 = np.fft.fft(cut_data)
        #d1_shifted = np.fft.fftshift(d1)
        #dd = np.fft.fftshift(np.fft.fft(cut_data))

        if not len(eq):
            eq = mod_data/cut_spec

        axs[0][0].set_title('Spectrum')
        
        axs[0][1].set_title('Constelation')
        axs[1][0].set_title('corr')
        axs[1][1].set_title('shifted')
        
        #axs[0][0].plot(np.absolute(d1))
        recov = cut_spec*eq
        axs[0][0].scatter(recov.real, recov.imag)
        axs[0][1].scatter(eq.real, eq.imag)
        axs[1][0].plot(np.absolute(cor))
        axs[1][1].plot(np.absolute(cut_spec*eq))
        rec_bits = mod.qpsk_demodulate(recov)
        print(rec_bits[:10])
        print(bits[:10])
        #print(data[:5])
        #input()
        plt.grid()
        plt.pause(1)
    print("Next iteration")
    sdr2.rx_destroy_buffer()
    sdr.tx_destroy_buffer()
    

stop = perf_counter()
#plt.show()
print(ar)
print(len(ar))
print(f'time: {stop-start}')
print(f'speed: {((config.BIT_SIZE/8)*iter_num)/(stop-start)} kBps')
'''

