import numpy as np
from time import sleep
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import config
import dprocessing as dp
import devices
import utils
import OFDM
import commpy 
from collections import deque


tails_FFT = {1024: 2047, 128: 255, 512: 1023}


fs          = config.SAMPLE_RATE
fftsize     = config.FOURIER_SIZE
scatter_color       = '#006699'
spectrum_color      = '#b35900'
correlation_color   = '#26734d'
spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


# Devices initialization
sdrtx, sdrrx = devices.initialize_sdr(single_mode=False, tx='RED_PIMPLE_RX')


sdrrx.tx_destroy_buffer()
sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()
sdrtx.rx_destroy_buffer()


# Signal formation
data_time_domain, carriersTuple, beets = OFDM.generate_ofdm_withpilots()

# Payload
payload = np.tile(data_time_domain, config.NUMBER_OF_OFDM_SYMBOLS)

# Preambula
preambula = OFDM.generate_ofdm_nopilots()
preambula = np.concatenate([preambula, preambula])

# Final signal (goes to TX pluto)
tx_signal = np.concatenate((preambula, payload))
tx_signal *= 2**14


# Make sure that preambulas are the same
assert (preambula[:fftsize] == preambula[fftsize:2*fftsize]).all(), 'preambulas part1 and 2 are different!'


# Transmission
sdrtx.tx(tx_signal)


#       V
#       |
#       |
#       |
#       |
#       |
#       |
#       V



'''
fig, axes = plt.subplots(5, 2)
fi2g, axes2 = plt.subplots(5, 2)
fig3, axes3 = plt.subplots(5, 2)
fig4, axes4 = plt.subplots(5, 2)
fig5, axes5 = plt.subplots()
'''

fig6, axes6 = plt.subplots(2, 2)

ar = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])*2
dq = deque(np.zeros(10), maxlen=10)


temp = config.FOURIER_SIZE*50 + tails_FFT[fftsize]
x = np.arange(100e3)
y = np.zeros(100_000)

x2 = np.arange(temp)
y2 = np.zeros(temp)




axes6[0][0].set_xlim([0, 100e3])
axes6[0][0].set_ylim([0, 3e5])
axes6[0][0].set_xticklabels([])
plot1, = axes6[0][0].plot(x, y)


scat1 = axes6[0][1].scatter(ar.real, ar.imag, marker='.')

axes6[1][0].set_xlim([0, temp])
axes6[1][0].set_ylim([0, 3e3])
axes6[1][0].set_xticklabels([])
plot2, = axes6[1][0].plot(x2, y2)

axes6[1][1].set_ylim([-.1, 0.6])
axes6[1][1].set_xticklabels([])

plot3, = axes6[1][1].plot(np.arange(10), np.zeros(10))



M  = commpy.modulation.QAMModem(4)

def update_func(frames, scat1, plot1, plot2, plot3):
            
    
    # Receiving
    data_recieved = sdrrx.rx()



    plot1.set_ydata(np.abs(spectrum_and_shift(data_recieved)))

    # Main correlation 
    cutted, abs_first_correlation = dp.correlation(preambula, data_recieved, fftsize*2)
    
    
    # Unpacking preambulas and payload from cutted data
    part1, part2, data = cutted[:fftsize], cutted[fftsize:fftsize*2], cutted[fftsize*2:]


    
    # Accumulate OFDM
    summed_ofdm_time_domain = utils.summ_ofdm(data, specs=False)
    

    # Degenerating OFDM with pilots
    boo, pilsAnddata = OFDM.degenerate_ofdm_withpilots(summed_ofdm_time_domain, carriersTuple)
    

    # 
    Hest = utils.channelEstimate(summed_ofdm_time_domain, carriersTuple)
    equalized_ofdm = boo/Hest

    eq_r = equalized_ofdm[carriersTuple[1]]
    
    
    demod_bits = M.demodulate(eq_r, 'hard')


    ber_pimple = utils.biterr(beets, demod_bits)
    dq.append(ber_pimple)
    

    E = utils.calculateEVM(equalized_ofdm, carriersTuple[1], 1+1j)
    snr = 1/(E**2)
    #snrdB += 10*np.log10(snr)


    plot2.set_ydata(abs_first_correlation)
    

    data = np.array([eq_r.real, eq_r.imag])
    scat1.set_offsets(data.T)

    plot3.set_ydata(dq)

    return scat1, plot1, plot2, plot3
    
      

anim = FuncAnimation(fig6,
                    func=update_func,
                    fargs=(scat1, plot1, plot2, plot3),
                    interval=10,
                    blit=True,
                    repeat=True)



plt.show()
