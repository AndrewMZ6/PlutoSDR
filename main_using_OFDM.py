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


fs          = config.SAMPLE_RATE
fftsize     = config.FOURIER_SIZE
scatter_color       = '#006699'
spectrum_color      = '#b35900'
correlation_color   = '#26734d'
spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


# Devices initialization
sdrtx, sdrrx = devices.initialize_sdr(single_mode=False, tx='RED_PIMPLE_TX')


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

ar = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])


temp = config.FOURIER_SIZE*50 + 255

axes6[1][0].set_xlim([0, temp])
axes6[1][0].set_ylim([0, 1000])

axes6[0][0].set_xlim([0, 100e3])
axes6[0][0].set_ylim([0, 2e5])

x = np.arange(100e3)
y = np.zeros(100_000)


x2 = np.arange(temp)
y2 = np.zeros(temp)

plot1, = axes6[0][0].plot(x, y)
plot2, = axes6[1][0].plot(x2, y2)
scat1 = axes6[0][1].scatter(ar.real, ar.imag, marker='.')



def update_func(frames, scat1, plot1, plot2):
            
            
    # Receiving
    data_recieved = sdrrx.rx()



    plot1.set_ydata(np.abs(spectrum_and_shift(data_recieved)))

    # Main correlation 
    cutted, abs_first_correlation = dp.correlation(preambula, data_recieved, fftsize*2)
    
    
    # Unpacking preambulas and payload from cutted data
    part1, part2, data = cutted[:fftsize], cutted[fftsize:fftsize*2], cutted[fftsize*2:]


    corr2 = np.correlate(part2, part1, 'full')

    abscorr2 = np.abs(corr2)
    maxx = abscorr2.argmax()
    complex_max = corr2[maxx]



        # variant2
    dphi = np.angle(complex_max)
    dt = 1/fs
    tau = config.FOURIER_SIZE*dt
    ocen_freq = dphi/(2*np.pi*tau)
    dphi_ocen = (ocen_freq*2*np.pi)/fs  
    freq_corrected_data = np.zeros(len(cutted), dtype=complex)
    for i in range(len(cutted)):
        freq_corrected_data[i] = cutted[i]*np.exp(1j*i*(-dphi_ocen))
    
    part1, part2, data = freq_corrected_data[:fftsize], freq_corrected_data[fftsize:fftsize*2], freq_corrected_data[fftsize*2:]




    # Accumulate OFDM
    summed_ofdm_time_domain = utils.summ_ofdm(data, specs=False)
    

    # Degenerating OFDM with pilots
    boo, pilsAnddata = OFDM.degenerate_ofdm_withpilots(summed_ofdm_time_domain, carriersTuple)
    

    # 
    Hest = utils.channelEstimate(summed_ofdm_time_domain, carriersTuple)
    equalized_ofdm = boo/Hest

    eq_r = equalized_ofdm[carriersTuple[1]]
    
    M  = commpy.modulation.QAMModem(4)
    demod_bits = M.demodulate(eq_r, 'hard')


    #ber_pimple = utils.biterr(beets, demod_bits)
    

    E = utils.calculateEVM(equalized_ofdm, carriersTuple[1], 1+1j)
    snr = 1/(E**2)
    #snrdB += 10*np.log10(snr)


    plot2.set_ydata(abs_first_correlation)

    data = np.array([eq_r.real, eq_r.imag])
    scat1.set_offsets(data.T)

    return scat1, plot1, plot2, 
    
      

anim = FuncAnimation(fig6,
                    func=update_func,
                    fargs=(scat1, plot1, plot2),
                    interval=1,
                    blit=True,
                    repeat=True)



plt.show()
