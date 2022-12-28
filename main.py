import numpy as np
from time import sleep
from matplotlib import pyplot as plt
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

fig6, axes6 = plt.subplots(2, 1)
fig4, axes4 = plt.subplots(1, 2)
fig7, axes7 = plt.subplots()

# containers for storing BER vs SNR data
biterrs_array = np.array([])
snr_array = np.array([])


flag = True
try:
    for ne in range(30):

        N = 1000
        ber = 0
        snrdB = 0
        for oo in range(N):

            # Receiving
            data_recieved = sdrrx.rx()
            
            # Clearing axes each iteration
            for ax in axes4:
                ax.cla(); ax.cla()

            axes7.cla()



            axes7.plot(np.abs(spectrum_and_shift(data_recieved)))

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


            ber_pimple = utils.biterr(beets, demod_bits)
            
            ber += ber_pimple

            E = utils.calculateEVM(equalized_ofdm, carriersTuple[1], 1+1j)
            snr = 1/(E**2)
            snrdB += 10*np.log10(snr)


            axes4[1].plot(abs_first_correlation, color=scatter_color)
            axes4[1].set_ylim([0, 9e3])
            axes4[1].grid()
            axes4[0].scatter(eq_r.real, eq_r.imag, color=scatter_color, marker='.')
            axes4[0].grid()
            axes4[0].set_title(f'SNR = {10*np.log10(snr):.2f}')
            
            plt.pause(0.1)
            
            
            
        
        ber /= N
        snrdB /= N
        
        print(f'-> BER = {ber}')
           
        print(f'-> SNR = {snrdB:.2f}')

        snr_array = np.append(snr_array, snrdB)
        biterrs_array = np.append(biterrs_array, ber)


        



        sdrrx.rx_destroy_buffer()
        print(f'\n-------- iteration {ne + 1} --------\n')

        HCG = sdrtx.tx_hardwaregain_chan0
        print(HCG)
        if flag:
            if HCG == -60:
                flag = False
            sdrtx.tx_hardwaregain_chan0 -= 1
        else:   
            if HCG == -1:
                flag = True
            sdrtx.tx_hardwaregain_chan0 += 1


        print(f'tx gain = {sdrtx.tx_hardwaregain_chan0}')
        #plt.pause(1)

    
    # BER vs SNR log
    axes6[0].set_yscale('log')
    #axes6[0].set_xlim([0, 35])
    axes6[0].plot(snr_array, biterrs_array, color=scatter_color, marker='.', linestyle='None')
    axes6[0].grid()
    axes6[0].set_title('BER/SNR log')


    #axes6[1].set_xlim([0, 35])
    axes6[1].plot(snr_array, biterrs_array, color=scatter_color, marker='.', linestyle='None')
    axes6[1].grid()
    axes6[1].set_title('BER/SNR linear')
    


    plt.show()
    sdrtx.tx_destroy_buffer()

except KeyboardInterrupt:
    print('\nExecution has been interrupted')
    sdrrx.tx_destroy_buffer()
    sdrtx.tx_destroy_buffer()
    sdrrx.rx_destroy_buffer()
    sdrtx.rx_destroy_buffer()
