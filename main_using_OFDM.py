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
sdrtx, sdrrx = devices.initialize_sdr(single_mode=False, tx='RED_PIMPLE_RX')

sdrrx.tx_destroy_buffer()
sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()
sdrtx.rx_destroy_buffer()


# Signal formation
data_time_domain, carriersTuple, beets = OFDM.generate_ofdm_withpilots()

print(f'bits len = {beets.size}')
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



fig6, axes6 = plt.subplots(2, 1)

# containers for storing BER vs SNR data
biterrs_array = np.array([])
snr_array = np.array([])


flag = True
try:
    
    # Outer for loop with changing gain
    for ne in range(30):

        N = 750
        ber = 0
        snrdB = 0
        N_initial_skip = 100


        accumulate_evm = np.array([])
        inner_loop_counter = 0
        inner_total_loop_counter = 0

        # Inner while loop repeating measurements N times
        while inner_loop_counter < N:

            inner_total_loop_counter += 1
            # Receiving
            data_recieved = sdrrx.rx()

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

            E = utils.calculateEVM(equalized_ofdm, carriersTuple[1], 1+1j)
            accumulate_evm = np.append(accumulate_evm, E)        
            s = np.sum(accumulate_evm)/inner_total_loop_counter

            if inner_total_loop_counter < N_initial_skip:
                continue

            if E > s:
                continue
            
            inner_loop_counter += 1
            eq_r = equalized_ofdm[carriersTuple[1]]
            
            M  = commpy.modulation.QAMModem(4)
            demod_bits = M.demodulate(eq_r, 'hard')


            ber_pimple = utils.biterr(beets, demod_bits)
            
            ber += ber_pimple

            

            snr = 1/(E**2)
            snrdB += 10*np.log10(snr)
            
            
        
        ber /= inner_loop_counter
        snrdB /= inner_loop_counter
        
        #print(f'-> BER = {ber}')
           
        #print(f'-> SNR = {snrdB:.2f}')

        snr_array = np.append(snr_array, snrdB)
        biterrs_array = np.append(biterrs_array, ber)


        sdrrx.rx_destroy_buffer()
        #print(f'\n-------- iteration {ne + 1} --------\n')




        HCG = sdrtx.tx_hardwaregain_chan0
        if flag:
            if HCG == -60:
                flag = False
            sdrtx.tx_hardwaregain_chan0 -= 1
        else:   
            if HCG == -1:
                flag = True
            sdrtx.tx_hardwaregain_chan0 += 1


        print(f'tx gain = {sdrtx.tx_hardwaregain_chan0}')
        print(f'inner loop counter = {inner_loop_counter}')
        print(f'inner total loop counter = {inner_total_loop_counter - N_initial_skip}')
        print(f'good / not_good = {inner_loop_counter/(inner_total_loop_counter - N_initial_skip):.2f}')
        #plt.pause(1)

    
    # BER vs SNR log
    axes6[0].set_yscale('log')
    axes6[0].plot(snr_array, biterrs_array, color=scatter_color, marker='.', linestyle='None')
    axes6[0].grid()
    axes6[0].set_title('BER/SNR log')


    axes6[1].plot(snr_array, biterrs_array, color=scatter_color, marker='.', linestyle='None')
    axes6[1].grid()
    axes6[1].set_title('BER/SNR linear')
    
    np.save(f'snr_array{config.FOURIER_SIZE}', snr_array)
    np.save(f'biterrs_array{config.FOURIER_SIZE}', biterrs_array)


    plt.show()
    sdrtx.tx_destroy_buffer()
    

except KeyboardInterrupt:
    print('\nExecution has been interrupted')
    sdrrx.tx_destroy_buffer()
    sdrtx.tx_destroy_buffer()
    sdrrx.rx_destroy_buffer()
    sdrtx.rx_destroy_buffer()
