
import adi
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import mod
import config
import utils
from devexceptions import NoDeviceFoundException
import dprocessing as dp
import devices
import scipy
import OFDM


fs          = config.SAMPLE_RATE
fftsize     = config.FOURIER_SIZE
scatter_color       = '#006699'
spectrum_color      = '#b35900'
correlation_color   = '#26734d'
spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))



sdrtx, sdrrx = devices.initialize_sdr(single_mode=False, tx='RED_PIMPLE_TX')

sdrrx.tx_destroy_buffer()
sdrtx.tx_destroy_buffer()
sdrrx.rx_destroy_buffer()
sdrtx.rx_destroy_buffer()

tx_signal = np.array([])
data_time_domain, carriersTuple = OFDM.generate_ofdm_withpilots()

for _ in range(config.NUMBER_OF_OFDM_SYMBOLS):
    
    tx_signal = np.append(tx_signal, data_time_domain)



reference_data = tx_signal
preambula = OFDM.generate_ofdm_nopilots()
preambula = np.concatenate([preambula, preambula])
print(f'-> preambula length: {len(preambula)}')
tx_signal = np.concatenate((preambula, tx_signal))
tx_signal *= 2**14

assert (preambula[:config.FOURIER_SIZE] == preambula[config.FOURIER_SIZE:2*config.FOURIER_SIZE]).all(), 'preambulas part1 and 2 are different!'



def channelEstimate(OFDM_TD):
    gsize = config.GUARD_SIZE    
    fftsize = config.FOURIER_SIZE
    K = fftsize - 2*gsize - 1 + 1 
    allCarriers = np.arange(K)

    removedZeros, (pilots, datas) = OFDM.degenerate_ofdm_withpilots(OFDM_TD, carriersTuple)
    pilotValue = 2+2j
    Hest_at_pilots = pilots/pilotValue

    pilotCarriers = carriersTuple[0]
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs*np.exp(1j*Hest_phase)

    return Hest




def summ_ofdm(data, specs=False):

    fftsize = config.FOURIER_SIZE
    K = len(data)/fftsize
    K = int(K)
    s = np.zeros(fftsize, dtype=complex)
    
    if not specs:
        for i in range(K):
            s += data[i*fftsize:(i+1)*fftsize]
    else:
        for i in range(K):
            chunk = data[i*fftsize:(i+1)*fftsize]
            s += spectrum_and_shift(chunk)

    return s/K


# transmission
sdrtx.tx(tx_signal)


fig, axes = plt.subplots(5, 2)
fi2g, axes2 = plt.subplots(5, 2)
fig3, axes3 = plt.subplots(5, 2)
fig4, axes4 = plt.subplots(5, 2)

for ne in range(1000):

    for ax in axes:
        ax[0].cla(); ax[1].cla()

    for ax in axes2:
        ax[0].cla(); ax[1].cla()

    for ax in axes3:
        ax[0].cla(); ax[1].cla()


    for ax in axes4:
        ax[0].cla(); ax[1].cla()
    
    # receiving
    data_recieved = sdrrx.rx()
    recived_data_length = len(data_recieved)

    #sleep(0.6)
    # creating spectrum of recieved data
    spectrum_data_recived = np.fft.fftshift(np.fft.fft(data_recieved))


    # first correlation 
    cutted, abs_first_correlation = dp.correlation(preambula, data_recieved, fftsize*4)

    # received spec, constellation and correlation graphs
    axes[0][0].plot(np.abs(spectrum_data_recived), spectrum_color)
    axes[0][0].set_title('received sig spec')

    axes[1][0].plot(abs_first_correlation, correlation_color)
    axes[1][0].set_title('correlation')


    # cutting off
    cut_data = cutted
    cut_data_spec = spectrum_and_shift(cut_data)


    # correlating part1 and part2 of cutted data
    part1, part2, data = cut_data[:fftsize], cut_data[fftsize:fftsize*2], cut_data[fftsize*2:]

    print(f'data len: {len(data)}')
    corr2 = np.correlate(part2, part1, 'full')

    abscorr2 = np.abs(corr2)
    maxx = abscorr2.argmax()
    complex_max = corr2[maxx]


    first_OFDM_symbol = spectrum_and_shift(data[:fftsize])
    axes[3][0].scatter(first_OFDM_symbol.real, first_OFDM_symbol.imag, color=scatter_color, marker='.')
    axes[3][0].set_title('first_OFDM_symbol before freq ')

    
    '''
    # variant 1
    ang2 = np.arctan2(complex_max.imag, complex_max.real)

    for i in range(len(data)):
        data[i] = data[i]*np.exp(1j*i*(ang2/fftsize))
    '''


    '''
    # variant2
    dphi = np.angle(complex_max)
    print(f'angle: {dphi}')
    dt = 1/fs
    tau = config.FOURIER_SIZE*dt
    ocen_freq = dphi/(2*np.pi*tau)
    dphi_ocen = (ocen_freq*2*np.pi)/fs  

    freq_corrected_data = np.zeros(len(cutted), dtype=complex)
    for i in range(len(cutted)):
        freq_corrected_data[i] = cutted[i]*np.exp(1j*i*(-dphi_ocen))
    

    part1, part2, data = freq_corrected_data[:fftsize], freq_corrected_data[fftsize:fftsize*2], freq_corrected_data[fftsize*2:]
    '''

    # Accumulate OFDM
    summed_ofdm_time_domain = summ_ofdm(data, specs=False)
    p = spectrum_and_shift(summed_ofdm_time_domain)
    summed_ofdm_freq_domain = summ_ofdm(data, specs=True)
    
    unzeroed_spec_freq_domain = utils.cut_data_from_spectrum(summed_ofdm_freq_domain, spectrum=True)
    unzeroed_spec_time_domain = utils.cut_data_from_spectrum(summed_ofdm_time_domain, spectrum=False)

    unzeroed_spec_time_domain_spectrum = unzeroed_spec_time_domain

    axes3[0][1].scatter(p.real, p.imag, color=scatter_color, marker='.')
    axes3[0][1].set_title('TD')

    axes3[1][1].scatter(summed_ofdm_freq_domain.real, summed_ofdm_freq_domain.imag, color=scatter_color, marker='.')
    axes3[1][1].set_title('FD')
    # -------------------------

    # Degenerating OFDM with pilots
    
    boo, pilsAnddata = OFDM.degenerate_ofdm_withpilots(summed_ofdm_time_domain, carriersTuple)
    axes4[0][0].scatter(boo.real, boo.imag, color=scatter_color, marker='.')
    axes4[0][1].plot(np.abs(boo))

    Hest = channelEstimate(summed_ofdm_time_domain)
    equalized_ofdm = boo/Hest

    axes4[1][0].scatter(equalized_ofdm.real, equalized_ofdm.imag, color=scatter_color, marker='.')
    axes4[1][1].plot(np.abs(equalized_ofdm))
    axes4[1][0].set_title('equalized')
    axes4[1][1].set_title('equalized')


    first_OFDM_symbol = spectrum_and_shift(data[:fftsize])
    axes[4][0].scatter(first_OFDM_symbol.real, first_OFDM_symbol.imag, color=scatter_color, marker='.')
    axes[4][0].set_title('first_OFDM_symbol after freq ')

    eq = utils.equalize(preambula[:fftsize], part1)
    data_eq = utils.remove_spectrum_zeros(data)

    axes[2][0].scatter(eq.real, eq.imag, color=scatter_color, marker='.')
    axes[2][0].set_title('equalizer')


    s = []
    for spectrum in data_eq:
        try:
            q = spectrum*eq
            q_abs = np.abs(q)
            m = np.max(q_abs)
            q_normilized = (q/m)*1.4142
            s.append(q_normilized)
        except ValueError:
            print('-> Value Error occured')



    q = unzeroed_spec_time_domain_spectrum*eq
    q_abs = np.abs(q)
    m = np.max(q_abs)
    q_normilized = (q/m)*1.4142
    axes3[0][0].scatter(q_normilized.real, q_normilized.imag, color=scatter_color, marker='.')
    axes3[0][0].set_title('TD eq')

    q = unzeroed_spec_time_domain*eq
    q_abs = np.abs(q)
    m = np.max(q_abs)
    q_normilized = (q/m)*1.4142
    axes3[1][0].scatter(q_normilized.real, q_normilized.imag, color=scatter_color, marker='.')
    axes3[1][0].set_title('FD eq')


    for i in range(5):
        try:
            axes[i][1].scatter(s[i].real, s[i].imag, color=scatter_color,marker='.')
            axes[i][1].set_title(f's[{i}]')
        except IndexError:
            print('Index Error ')

    
    a1 = data[0:fftsize]
    a2 = data[fftsize:fftsize*2]
    a3 = data[fftsize*2:fftsize*3]
    a4 = data[fftsize*3:fftsize*4]
    a5 = data[fftsize*4:fftsize*5]
    
    a1 = np.fft.fft(a1)
    a2 = np.fft.fft(a2)
    a3 = np.fft.fft(a3)
    a4 = np.fft.fft(a4)
    a5 = np.fft.fft(a5)
    

    try:
        axes2[0][1].scatter(a1.real, a1.imag, color=scatter_color,marker='.')
        axes2[0][1].set_title(f's[0]')

        axes2[1][1].scatter(a2.real, a2.imag, color=scatter_color,marker='.')
        axes2[1][1].set_title(f's[1]')

        axes2[2][1].scatter(a3.real, a3.imag, color=scatter_color,marker='.')
        axes2[2][1].set_title(f's[2]')

        axes2[3][1].scatter(a4.real, a4.imag, color=scatter_color,marker='.')
        axes2[3][1].set_title(f's[3]')


        axes2[4][1].scatter(a5.real, a5.imag, color=scatter_color,marker='.')
        axes2[4][1].set_title(f's[4]')

        axes2[3][1].plot(data_recieved[:10].real)
        axes2[2][1].plot(data_recieved[:10].imag)
        
    except IndexError:
        print('Index Error ')

    
    for i in range(5):
        try:
            axes2[i][0].scatter(s[i+5].real, s[i+5].imag, color=scatter_color,marker='.')
            axes2[i][0].set_title(f's[{i+5}]')
        except IndexError:
            print('Index Error ')


    #demod_data = mod.qpsk_demodulate(s[0])
    sdrrx.rx_destroy_buffer()
    print('\n-------- next iteration --------\n')
    plt.pause(1)



plt.show()
sdrtx.tx_destroy_buffer()

