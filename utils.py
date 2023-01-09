import os
import re
import config
import numpy as np
import scipy
import OFDM



def detect_devices() -> dict | None:
    '''Detects plutos '''

    x = os.popen('iio_info -s').read()
    s = re.findall(r"serial=(\w+)\s*\[(usb:.*)\]", x)
    l = len(s)
    print(f"Devices detected: {l}")

    if not l:
        return None
    
    assert l >= 1

    devices = {}

    for device in s:

        serial_number, usb_number = device
        devices.update({config.devices[serial_number]:usb_number})

    return devices



spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


def equalize(transmitted_sig, received_sig):  #input signals in time domain
    '''
        Makes channel estimation and returns estimation array
    '''

    transmitted_data = cut_data_from_spectrum(transmitted_sig)
    received_data = cut_data_from_spectrum(received_sig)

    eq = transmitted_data/received_data
    return eq


def remove_spectrum_zeros(time_domain_sig: np.ndarray) -> np.ndarray:
    '''
        Removes guards intervals and central zero sample. Returns spectrum samples
        function 'cut_data_from_spectrum' now does this so... ???
    '''
    fftsize = config.FOURIER_SIZE
    guardsize = config.GUARD_SIZE
    
    L = len(time_domain_sig)
    K = L/fftsize
    #if len(time_domain_sig) > 1024:


    part1, part2, part3, part4, part5 = time_domain_sig[:fftsize], time_domain_sig[fftsize:fftsize*2], time_domain_sig[fftsize*2:fftsize*3], time_domain_sig[fftsize*3:fftsize*4], time_domain_sig[fftsize*4:fftsize*5]
    part6, part7, part8, part9, part10 = time_domain_sig[fftsize*5:fftsize*6], time_domain_sig[fftsize*6:fftsize*7], time_domain_sig[fftsize*7:fftsize*8], time_domain_sig[fftsize*8:fftsize*9], time_domain_sig[fftsize*9:fftsize*10]
    t = (part1, part2, part3, part4, part5, part6, part7, part8, part9, part10)
    
    result = []
    for part in t:
        spectrum = spectrum_and_shift(part)
    
    
    
        left_part = spectrum[guardsize:int(fftsize/2)] 
        right_part = spectrum[int(fftsize/2)+1:fftsize - (guardsize - 1)]
        result.append(np.concatenate((left_part, right_part)))

    '''else:
        spectrum = spectrum_and_shift(time_domain_sig)
        
               
        left_part = spectrum[100:int(1024/2)] 
        right_part = spectrum[int(1024/2)+1:924+1]
        return np.concatenate((left_part, right_part))
    '''
    
    return result


def cut_data_from_spectrum(time_domain_signal: np.ndarray, spectrum=False) -> np.ndarray:
    '''
        Cuts data complex vectors from input time domatin signal using
        config.FOURIER_SIZE and config.GUARD_SIZE parameters.

        Expects time domain data as input.
        Returns frequency domain complex vectors without guard zeros and central zero

        _|-|-|_   ->   --
    '''


    fftsize = config.FOURIER_SIZE
    guardsize = config.GUARD_SIZE

    
    # If we need to strip single ofdm symbol
    if len(time_domain_signal) == fftsize:

        if not spectrum:
            spectrum = spectrum_and_shift(time_domain_signal)
        else:
            spectrum = time_domain_signal

        central_zero_index = int(fftsize/2)
        left_part = spectrum[guardsize:central_zero_index]
        right_part = spectrum[central_zero_index + 1:fftsize - (guardsize - 1)]

        result = np.concatenate((left_part, right_part))

        assert len(result) == fftsize - 2*guardsize, 'the cutted data has wrong size'
    else:
        print(f'else worker in module {__name__}')
        raise('YOU hacked TOO FAR!')

    return result



def biterr(bits1, bits2):
    y = np.logical_xor(bits1, bits2)
    L = len(bits1)
    s = np.sum(y)
    ber = s/L

    return ber


def find_evm(pos_data, ref):

    L = len(pos_data)
    ref_array = np.zeros(L, dtype=complex)
    ref_array[:] = ref
    
    I_err_array = pos_data.real - ref_array.real
    Q_err_array = pos_data.imag - ref_array.imag

    E_module_squared = I_err_array**2 + Q_err_array**2
    P_ref = np.abs(ref)**2

    EVM_number = (np.sqrt(np.sum(E_module_squared)/L))/P_ref
    

    return EVM_number




def calculateEVM(spectrum_with_pilots, data_carrs, ref_complex):
    '''
        Returns EVM number.
        Input params:
            spectrum_with_pilots - spectrum with pilots but without any zeros
            data_carrs           - array with data carriers indicies
            ref_complex          - complex number that represents qpsk modulation vector. 1+1j
    '''
    data = spectrum_with_pilots[data_carrs]
    pos_data = OFDM.positivise(data)
    evm = find_evm(pos_data, ref_complex)

    return evm


def channelEstimate(OFDM_TD, carriersTuple):
    
    fftsize = config.FOURIER_SIZE
    gsize, P = config.GUARDS_AND_PILOTS[fftsize]
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



if __name__ == '__main__':
    print(detect_devices())
