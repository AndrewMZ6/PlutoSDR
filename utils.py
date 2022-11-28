import os
import re
import config
import numpy as np




spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))


def equalize(transmitted_sig, received_sig):  #input signals in time domain
    '''
        Makes channel estimation and returns estimation array
    '''

    transmitted_data = remove_spectrum_zeros(transmitted_sig)
    received_data = remove_spectrum_zeros(received_sig)

    eq = transmitted_data/received_data
    return eq


def remove_spectrum_zeros(time_domain_sig: np.ndarray) -> np.ndarray:
    '''
        Removes guards intervals and central zero sample. Returns spectrum samples
        function 'cut_data_from_spectrum' now does this so... ???
    '''
    
    if len(time_domain_sig) > 1024:


        part1, part2, part3, part4, part5 = time_domain_sig[:1024], time_domain_sig[1024:2048], time_domain_sig[2048:3072], time_domain_sig[3072:4096], time_domain_sig[4096:]
        t = (part1, part2, part3, part4, part5)
        result = []
        for part in t:
            spectrum = spectrum_and_shift(part)
        
        
        
            left_part = spectrum[100:int(1024/2)] 
            right_part = spectrum[int(1024/2)+1:924+1]
            result.append(np.concatenate((left_part, right_part)))

    else:
        spectrum = spectrum_and_shift(time_domain_sig)
        
               
        left_part = spectrum[100:int(1024/2)] 
        right_part = spectrum[int(1024/2)+1:924+1]
        return np.concatenate((left_part, right_part))

    
    return result


def cut_data_from_spectrum(fftsize: int, guardsize: int, spectrum: np.ndarray) -> np.ndarray:
    '''
        Cuts data complex vectors from input spectum using fftsize and guardsize.
        Expects frequency domain spectrum as input variable 'spectrum'.
        Returns frequency domain complex vectors without guard zeros and central zero

        _|-|-|_   ->   --
    '''

    
    central_zero_index = int(fftsize/2)
    left_part = spectrum[guardsize:central_zero_index]
    right_part = spectrum[central_zero_index + 1:fftsize - (guardsize - 1)]

    result = np.concatenate((left_part, right_part))

    assert len(result) == fftsize - 2*guardsize, 'the cutted data has wrong size'

    return result







if __name__ == '__main__':
    print(detect_devices())
