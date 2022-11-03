from operator import mod
import commpy as cp
import numpy as np
from matplotlib import pyplot as plt
import string




def create_bit_sequence_from_letters(t: tuple) -> np.ndarray:
    '''Returns concatenated numpy array from input tuple.
        ('01101000', '01100101') -> array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1])'''

    result = np.array([])

    for letters in t:
        nparr = np.fromiter(letters, dtype=int)
        result = np.append(result, nparr)
    

    result = np.array(result, dtype=int)
    return result


def create_bits(size):
    return np.random.randint(low=0, high=2, size=size)


def qpsk_modualte(bits):
    qpsk = cp.modulation.QAMModem(4)
    return qpsk.modulate(bits)


def qpsk_demodulate(complex_data):
    qpsk = cp.modulation.QAMModem(4)
    return qpsk.demodulate(complex_data, 'hard')


#def qpsk_demod2(complex_data)


def _zeros(size):
    return np.zeros(size, dtype=complex)


class put_data_to_zeros:

    def __init__(self, N_fourier: int, Guard_size: int, mod_data: np.ndarray) -> None:
        self.N_fourier = N_fourier
        self.Guard_size = Guard_size
        self.zero_index = int(self.N_fourier/2)
        self.N_complex_points = int((self.N_fourier/2 - self.Guard_size)*2)
        self.mod_data = mod_data[:self.N_complex_points]

    
    def get_spectrum(self):
        spec = _zeros(self.N_fourier)
        
        left_point = self.Guard_size
        r = int(self.N_complex_points/2)
        spec[left_point:self.zero_index] = self.mod_data[:r]
        spec[self.zero_index+1:self.N_fourier - (self.Guard_size - 1)] = self.mod_data[r:]

        # If we want to change central zero 
        # spec[self.zero_index] = 1+ 1j
        return spec
    
    def get_time_samples(self):
        time_samples = np.fft.ifft(np.fft.fftshift(self.get_spectrum()))
        mods = np.absolute(time_samples)
        m = mods.max()
        time_samples = (time_samples/m)*(2**14)
        return time_samples


def normalize_for_pluto(complex_time_data):
    m = np.max(np.abs(complex_time_data))
    normalized_complex_time_data = (complex_time_data/m)*(2**14)
    return normalized_complex_time_data


def get_preambula():
    preambula_word = string.printable + string.ascii_letters + string.printable[::-1]

    # converts ascii letters to binary strings. 'he' -> ('01101000', '01100101')
    mapped = tuple(map(lambda x: f"{ord(x):08b}", preambula_word))
    pream_bits = create_bit_sequence_from_letters(mapped)
    q = qpsk_modualte(pream_bits)
    pre = put_data_to_zeros(1024, 100, q)
    return pre


if __name__ == '__main__':
    
    pass