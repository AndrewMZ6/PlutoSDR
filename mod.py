from operator import mod
import commpy as cp
import numpy as np
from matplotlib import pyplot as plt
import string




def create_bit_sequence_from_letters(t: tuple) -> np.ndarray:
    '''
        Returns concatenated numpy array from input tuple.
        ('01101000', '01100101') -> array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    '''

    result = np.array([])

    for letters in t:
        nparr = np.fromiter(letters, dtype=int)
        result = np.append(result, nparr)
    
    result = np.array(result, dtype=int)
    return result


def create_bits(size):
    return np.random.randint(low=0, high=2, size=size)


def qpsk_modualte(bits: np.ndarray) -> np.ndarray:
    '''
        Modulates input bits 'bits'.
        Returns numpy array of complex vectors.

            [0, 1, 0, 0, 1, ...]  ->  [1+1j, 1-1j, -1+1j, ...]
    '''
    qpsk = cp.modulation.QAMModem(4)
    return qpsk.modulate(bits)


def qpsk_demodulate(complex_data: np.ndarray) -> np.ndarray:
    '''
        Demodulates input complex vectors 'complex_data'.
        Returns bits as numpy array.

        [1+1j, 1-1j, -1+1j, ...]    ->  [0, 1, 0, 0, 1, ...]
    '''
    qpsk = cp.modulation.QAMModem(4)

    return qpsk.demodulate(complex_data, 'hard')





def _zeros(size):
    return np.zeros(size, dtype=complex)


class PutDataToZeros:
    '''
        Accepts complex vectors (which are modulated bits).

            method: get_spectrum()
                Places complex vectors into zeros array,
                thus creating OFDM spectrum

            method: get_time_samplex()
                1. calls get_spectrum() to create OFDM spectrum
                2. fftshifts it
                3. makes inverse FFT to get time samplex
                4. normalizes the time samples
    '''


    def __init__(self, N_fourier: int, Guard_size: int, mod_data: np.ndarray) -> None:
        self.N_fourier = N_fourier
        self.Guard_size = Guard_size

        self.zero_index = int(self.N_fourier/2)
        self.N_complex_vectors = int(self.N_fourier - self.Guard_size*2)

        assert mod_data.size >= self.N_complex_vectors, 'not enough complex vectors to create OFDM symbol'

        self.mod_data = mod_data[:self.N_complex_vectors]

    
    def get_spectrum(self) -> np.ndarray:

        spec = _zeros(self.N_fourier)
        
        left_point = self.Guard_size
        r = int(self.N_complex_vectors/2)
        spec[left_point:self.zero_index] = self.mod_data[:r]
        spec[self.zero_index+1:self.N_fourier - (self.Guard_size - 1)] = self.mod_data[r:]

        return spec
    
    def get_time_samples(self) -> np.ndarray:
        time_samples = np.fft.ifft(np.fft.fftshift(self.get_spectrum()))
        time_samples = _normalize_for_pluto(time_samples)

        return time_samples


def _normalize_for_pluto(complex_time_data):

    m = np.max(np.abs(complex_time_data))
    normalized_complex_time_data = (complex_time_data/m)*(2**14)
    return normalized_complex_time_data


def get_preambula() -> PutDataToZeros:
    '''
        Returns instance of preambula 'pre'. 
        The instance has methods to generate time dommain complex vectors
    '''
    preambula_word = string.printable + string.ascii_letters + string.printable[::-1]

    # converts ascii letters to binary strings. 'he' -> ('01101000', '01100101')
    mapped = tuple(map(lambda x: f"{ord(x):08b}", preambula_word))
    pream_bits = create_bit_sequence_from_letters(mapped)
    q = qpsk_modualte(pream_bits)
    pre = PutDataToZeros(1024, 100, q)


    return pre


if __name__ == '__main__':
    pre = get_preambula()
    pre_spec = pre.get_spectrum()

    fig, (ax1, ax2) = plt.subplots(2, 2)
    ax1[0].scatter(pre_spec.real, pre_spec.imag)
    ax1[1].plot(np.abs(pre_spec))

    


    plt.show()