import unittest
import numpy as np
import commpy as cp
from unittest import TestCase


bits = np.random.randint(low=0, high=2, size=3700)
print(bits)

# input values for class
modulation_index = 16
fftsize = 1024
guardsize = 100


def modulate(bits, fftsize=1024, guardsize=100, modulation_index=4):
    '''The function converts and array of bits into a sequence of 
        ofdm symbols. Number of ofdm symbols and the size of one symbol
        depends on the input arguments'''


    # {modulation index: number of bits per complex vector}
    bits_per_complex_vector = {4: 2, 16: 4}

    # available number of complex data vectors
    complex_vectors_number_per_ofdm_symbol = fftsize - guardsize*2       # in reality the right guard has 1 less zero than the left one, but we also consider the carrier frequency zero


    # available number of bits that could contains in one ofdm symbol
    bits_per_one_ofdm_symbol = bits_per_complex_vector[modulation_index]*complex_vectors_number_per_ofdm_symbol

    # stop executon if not enough input bits for creating at least one ofdm symbol
    assert len(bits) >= bits_per_one_ofdm_symbol, f'too few bits for one ofdm symbol, bits number must be not less than {bits_per_one_ofdm_symbol}'

    number_of_ofdm_symbols = len(bits)//bits_per_one_ofdm_symbol
    print(f'number of ofdm symbols: {number_of_ofdm_symbols}')
    print(f"number of complex vectors={complex_vectors_number_per_ofdm_symbol*number_of_ofdm_symbols}, number of available bits={bits_per_one_ofdm_symbol*number_of_ofdm_symbols}")
    available_bits = bits[:bits_per_one_ofdm_symbol*number_of_ofdm_symbols]

    qam_modulator = cp.modulation.QAMModem(modulation_index)
    mod = qam_modulator.modulate(available_bits)
    return mod, number_of_ofdm_symbols


mod, n = modulate(bits, modulation_index=16)
print(f"mod length: {len(mod)}, mod type: {type(mod)}")


def define_spectrum_data_indexes(fftsize=1024, guardsize=100):
    '''
        Utility function.
        Calculates indexes for complex vectors placement in ofdm spectrum.
        Returns tuple with 2 'range' generators containing indexes.
        The first 'range' contains indexes for placement complex vectors before
        the central zero, the second - after the central zero respectively.

        Usage example: 

            l, r = define_spectrum_data_indexes(fftsize=512, guardsize=100)
            spectrum = numpy.zeros(512)
            spectrum[l] = complex_vectors_before_central_zero
            spectrum[r] = complex_vectors_after_central_zero

        OR

            for i, index in enumerate(l):
                spectrum[index] = complex_vectors_before_central_zero[i]

            for j, index in enumerate(l):
                spectrum[index] = complex_vectors_before_central_zero[j]
    '''


    assert fftsize%2 == 0, 'fftsize should be an even number'
    assert fftsize - 2*guardsize >= 2, 'spectrum cannot contain only guards. Theres no place for data! Are you mad?'

    central_zero_index = int(fftsize/2)
    left_data_indexes = range(guardsize, central_zero_index)
    right_data_indexes = range(central_zero_index + 1, fftsize - (guardsize - 1))

    assert len(left_data_indexes) == len(right_data_indexes), f"{len(left_data_indexes)} != {len(right_data_indexes)}"

    return left_data_indexes, right_data_indexes


l, r = define_spectrum_data_indexes(fftsize=512)
print(l[0], tuple(r[:5]))


# testing
class test_modulate(TestCase):
    def setUp(self) -> None:
        self.bits = np.random.randint(low=0, high=2, size=3700)
        self.mod, self.n = modulate(self.bits, modulation_index=16)

    def test_number_of_symbols(self):
        self.assertEqual(self.n, 1)

    def test_ofdm_symbol_length(self):
        self.assertEqual(len(self.mod), 824)

    def test_ofdm_type(self):
        self.assertEqual(type(self.mod), np.ndarray)

    def test_ofdm_vector_type(self):
        self.assertEqual(type(self.mod[0]), np.complex128)

    def test_not_enough_bits(self):
        self.assertRaises(AssertionError, modulate, bits = np.random.randint(low=0, high=2, size=700))


class test_define_spectrum_data_indexes(TestCase):

    def test_output_range_length(self):
        l, r = define_spectrum_data_indexes(fftsize=512)
        self.assertEqual(len(l), len(r))

    def test_first_left_data_index(self):
        l, r = define_spectrum_data_indexes(fftsize=1024, guardsize=100)
        self.assertEqual(l[0], 100)

    def test_last_left_data_index(self):
        l, r = define_spectrum_data_indexes(fftsize=1024, guardsize=100)
        self.assertEqual(l[-1], 511)

    def test_first_right_data_index(self):
        fftsize=1024
        central_zero_index = int(1024/2) # 512
        l, r = define_spectrum_data_indexes(fftsize=fftsize, guardsize=100)
        self.assertEqual(r[0], 512 + 1)

    def test_last_right_data_index(self):
        l, r = define_spectrum_data_indexes(fftsize=1024, guardsize=100)
        last_index = 1024 - 100  # +1 is not in 'r' since we never reach the last index
                                 # right guard zeros start with 925 index
        self.assertEqual(r[-1], last_index)

    def test_fftsize_and_guardsize(self):
        self.assertRaises(AssertionError, define_spectrum_data_indexes, fftsize=1000, guardsize=500)

    def test_fftsize_is_not_even(self):
        self.assertRaises(AssertionError, define_spectrum_data_indexes, fftsize=103)


if __name__ == '__main__':
    unittest.main()