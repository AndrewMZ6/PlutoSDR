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


def modulate(bits, fftsize=1024, guardsize=100, modulation_index=4):     #  output sequence of time domain qam modulated ofdm symbols
    # modulation index: number of bits per complex vector
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





# testing
class test_modulate(TestCase):
    def setUp(self) -> None:
        self.bits = np.random.randint(low=0, high=2, size=3700)
        self.mod = modulate(self.bits, modulation_index=16)

    def test_number_of_symbols(self):
        self.assertEqual(self.mod[1], 1)

    def test_ofdm_symbol_length(self):
        self.assertEqual(len(self.mod[0]), 824)

    def test_ofdm_type(self):
        self.assertEqual(type(self.mod[0]), np.ndarray)

    def test_ofdm_vector_type(self):
        self.assertEqual(type(self.mod[0][0]), np.complex128)

    def test_not_enough_bits(self):
        self.assertRaises(AssertionError, modulate, bits = np.random.randint(low=0, high=2, size=700))





if __name__ == '__main__':
    unittest.main()