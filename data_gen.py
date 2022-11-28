import mod
import numpy as np
import config



def _generate_preambula():
    '''Generates 2 identical preambulas with size 1024 and concatenates them'''
    preambula = mod.get_preambula()
    preambula_time_domain = preambula.get_time_samples()
    preambula_time_domain = np.concatenate((preambula_time_domain, preambula_time_domain))

    return preambula_time_domain


def generate_tx_data(frames=10, use_dpd=False, dpd_measure_mode=False) -> tuple:

    # create preambula
    preambula = _generate_preambula()
    
    
    # create user signal
    tx_signal = np.array([])
    for _ in range(frames):
        # data payload
        data_bits = mod.create_bits(config.BIT_SIZE)[:1648]
        data_modulated = mod.qpsk_modualte(data_bits)
        if _ == 0:
            data_compare = data_bits

        # create spectrum and time samples
        data_spectrum = mod.PutDataToZeros(config.FOURIER_SIZE, config.GUARD_SIZE, data_modulated)
        data_time_domain = data_spectrum.get_time_samples()
        tx_signal = np.append(tx_signal, data_time_domain)

    tx_signal = np.concatenate((preambula, tx_signal))

    return data_compare, tx_signal


if __name__ == '__main__':
    f = _generate_preambula()
    print(f[:2])    # [-11443.99475576+4087.1409842j   -2142.80634916 +668.22613374j]
    print(len(f))   # 2048