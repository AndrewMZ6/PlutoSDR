import numpy as np
import config


def correlation(reference, received, shift):
    '''
        Input values are time domain numpy arrays.

        reference is shorter than received, thus:
            correlate(received, reference) -> corr maximum x coordinate
            indicates the end of reference signal in received array
        otherwise:
            corrleate(reference, received) -> len(received) - corr maximum x coordinate
            indicates the start of reference signal in received array

        see "correlation_testing.py" for graphs
    '''
    
    # reference sig consists of 2 preambulas
    pream_length = reference.size

    # number of payload data points
    payload_length = config.NUMBER_OF_OFDM_SYMBOLS*config.FOURIER_SIZE

    # the whole tx signal length
    tx_length = pream_length + payload_length

    # cut piece of received data for correlation
    received_data = received[shift:shift + tx_length*3]

    
    corr = np.correlate(received_data, reference, 'full')
    abs_corr = np.abs(corr)
    max_x = abs_corr.argmax()

    # step back 2 preambula lengths to find the starting index
    left_cut_index = max_x + shift - pream_length + 1

    #
    right_cut_index = left_cut_index + tx_length


    cutted = received[left_cut_index:right_cut_index]

    return cutted, abs_corr

