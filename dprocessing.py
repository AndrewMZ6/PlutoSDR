import numpy as np
from matplotlib import pyplot as plt


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
    pream_length = int(len(reference)/2)
    corr = np.correlate(received[shift:shift + pream_length*15], reference, 'full')
    abs_corr = np.abs(corr)
    max_x = abs_corr.argmax()


    # cut index validation

    '''
    # if left spike if found as maximum, make the right spike new maximum
    if (abs_corr[max_x]*0.8) > abs_corr[max_x - pream_length]:
        print(f"left index IF works, found max: {max_x}")
        max_x += pream_length
    '''

    
    print(f"found corr max: {max_x}")

    left_cut_index = max_x - pream_length*2 + shift
    right_cut_index = left_cut_index + pream_length*7
    print(f"left cut index: {left_cut_index}")
    print(f"right cut index: {right_cut_index}")


    cutted = received[left_cut_index:right_cut_index]

    return cutted, abs_corr

