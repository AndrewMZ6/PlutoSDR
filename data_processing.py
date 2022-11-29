import numpy as np
import data_gen
import utils
import config


def process_data(receivced_data, show_graphs=False):


    spectrum_and_shift = lambda x: np.fft.fftshift(np.fft.fft(x))
    # creating spectrum of recieved data
    spectrum_data_recived = np.fft.fftshift(np.fft.fft(receivced_data))
    preambula = data_gen._generate_preambula()
    cut_data, abs_first_correlation = correlation(preambula, receivced_data, 0)
    cut_data_spec = spectrum_and_shift(cut_data)

    part1, part2, data = cut_data[:1024], cut_data[1024:2048], cut_data[2048:]

    corr2 = np.correlate(part2, part1, 'full')

    abscorr2 = np.abs(corr2)
    maxx = abscorr2.argmax()
    complex_max = corr2[maxx]


    first_OFDM_symbol = spectrum_and_shift(data[:1024])


    eq = utils.equalize(preambula[1024:2048], part2)
    data_eq = utils.remove_spectrum_zeros(data)



def correlation(reference:np.ndarray, received:np.ndarray, shift:int) -> tuple:
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

    left_cut_index = max_x - pream_length*2 + shift
    right_cut_index = left_cut_index + pream_length*7


    cutted = received[left_cut_index:right_cut_index]

    return cutted, abs_corr


def get_pilot_and_data_indexes(fftsize=None, guard_size=None, pilot_step:int=10) -> tuple:
    if fftsize is None:
        fftsize = config.FOURIER_SIZE

    if guard_size is None:
        guard_size = config.GUARD_SIZE

    points_available = fftsize - 2*guard_size
    pilot_indexes = np.array(range(0, points_available, pilot_step))
    data_indexes = np.array([i for i in range(points_available) if not i%pilot_step==0])

    return pilot_indexes, data_indexes


def put_data_to_indexes(pilot_indexes:np.ndarray,
                        data_indexes:np.ndarray,
                        pilots:np.ndarray,
                        data:np.ndarray):

    z = np.zeros(pilot_indexes.size + data_indexes.size)
    z[pilot_indexes] = pilots
    z[data_indexes] = data

    return z



if __name__ == '__main__':
    p, d = get_pilot_and_data_indexes()
    print(f"pilot_indexes:\n{p}\ndata_indexes:\n{d}")
    print(f"length p = {len(p)}. length d = {len(d)}")

    pilots = np.ones(83)
    data = np.ones(741)*0.4

    z = put_data_to_indexes(p, d, pilots, data)
    print(z)
    print(len(z[:412]))

    g = z[d]
    print(g)
    print(len(g))