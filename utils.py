import os
import re
import config
import numpy as np



def detect_devices():
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
    '''Makes channel estimation and returns estimation array'''

    transmitted_data = remove_spectrum_zeros(transmitted_sig)
    received_data = remove_spectrum_zeros(received_sig)

    eq = transmitted_data/received_data
    return eq




def remove_spectrum_zeros(time_domain_sig: np.ndarray) -> np.ndarray:
    '''Removes guards intervals and central zero sample. Returns spectrum samples'''

    spectrum = spectrum_and_shift(time_domain_sig)
    left_part = spectrum[100:int(1024/2)] 
    right_part = spectrum[int(1024/2)+1:924+1]

    result = np.concatenate((left_part, right_part))
    return result





if __name__ == '__main__':
    print(detect_devices())
