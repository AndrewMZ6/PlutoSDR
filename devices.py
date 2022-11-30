import os
import re
import config
from devexceptions import NoDeviceFoundException
import adi

"""
    The module provides functions for establishing connection
    to plugged plutos and initializing them
"""


def _detect_devices() -> dict | None:
    '''Detects plutos and returns dictinary {serial_number:usb_number} of devices found'''

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


def _configure_devices(sdrtx, sdrrx) -> tuple:
    '''Set transmitter and receiver attributes and return them configured'''

    # configure transmitter
    sdrtx.sample_rate = int(config.SAMPLE_RATE)
    sdrtx.tx_rf_bandwidth = int(config.SAMPLE_RATE)
    sdrtx.tx_lo = int(config.CENTRAL_FREQUENCY)
    sdrtx.tx_hardwaregain_chan0 = config.TX_HARDWARE_GAIN_CHAN_0
    sdrtx.tx_cyclic_buffer = config.TX_USE_CYCLIC_BUFFER

    # configure receiver
    sdrrx.rx_lo = int(config.CENTRAL_FREQUENCY)
    sdrrx.rx_rf_bandwidth = int(config.SAMPLE_RATE)
    sdrrx.rx_buffer_size = config.COMPLEX_SAMPLES_NUMBER
    sdrrx.gain_control_mode_chan0 = config.RX_GAIN_CONTROL_MODE_CHAN_0
    sdrrx.rx_hardwaregain_chan0 = config.RX_HARDWARE_GAIN_CHAN_0

    return sdrtx, sdrrx


def initialize_sdr(single_mode=False, tx='ANGRY', swap=False) -> tuple:
    '''
       Initialize detected devices and return
       configuration depending on the input arguments
    '''

    devices = _detect_devices()
    if devices is None:
        raise NoDeviceFoundException('No connected devices found')
    
    
    try:
        sdrtx = adi.Pluto(devices[tx])
    except KeyError:
        raise KeyError(f'''No device with name "{tx}" found. List of available names: {list(devices.keys())}''')


    if single_mode:
        sdrrx = sdrtx
    else:
        assert 1 < len(devices) and len(devices) < 3, 'not enough devices for NOT single mode. Must be at least two'
        for key in devices.keys():
            if key != tx:
                sdrrx = adi.Pluto(devices[key])


    

    if swap:
        sdrrx, sdrtx = sdrtx, sdrrx
    
    _configure_devices(sdrtx, sdrrx)

    return sdrtx, sdrrx



if __name__ == '__main__':
    tup = initialize_sdr(single_mode=0, tx='ANGRY', swap=False)
    print(tup)