import os
import re
import config
from devexceptions import NoDeviceFoundException
import adi

"""
    The module provides function for establishing connection
    to plugged plutos and initializing them
"""


def detect_devices() -> dict | None:
    '''Detects plutos and return dictinary of devices'''

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


def initialize_sdr(single_mode=False, tx='ANGRY', swap=False) -> tuple:
    '''
       Initialize detected devices and return
       configuration depending on the input arguments
    '''

    devices = detect_devices()
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

    return sdrtx, sdrrx



if __name__ == '__main__':
    tup = initialize_sdr(single_mode=0, tx='ANGRY', swap=False)
    print(tup)