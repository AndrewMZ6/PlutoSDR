import os
import re
import config


def detect_devices():
    '''Detects'''

    x = os.popen('iio_info -s').read()
    s = re.findall(r"serial=.*", x)
    l = len(s)
    devices = {}

    for i in range(l-1 if l > 1 else 1):

        serial, usb = s[i].split(' ')
        serail_number = serial.strip()[7:]   # remove 'serial='
        usb_number = usb.strip()[1:-1]       # remove '[' and ']'
        devices.update({config.devices[serail_number]:usb_number})

    return devices


if __name__ == '__main__':
    print(detect_devices())