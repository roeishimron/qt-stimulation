from serial import Serial, PortNotOpenError
from serial.serialutil import SerialException


class SoftSerial(Serial):
    EVENT_PORT_NAME = "/dev/ttyUSB0"
    BAUDRATE = 115200

    def __init__(self):
        try:
            super().__init__(self.EVENT_PORT_NAME, self.BAUDRATE)
        except SerialException as e:
            print(f"WARNING: can't send events: {e}")

    def write_int(self, value: int):
        try:
            super().write(value.to_bytes())
        except PortNotOpenError:
            pass
