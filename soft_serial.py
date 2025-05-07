from serial import Serial, PortNotOpenError
from serial.serialutil import SerialException
from enum import IntEnum
from threading import Thread

class Codes(IntEnum):
    FrameChangeToOddball = 0x1
    FrameChangeToBase = 0x2

    TrialEnd = 0x10
    BreakEnd = 0x11

    Quit = 0xFF

class SoftSerial(Serial):
    EVENT_PORT_NAME = "/dev/ttyUSB0"
    BAUDRATE = 9600

    def __init__(self):
        try:
            super().__init__(baudrate=self.BAUDRATE)
            self.port = self.EVENT_PORT_NAME
            
        except SerialException as e:
            print(f"WARNING: can't send events: {e}")

    def write_int(self, value: int):
        # only at trial end until we'll have a proper event:
        try:
            if value == Codes.BreakEnd:
                self.open()
                self.write(bytes((0x50)))
                self.close()
                # super().write(value.to_bytes())
        except (PortNotOpenError, SerialException):
            pass

    def parallel_write_int(self, value: int):
        # Assume that the last thread is already done
        t = Thread(target=self.write_int, args=[value])
        t.start()
        
