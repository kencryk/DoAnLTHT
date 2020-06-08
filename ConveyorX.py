import serial
import time

# The following line is for serial over GPIO
port = '/dev/ttyUSB0'  # note I'll use Mac OS-X if i had money

ard = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # wait for Arduino


def ConveyorX(signal, ard):
    # Serial write section
    port = '/dev/ttyUSB0'  # note I'll use Mac OS-X if i had money

    ard = serial.Serial(port, 115200, timeout=1)
    time.sleep(1)  # wait for Arduino
    setTempCar1 = signal + '\r\n'
    # setTempCar2 = 'M312 100\r\n'
    ard.flush()
    setTemp1 = str(setTempCar1)
    # setTemp2 = str(setTempCar2)
    print("Python value sent: ")
    print(setTemp1)
    ard.write(setTemp1.encode())
    # I shortened this to match the new value in your Arduino code
    time.sleep(1)

    msg = ard.read(ard.inWaiting())  # read all characters in buffer
    print("AlreadySent")
    return



# if __name__ == "__main__":
#     ConveyorX('M310 1')
#     OpenCamera()
#     database()
