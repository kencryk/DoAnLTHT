import serial
import time
from firebase import firebase
import cv2
from Camera import OpenCamera

# The following line is for serial over GPIO
port = 'COM7'  # note I'll use Mac OS-X if i had money

ard = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # wait for Arduino

firebase = firebase.FirebaseApplication("https://pythondbtest-d6805.firebaseio.com/", None)
cap = cv2.VideoCapture(0)


def ConveyorUsingSerialMode():
    # Serial write section

    setTempCar1 = 'M310 1\r\n'
    # setTempCar2 = 'M312 100\r\n'
    ard.flush()
    setTemp1 = str(setTempCar1)
    # setTemp2 = str(setTempCar2)
    print("Python value sent: ")
    print(setTemp1)
    ard.write(setTemp1.encode())
    # I shortened this to match the new value in your Arduino code
    time.sleep(2)

    msg = ard.read(ard.inWaiting())  # read all characters in buffer
    # print("Message from arduino: ")
    # print(msg)
    # print("Python value sent: ")
    # print(setTemp2)
    # ard.write(setTemp2.encode())
    # I shortened this to match the new value in your Arduino code
    # time.sleep(1)
    # Serial read section
    # msg = ard.read(ard.inWaiting())
    # print("Message from arduino: ")
    # print(msg)
    print("AlreadySent")
    return


def ConveyorXSendSignalPosition():
    # setTempCar1 = 'M310 1\r\n'
    setTempCar2 = 'M312 -200\r\n'
    ard.flush()
    setTemp1 = str(setTempCar2)
    print("Python value sent: ")
    print(setTemp1)
    ard.write(setTemp1.encode())
    time.sleep(2)

    msg = ard.read(ard.inWaiting())  # read all characters in buffer
    # print("Message from arduino: ")
    # print(msg)

    print("AlreadySent")


def database():
    name = "Keys"
    if name == firebase.get('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieqDY', 'Name'):
        count = firebase.get('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieqDY', 'Count')
        # name = firebase.get('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieqDY', 'Name')
        new_count = count + 1
        firebase.put('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieqDY', 'Count', new_count)
        if count < new_count:
            print(name)
            print(new_count)
            ConveyorXSendSignalPosition()
    return


if __name__ == "__main__":
    ConveyorUsingSerialMode()
    OpenCamera()
    database()
