import RPi.GPIO as GPIO
import time

try:
   class servo:
         def __init__(self, ID):
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            self.ID = ID
            self.angle = 0        
            GPIO.setup(ID, GPIO.OUT)
            self.p = GPIO.PWM(ID, 50) # GPIO 17 for PWM with 50Hz
            self.p.start(7)

         def getID(self):
            return self.ID

         def openDoor(self):   
            self.p.ChangeDutyCycle(12)
            time.sleep(0.5)

         def closeDoor(self):
            self.p.ChangeDutyCycle(7)
            time.sleep(0.5)

except KeyboardInterrupt:
   p = GPIO.PWM(ID, 50) # GPIO 17 for PWM with 50Hz
   p.stop()
   GPIO.cleanup()