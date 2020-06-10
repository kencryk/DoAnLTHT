import RPi.GPIO as GPIO
import time

# GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(37, GPIO.OUT)

GPIO.setup(29, GPIO.OUT)
GPIO.setup(31, GPIO.OUT)
p = GPIO.PWM(37, 50)
p1 = GPIO.PWM(29, 50)
p2 = GPIO.PWM(31, 50)

p.start(2.5)
p1.start(2.5)
p2.start(2.5)

def servoVat0():
    p.ChangeDutyCycle(2.5)  # turn towards 0 degree
    time.sleep(1) # sleep 1 second
    
    p.ChangeDutyCycle(9.5)  # turn towards -70 degree To vat 1
    time.sleep(0.5) # sleep 1 second
    
    p1.ChangeDutyCycle(9.5)  # turn towards 70 degree TO VAT 1
    time.sleep(1.5) # sleep 1 second
    
def servoVat1():
    
    p.ChangeDutyCycle(2.5)  # turn towards 0 degree
    time.sleep(1) # sleep 1 second
    
    p.ChangeDutyCycle(9.5)  # turn towards -70 degree To vat 1
    time.sleep(0.5) # sleep 1 second
    
    p1.ChangeDutyCycle(6.25)  # turn towards 70 degree TO VAT 1
    time.sleep(1.5) # sleep 1 second
        

def servoVat2():
    
    p.ChangeDutyCycle(2.5)  # turn towards 0 degree
    time.sleep(1) # sleep 1 second
    
    p.ChangeDutyCycle(6.25)  # turn towards -70 degree To vat 1
    time.sleep(0.5) # sleep 1 second
    
    p2.ChangeDutyCycle(9.5)  # turn towards 70 degree TO VAT 1
    time.sleep(1.5) # sleep 1 second
        

def servoVat3():
    
    p.ChangeDutyCycle(2.5)  # turn towards 0 degree
    time.sleep(1) # sleep 1 second
    
    p.ChangeDutyCycle(6.25)  # turn towards -70 degree To vat 1
    time.sleep(0.5) # sleep 1 second
    
    p2.ChangeDutyCycle(6.25)  # turn towards 70 degree TO VAT 1
    time.sleep(1.5) # sleep 1 second
 
