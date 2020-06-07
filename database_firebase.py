from firebase import firebase
from datetime import datetime
from testLCD import DisplayLCD
firebase = firebase.FirebaseApplication("https://pythondbtest-d6805.firebaseio.com/", None)

def goToFireBase():
    now = datetime.now()
    currentDT = datetime.now()

    
    other_data = {
        'Time' : now.strftime("%m/%d/%Y, %H:%M:%S"),
        'Order' : 1
    }

    
    count = firebase.get('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieq', 'Total_Count')
    count = count +1
    firebase.put('/pythondbtest-d6805/Customer/-M8tXnpZZlLKBVOieq', 'Total_Count', count)
    name = firebase.get('/pythondbtest-d6805/Customer/', '-M8tXnpZZlLKBVOieq')
    
    DisplayLCD(list(name.keys())[0], count)
    return True
