from firebase import firebase
from datetime import datetime
from testLCD import DisplayLCD
firebase = firebase.FirebaseApplication("https://pythondbtest-d6805.firebaseio.com/", None)

def goToFireBase(data_string_firebase):
    now = datetime.now()
    currentDT = datetime.now()

    
    other_data = {
        'Time' : now.strftime("%m/%d/%Y, %H:%M:%S"),
    }

    
    count = firebase.get('/pythondbtest-d6805/Customer/'+data_string_firebase+'', 'Total_Count')
    count = count +1
    firebase.put('/pythondbtest-d6805/Customer/'+data_string_firebase+'', 'Total_Count', count)
    name_dict = firebase.get('/pythondbtest-d6805/Customer/', data_string_firebase)
    name  = list(name_dict)[0]
    firebase.post('/pythondbtest-d6805/Customer/'+data_string_firebase+'/'+name+'', other_data)
    DisplayLCD(name, count)
    return True
