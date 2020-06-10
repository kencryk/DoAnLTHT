import cv2
import time
# Opens the Video file

#cap1 = cv2.VideoCapture(0)


# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('img1',frame) #display the captured image
#     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
#         cv2.imwrite('c1.jpg',frame)
#         cv2.destroyAllWindows()
#         break
# cap.release()
cap = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0)
def conket1():

    i = 0
    while (cap.isOpened()):
        for i in range(5):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite('phuoc' + str(i) + '.jpg', frame)
            i += 1

        break
    cap.release()
    cv2.destroyAllWindows()

def conket3():
    ket = 0
    while(True):
        ret, img = camera.read()
        cv2.imshow('img1', img)
        for i in range(5):
            if ret == False or ket == 5:
                break
            cv2.imwrite('phuoc' + str(i) + '.jpg', img)
            ket+=1

        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.destroyAllWindows()
            break
    camera.release()
if __name__ == "__main__":
    conket3()
    #time.sleep(10)
    #conket1()

