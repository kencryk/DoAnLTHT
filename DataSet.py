import cv2

from SVMDetect import cap

n = 0

while True:
    ret, frame = cap.read()

    cv2.imshow('abc', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        n+=1
        dim = (1280, 960)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #(thresh, binary) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('TestSet2/{:02}.jpg'.format(n), binary)
        cv2.imwrite('TRAINSET/leds/{:03}.jpg'.format(n), frame)
        print(n)
        # cv2.destroyAllWindows()
        # break
#cap.release()