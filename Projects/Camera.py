import numpy as np
import cv2

# video frame
# width, height = 400, 600

camera = cv2.VideoCapture(1)

# camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
sucess, frame = camera.read()

# red color boundaries [B, G, R]
lower = [1, 0, 20]
upper = [60, 40, 220]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

while True:  
    sucess, frame = camera.read()
    if not sucess: break
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    lab = cv2.split(lab)

    binary = cv2.adaptiveThreshold(lab[2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel,iterations=3)

    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    points = np.concatenate(contours)

    (x,y,w,h) = cv2.boundingRect(points)

    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
    
    cv2.imshow("Result", frame)

    # cv2.imshow("Result", np.hstack([frame, output]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

camera.release()
cv2.destroyAllWindows()