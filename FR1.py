# Face Recognition - Attempt 1

# import openCV 2
import cv2

# loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Detection happens here using the cascades above, the image and greyed image
def detect(grey, frame):
    faces = face_cascade.detectMultiScale(grey, 1.3, 5) # the image will be rediced 1.3 times and will accept 5 neighboar zones
    for (x, y, w, h) in faces: # go through all faces detected and pain over them
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # draw rectangle
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes: # go through all eyes detected and pain over them
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # draw rectangle
    
    return frame

# Get video from webcame

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()