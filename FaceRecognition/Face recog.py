import cv2
from simple_facerec import SimpleFacerec

# load webcam
cap = cv2.VideoCapture(0)

# uncomment load from video file and comment load webcam if you want to load video

# load from video file
# cap = cv2.VideoCapture(filename.mp4)

# Encode faces from a folder
# name using the image file name
sfr = SimpleFacerec()
sfr.load_encoding_images('images/')

while True:
    _, img = cap.read()
    
    # Detect faces
    face_location, face_names = sfr.detect_known_faces(img)
    for face_loc, name in zip(face_location, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        # face name
        cv2.putText(img, name,(x1,y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,200),2)
        # rectangle box
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,200),4)
    
    cv2.imshow('img', img)
    
    # check to stop loop with 'q' button
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()