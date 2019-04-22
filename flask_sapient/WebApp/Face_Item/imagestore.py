import os
import cv2

# Capture video from camera
cap = cv2.VideoCapture(0)

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# store the screen shots to the image library
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/imageLibrary")
image_id = 1


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        out.write(frame)

        cv2.imshow('frame',frame)

        # write the image
        cv2.imwrite(os.path.join(image_dir, "%d.png" %image_id), frame)
        image_id += 1
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

