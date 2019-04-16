import time
import os
import numpy as np
import cv2
import pickle


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.out = cv2.VideoWriter('true.mp4', fourcc, 20.0, (width, height))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()
        self.out.release()

    def get_frame(self):
        success, frame = self.video.read()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        face_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/team/flask_web/flask_sapient/WebApp/cascades/data/haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/team/flask_web/flask_sapient/WebApp/cascades/data/haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/team/flask_web/flask_sapient/WebApp/cascades/data/haarcascade_smile.xml')

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('/Users/shunshao/Desktop/team/flask_web/flask_sapient/WebApp/recognizers/face-trainner.yml')

        labels = {"person_name": 1}
        with open("/Users/shunshao/Desktop/team/flask_web/flask_sapient/WebApp/pickles/face-labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}

        font = cv2.FONT_HERSHEY_SIMPLEX
        stroke = 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            color = (255, 0, 0)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 2:
                name = labels[id_]
                color = (255, 255, 255)

                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        self.out.write(frame)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

