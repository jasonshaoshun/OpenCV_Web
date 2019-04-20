import time
import os
import numpy as np
import cv2
import pickle
from google.cloud import vision


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/true.mp4', fourcc, 20.0, (width, height))
        self.image_id = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.stroke = 2
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(BASE_DIR, "/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/analysis")
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()
        self.out.release()

    def get_frame(self):
        success, frame = self.video.read()

        face_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_smile.xml')

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/recognizers/face-trainner.yml')

        labels = {"person_name": 1}
        with open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/pickles/face-labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            color = (255, 0, 0)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, self.stroke)

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 2:
                name = labels[id_]
                color = (255, 255, 255)

                cv2.putText(frame, name, (x, y), self.font, 1, color, self.stroke, cv2.LINE_AA)

        if self.image_id % 10 == 0:
            cv2.imwrite(os.path.join(self.image_dir, "%d.png" % self.image_id), frame)

            color = (123, 222, 0)

            client = vision.ImageAnnotatorClient()

            with open(os.path.join(self.image_dir, "%d.png" % self.image_id), 'rb') as image_file:
                content = image_file.read()
            image = vision.types.Image(content=content)

            objects = client.object_localization(image=image).localized_object_annotations

            # f.write('Number of objects found: {}\n\n\n'.format(len(objects)))
            for object_ in objects:

                # f.write('{} (confidence: {})\n'.format(object_.name, object_.score))
                # f.write('Normalized bounding polygon vertices:\n')
                pts = []
                count = 0
                for vertex in object_.bounding_poly.normalized_vertices:
                    if count == 3:
                        cv2.putText(frame, '{}'.format(object_.name), (int(vertex.x * 1280), int(vertex.y * 720 + 28)),
                                    self.font, 1, color, self.stroke, cv2.LINE_AA)
                    # f.write(' - ({}, {})\n'.format(vertex.x, vertex.y))
                    pts.append([vertex.x * 1280, vertex.y * 720])
                    count += 1
                # print("pts", pts)
                a = np.asarray(pts, np.int32)
                # print("a", a)
                a = a.reshape((-1, 1, 2))
                cv2.polylines(frame, [a], True, (0, 255, 255))

            cv2.imwrite(os.path.join(self.image_dir, "test%d.png" % self.image_id), frame)
            # print("sucess")

        self.image_id += 1
        self.out.write(frame)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

