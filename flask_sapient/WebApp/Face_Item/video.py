import time
import os
import numpy as np
import cv2
import pickle
import sys
from google.cloud import vision


# Using OpenCV to capture from device 0. If you have trouble capturing
# from a webcam, comment the line below out and use a video file
# instead.


def gen():
    video = cv2.VideoCapture('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/input.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    image_id = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    stroke = 2
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR,
                                  "/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/analysis")

    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/output.mp4', fourcc, 20.0, (width, height))
    # If you decide to use video.mp4, you must have this file in the folder

    # open the text to write recognition information
    f = open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/output.txt", "w")
    f.write('Video start at {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    while (video.isOpened()):
        success, frame = video.read()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        face_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/cascades/data/haarcascade_smile.xml')

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/recognizers/face-trainner.yml')

        labels = {"person_name": 1}
        with open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Face_Item/pickles/face-labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            out.release()
            sys.exit("face/item video finished!")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # outline the faces within the frame
            color = (255, 0, 0)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            # if the confidence rate is high enough, put the name in blue onto the frame
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 2:
                name = labels[id_]
                color = (255, 255, 255)

                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # process the image recognition every five frames
        if image_id % 5 == 0:
            cv2.imwrite(os.path.join(image_dir, "%d.png" % image_id), frame)

            color = (123, 222, 0)

            client = vision.ImageAnnotatorClient()

            with open(os.path.join(image_dir, "%d.png" % image_id), 'rb') as image_file:
                content = image_file.read()
            image = vision.types.Image(content=content)

            objects = client.object_localization(image=image).localized_object_annotations

            # Item recognition returns the objects
            for object_ in objects:
                # f.write('{} (confidence: {})\n'.format(object_.name, object_.score))
                # f.write('Normalized bounding polygon vertices:\n')
                f.write('The object detected is {} with confidence rate of {}\n'.format(object_.name, round(object_.score, 2)))
                pts = []
                count = 0
                for vertex in object_.bounding_poly.normalized_vertices:
                    if count == 3:
                        cv2.putText(frame, '{}'.format(object_.name), (int(vertex.x * 1280), int(vertex.y * 720 + 28)),
                                    font, 1, color, stroke, cv2.LINE_AA)
                    # f.write(' - ({}, {})\n'.format(vertex.x, vertex.y))
                    pts.append([vertex.x * 1280, vertex.y * 720])
                    count += 1
                # print("pts", pts)
                a = np.asarray(pts, np.int32)
                # print("a", a)
                a = a.reshape((-1, 1, 2))
                # put the outlines of the objects onto the frame
                cv2.polylines(frame, [a], True, (0, 255, 255))

            cv2.imwrite(os.path.join(image_dir, "test%d.png" % image_id), frame)

        image_id += 1
        out.write(frame)

    # We are using Motion JPEG, but OpenCV defaults to capture raw images,
    # so we must encode it into JPEG in order to correctly display the
    # video stream.

    # close and save the text and video writing on, close the video capturing
    f.close()
    video.release()
    out.release()

