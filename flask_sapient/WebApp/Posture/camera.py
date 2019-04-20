import time
import os
import numpy as np
import cv2
import pickle
import argparse
import logging
import time
import sys
import cv2
from sklearn.externals import joblib
from flask_sapient.WebApp.Posture.featureExtraction.tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from flask_sapient.WebApp.Posture.featureExtraction.tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh
from flask_sapient.WebApp.Posture.featureExtraction.tf_pose_estimation import helpers
import datetime


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.f = open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/output.txt", "w")
        self.f.write('Video start at {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.video = cv2.VideoCapture(0)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/true.mp4', fourcc, 20.0, (width, height))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.f.close()
        self.video.release()
        self.out.release()

    def get_frame(self):
        parser = argparse.ArgumentParser(description='tf_pose_estimation realtime webcam')
        parser.add_argument('--camera', type=int, default=0)

        parser.add_argument('--resize', type=str, default='0x0',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
        parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')
        args, _ = parser.parse_known_args()

        logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
        w, h = model_wh(args.resize)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        logger.debug('cam read+')

        ret_val, image = self.video.read()
        logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

        array_humans = []

        clf = joblib.load('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/training3.pkl')

        # output = open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/output.txt", "w")
        # output.write('')
        # output.close()

        logger.debug('image process+')
        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        except Exception:
            sys.exit("video finished!")

        # Go over all the humans in the photo
        for human in humans:
            array_human = []
            # For every single limb that each human has
            for i in range(0, 16):
                try:
                    # record the limb coordinates
                    array_human.append(human.body_parts[i].x)
                    array_human.append(human.body_parts[i].y)
                # If there are no coordinates for a certain limb record null
                except KeyError:
                    array_human.append(0.0)
                    array_human.append(0.0)
            # Improve the generated feature extraction by removing unclear data points
            if helpers.clean_data(array_human) is None:
                array_human = []
            else:
                array_humans.append(array_human)
                array_human = []

        # Print the current classification
        if len(array_humans) > 0:
            # print(clf.predict(array_humans))

            self.f.write('System Time of the Machine Running: {}, People under CCTV are {} respectively\n'.format(
                datetime.datetime.now(), clf.predict(array_humans)))
            print('System Time of the Machine Running: {}, People under CCTV are {} respectively\n'.format(
                datetime.datetime.now(), clf.predict(array_humans)))

        del array_humans[:]
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        logger.debug('finished+')
        self.out.write(image)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

