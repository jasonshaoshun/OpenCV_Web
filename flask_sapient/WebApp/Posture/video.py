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


def gen():
    # analyse the live stream and store it in the static folder only works for the Chrome
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
        video = cv2.VideoCapture('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/input.mp4')
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video/output.mp4', fourcc, 20.0, (width, height))

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

        ret_val, image = video.read()
        logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

        array_humans = []

        clf = joblib.load('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/training3.pkl')

        # output = open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/output.txt", "w")
        # output.write('')
        # output.close()

        f = open("/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/Posture/output.txt", "w")
        f.write('Video start at {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        while (video.isOpened()):
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

                f.write('System Time of the Machine Running: {}, People under CCTV are {} respectively\n'.format(
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
            out.write(image)

    # close and save the text and video writing on, close the video capturing
        f.close()
        video.release()
        out.release()

