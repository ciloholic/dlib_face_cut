from os.path import splitext
from sys import argv

import cv2

from dlib import get_frontal_face_detector, shape_predictor

from imutils import face_utils

from scipy.spatial.distance import euclidean


class Dlib():
    def __init__(self):
        self.image_path = ''
        self.predictor = shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.detector = get_frontal_face_detector()

    def set_image_path(self, image_path=''):
        if not image_path:
            raise Exception('image_path empty')
        self.image_path = image_path

    def get_output_path(self, image_cnt=0, eye_cnt=0):
        return f'{splitext(self.image_path)[0]}_i{image_cnt}_e{eye_cnt}{splitext(self.image_path)[1]}'

    def calc_ear(self, eyes):
        eye_cnt = 0
        for eye in [eyes[36:42], eyes[42:48]]:
            A = euclidean(eye[1], eye[5])
            B = euclidean(eye[2], eye[4])
            C = euclidean(eye[0], eye[3])
            if round((A + B) / (2.0 * C), 3) > 0.2:
                eye_cnt += 1
        return eye_cnt

    def run(self):
        image_org = cv2.imread(self.image_path)
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        dets, scores, idx = self.detector.run(image, 0)
        if len(dets) == 0:
            raise Exception('face empty')
        for i, rect in enumerate(dets):
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            copy = image_org.copy()
            for (x, y) in shape:
                cv2.circle(copy, (x, y), 1, (0, 255, 0), -1)
            cv2.imwrite(self.get_output_path(image_cnt=i, eye_cnt=self.calc_ear(shape)), copy)


if __name__ == "__main__":
    dlib = Dlib()
    dlib.set_image_path(argv[1])
    dlib.run()
