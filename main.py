from os import getenv
from os.path import splitext
from sys import argv

import cv2


class OpenCV():
    def __init__(self):
        self.image_path = ''
        self.haarcascade = f'{getenv("VIRTUAL_ENV")}/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'

    def set_image_path(self, image_path=''):
        if not image_path:
            raise Exception('image_path empty')
        self.image_path = image_path

    def get_output_path(self, cnt=0):
        return f'{splitext(self.image_path)[0]}_{cnt}{splitext(self.image_path)[1]}'

    def run(self):
        image_org = cv2.imread(self.image_path)
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(self.haarcascade)
        faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50))
        if len(faces) == 0:
            raise Exception('face empty')
        for i, (x, y, w, h) in enumerate(faces):
            padding = int(h / 4)
            x1, x2, y1, y2 = x, x + w, y - padding, y + h + padding
            cv2.imwrite(self.get_output_path(cnt=i), image_org[y1:y2, x1:x2])


if __name__ == "__main__":
    cv = OpenCV()
    cv.set_image_path(argv[1])
    cv.run()