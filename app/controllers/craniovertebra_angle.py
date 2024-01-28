import base64
import json
import time
from datetime import datetime
import cv2 as cv
import mediapipe as mp
import math

from utils.color_detection import ColorDetection, Point
from utils.theme import Colors
from utils.camera import Frame, global_colors

class CraniovertebraAngle:
    def __init__(self):
        self.isValid = False
        self.angle = 0
        self.interpretation= ''
        self.results = []

    def calc_angle(self, point1: Point, point2: Point):
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y

        theta = math.atan2(y2 - y1, x2 - x1)
        degree = int(math.degrees(theta))

        return degree

    def interpret(self, angle):
        if (angle <= 48):
            return 'forward'
        elif (angle <= 90):
            return 'normal'
        else:
            return ''

    def run(self, img):
        colors = Colors()
        detect = ColorDetection()
        frame = cv.imread(img)
        fr = Frame()

        contours = detect.get_objects(frame)
        # Sort contours based on y-coord (top to bottom)
        contours = sorted(contours, key=lambda c: cv.boundingRect(c)[1])

        keypoints = []

        for i, contour in enumerate(contours, start=1):
            point_x, point_y, w, h = cv.boundingRect(contour)

            point = Point(point_x + w // 2, point_y + h // 2)

            keypoints.append(point)

            cv.rectangle(frame, (point_x, point_y), (point_x + w, point_y + h), colors.green, 2)
            fr.circle(frame, (point.x, point.y))
            fr.put_text(frame, str(i), (point.x + 10, point.y), color=colors.white)
        # endfor

        # Connect keypoints with lines
        if (len(keypoints) == 2):
            self.isValid= True
            # Draw horizontal line from C7
            fr.line(frame, (keypoints[1].x, keypoints[1].y), (keypoints[1].x - 200, keypoints[1].y))
            fr.line(frame, (keypoints[1].x, keypoints[1].y), (keypoints[1].x + 200, keypoints[1].y))

            # Draw line to each keypoints
            for i in range(len(keypoints) - 1):
                fr.line(frame, (keypoints[i].x, keypoints[i].y), (keypoints[i + 1].x, keypoints[i + 1].y))

            # Calculate cv angle
            cv_angle = self.calc_angle(keypoints[0], keypoints[1])
            fr.put_text(frame, str(int(cv_angle)), (keypoints[1].x + 10, keypoints[1].y + 50), fontSize=1)
            self.angle= int(cv_angle)
            # fr.put_text(frame, 'Craniovertebra angle: ' + str(int(cv_angle)), (10, 50), fontSize=0.5)
            # fr.put_text(frame, 'Condition: ' + self.interpret(cv_angle), (10, 100), fontSize=0.5)
            self.interpretation = self.interpret(cv_angle)
            # Save results
            self.results.append((int(cv_angle), datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        # endif

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        with open('result_image.jpg', 'wb') as file:
            file.write(frame)

        # yield frame
        dt = {
            "isValid": self.isValid,
            "angle": self.angle,
            "interpretation": self.interpretation,
            "reqPoint": 2,
            "img": base64.b64encode(frame).decode("utf-8")
        }

        yield json.dumps(dt)
