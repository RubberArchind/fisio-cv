import base64
import json
from datetime import datetime
import time
import cv2 as cv
import mediapipe as mp
import math
import numpy as np
from utils.color_detection import ColorDetection, Point
from utils.theme import Colors
from utils.camera import Frame, global_colors


class CarryingAngle:
    def __init__(self):
        self.interpretation = ''
        self.isValid = False
        self.angle = 0
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles
        self.results = []

    def calc_angle(self, point1: Point, point2: Point):
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y

        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / math.pi) * theta

        return degree

    def calc_angle_mid(self, p1: Point, p2: Point, p3: Point):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        # Vektor BA dan BC
        vec_BA = (x1 - x2, y1 - y2)
        vec_BC = (x3 - x2, y3 - y2)

        # Dot product
        dot_product = vec_BA[0] * vec_BC[0] + vec_BA[1] * vec_BC[1]

        # Magnitudo BA dan BC
        mag_BA = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        mag_BC = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        # Cosinus theta
        cos_theta = dot_product / (mag_BA * mag_BC)
        # Ensure the argument is within the valid range [-1, 1]
        cos_theta = max(min(cos_theta, 1), -1)

        # Sudut dalam radian
        theta_rad = math.acos(cos_theta)
        degree = math.degrees(theta_rad)

        return degree

    def calc_distance(self, p1: Point, p2: Point):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_endpoint(self, point: Point, distance, angle):
        rad = math.radians(angle + 180)

        x = int(point.x + distance * np.sin(rad))
        y = int(point.y + distance * np.cos(rad))

        return Point(x, y)

    def interpret(self, angle):
        if (angle < 5):
            return 'cubitus varus'
        elif (angle >= 5 and angle <= 15):
            return 'normal'
        else:
            return 'cubitus valgus'

    def draw_line_over(self, frame, center: Point, radius, end_angle, direction, color=global_colors.yellow):
        # Convert angles to radians
        end_angle_rad = math.radians(end_angle)

        # Calculate end points of the line
        if direction == 1:
            end_point = (
                int(center.x - radius * np.sin(end_angle_rad)),
                int(center.y - radius * np.cos(end_angle_rad))
            )
        elif direction == -1:
            end_point = (
                int(center.x + radius * np.sin(end_angle_rad)),
                int(center.y + radius * np.cos(end_angle_rad))
            )

        cv.line(frame, (center.x, center.y), end_point, color, 2)

    def run(self, img):
        colors = Colors()
        detect = ColorDetection()
        frame = cv.imread(img)
        fr = Frame()
        # frame = fr.frame
        contours = detect.get_objects(frame)
        # Sort contours based on y-coordinate (top to bottom)
        contours = sorted(contours, key=lambda c: cv.boundingRect(c)[1])

        keypoints = []

        for i, contour in enumerate(contours, start=1):
            # Get the bounding box of the contour
            point_x, point_y, w, h = cv.boundingRect(contour)

            point = Point(point_x + w // 2, point_y + h // 2)

            keypoints.append(point)

            # Draw a rectangle around the detected object and label
            cv.rectangle(frame, (point_x, point_y), (point_x + w, point_y + h), colors.green, 2)
            fr.circle(frame, (point.x, point.y))
            fr.put_text(frame, str(i), (point.x + 10, point.y), color=colors.white)
        # endfor

        # fr.meta_info(frame, 'Keypoints req: 3', 'bottom_left', (0, -100), fontSize=1.2)
        # fr.meta_info(frame, 'Keypoints count: ' + str(len(keypoints)), 'bottom_left', (0, -50), fontSize=1.2)

        # Connect keypoints with lines
        if len(keypoints) == 3:
            self.isValid=True
            # fr.meta_info(frame, 'Keypoints status: valid', 'bottom_left', fontSize=1.2, color=colors.green)

            # Draw line to each keypoints
            for i in range(len(keypoints) - 1):
                fr.line(frame, (keypoints[i].x, keypoints[i].y), (keypoints[i + 1].x, keypoints[i + 1].y))

            humerus = Point(keypoints[0].x, keypoints[0].y)
            elbow = Point(keypoints[1].x, keypoints[1].y)
            wrist = Point(keypoints[2].x, keypoints[2].y)

            humerus_elbow_angle = self.calc_angle(humerus, elbow)
            elbow_wrist_angle = self.calc_angle(elbow, wrist)

            # Draw line over upper arm
            self.draw_line_over(frame, elbow, 400, humerus_elbow_angle, 1)
            self.draw_line_over(frame, humerus, 400, humerus_elbow_angle, -1)

            # Draw line over lower arm
            self.draw_line_over(frame, wrist, 400, elbow_wrist_angle, 1)
            self.draw_line_over(frame, elbow, 400, elbow_wrist_angle, -1)

            dist_elbow_wrist = self.calc_distance(elbow, wrist)

            point_hmrs_over = self.find_endpoint(elbow, dist_elbow_wrist, humerus_elbow_angle)

            fr.circle(frame, (point_hmrs_over.x, point_hmrs_over.y))

            carry_angle = self.calc_angle_mid(wrist, elbow, point_hmrs_over)
            if wrist.x > point_hmrs_over.x:
                carry_angle *= -1

            # Object information
            fr.put_text(frame, str(int(carry_angle)), (elbow.x + 10, elbow.y + 50), fontSize=1)
            self.angle = int(carry_angle)
            fr.meta_info(frame, 'Carrying angle: ' + str(int(carry_angle)), fontSize=0.5)
            fr.meta_info(frame, 'Condition: ' + self.interpret(carry_angle), 'top_left', (0, 50), fontSize=0.5)
            self.interpretation = self.interpret(carry_angle)
            # Save results
            self.results.append((int(carry_angle), datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

        else:
            self.isValid = False
            # fr.meta_info(frame, 'Keypoints status: invalid', 'bottom_left', fontSize=1.2, color=colors.red)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        with open('result_image.jpg', 'wb') as file:
            file.write(frame)

        # yield frame
        dt = {
            "isValid": self.isValid,
            "angle": self.angle,
            "interpretation": self.interpretation,
            "reqPoint": 3,
            "img": base64.b64encode(frame).decode("utf-8")}
        yield json.dumps(dt)
