from datetime import datetime
import time
import cv2 as cv
import mediapipe as mp
import math
import numpy as np
from PIL import Image


class QAngle:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles
        self.results = []
        self.img = []
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
            "cyan": (0, 255, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "pink": (255, 192, 203)
        }

    def find_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_endpoint(self, point, distance, angle):
        rad = math.radians(angle)

        x = int(point[0] + distance * np.sin(rad))
        y = int(point[1] + distance * np.cos(rad))

        return x, y

    def find_angle(self, x1, y1, x2, y2):
        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / math.pi) * theta

        return degree

    def find_angle_mid(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

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

        # Sudut dalam radian
        theta_rad = math.acos(cos_theta)
        degree = math.degrees(theta_rad)

        return degree

    def get_landmarks(self, frame):
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = self.pose.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        lm = results.pose_landmarks
        lm_pose = self.mp_pose.PoseLandmark

        return lm, lm_pose

    def get_keypoints(self, lm, lm_pose, w, h):
        keypoints = {}
        # Left hip
        keypoints["l_hip_x"] = int(lm.landmark[lm_pose.LEFT_HIP].x * w)
        keypoints["l_hip_y"] = int(lm.landmark[lm_pose.LEFT_HIP].y * h)
        # Left knee
        keypoints["l_knee_x"] = int(lm.landmark[lm_pose.LEFT_KNEE].x * w)
        keypoints["l_knee_y"] = int(lm.landmark[lm_pose.LEFT_KNEE].y * h)
        # Left ankle
        keypoints["l_ankle_x"] = int(lm.landmark[lm_pose.LEFT_ANKLE].x * w)
        keypoints["l_ankle_y"] = int(lm.landmark[lm_pose.LEFT_ANKLE].y * h)

        return keypoints

    def show_landmarks(self, frame, lm):
        self.mp_draw.draw_landmarks(
            frame,
            lm,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_draw_style.get_default_pose_landmarks_style()
        )

    def draw_line_over(self, frame, center, radius, end_angle, direction, color='yellow'):
        # Convert angles to radians
        end_angle_rad = math.radians(end_angle)

        # Calculate end points of the line
        if direction == 1:
            end_point = (
                int(center[0] - radius * np.sin(end_angle_rad)),
                int(center[1] - radius * np.cos(end_angle_rad))
            )
        elif direction == -1:
            end_point = (
                int(center[0] + radius * np.sin(end_angle_rad)),
                int(center[1] + radius * np.cos(end_angle_rad))
            )

        cv.line(frame, center, end_point, color, 2)

    def draw_angle_indicator(self, frame, center, radius, start_angle, end_angle, color):
        # Convert angles to radians
        start_angle_rad = math.radians(start_angle)
        end_angle_rad = math.radians(end_angle)

        # Calculate start and end points of the arc
        start_point = (
            int(center[0] - radius * np.cos(start_angle_rad)),
            int(center[1] - radius * np.sin(start_angle_rad))
        )
        end_point = (
            int(center[0] - radius * np.cos(end_angle_rad)),
            int(center[1] - radius * np.sin(end_angle_rad))
        )

        # Draw the arc on the frame
        cv.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)

        # Draw lines connecting the center to the start and end points
        cv.line(frame, center, start_point, color, 2)
        cv.line(frame, center, end_point, color, 2)

    def get_angle(self, fr, frame, keypoints, w, h, font, colors):
        l_hip_x = keypoints["l_hip_x"]
        l_hip_y = keypoints["l_hip_y"]
        l_knee_x = keypoints["l_knee_x"]
        l_knee_y = keypoints["l_knee_y"]
        l_ankle_x = keypoints["l_ankle_x"]
        l_ankle_y = keypoints["l_ankle_y"]

        green = colors["green"]
        red = colors["red"]
        yellow = colors["yellow"]
        pink = colors["pink"]

    def save_img(self, img):
        self.img = img

    def run(self, img):
        fr = cv.imread("tes.jpg")
        frame = np.array(fr)
        lm, lm_pose = self.get_landmarks(frame)
        print(lm, lm_pose)

        if lm:
            keypoints = self.get_keypoints(lm, lm_pose, frame.shape[1], frame.shape[0])
            print(keypoints)
            self.get_angle(frame, frame, keypoints, frame.shape[1], frame.shape[0], self.font, self.colors)

        self.show_landmarks(frame, lm)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        with open('result_image.jpg', 'wb') as file:
            file.write(frame)

        yield frame
