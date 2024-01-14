import base64
import json
import time
from datetime import datetime
import cv2 as cv
import mediapipe as mp
import math

import numpy as np
from controllers.camera import Camera, Frame, Record

class CraniovertebraAngle:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles
        self.results = []
        self.aligned = False
        self.angle = -1

    def find_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def find_angle(self, x1, y1, x2, y2, facing):
        theta = math.atan2(y2 - y1, x2 - x1)
        degree = int(math.degrees(theta))

        if facing == "left":
            degree += 180
        else:
            degree *= -1

        return degree
    
    def find_midpoint(self, x1, y1, x2, y2):
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
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
        # Left shoulder
        keypoints["l_shldr_x"] = int(lm.landmark[lm_pose.LEFT_SHOULDER].x * w)
        keypoints["l_shldr_y"] = int(lm.landmark[lm_pose.LEFT_SHOULDER].y * h)
        # Right shoulder
        keypoints["r_shldr_x"] = int(lm.landmark[lm_pose.RIGHT_SHOULDER].x * w)
        keypoints["r_shldr_y"] = int(lm.landmark[lm_pose.RIGHT_SHOULDER].y * h)
        # Left ear
        keypoints["l_ear_x"] = int(lm.landmark[lm_pose.LEFT_EAR].x * w)
        keypoints["l_ear_y"] = int(lm.landmark[lm_pose.LEFT_EAR].y * h)
        # Right ear
        keypoints["r_ear_x"] = int(lm.landmark[lm_pose.RIGHT_EAR].x * w)
        keypoints["r_ear_y"] = int(lm.landmark[lm_pose.RIGHT_EAR].y * h)
        # Left outer eye
        keypoints["l_eye_x"] = int(lm.landmark[lm_pose.LEFT_EYE_OUTER].x * w)
        keypoints["l_eye_y"] = int(lm.landmark[lm_pose.LEFT_EYE_OUTER].y * h)
        # Right outer eye
        keypoints["r_eye_x"] = int(lm.landmark[lm_pose.RIGHT_EYE_OUTER].x * w)
        keypoints["r_eye_y"] = int(lm.landmark[lm_pose.RIGHT_EYE_OUTER].y * h)
        # Nose
        keypoints["nose_x"] = int(lm.landmark[lm_pose.NOSE].x * w)
        
        return keypoints
    
    def show_landmarks(self, frame, lm):
        self.mp_draw.draw_landmarks(
            frame, 
            lm, 
            self.mp_pose.POSE_CONNECTIONS, 
            landmark_drawing_spec=self.mp_draw_style.get_default_pose_landmarks_style()
        )
    
    def get_angle(self, fr, frame, keypoints, w, h, font, colors):
        l_shldr_x, l_shldr_y = keypoints["l_shldr_x"], keypoints["l_shldr_y"]
        r_shldr_x, r_shldr_y = keypoints["r_shldr_x"], keypoints["r_shldr_y"]
        l_ear_x, l_ear_y = keypoints["l_ear_x"], keypoints["l_ear_y"]
        r_ear_x, r_ear_y = keypoints["r_ear_x"], keypoints["r_ear_y"]
        l_eye_x, l_eye_y = keypoints["l_eye_x"], keypoints["l_eye_y"]
        r_eye_x, r_eye_y = keypoints["r_eye_x"], keypoints["r_eye_y"]
        nose_x = keypoints["nose_x"]
        
        green = colors["green"]
        red = colors["red"]
        yellow = colors["yellow"]
        pink = colors["pink"]
        
        # Calculate distance between left shoulder and right shoulder points.
        offset = self.find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        print("OFFsET", offset)
        # Assist to align the camera to point at the side view of the person.
        if offset < 50:
            self.aligned = True
            cv.putText(frame, str(int(offset)) + ' Aligned', (10, h -50), font, 0.9, green, 2)
        else:
            self.aligned = False
            cv.putText(frame, str(int(offset)) + ' Not Aligned', (10, h -50), font, 0.9, red, 2)
        
        # Get midpoint of left and right shoulder points.
        mx, my = self.find_midpoint(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
        cv.line(frame, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), yellow, 2)

        # Determine whether the person is self.facing left or right.
        if mx > nose_x:
            self.facing = 'left'
        else:
            self.facing = 'right'
            
        cv.putText(frame, 'facing: ' + self.facing, (10, h - 20), font, 0.9, green, 2)

        # Get tragus
        if self.facing == 'left':
            l_eye_to_ear = self.find_distance(l_eye_x, l_eye_y, l_ear_x, l_ear_y)
            trg_x, trg_y = int(l_ear_x + (l_eye_to_ear / 2)), int(l_ear_y - (l_eye_to_ear / 4))
        else:
            r_eye_to_ear = self.find_distance(r_eye_x, r_eye_y, r_ear_x, r_ear_y)
            trg_x, trg_y = int(r_ear_x - (r_eye_to_ear / 2)), int(r_ear_y - (r_eye_to_ear / 4))

        cv.circle(frame, (trg_x, trg_y), 7, pink, -1)

        # Calculate distance between tragus and midpoint.
        trg_to_mid = self.find_distance(trg_x, trg_y, mx, my)
        neck_ratio = 0.25
        if self.facing == 'left':
            c7_x, c7_y = int(mx + (trg_to_mid * neck_ratio)), int(my - (trg_to_mid * neck_ratio))
        else:
            c7_x, c7_y = int(mx - (trg_to_mid * neck_ratio)), int(my - (trg_to_mid * neck_ratio))
        
        cv.circle(frame, (c7_x, c7_y), 7, pink, -1)
        cv.circle(frame, (mx, my), 7, yellow, -1)
        
        # Draw imaginary horizontal line from midpoint.
        cv.line(frame, (c7_x - 300, c7_y), (c7_x + 300, c7_y), yellow, 2)
        cv.circle(frame, (c7_x - 300, c7_y), 7, yellow, -1)
        cv.circle(frame, (c7_x + 300, c7_y), 7, yellow, -1)
        
        # Connect shoulder to c7 point
        cv.line(frame, (l_shldr_x, l_shldr_y), (c7_x, c7_y), yellow, 2)
        cv.line(frame, (r_shldr_x, r_shldr_y), (c7_x, c7_y), yellow, 2)
        
        # Draw imaginary vertical line from midpoint.
        cv.line(frame, (c7_x, c7_y - 50), (c7_x, c7_y + 50), yellow, 2)
        
        # Calculate angles.
        neck_inclination = self.find_angle(c7_x, c7_y, trg_x, trg_y, self.facing)
        
        # Save results
        self.results.append((int(neck_inclination), datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

        # Draw landmarks.
        cv.circle(frame, (l_shldr_x, l_shldr_y), 7, green, -1)

        # cv.circle(frame, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv.circle(frame, (r_shldr_x, r_shldr_y), 7, red, -1)

        fr.draw_angle_indicator(frame, (c7_x, c7_y), 50, 0, neck_inclination, yellow)

        # Put text, Posture and angle inclination.
        # Text string for display.
        self.angle = str(int(neck_inclination))
        angle_text_string = 'Angle: ' + str(int(neck_inclination))

        # The threshold angles to determine posture condition.
        if neck_inclination >= 48 and neck_inclination <= 90:
            cv.putText(frame, angle_text_string, (10, 30), font, 0.9, green, 2)
            cv.putText(frame, str(int(neck_inclination)), (c7_x + 10, c7_y - 10), font, 0.9, green, 2)

            # Join landmarks.
            cv.line(frame, (c7_x, c7_y), (trg_x, trg_y), green, 4)

        else:
            cv.putText(frame, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv.putText(frame, str(int(neck_inclination)), (c7_x + 10, c7_y - 10), font, 0.9, red, 2)

            # Join landmarks.
            cv.line(frame, (c7_x, c7_y), (trg_x, trg_y), red, 4)

    def run(self, img):
        frame = cv.imread(img)
        fr = Frame(frame)
        frame = fr.frame
        lm, lm_pose = self.get_landmarks(frame)
        if lm:
            keypoints = self.get_keypoints(lm, lm_pose, fr.width, fr.height)
            self.get_angle(fr, frame, keypoints, fr.width, fr.height, fr.font, fr.colors)

        # self.show_landmarks(frame, lm)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        with open('result_image.jpg', 'wb') as file:
            file.write(frame)

        # yield frame
        dt = {"facing": self.facing, "isAligned": self.aligned, "angle":self.angle, "img":base64.b64encode(frame).decode("utf-8")}
        yield json.dumps(dt)
