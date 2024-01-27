import base64
import json
import time
from datetime import datetime
import cv2 as cv
import math
from utils.theme import Colors
from utils.camera import Camera, Frame
from utils.color_detection import ColorDetection, Point

class ClarkAngle:
    def __init__(self):
        self.interpretation = ''
        self.angle = 0
        self.isValid = True
        self.results = []
    
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
        mag_BA = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        mag_BC = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)

        # Cosinus theta
        cos_theta = dot_product / (mag_BA * mag_BC)
        # Ensure the argument is within the valid range [-1, 1]
        cos_theta = max(min(cos_theta, 1), -1)

        # Sudut dalam radian
        theta_rad = math.acos(cos_theta)
        degree = math.degrees(theta_rad)

        return degree 
    
    def interpret(self, angle):
        if (angle < 31):
            return 'flat foot'
        elif (angle >= 31 and angle <= 45):
            return 'normal'
        else:
            return 'cavus foot'
    
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

            cv.rectangle(frame, (point_x, point_y), (point_x+w, point_y+h), colors.green, 2)
            fr.circle(frame, (point.x, point.y))
            fr.put_text(frame, str(i), (point.x + 10, point.y), color=colors.white)
        #endfor

        fr.meta_info(frame, 'Keypoints req: 3', 'bottom_left', (0, -100), fontSize=1.2)
        fr.meta_info(frame, 'Keypoints count: ' + str(len(keypoints)), 'bottom_left', (0, -50), fontSize=1.2)

        # Connect keypoints with lines
        if len(keypoints) == 3:
            self.isValid= True
            fr.meta_info(frame, 'Keypoints status: valid', 'bottom_left', fontSize=1.2, color=colors.green)

            medCap = Point(keypoints[0].x, keypoints[0].y)
            arkus = Point(keypoints[1].x, keypoints[1].y)
            tumit = Point(keypoints[2].x, keypoints[2].y)

            # Draw line
            fr.line(frame, (medCap.x, medCap.y), (arkus.x, arkus.y), color=colors.yellow)
            fr.line(frame, (medCap.x, medCap.y), (tumit.x, tumit.y), color=colors.yellow)

            clark_angle = self.calc_angle_mid(arkus, medCap, tumit)

            # Object information
            fr.put_text(frame, str(int(clark_angle)), (arkus.x + 10, arkus.y + 50), fontSize=1)
            self.angle= int(clark_angle)
            fr.meta_info(frame, 'Clarke\'s angle: ' + str(int(clark_angle)))
            fr.meta_info(frame, 'Condition: ' + self.interpret(clark_angle), 'top_left', (0, 50), fontSize=1.5)
            self.interpretation= self.interpret(clark_angle)
            # Save results
            self.results.append((int(clark_angle), datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

        else:
            fr.meta_info(frame, 'Keypoints status: invalid', 'bottom_left', fontSize=1.2, color=colors.red)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        with open('result_image.jpg', 'wb') as file:
            file.write(frame)

        dt = {
            "isValid": self.isValid,
            "angle": self.angle,
            "interpretation": self.interpretation,
            "reqPoint": 3,
            "img": base64.b64encode(frame).decode("utf-8")
        }
        yield json.dumps(dt)
