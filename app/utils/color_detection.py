import cv2 as cv
import numpy as np

class ColorDetection:
    def get_objects(self, frame):
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        
        # Convert the frame from BGR to HSV color space
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # lower boundary GREEN color range values
        lower = np.array([35, 100, 60])
        upper = np.array([70, 255, 255])
        
        mask = cv.inRange(hsv, lower, upper)
        
        # Get only masked color
        # result = cv.bitwise_and(frame, frame, mask=mask)
        
        # Morphological operations
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (5, 5))
        
        # Find contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        return contours
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y