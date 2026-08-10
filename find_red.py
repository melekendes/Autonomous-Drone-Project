import cv2
import numpy as np
img = cv2.imread("test_cam.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
mask = mask1 | mask2
coords = np.argwhere(mask > 0)
if len(coords) > 0:
    cy = int(np.mean(coords[:, 0]))
    cx = int(np.mean(coords[:, 1]))
    print(f"Red sphere found at: cx={cx}, cy={cy}")
else:
    print("No red sphere found")
