import cv2
from src.alpr.enhance import enhance_plate

plate = cv2.imread("plate.jpg")

result = enhance_plate(plate)

cv2.imwrite("plate_enhanced.jpg", result)

print("Enhanced plate saved!")