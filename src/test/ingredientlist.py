import cv2
import pytesseract

image = cv2.imread("D:/newthing/Robot-Barista/src/test/screenshot1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, lang="eng")

robot_position = {"x": 100, "y": 200}
data = {"detected_text": text, "robot_position": robot_position}

print(data)