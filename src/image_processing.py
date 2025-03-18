# Communicates with the Doosan robot that is listening within the while loop
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class CameraInterface:    
    def __init__(self, url):
        self.cam2 = 0
        self.cap = cv2.VideoCapture(self.cam2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        self.url = url

    def capture_image(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        ret, img = self.cap.read()
        img = cv2.resize(img, (1280, 720))
        cv2.imwrite(self.url, img)

        return

# Returns image coordinates of target objects
class ImageProcessing:
    def __init__(self, url='saved.jpg'):
        self.lower_red1 = np.array([0, 120, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.upper_white = np.array([255, 255, 255])
        self.lower_white = np.array([0, 0, 200])

        self.url = url

    def detect_circle(image_path, display=False):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Can't load this image: {image_path}")
            return False, None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Hough Circle Transform 적용
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = circles[0][0]
            cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
            if display:
                cv2.circle(image, (cx, cy), 2, (0, 0, 255), 3)       
                cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)  
                cv2.imshow("Detected Circle", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return True, cx, cy, radius
        else:
            return False, None, None, None

    def find_ingredient_cup(self, image_path="saved.jpg", target_ingredient="Water"):
        ci = CameraInterface('saved.jpg')
        found = False
        count = 0
        target_x = None

        while not found and count < 10:
            # ci.capture_image()
            text_list = []
            image = cv2.imread(image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Optional: Apply threshold
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            cv2.imwrite('processed.jpg', thresh)
            data = pytesseract.image_to_data(thresh, config='--psm 6', output_type=pytesseract.Output.DICT)
            print(data['text'])

            for i in range(len(data['text'])):
                if data['text'][i].lower() == target_ingredient.lower():
                    x = data['left'][i]
                    w = data['width'][i]
                    centre_x = x + w // 2
                    target_x = centre_x
                    found = True

            count = count + 1

        print(target_x)
        
        if target_x == None:
            return 
        
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Hough Circle Transform 적용
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=120,
            maxRadius=150
        )
        
        print(circles)
        display = True

        if circles is not None:
            circles = circles[0].astype(np.float32)
            best_circle = min(circles, key=lambda circle: abs(circle[0] - target_x))
            cx, cy, radius = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            print("Closest circle center (x, y):", cx, cy, "with radius:", radius)
            
            if display:
                # Draw the circle center.
                cv2.circle(image, (cx, cy), 2, (0, 0, 255), 3)
                # Draw the circle outline.
                cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
                cv2.imshow("Detected Circle", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return cx, cy


if __name__ == '__main__' :
    ip = ImageProcessing('saved.jpg')
    data = ip.find_ingredient_cup(target_ingredient="syrup")
