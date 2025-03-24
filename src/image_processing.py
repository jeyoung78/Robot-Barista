import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np

# Access camera and save the image in jpg format
class CameraInterface:    
    def __init__(self, url='saved.jpg'):
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

    def return_image(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        ret, img = self.cap.read()
        img = cv2.resize(img, (1280, 720))
        return img

# Returns image coordinates of target objects from a saved image
class ImageProcessing:
    def __init__(self, url='saved.jpg'):
        self.lower_red1 = np.array([0, 120, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.upper_white = np.array([255, 255, 255])
        self.lower_white = np.array([0, 0, 200])

        self.url = url
        self.ci = CameraInterface()

    def detect_circle(image_path='saved.jpg', display=False):
        image = cv2.imread('saved.jpg')
        
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
    
    def find_ingredient_cup(self, target_word="Water"):
        count = 0
        target_x = None

        # Initialize the docTR OCR model (pretrained weights are downloaded automatically)
        model = ocr_predictor(pretrained=True)
        # Attempt up to 10 times to capture an image and find the ingredient
        while target_x == None:
            self.ci.capture_image()
            doc = DocumentFile.from_images(self.url)
            result = model(doc)
            ocr_data = result.export()

            for page in ocr_data["pages"]:
                # The dimensions are given as (height, width)
                page_height, page_width = page["dimensions"]
                for block in page["blocks"]:
                    for line in block["lines"]:
                        for word in line["words"]:
                            if word["value"].lower() == target_word.lower():
                                # Retrieve the normalized bounding box:
                                # For example: ((0.6084, 0.6476), (0.7021, 0.6944))
                                norm_box = word["geometry"]
                                top_left, bottom_right = norm_box

                                # Convert to pixel coordinates:
                                x1 = top_left[0] * page_width
                                y1 = top_left[1] * page_height
                                x2 = bottom_right[0] * page_width
                                y2 = bottom_right[1] * page_height

                                target_x = (x1 + x2)/2

                                print(f"Word: '{word['value']}'")
                                print(f"Normalized Coordinates: {norm_box}")
                                print(f"Pixel Coordinates: Top-Left: ({x1:.0f}, {y1:.0f}), Bottom-Right: ({x2:.0f}, {y2:.0f})")
                                found = True
            # result.show()
            print("x: ", target_x)
            count = count + 1

        # Use the grayscale image for circle detection
        image = cv2.imread(self.url)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=100,
            maxRadius=150
        )
        
        # print("Detected circles:", circles)
        
        display = True

        if circles is not None:
            # Convert circles to float to avoid overflow issues
            circles = circles[0].astype(np.float32)
            # Find the circle with x-coordinate closest to the target ingredient's center
            best_circle = min(circles, key=lambda circle: abs(circle[0] - target_x))
            cx, cy, radius = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            # print("Closest circle center (x, y):", cx, cy, "with radius:", radius)
            
            if display:
                # Draw the detected circle and its center on the original image
                cv2.circle(image, (cx, cy), 2, (0, 0, 255), 3)
                cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
                cv2.imwrite(self.url, image)
                # cv2.imshow("Detected Circle", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            
            return cx, cy

if __name__ == '__main__' :
    ip = ImageProcessing('saved.jpg')
    data = ip.find_ingredient_cup('water')
    
