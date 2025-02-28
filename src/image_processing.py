# Communicates with the Doosan robot that is listening within the while loop
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
# from transformers import DPTImageProcessor, DPTForDepthEstimation

# image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
class CameraInterface:    
    def __init__(self, url) : # -> return 값
        self.cam2 = 0
        self.cap = cv2.VideoCapture(self.cam2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        self.url = url

    def capture_iamge(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        ret, img = self.cap.read()
        img = cv2.resize(img, (1280, 720))
        cv2.imwrite(self.url, img)

        return

    def robot_camera_stream(self):
        # cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)

        # show
        while True:
            ret, img = self.cap.read() #ret : 성공하면 true, 실패하면 false
            if not ret :
                print("can't read cap")
                break
            img = cv2.resize(img, (1280, 720))
            cv2.imshow("video", img)
            k = cv2.waitKey(1)
            if k == ord('s') :
                cv2.imwrite(time.strftime("%H%M%S")+'.jpg', img)
            elif k == ord('q') :
                break
        
        cv2.destroyAllWindows()

class ImageProcessing:
    def __init__(self, url='saved.jpg'):
        self.lower_red1 = np.array([0, 120, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 150])
        self.upper_red2 = np.array([180, 255, 255])

        self.url = url

    def detect_red_dot(self):
        # Read the image
        image = cv2.imread(self.url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of red areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        output = image.copy()
        # print(len(contours))
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 100 < area:  # Adjust for small dots, removing large areas
                # x, y, w, h = cv2.boundingRect(contour)

                # Ensure it's roughly circular
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                # print(cx, cy)
                # radius = int(radius)
                # if 0.8 < (w / h) < 1.2:  # Ensure width and height are similar (circular)
                #     cv2.circle(output, (int(cx), int(cy)), radius,(0, 128, 0), thickness=20)
        
        # return (cx, cy)

        # Show images
        '''
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Detected Red Dot")
        plt.imshow(output)
        plt.axis("off")

        plt.show()
        '''
        return cx, cy        

    def find_cup(self):
        pass

if __name__ == '__main__' :
    # grab()
    image = CameraInterface('saved.jpg')
    image.capture_iamge()
    ip = ImageProcessing('saved.jpg')
    cx, cy = ip.detect_red_dot()
    print(cx, cy)