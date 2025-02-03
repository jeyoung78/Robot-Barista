# Communicates with the Doosan robot that is listening within the while loop
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation

image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)

class ImageProcessing:
    def __init__(self, url):
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
                x, y, w, h = cv2.boundingRect(contour)

                # Ensure it's roughly circular
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                # print(cx, cy)
                radius = int(radius)
                if 0.8 < (w / h) < 1.2:  # Ensure width and height are similar (circular)
                    cv2.circle(output, (int(cx), int(cy)), radius, (255, 0, 0), 2)
        
        return (cx, cy)

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

    def find_cup(self):
        pass

