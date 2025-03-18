from image_processing import ImageProcessing, CameraInterface
from control import Communicate
import time

class robot_control:
    def __init__(self, url):
        self.url = url
        self.co = Communicate()
        self.ci = CameraInterface(url=self.url)

    def y_alligned(self):
        while True:
            self.ci.capture_image()
            found, cx, cy, radius = ImageProcessing.detect_circle(self.url, display=False)
            
            if not found or cy is None:
                print("cannot detect cup... retry...")
                time.sleep(1)
                continue

            print(f"Detected circle: center=({cx}, {cy})")
            if 450 <= cy <= 500:
                print("Y-axis aligned.")
                break
            elif cy > 500:
                self.co.move_y(False)
                print("Moving y negative.")
            elif cy < 450:
                self.co.move_y(True)
                print("Moving y positive.")
            time.sleep(1)

    def x_alligned(self):
        while True:
            self.ci.capture_image()
            found, cx, cy, radius = ImageProcessing.detect_circle(self.url, display=False)
            if not found or cx is None:
                print("cannot detect cup... retry...")
                time.sleep(1)
                continue

            print(f"Detected circle: center=({cx}, {cy})")
            if 475 <= cx <= 525:
                print("X-axis aligned.")
                break
            elif cx > 525:
                self.co.move_x(False)
                print("Moving x negative.")
            elif cx < 475:
                self.co.move_x(True)
                print("Moving x positive.")
            time.sleep(1)