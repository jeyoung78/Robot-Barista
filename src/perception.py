import cv2
from pypylon import pylon

class DualCameraCapture:
    def __init__(self):
        self.webcam = None
        self.basler_cam = None

    def capture_webcam(self, output_path='webcam_image.jpg', camera_index=0):
        """Capture image from standard webcam using OpenCV"""
        try:
            self.webcam = cv2.VideoCapture(camera_index)
            if not self.webcam.isOpened():
                raise RuntimeError("Could not open webcam")

            # Allow camera to warm up
            for _ in range(5):
                self.webcam.read()

            ret, frame = self.webcam.read()
            if ret:
                cv2.imwrite(output_path, frame)
                return True
            return False
            
        finally:
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()

    def capture_basler(self, output_path='basler_image.jpg'):
        """Capture image from Basler industrial camera using pypylon"""
        try:
            # Create camera instance
            self.basler_cam = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.basler_cam.Open()
            
            # Single shot mode
            self.basler_cam.StartGrabbingMax(1)
            grabResult = self.basler_cam.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                img = grabResult.Array
                cv2.imwrite(output_path, img)
                return True
            return False
            
        finally:
            if self.basler_cam and self.basler_cam.IsOpen():
                self.basler_cam.Close()

# Example usage
if __name__ == '__main__':
    cam = DualCameraCapture()
    
    # Capture from webcam
    if cam.capture_webcam(output_path='my_webcam.jpg'):
        print("Webcam capture successful")
    
    # Capture from Basler camera
    if cam.capture_basler(output_path='my_basler.jpg'):
        print("Basler capture successful")