# Interprets the visual input from the two cameras
from pypylon import pylon
import cv2

# Access Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbingMax(1)
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
if grabResult.GrabSucceeded():
    img = grabResult.Array
    cv2.imwrite('basler_image.jpg', img)  # Requires OpenCV for saving
camera.Close()