from google import genai
from google.genai.types import HttpOptions, Part
import sys
import os 
import PIL.Image

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_path)

from image_processing import ImageProcessing, CameraInterface
from control import Communicate

client = genai.Client(api_key="AIzaSyDY0QuJXHlODt2bJCTCrY2NbhGxw703h6U")


words = ['', '']

url_save = 'saved.jpg'
ip = ImageProcessing(url = url_save)
ci = CameraInterface(url = url_save)
co = Communicate()

print(words)

prompt = prompt = """
The image is taken by a camera mounted on the robot end effector, looking downward. 
The center of the image represents the end effector’s current position. 
If the cup appears left relative to the center the image, the robot should move left. 
If the cup appears right relative to the center the image, the robot should move right. 
If the cup appears top relative to the center the image, the robot should move forwards. 
If the cup appears bottom relative to the center the image, the robot should move backwards. 
If the cup is already centered, the robot should not move, meaning it should stop.
Which way the robot should move? 
first word should be one of right, left, stop, and the second word should be one of forward, backward, and stop.
Make sure to stop if the robot is aligned with the cup.
"""

while words[1] != 'stop':
    ci.capture_iamge()
    print('captured image!')
    image = PIL.Image.open('saved.jpg')

    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=[prompt, image])
    
    words = response.text.split()

    print(words)

    if words[1] == 'backward':
        co.move_y(True)
        print('move y pos')

    if words[1] == 'forward':
        co.move_y(False)
        print('move y neg')
        

while words[0] != 'stop':
    ci.capture_iamge()
    print('captured image!')
    image = PIL.Image.open('saved.jpg')

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image])
    
    words = response.text.split()

    print(words)

    if words[0] == 'right':
        co.move_x(False)
        print('move x neg')

    if words[0] == 'left':
        co.move_x(True)
        print('move x pos')
    
co.communicate("alignment_complete")

print(response.text)
