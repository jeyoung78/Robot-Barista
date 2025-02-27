from google import genai
from google.genai.types import HttpOptions, Part
import PIL.Image

client = genai.Client(api_key="AIzaSyDY0QuJXHlODt2bJCTCrY2NbhGxw703h6U")

image = PIL.Image.open('saved.jpg')

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What is paper cup's relative location from the center of the image? Answer in two words. first word should be one of right, left, center, and the second word should be one of up, down, and center", image])


print(response.text)