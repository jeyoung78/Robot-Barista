# Goal now: move robot to grab a cup and move to a different location
# Monday: Implement robot code and Communicate class so that robot can be controlled from python script
import pyttsx3
import time
import PIL.Image

from google import genai
from image_processing import ImageProcessing, CameraInterface
from control import Communicate
from llm_highlevel import llmRecipeGeneration
from slm_highlevel import slmRecipeGeneration
from checkslm import DrinkCheck
from circle import detect_circle

def main():
    url_save = 'saved.jpg'
    ci = CameraInterface(url = url_save)
    co = Communicate()

    while True:
        ci.capture_iamge()
        found, cx, cy, radius = detect_circle(url_save, display=False)
        if not found or cy is None:
            print("cannot detect cup... retry...")
            time.sleep(1)
            continue

        print(f"Detected circle: center=({cx}, {cy})")
        if 450 <= cy <= 500:
            print("Y-axis aligned.")
            break
        elif cy > 500:
            co.move_y(False)
            print("Moving y negative.")
        elif cy < 450:
            co.move_y(True)
            print("Moving y positive.")
        time.sleep(1)

    while True:
        ci.capture_iamge()
        found, cx, cy, radius = detect_circle(url_save, display=False)
        if not found or cx is None:
            print("cannot detect cup... retry...")
            time.sleep(1)
            continue

        print(f"Detected circle: center=({cx}, {cy})")
        if 475 <= cx <= 525:
            print("X-axis aligned.")
            break
        elif cx > 525:
            co.move_x(False)
            print("Moving x negative.")
        elif cx < 475:
            co.move_x(True)
            print("Moving x positive.")
        time.sleep(1)

    user_request = input("Enter your request: ")
    checker = DrinkCheck(user_request)
    beverage = checker.generate()

    if beverage.lower() == "none":
        rgllm = llmRecipeGeneration(user_request)
        ingredients = rgllm.generate()
    
    else:
        rgslm = slmRecipeGeneration(beverage)
        ingredients = rgslm.generate()

    target_word = "proceed"

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    
    for ingredient in ingredients:
        print(ingredient)
        co.prepare(True)
        ingredient_cleaned = ingredient.replace("_", " ")
        engine.say("pour " + ingredient_cleaned)
        engine.runAndWait()
        
        
        while True:
            user_input = input("Type a word: ").strip()
            if user_input.lower() == target_word.lower():
                print("Proceeding...")
                break
            else:
                print("Please type 'proceed' to continue.")

        co.communicate("pour")
        time.sleep(20)
    
    print("complete!")
    engine.say("Complete!")
    engine.runAndWait()

if __name__ == "__main__":
    main()