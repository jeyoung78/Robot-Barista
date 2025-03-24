import pyttsx3
import time

from image_processing import ImageProcessing, CameraInterface
from control import Communicate
from task_planning import Scoring, llmRecipeGeneration

beverage = 'capuccino'

co = Communicate()
url_save = 'saved.jpg'
ci = CameraInterface(url = url_save)
ip = ImageProcessing('saved.jpg')
llm_gen = llmRecipeGeneration(beverage)

scoring = Scoring()

global_target_x = 500
global_target_y = 480
global_margin = 20

llm = False

def main():
    if llm:
        ingredients = llm_gen.generate()
    else:
        ingredients = scoring.generate_prompt(beverage)
    
    print("Beverage ingredients list:", ingredients)

    while True:
        ci.capture_image()
        found, cx, cy, radius = ip.detect_circle()
        alignment = move(cx, cy)
        if alignment:
            break
    
    # Save current position of the target cup
    co.communicate("save")

    

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)

    for ingredient in ingredients:
        print(ingredient)
        co.prepare(True)
        
        while True:
            ci.capture_image()
            cx, cy = ip.find_ingredient_cup(ingredient)
            alignment = move(cx, cy)
            if alignment:
                break
        
        engine.say("pouring " + ingredient)
        engine.runAndWait()
        co.communicate("pour")        
        time.sleep(25)
    
    print("complete!")
    engine.say("Complete!")
    engine.runAndWait()

def move(cx: int, cy: int, target_x=global_target_x, target_y=global_target_y, margin=global_margin):
    alignment_x = False
    alignment_y = False

    if (target_x - margin) <= cx <= (target_x + margin):
        print("x-axis aligned.")
        alignment_x = True
    elif cx >= (target_x + margin):
        co.move_x(False)
        print("Moving x negative.")
    elif cx < (target_x - margin):
        co.move_x(True)
        print("Moving x positive.")
    
    
    if alignment_x:    
        if (target_y - margin) <= cy <= (target_x + margin):
            print("y-axis aligned.")
            alignment_y = True
        elif cy > (target_x + margin):
            co.move_y(False)
            print("Moving y negative.")
        elif cy < (target_x - margin):
            co.move_y(True)
            print("Moving y positive.")

    if alignment_x and alignment_y:
        return True
    else:
        return False

if __name__ == "__main__":
    main()