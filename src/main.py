import pyttsx3
import time

from image_processing import ImageProcessing, CameraInterface
from control import Communicate
from task_planning import slmRecipeGeneration, llmRecipeGeneration, DrinkCheck

def main():
    url_save = 'saved.jpg'
    ci = CameraInterface(url = url_save)
    co = Communicate()

    while True:
        ci.capture_iamge()
        found, cx, cy, radius = ImageProcessing.detect_circle(url_save, display=False)
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
        found, cx, cy, radius = ImageProcessing.detect_circle(url_save, display=False)
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
        satisfied = False
        current_request = user_request  
        while not satisfied:
            rgllm = llmRecipeGeneration(current_request)
            recipe_result = rgllm.generate()
            if recipe_result is None:
                print("Failed to generate beverage. Please try again.")
                current_request = input("Enter a new beverage request: ")
                continue

            beverage_name, ingredients = recipe_result
            
            # ingredients가 None인 경우 재생성을 시도
            if ingredients is None:
                print("Failed to generate ingredients for the beverage. Let's try again.")
                current_request = input("Enter a new beverage request: ")
                continue

            print("Recommended beverage name:", beverage_name)
            user_choice = input("Are you satisfied with this beverage name? (yes/no): ")
            if user_choice.lower() in ['yes', 'y']:
                satisfied = True
            else:
                print("Let's try generating a new beverage name.")
                current_request = input("Enter a new beverage description: ")
        
    else:
        rgslm = slmRecipeGeneration(beverage)
        ingredients = rgslm.generate()
        # ingredients가 None이면 새로운 음료 입력을 받아 다시 생성
        while ingredients is None:
            print("Failed to generate ingredients for the beverage. Please try again.")
            beverage = input("Enter a new beverage: ")
            rgslm = slmRecipeGeneration(beverage)
            ingredients = rgslm.generate()

    print("Beverage ingredients list:", ingredients)

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
            user_input = input("After adding the ingredients, type 'proceed' : ").strip()
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