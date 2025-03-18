import pyttsx3
import time

from control import Communicate
from task_planning import slmRecipeGeneration, llmRecipeGeneration, DrinkCheck
from robot_move import robot_control

def main():
    url_save = 'saved.jpg'
    co = Communicate()
    robot = robot_control(url_save)
    robot.y_alligned()
    robot.x_alligned()
    
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