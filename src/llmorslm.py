import time
import PIL.Image
from llm_highlevel import llmRecipeGeneration
from slm_highlevel import slmRecipeGeneration
import google.generativeai as genai
from checkslm import DrinkCheck

def main():
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

    for ingredient in ingredients:
        print(ingredient)
        ingredient_cleaned = ingredient.replace("_", " ")

        while True:
            user_input = input("Type a word: ").strip()
            if user_input.lower() == target_word.lower():
                print("Proceeding...")
                break
            else:
                pass
        time.sleep(5)
    
    print("complete!")

if __name__ == "__main__":
    main() 