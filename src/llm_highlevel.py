import google.generativeai as genai
import PIL.Image
import json
import ast

class llmRecipeGeneration():
    def __init__(self, beverage):
        self.api_key = "AIzaSyAbZpHttVawCw_I-K68XQgHPlKQZ4XXSQg"
        genai.configure(api_key=self.api_key)

        self.beverage = beverage
        self.prompt = f"""
        Provide a valid Python list containing only liquid ingredients and ice for this beverage, to be poured: {self.beverage}. 
        Skip those that are not liquid or ice. 
        If syrup is to go in, add the word syrup. So if vanilla syrup has to go in, for example, one element within an array should "vanilla_syrup"
        Do not include solid ingredients, amounts, measurements, or explanations. 
        Strictly output a valid Python list with ingredient names as strings.
        Nothing more than a Python array.
        Order to ingredients in an order that it needs to be poured.
        """

        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

    def generate(self):
        try:
            response = self.model.generate_content(self.prompt)  
            response.resolve() 

            words = response.text.split()
            print(words)

        except Exception as e:
            print(f"An error occurred: {e}")

        # 1. Filter out the code-fence lines
        filtered_lines = [line for line in words if not line.startswith("```")]

        # 2. Join the remaining lines into a single string
        raw_string = "".join(filtered_lines)

        # 3. Safely parse the string as a Python literal
        my_array = ast.literal_eval(raw_string)

        print(my_array)

        return my_array
    
def main():
        beverage = input("Enter your beverage: ")
        recipe_gen = llmRecipeGeneration(beverage)
        ingredients = recipe_gen.generate()
        
        if ingredients is not None:
            print("Extracted ingredients list:", ingredients)
        else:
            print("Failed to generate a valid ingredients list.")

if __name__ == "__main__":
        main()