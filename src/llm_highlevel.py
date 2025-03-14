import openai
import ast
import re

class llmRecipeGeneration:
    def __init__(self, beverage):
        # OpenAI API 키 설정 (실제 API 키로 교체하세요)
        self.api_key = "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"
        openai.api_key = self.api_key

        self.beverage = beverage
        self.prompt = f"""
        Below are examples of extracting a valid Python list containing only the liquid ingredients and ice for a specific beverage, in the exact order they should be poured.
If the provided beverage name is ambiguous or generic (for example, "some sweet beverage"), please determine a specific beverage that matches the description and use its appropriate recipe.
Skip any ingredients that are not liquid or ice.
If a syrup is needed, include the word "syrup" (for example, if vanilla syrup is required, output "vanilla_syrup").
Do not include any solid ingredients, amounts, measurements, or explanations.
Strictly output only a valid Python list with ingredient names as strings—nothing more than a Python array.

Example:
Request: "Cappuccino"
Answer: ['ice', 'espresso', 'milk']

Example:
Request: "chocolate latte"
Answer: ["chocolate_syrup", "espresso", "milk", "ice"]

Now, based on the request below, output the Python list.
Request: {self.beverage}
Answer:
"""
        # 사용할 모델 이름 (채팅 모델임)
        self.model = "gpt-4o"

    def generate(self):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.prompt}
                ],
                max_tokens=100,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            generated_text = response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        # 후처리: 줄 단위로 분리 후 하나의 문자열로 결합
        generated_text = re.sub(r"```python", "", generated_text)
        generated_text = re.sub(r"```", "", generated_text)
        generated_text = generated_text.strip()

        lines = generated_text.splitlines()
        raw_string = "\n".join(lines).strip()
        if not raw_string.startswith('['):
            print("Output is not a valid Python list.")
            return None
        try:
            my_array = ast.literal_eval(raw_string)
            print("LLM Parsed array:", my_array)
            return my_array
        except Exception as e:
            print(f"Error parsing the generated text as a Python list: {e}")
            return None

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