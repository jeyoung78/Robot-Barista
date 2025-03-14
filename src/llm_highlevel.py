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
Below are examples of extracting a valid output with two parts:
1. The first line must contain the beverage name, prefixed with "Beverage Name:".
2. The subsequent line(s) must contain only a valid Python list with liquid ingredients and ice for that beverage, in the exact order they should be poured.
If the provided beverage name is ambiguous or generic (for example, "some salty caffeine"), please determine a specific beverage that satisfies the request and output its proper name in the first line.
Skip any ingredients that are not liquid or ice.
If a syrup is needed, include the word "syrup" (for example, if vanilla syrup is required, output "vanilla_syrup").
Do not include any solid ingredients, amounts, measurements, or explanations.
Strictly output only the two parts as described below.

Example:
Request: "Cappuccino"
Answer:
Beverage Name: Cappuccino
['ice', 'espresso', 'milk']

Example:
Request: "chocolate latte"
Answer:
Beverage Name: chocolate latte
["chocolate_syrup", "espresso", "milk", "ice"]

Now, based on the request below, output in the same format.
Request: {self.beverage}
Answer:
"""
        # 사용할 모델 이름 (채팅 모델)
        self.model = "gpt-4o"

    def generate(self):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.prompt}
                ],
                max_tokens=150,
                temperature=0.9,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            generated_text = response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        # 코드 블록 제거 (예: ```python, ```)
        generated_text = re.sub(r"```python", "", generated_text)
        generated_text = re.sub(r"```", "", generated_text)
        generated_text = generated_text.strip()

        # 응답을 줄 단위로 분리
        lines = generated_text.splitlines()
        if len(lines) < 2:
            print("Output does not have enough lines.")
            return None

        # 첫 번째 줄에서 음료 이름 추출
        beverage_line = lines[0].strip()
        if not beverage_line.lower().startswith("beverage name:"):
            print("The first line does not start with 'Beverage Name:'.")
            return None
        recommended_beverage_name = beverage_line[len("Beverage Name:"):].strip()

        # 나머지 줄들을 합쳐서 리스트 형태의 문자열로 만듦
        list_string = "\n".join(lines[1:]).strip()
        if not list_string.startswith('['):
            print("Ingredients list not found or not valid.")
            return None

        try:
            beverage_ingredients = ast.literal_eval(list_string)
        except Exception as e:
            print(f"Error parsing the ingredients list: {e}")
            return None

        # 필요한 경우, 음료 이름 부분은 이후에 사용하지 않도록 저장 후 삭제 가능함
        return recommended_beverage_name, beverage_ingredients

def main():
    beverage_input = input("Enter your beverage: ")
    recipe_gen = llmRecipeGeneration(beverage_input)
    result = recipe_gen.generate()
    
    if result is not None:
        recommended_beverage_name, beverage_ingredients = result
        print("\nFinal Results:")
        print("Recommended beverage name:", recommended_beverage_name)
        print("Beverage ingredients list:", beverage_ingredients)
    else:
        print("Failed to generate a valid beverage and ingredients list.")

if __name__ == "__main__":
    main()
