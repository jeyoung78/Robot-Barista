import openai
import ast

class slmRecipeGeneration:
    def __init__(self, beverage):
        # OpenAI API 키 설정 (실제 API 키로 교체하세요)
        self.api_key = "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"
        openai.api_key = self.api_key

        self.beverage = beverage
        self.prompt = f"""
Below are examples of extracting a valid Python list containing only liquid ingredients and ice for a beverage, in the order they need to be poured.
Skip any ingredients that are not liquid or ice.
If syrup is needed, include the word "syrup" (for example, if vanilla syrup is required, output "vanilla_syrup").
Do not include solid ingredients, amounts, measurements, or any explanations.
Strictly output only a valid Python list with ingredient names as strings—nothing more than a Python array.

Example:
Request: "Cappuccino"
Answer: ['ice', 'espresso','milk']

Example:
Request: "chocolate latte"
Answer: ["chocolate_syrup", "espresso", "milk", "ice"]

Now, based on the request below, output the Python list.
Request: {self.beverage}
Answer:
"""

        # 사용할 모델 이름
        self.model = "babbage-002"

    def generate(self):
        try:
            response = openai.Completion.create(
                engine=self.model,
                prompt=self.prompt,
                max_tokens=100,       # 충분한 토큰 수 설정 (필요한 만큼 조절)
                temperature=0.6,      # 결정론적 출력을 위해 0으로 설정
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"]          # 개행 문자가 나오면 생성을 중단
            )
            generated_text = response.choices[0].text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")

# 올바르게 각 줄을 처리하도록 수정
        lines = generated_text.splitlines()
        raw_string = "\n".join(lines).strip()
        if not raw_string.startswith('['):
            print("Output is not a valid Python list.")
            return None
        try:
            my_array = ast.literal_eval(raw_string)
            return my_array
        except Exception as e:
            print(f"Error parsing the generated text as a Python list: {e}")
            return None


def main():
        beverage = input("Enter your beverage: ")
        recipe_gen = slmRecipeGeneration(beverage)
        ingredients = recipe_gen.generate()
        
        if ingredients is not None:
            print("Extracted ingredients list:", ingredients)
        else:
            print("Failed to generate a valid ingredients list.")

if __name__ == "__main__":
        main()
