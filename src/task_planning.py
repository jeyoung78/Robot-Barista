import openai
import ast
import re

class DrinkCheck:
    def __init__(self, beverage):
        self.api_key = "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"
        openai.api_key = self.api_key

        self.beverage = beverage
        self.prompt = (
            "Below are examples of extracting a beverage name from a request. "
            "If the beverage is not clearly defined, output \"none\".\n\n"
            "Request: make me a Macchiato\n"
            "Answer: Macchiato\n\n"
            "Request: give me water\n"
            "Answer: water\n\n"
            "Request: give me a cafe latte\n"
            "Answer: cafe latte\n\n"
            "Request: some sweet beverage\n"
            "Answer: none\n\n"
            f"Request: {self.beverage}\n"
            "Answer:"
        )
        # 사용할 모델
        self.model = "babbage-002"

    def generate(self):
        try:
            response = openai.Completion.create(
                engine=self.model,
                prompt=self.prompt,
                max_tokens=10,      # 음료 이름 정도로 충분한 길이
                temperature=0.5,    # 결정론적 출력
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"]
            )
            answer = response.choices[0].text.strip()
            return answer
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

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

# Example usage case: slmRecipeGeneration
'''
def main():
        beverage = input("Enter your beverage: ")
        recipe_gen = slmRecipeGeneration(beverage)
        ingredients = recipe_gen.generate()
        
        if ingredients is not None:
            print("Extracted ingredients list:", ingredients)
        else:
            print("Failed to generate a valid ingredients list.")
'''

# Example usage: llmRecipeGeneration
'''
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
'''

if __name__ == "__main__":
    main()
