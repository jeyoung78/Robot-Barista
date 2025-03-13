import openai

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
                temperature=0.0,    # 결정론적 출력
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

'''
if __name__ == "__main__":
    user_request = input("Enter your request: ")
    checker = DrinkCheck(user_request)
    beverage = checker.generate()
    print("Extracted beverage:", beverage)
'''

"""
# 내부 SLM을 사용해 레시피를 생성하는 클래스 (예시)
class RecipeGeneration:
    def __init__(self, beverage):
        self.beverage = beverage

    def generate(self):
        # 실제 레시피 생성 로직 구현. 여기서는 간단한 예시로 반환.
        return f"{self.beverage} 레시피: [재료1, 재료2, 재료3]"

# 외부 LLM API 호출 예시 함수
def call_llm_api(request):
    # 실제 외부 LLM API 호출 로직 구현 필요
    return "외부 LLM 결과: [레시피 생성 내용]"

def process_request(request):
    beverage = extract_beverage_from_request(request)
    if beverage:
        # 음료가 감지되면 내부 SLM(RecipeGeneration) 사용
        recipe = RecipeGeneration(beverage).generate()
        return recipe
    else:
        print("no")
        recipe = call_llm_api(request)
        return recipe"
"""