from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# flan-t5-small 모델 초기화
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class drink_check():
    def __init__(self, request):
        self.request = request
        self.prompt = f"""
        Provide a valid recipe for the following request: {self.request}.
        If the request contains a beverage, output the beverage recipe in English.
        If no beverage is present in the request, output "none".
        """
    def generate(self):
        input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=20)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    
if __name__ == "__main__":
    user_request = input("Enter your request: ")
    checker = drink_check(user_request)
    beverage = checker.generate()
    print("Extracted beverage:", beverage)

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