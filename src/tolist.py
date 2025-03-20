from llm_scoring import generate_prompt
import re

def extract_last_line(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1]


def parse_robot_line(text):
    if "Robot:" in text:
        text = text.split("Robot:")[-1].strip()
    parts = re.split(r'\d+\.\s*', text)
    ingredients = []
    for part in parts:
        if not part.strip():
            continue
        cleaned = part.strip().rstrip('.').strip()
        if cleaned:
            ingredients.append(cleaned)
    
    return ingredients

def parse_robot_line_until_done(text):
    if "Robot:" in text:
        text = text.split("Robot:", 1)[-1].strip()
    parts = re.split(r'\d+\.\s*', text)
    ingredients = []
    for part in parts:
        cleaned = part.strip().rstrip('.').strip()
        if not cleaned:
            continue
        if cleaned.lower() == 'done':
            break
        ingredients.append(cleaned)
    
    return ingredients

def main():
    beverage = "americano"
    score_threshold = -25.0
    final_prompt = generate_prompt(beverage, score_threshold=score_threshold, verbose=True)
    
    last_line = extract_last_line(final_prompt)
    print("\n last line : ",last_line)
    
    ingredients = parse_robot_line(last_line)
    print("\n ingredients : ",ingredients)
    
    example_prompt = "Robot:  1. caramel syrup. 2. vanilla syrup. 3. water. 4. ice. 5. milk. 6. done. 7. espresso."
    example_last_line = extract_last_line(example_prompt)
    print("\n example last line : ",example_last_line)
    ingredients_until_done = parse_robot_line_until_done(example_last_line)
    print("\n ingredients until done : ",ingredients_until_done)


if __name__ == "__main__":
    main()