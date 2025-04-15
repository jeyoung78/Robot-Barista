import json
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGPromptGenerator:
    def __init__(self, recipe_file: str, embedder_model: str = "all-MiniLM-L6-v2"):
        self.recipe_file = recipe_file
        self.embedder = SentenceTransformer(embedder_model)
        self.library = self._load_recipes(recipe_file)

    def _load_recipes(self, recipe_file: str):
        with open(recipe_file, "r", encoding="utf-8") as f:
            recipe_data = json.load(f)
        library = []
        for entry in recipe_data:
            beverage_name = entry.get("prompt", "")
            recipe_steps = entry.get("response", "")
            # Combine beverage name and recipe text for a more robust embedding
            combined_text = f"{beverage_name}. {recipe_steps}"
            embedding = self.embed_text(combined_text)
            library.append({
                "beverage": beverage_name,
                "recipe": recipe_steps,
                "embedding": embedding
            })
        return library

    def embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def retrieve_best_recipe(self, user_query: str, top_k: int = 1):
        query_emb = self.embed_text(user_query)
        scores = []
        for item in self.library:
            item_emb = item["embedding"]
            similarity = np.dot(query_emb, item_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(item_emb))
            scores.append((similarity, item))
        
        # Sort recipes by descending similarity
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = [s[1] for s in scores[:top_k]]
        # print(top_matches)
        return top_matches

    def build_prompt(self, user_query: str, recipe_entry: dict) -> str:
        beverage = recipe_entry["beverage"]
        original_recipe = recipe_entry["recipe"]
        
        prompt = f"""
            You are a coffee robot task planner. You have a recipe for a beverage called '{beverage}'.

            Original Recipe Steps:
            {original_recipe}

            User Request: "{user_query}"

            Important instructions:
            1. Use only the following action verbs in the plan: "place", "pour", "serve", and "done".
            2. All additional ingredient modifications must be added only in syrup form. For example:
            - If vanilla is requested, output it as "vanilla syrup".
            - If watermelon is requested, output it as "watermelon syrup".
            3. If the user asks for additions (e.g., "extra shot of espresso"), adjust the recipe by repeating the "pour espresso" step as needed.
            4. Preserve the overall structure of the original recipe as much as possible.
            5. Output a step-by-step list of actions, one per line, without any extra commentary.

            Example:
            If the original recipe steps are:
            "1. place cup
            2. pour espresso
            3. serve
            4. done"

            And the user request is for an extra shot of espresso, the output should be:
            "1. place cup
            2. pour espresso
            3. pour espresso
            4. serve
            5. done"

            Now, generate the final plan.
        """
        return prompt.strip()

    def generate_rag_prompt(self, user_query: str) -> str:
        best_recipe = self.retrieve_best_recipe(user_query, top_k=1)[0]
        prompt = self.build_prompt(user_query, best_recipe)
        return prompt


if __name__ == "__main__":
    generator = RAGPromptGenerator(recipe_file="label.json")
    sample_query = "I want a Thai Tea Latte with an extra shot of espresso"
    final_prompt = generator.generate_rag_prompt(sample_query)
    print("=== Generated RAG Prompt ===")
    print(final_prompt)
