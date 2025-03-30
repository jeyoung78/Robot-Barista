import os
import json
import random
import time

from datasets import Dataset
from google import genai
from google.genai.errors import ClientError


# Available actions: pour, grab, place, serve, done
class DataGeneration():
    def __init__(self, dataset_filename = "data_collection.json"):
        self.client = genai.Client(api_key="AIzaSyAbZpHttVawCw_I-K68XQgHPlKQZ4XXSQg")
        self.dataset_filename = dataset_filename
        if os.path.exists(dataset_filename):
            with open(dataset_filename, "r") as f:
                self.distillation_data = json.load(f)
        else:
            self.distillation_data = []

    def generate_caramel_macchiato(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an iced caramel macchiato order with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label =  "1. Place Cup 2. Pour Vanilla Syrup 3. Pour Milk 4. Pour Ice 5. Pour Espresso 6. Pour Espresso 7. Pour Caramel Syrup 8. Serve Beverage 9. Done"
            else:
                prompt = "Produce a user request for an iced caramel macchiato order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Vanilla Syrup 3. Pour Milk 4. Pour Ice 5. Pour Espresso 6. Pour Caramel Syrup 7. Serve Beverage 8. Done"
        else: 
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for a caramel macchiato order with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Vanilla Syrup 3. Pour Milk 4. Pour Espresso 5. Pour Espresso 6. Pour Caramel Syrup 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for a caramel macchiato order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Vanilla Syrup 3. Pour Milk 4. Pour Espresso 5. Pour Caramel Syrup 6. Serve Beverage 7. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt])
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            # "response": ["grab cup", "place cup", "pour espresso", "pour milk", "pour caramel", "serve", "done"]
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
        
    def generate_americano(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an iced americano order with an extra shot of espresso in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Pour Ice 4. Pour Espresso 5. Pour Espresso 6. Pour Espresso 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for an iced americano order in a cafe. Return nothing but the order."
                label =  "1. Place Cup 2. Pour Water 3. Pour Ice 4. Pour Espresso 5. Pour Espresso 6. Serve Beverage 7. Done"
        else: 
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an americano order with an extra shot of espresso in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Pour Espresso 5. Pour Espresso 6. Serve Beverage 7. Done"
            else:
                prompt = "Produce a user request for an americano order in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Pour Espresso 5. Serve Beverage 6. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt])
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
    
    def generate_cafe_mocha(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for an iced cafe mocha order "
                        "with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Chocolate 5. Pour Milk 6. Pour Ice 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for an iced cafe mocha order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Chocolate 4. Pour Milk 5. Pour Ice 6. Serve Beverage 7. Done"
        else:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for a cafe mocha order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Chocolate 5. Pour Milk 6. Serve Beverage 7. Done"
            else:
                prompt = "Produce a user request for a cafe mocha order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Chocolate 4. Pour Milk 5. Serve Beverage 6. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)

    def generate_cafe_latte(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for an iced cafe latte order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Pour Ice 6. Serve Beverage 7. Done"
            else:
                prompt = "Produce a user request for an iced cafe latte order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Milk 4. Pour Ice 5. Serve Beverage 6. Done"
        else:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for a cafe latte order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Serve Beverage 6. Done"
            else:
                prompt = "Produce a user request for a cafe latte order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Milk 4. Serve Beverage 5. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)

    def generate(self):
        recipe_selection = random.randint(1, 4)
        ice = random.randint(1,10) > 7
        extra_espresso = random.randint(1,10) > 7
        
        if recipe_selection == 1:
            self.generate_americano(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection == 2:
            self.generate_cafe_latte(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection == 3:
            self.generate_cafe_mocha(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection == 4:
            self.generate_caramel_macchiato(ice=ice, extra_espresso=extra_espresso)
        else:
            pass

    def data_collection(self):
        for i in range(1, 1000):
            print(f"Processing: {i}...")
            self.generate()
            
dg = DataGeneration()
dg.data_collection()
