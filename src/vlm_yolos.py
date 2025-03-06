# yolos_cup_detector.py

import torch
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw

class CupDetector:
    def __init__(self, model_name="hustvl/yolos-tiny", image_path="saved.jpg"):
        """
        모델 이름과 이미지 경로를 지정하여 객체를 초기화합니다.
        """
        self.image_path = image_path
        self.feature_extractor = YolosFeatureExtractor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name)

    def detect_cup(self, display=False):
        image = Image.open(self.image_path)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        # target_sizes는 (height, width) 순서로 전달
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        cup_found = False
        cx, cy = None, None

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()].lower()
            if label_name == "cup" and score > 0.5:
                x_min, y_min, x_max, y_max = box.tolist()
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cup_found = True

                if display:
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=3)
                    draw.text((x_min, y_min), f"cup: {score:0.2f}", fill="green")
                    image.show()
                break

        return cup_found, cx, cy
