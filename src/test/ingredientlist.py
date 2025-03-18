'''
import cv2
import pytesseract

image = cv2.imread("D:/newthing/Robot-Barista/src/test/screenshot1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, lang="eng")

robot_position = {"x": 100, "y": 200}
data = {"detected_text": text, "robot_position": robot_position}

print(data)
'''

'''
import cv2
import time
import pytesseract

from image_processing import CameraInterface

def main():
    url_save = 'saved.jpg'
    ci = CameraInterface(url=url_save)
    
    while True:
        ci.capture_iamge()
        image = cv2.imread(url_save)
        if image is None:
            print("cannot load image... retry...")
            time.sleep(1)
            continue

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        found_text = False

        for i in range(n_boxes):
            try:
                conf = int(data['conf'][i])
            except ValueError:
                continue

            if conf > 60:
                found_text = True
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                # 빨간색 사각형 그리기
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # 중심 좌표 계산
                center_x = x + w // 2
                center_y = y + h // 2
                # 중심에 초록색 원 표시
                cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)
                print(f"Text box {i} center: ({center_x}, {center_y})")

        if found_text:
            cv2.imshow("Detected Text", image)
            cv2.waitKey(0)
            break
        else:
            print("no text detected... retry...")
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
import cv2
import pytesseract

def process_image(image_path):
    # 이미지 파일 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    # pytesseract를 사용하여 이미지 내 텍스트 데이터 추출
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        try:
            conf = int(data['conf'][i])
        except ValueError:
            continue

        if conf > 60:
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            # 빨간색 네모(테두리 두께 2)를 그립니다.
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # 중심 좌표 계산 (정수로 변환)
            center_x = x + w // 2
            center_y = y + h // 2
            # 중심에 초록색 원 표시
            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
            print(f"텍스트 박스 {i}: 중심 좌표 = ({center_x}, {center_y})")
    
    cv2.imshow("Detected Text with Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "D:/newthing/Robot-Barista/src/test/screenshot.png" 
    process_image(image_path)

