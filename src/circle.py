# circle.py

import cv2
import numpy as np

def detect_circle(image_path, display=False):

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 파일을 불러올 수 없습니다: {image_path}")
        return False, None, None, None

    # 그레이스케일 변환 및 가우시안 블러 적용
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circle Transform 적용
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 첫 번째 검출된 원 사용
        circle = circles[0][0]
        cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
        if display:
            # 검출된 원의 중심과 외곽선을 이미지에 표시
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), 3)       # 중심 표시 (빨간 점)
            cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)    # 원 둘레 표시 (초록 선)
            cv2.imshow("Detected Circle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return True, cx, cy, radius
    else:
        return False, None, None, None
