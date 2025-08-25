
#yolov8_test.py
import cv2
import urllib.request
from ultralytics import YOLO

# 예제 이미지 다운로드
image_url = "https://ultralytics.com/images/bus.jpg"
image_path = "test.jpg"
urllib.request.urlretrieve(image_url, image_path)

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 읽기
image = cv2.imread(image_path)

# 객체 탐지
results = model(image)

# 결과 시각화
annotated_frame = results[0].plot()

# OpenCV 창에 띄우기
cv2.imshow("YOLOv8 Detection", annotated_frame)

print("🔍 ESC 키를 누르면 창이 닫힙니다...")

# ESC(27) 키가 눌릴 때까지 기다림
while True:
    key = cv2.waitKey(100)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
