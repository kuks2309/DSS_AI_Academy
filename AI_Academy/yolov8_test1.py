#yolov8_test1.py
import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 경로 (현재 폴더에 있는 car.jpg)
image_path = "car.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 정상적으로 읽혔는지 확인
if image is None:
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
    exit()

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
