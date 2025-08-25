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

# 자동차 클래스 ID들 (COCO 데이터셋 기준)
# 2: car, 3: motorcycle, 5: bus, 7: truck
car_classes = [2, 3, 5, 7]  # 차량 관련 클래스들

# 자동차 개수 세기
car_count = 0
detected_objects = []

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # 자동차 관련 클래스이고 신뢰도가 0.5 이상인 경우
            if class_id in car_classes and confidence >= 0.5:
                car_count += 1
                class_name = model.names[class_id]
                detected_objects.append(f"{class_name} (신뢰도: {confidence:.2f})")

# 결과 출력
print(f"🚗 탐지된 차량 수: {car_count}대")
if detected_objects:
    print("📋 탐지된 차량 목록:")
    for i, obj in enumerate(detected_objects, 1):
        print(f"  {i}. {obj}")

# 결과 시각화
annotated_frame = results[0].plot()

# 이미지에 차량 개수 텍스트 추가
text = f"Cars: {car_count}"
cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA)

# OpenCV 창에 띄우기
cv2.imshow("YOLOv8 Car Detection", annotated_frame)
print("🔍 ESC 키를 누르면 창이 닫힙니다...")

# ESC(27) 키가 눌릴 때까지 기다림
while True:
    key = cv2.waitKey(100)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()