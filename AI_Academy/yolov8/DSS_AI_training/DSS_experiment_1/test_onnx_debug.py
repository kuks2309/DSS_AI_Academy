import sys
import traceback
from pathlib import Path

print("=== 디버깅 시작 ===")

# 1. 기본 패키지 확인
try:
    import cv2
    print("✓ OpenCV 설치됨:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV 설치 필요:", e)

try:
    import numpy as np
    print("✓ NumPy 설치됨:", np.__version__)
except ImportError as e:
    print("✗ NumPy 설치 필요:", e)

try:
    import onnxruntime as ort
    try:
        version = ort.__version__
    except AttributeError:
        version = "버전 정보 없음 (설치됨)"
    print("✓ ONNX Runtime 설치됨:", version)
except ImportError as e:
    print("✗ ONNX Runtime 설치 필요:", e)

# 2. 모델 파일 확인
model_dir = Path("C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1/weights/onnx_models")
print(f"\n모델 디렉토리: {model_dir}")
print(f"디렉토리 존재: {model_dir.exists()}")

if model_dir.exists():
    onnx_files = list(model_dir.glob("*.onnx"))
    print(f"ONNX 파일들: {onnx_files}")
    
    if onnx_files:
        model_path = onnx_files[0]
        print(f"선택된 모델: {model_path}")
        print(f"모델 파일 크기: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 3. ONNX 모델 로드 테스트
        try:
            session = ort.InferenceSession(str(model_path))
            print("✓ ONNX 모델 로드 성공")
            
            # 모델 입출력 정보
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            print(f"입력 이름: {input_info.name}, 형태: {input_info.shape}")
            print(f"출력 이름: {output_info.name}, 형태: {output_info.shape}")
            
        except Exception as e:
            print("✗ ONNX 모델 로드 실패:", e)
            traceback.print_exc()

# 4. 웹캠 테스트
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ 웹캠 접근 가능")
        ret, frame = cap.read()
        if ret:
            print(f"✓ 프레임 읽기 성공, 크기: {frame.shape}")
        else:
            print("✗ 프레임 읽기 실패")
        cap.release()
    else:
        print("✗ 웹캠 접근 불가")
except Exception as e:
    print("✗ 웹캠 테스트 실패:", e)

print("\n=== 디버깅 완료 ===")
