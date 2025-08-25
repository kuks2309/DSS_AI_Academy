#!/usr/bin/env python3

print("=== ONNX Runtime 테스트 ===")

try:
    import onnxruntime as ort
    print("✓ onnxruntime import 성공")
    
    # InferenceSession 확인
    from onnxruntime import InferenceSession
    print("✓ InferenceSession import 성공")
    
    # 사용 가능한 프로바이더 확인
    available_providers = ort.get_available_providers()
    print(f"사용 가능한 프로바이더: {available_providers}")
    
    # GPU 지원 확인
    if 'CUDAExecutionProvider' in available_providers:
        print("✓ CUDA GPU 지원 가능")
    else:
        print("○ CPU만 사용 가능")
    
except ImportError as e:
    print(f"✗ Import 실패: {e}")
    exit(1)

print("\n=== 모델 로드 테스트 ===")

try:
    model_path = "C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1/weights/onnx_models/best.onnx"
    
    # GPU 우선, CPU 백업 프로바이더
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = InferenceSession(model_path, providers=providers)
    print("✓ 모델 로드 성공")
    print(f"실제 사용 프로바이더: {session.get_providers()}")
    
    # 입출력 정보
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"입력: {input_info.name}, 형태: {input_info.shape}")
    print(f"출력: {output_info.name}, 형태: {output_info.shape}")
    
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== 이미지 파일 확인 ===")

from pathlib import Path
import os

# 이미지 파일 찾기
search_paths = [
    Path("C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1"),
    Path("./test_images"),
    Path("./images"),
    Path(".")
]

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
found_images = []

for search_path in search_paths:
    if search_path.exists():
        for ext in image_extensions:
            images = list(search_path.glob(f"*{ext}"))
            found_images.extend(images)

if found_images:
    print(f"✓ {len(found_images)}개의 이미지 파일 발견:")
    for img in found_images[:5]:  # 처음 5개만 표시
        print(f"  - {img}")
    if len(found_images) > 5:
        print(f"  ... 외 {len(found_images)-5}개")
else:
    print("○ 이미지 파일을 찾을 수 없습니다.")
    print("테스트용 이미지를 다음 위치에 넣어주세요:")
    for path in search_paths:
        print(f"  - {path}")

print("\n=== 모든 테스트 완료 ===")
print("문제가 없다면 메인 프로그램을 실행할 수 있습니다:")
print("python dss_yolo_image_test_onnx.py")
