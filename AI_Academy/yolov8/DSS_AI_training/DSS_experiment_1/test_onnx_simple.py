import cv2
import numpy as np
import time
from pathlib import Path

def test_camera():
    """다양한 카메라 인덱스 테스트"""
    print("=== 카메라 테스트 ===")
    
    for i in range(5):  # 0~4번 카메라 테스트
        print(f"카메라 인덱스 {i} 테스트 중...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ 카메라 인덱스 {i}: 작동 중, 해상도: {frame.shape}")
                cap.release()
                return i
            else:
                print(f"✗ 카메라 인덱스 {i}: 프레임 읽기 실패")
        else:
            print(f"✗ 카메라 인덱스 {i}: 접근 불가")
        
        cap.release()
    
    print("✗ 사용 가능한 카메라를 찾을 수 없습니다.")
    return None

def test_onnx_simple():
    """간단한 ONNX 테스트"""
    print("\n=== ONNX Runtime 재테스트 ===")
    
    try:
        # 다른 방법으로 import 시도
        from onnxruntime import InferenceSession
        print("✓ InferenceSession import 성공")
        
        model_path = "C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1/weights/onnx_models/best.onnx"
        
        # 세션 생성 시도
        session = InferenceSession(model_path)
        print("✓ ONNX 모델 로드 성공")
        
        # 입출력 정보 확인
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"입력: {input_info.name}, 형태: {input_info.shape}")
        print(f"출력: {output_info.name}, 형태: {output_info.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ InferenceSession import 실패: {e}")
        print("ONNX Runtime을 재설치해야 합니다:")
        print("pip uninstall onnxruntime")
        print("pip install onnxruntime")
        return False
    except Exception as e:
        print(f"✗ ONNX 모델 로드 실패: {e}")
        return False

if __name__ == "__main__":
    # 1. ONNX 테스트
    onnx_ok = test_onnx_simple()
    
    # 2. 카메라 테스트
    camera_index = test_camera()
    
    if onnx_ok and camera_index is not None:
        print(f"\n✓ 모든 테스트 통과! 카메라 인덱스: {camera_index}")
        print("메인 프로그램을 실행할 준비가 되었습니다.")
    else:
        print("\n✗ 문제가 있습니다. 위의 지시사항을 따라 해결해주세요.")
