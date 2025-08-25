import os
import sys
from pathlib import Path
import logging
import cv2
import numpy as np

def check_dependencies():
    """필요한 라이브러리가 설치되어 있는지 확인"""
    missing_packages = []
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_packages.append('ultralytics')
    
    try:
        import onnx
    except ImportError:
        missing_packages.append('onnx')
    
    try:
        import onnxruntime
    except ImportError:
        missing_packages.append('onnxruntime')
    
    if missing_packages:
        print("❌ 다음 라이브러리가 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n설치 명령어:")
        print("pip install ultralytics onnx onnxruntime")
        return False
    
    return True

def validate_onnx_model(onnx_path, test_image_path=None):
    """ONNX 모델 검증"""
    try:
        import onnxruntime as ort
        import onnx
        
        print(f"🔍 ONNX 모델 검증 중: {onnx_path}")
        
        # ONNX 모델 로드 및 구조 확인
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✅ ONNX 모델 구조 검증 성공")
        
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 입출력 정보 출력
        print(f"📋 모델 정보:")
        print(f"   입력: {[inp.name + str(inp.shape) for inp in session.get_inputs()]}")
        print(f"   출력: {[out.name + str(out.shape) for out in session.get_outputs()]}")
        
        # 더미 데이터로 추론 테스트
        input_shape = session.get_inputs()[0].shape
        if input_shape[0] == 'batch':
            input_shape = [1] + input_shape[1:]
        elif input_shape[0] is None:
            input_shape = [1] + input_shape[1:]
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        print(f"✅ 추론 테스트 성공")
        print(f"   출력 개수: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"   출력 {i} 형태: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX 모델 검증 실패: {e}")
        return False

def compare_models(pt_path, onnx_path, test_image_path=None):
    """PT와 ONNX 모델 결과 비교"""
    try:
        from ultralytics import YOLO
        import onnxruntime as ort
        
        if test_image_path and os.path.exists(test_image_path):
            test_image = cv2.imread(test_image_path)
        else:
            # 더미 이미지 생성
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print(f"🔄 모델 비교 테스트 시작...")
        
        # PT 모델 테스트
        pt_model = YOLO(pt_path)
        pt_results = pt_model(test_image, conf=0.25, verbose=False)
        pt_has_masks = pt_results and pt_results[0].masks is not None
        print(f"📊 PT 모델 결과: 마스크 존재 = {pt_has_masks}")
        
        # ONNX 모델 테스트
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 이미지 전처리
        resized = cv2.resize(test_image, (640, 640))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # ONNX 추론
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        print(f"📊 ONNX 모델 결과: {len(outputs)}개 출력")
        
        if pt_has_masks and len(outputs) >= 2:
            print("✅ 모델 비교: 둘 다 세그멘테이션 출력 있음")
        elif not pt_has_masks and len(outputs) == 1:
            print("✅ 모델 비교: 둘 다 검출만 수행")
        else:
            print("⚠️  모델 비교: 출력 형태가 다를 수 있음")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 비교 실패: {e}")
        return False

def convert_best_weight_to_onnx_improved():
    """개선된 ONNX 변환 함수"""
    # 의존성 체크
    if not check_dependencies():
        return
    
    from ultralytics import YOLO
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 폴더 경로 설정
    INPUT_FOLDER = r"C:\Project\DSS\AI_Academy\yolov8\DSS_AI_training\DSS_experiment_1\weights"
    OUTPUT_FOLDER = r"C:\Project\DSS\AI_Academy\yolov8\DSS_AI_training\DSS_experiment_1\weights\onnx_models"
    
    print("=" * 70)
    print("🔧 개선된 YOLOv8 Best Weight to ONNX Converter")
    print("=" * 70)
    print(f"입력 폴더: {INPUT_FOLDER}")
    print(f"출력 폴더: {OUTPUT_FOLDER}")
    print("=" * 70)
    
    # best.pt 파일 경로
    best_pt_path = os.path.join(INPUT_FOLDER, "best.pt")
    
    # best.pt 파일 존재 확인
    if not os.path.exists(best_pt_path):
        print(f"❌ best.pt 파일을 찾을 수 없습니다: {best_pt_path}")
        return
    
    print(f"📁 변환할 파일: best.pt ({os.path.getsize(best_pt_path) / (1024*1024):.1f} MB)")
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    try:
        print("🔄 변환 시작: best.pt → best.onnx")
        
        # YOLO 모델 로드
        model = YOLO(best_pt_path)
        
        # 모델 정보 확인
        print(f"📋 모델 정보:")
        print(f"   모델 종류: {model.task}")
        print(f"   클래스 수: {len(model.names) if hasattr(model, 'names') else '불명'}")
        
        # ONNX 파일 경로
        onnx_filename = "best.onnx"
        onnx_path = os.path.join(OUTPUT_FOLDER, onnx_filename)
        
        print("⚙️  모델 로드 완료, ONNX 변환 중...")
        print("   (세그멘테이션 모델의 경우 시간이 오래 걸릴 수 있습니다)")
        
        # 개선된 ONNX 변환 설정
        success = model.export(
            format='onnx',
            imgsz=640,
            dynamic=False,
            simplify=True,
            opset=11,
            half=False,  # FP16 비활성화 (호환성을 위해)
            int8=False,  # INT8 양자화 비활성화
            device='cpu'  # CPU에서 변환
        )
        
        # 변환된 파일을 지정 폴더로 이동
        default_onnx_path = best_pt_path.replace('.pt', '.onnx')
        if os.path.exists(default_onnx_path):
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            os.rename(default_onnx_path, onnx_path)
            
            onnx_size = os.path.getsize(onnx_path) / (1024*1024)
            print(f"✅ 변환 성공!")
            print(f"📂 출력 파일: {onnx_filename} ({onnx_size:.1f} MB)")
            print(f"📂 저장 경로: {onnx_path}")
            
            # ONNX 모델 검증
            print("\n" + "=" * 70)
            if validate_onnx_model(onnx_path):
                print("✅ ONNX 모델 검증 완료")
            else:
                print("❌ ONNX 모델 검증 실패")
                
            # 모델 비교
            print("\n" + "=" * 70)
            if compare_models(best_pt_path, onnx_path):
                print("✅ 모델 비교 완료")
            else:
                print("❌ 모델 비교 실패")
            
        else:
            print("❌ 변환된 ONNX 파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"❌ 변환 실패: {str(e)}")
        logging.error(f"변환 실패: {best_pt_path} - {str(e)}")
        print("\n💡 해결 방법:")
        print("1. 최신 ultralytics 설치: pip install -U ultralytics")
        print("2. ONNX 라이브러리 재설치: pip install -U onnx onnxruntime")
        print("3. PyTorch 버전 확인: pip install torch torchvision")
        print("4. 메모리 부족 시 다른 프로그램 종료")
    
    print("\n" + "=" * 70)
    print("🔧 개선된 변환 작업 완료!")
    print("=" * 70)

if __name__ == "__main__":
    convert_best_weight_to_onnx_improved()
