# yolov8_custom_train.py
# 640x 480 image 학습용

import os
import zipfile
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import torch

class YOLOv8Trainer:
    def __init__(self, zip_path, extract_path="./dataset"):
        """
        YOLOv8 훈련을 위한 클래스 초기화
        
        Args:
            zip_path (str): Roboflow에서 다운받은 zip 파일 경로
            extract_path (str): 데이터셋을 압축 해제할 경로
        """
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.dataset_path = None
        self.yaml_path = None
        
    def extract_dataset(self):
        """Roboflow zip 파일 압축 해제"""
        print("데이터셋 압축 해제 중...")
        
        # 압축 해제 디렉토리 생성
        os.makedirs(self.extract_path, exist_ok=True)
        
        # zip 파일 압축 해제
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)
        
        print(f"데이터셋이 {self.extract_path}에 압축 해제되었습니다.")
        
        # data.yaml 파일 찾기
        for root, dirs, files in os.walk(self.extract_path):
            if 'data.yaml' in files:
                self.yaml_path = os.path.join(root, 'data.yaml')
                self.dataset_path = root
                break
        
        if not self.yaml_path:
            raise FileNotFoundError("data.yaml 파일을 찾을 수 없습니다.")
        
        print(f"data.yaml 파일 위치: {self.yaml_path}")
        return self.yaml_path
    
    def check_dataset_structure(self):
        """데이터셋 구조 확인 및 자동 수정"""
        print("\n=== 데이터셋 구조 확인 ===")
        
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"클래스 수: {data.get('nc', 'Unknown')}")
        print(f"클래스 이름: {data.get('names', 'Unknown')}")
        
        # 실제 폴더 구조 확인
        print(f"\n실제 폴더 구조:")
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # 하위 폴더 확인
                try:
                    sub_items = os.listdir(item_path)
                    for sub_item in sub_items:
                        sub_path = os.path.join(item_path, sub_item)
                        if os.path.isdir(sub_path):
                            img_count = len([f for f in os.listdir(sub_path) 
                                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            print(f"    📁 {sub_item}/ ({img_count}개 이미지)")
                except:
                    pass
        
        # 이미지 개수 확인 및 폴더명 매핑
        folder_mapping = {}
        
        # 가능한 폴더명들 확인
        possible_folders = ['train', 'val', 'valid', 'validation', 'test']
        actual_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        print(f"\n📂 발견된 폴더들: {actual_folders}")
        
        # 폴더명 매핑 생성
        for folder in actual_folders:
            if 'train' in folder.lower():
                folder_mapping['train'] = folder
            elif any(val_name in folder.lower() for val_name in ['val', 'valid']):
                folder_mapping['val'] = folder
            elif 'test' in folder.lower():
                folder_mapping['test'] = folder
        
        print(f"📋 폴더 매핑: {folder_mapping}")
        
        # 각 폴더의 이미지 개수 확인
        for standard_name, actual_name in folder_mapping.items():
            images_path = os.path.join(self.dataset_path, actual_name, 'images')
            if os.path.exists(images_path):
                img_count = len([f for f in os.listdir(images_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✅ {standard_name} ({actual_name}): {img_count}개 이미지")
            else:
                print(f"❌ {standard_name} ({actual_name}): images 폴더 없음")
        
        return folder_mapping
    
    def copy_train_to_valid_test(self):
        """
        train 폴더의 모든 데이터를 valid와 test 폴더에 복사
        (분할하지 않고 전체 데이터를 각 폴더에 복사)
        """
        print(f"\n=== Train 데이터를 Valid/Test로 복사 ===")
        
        train_images_path = os.path.join(self.dataset_path, 'train', 'images')
        train_labels_path = os.path.join(self.dataset_path, 'train', 'labels')
        
        if not os.path.exists(train_images_path):
            raise FileNotFoundError(f"Train images 폴더를 찾을 수 없습니다: {train_images_path}")
        
        # 이미지 파일 목록 가져오기
        image_files = [f for f in os.listdir(train_images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            raise ValueError("Train 폴더에 이미지가 없습니다!")
        
        print(f"📁 복사할 이미지 수: {len(image_files)}개")
        
        # valid와 test 폴더에 전체 데이터 복사
        folders_to_create = ['valid', 'test']
        
        for folder_name in folders_to_create:
            # 폴더 생성
            folder_images_path = os.path.join(self.dataset_path, folder_name, 'images')
            folder_labels_path = os.path.join(self.dataset_path, folder_name, 'labels')
            
            os.makedirs(folder_images_path, exist_ok=True)
            os.makedirs(folder_labels_path, exist_ok=True)
            
            print(f"📂 {folder_name} 폴더에 데이터 복사 중...")
            
            # 모든 파일 복사
            for filename in image_files:
                # 이미지 파일 복사
                src_img = os.path.join(train_images_path, filename)
                dst_img = os.path.join(folder_images_path, filename)
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 복사 (있는 경우)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                src_label = os.path.join(train_labels_path, label_filename)
                dst_label = os.path.join(folder_labels_path, label_filename)
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
            
            print(f"✅ {folder_name} 폴더 복사 완료: {len(image_files)}개 파일")
        
        print(f"🎉 Train 데이터 복사 완료!")
        print(f"📁 Train: {len(image_files)}개")
        print(f"📁 Valid: {len(image_files)}개 (Train과 동일)")
        print(f"📁 Test: {len(image_files)}개 (Train과 동일)")
        
        return {
            'train': len(image_files),
            'valid': len(image_files),
            'test': len(image_files)
        }
    
    def convert_segmentation_to_detection_labels(self):
        """
        세그멘테이션 라벨을 객체 감지용 바운딩 박스로 변환
        """
        print(f"\n=== 세그멘테이션 라벨을 객체 감지용으로 변환 ===")
        
        folders = ['train', 'valid', 'test']
        converted_count = 0
        
        for folder in folders:
            labels_path = os.path.join(self.dataset_path, folder, 'labels')
            if not os.path.exists(labels_path):
                continue
                
            print(f"📁 {folder} 폴더 처리 중...")
            
            for label_file in os.listdir(labels_path):
                if not label_file.endswith('.txt'):
                    continue
                    
                file_path = os.path.join(labels_path, label_file)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3:  # 최소 클래스 + 2개 좌표
                        continue
                        
                    class_id = parts[0]
                    
                    # 좌표들 추출 (클래스 제외)
                    coords = [float(x) for x in parts[1:]]
                    
                    if len(coords) % 2 != 0:  # x,y 쌍이 아닌 경우
                        coords = coords[:-1]  # 마지막 하나 제거
                    
                    # x, y 좌표 분리
                    x_coords = coords[::2]  # 짝수 인덱스 (x)
                    y_coords = coords[1::2]  # 홀수 인덱스 (y)
                    
                    if len(x_coords) < 2:
                        continue
                    
                    # 바운딩 박스 계산
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # YOLO 형식으로 변환 (center_x, center_y, width, height)
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # 새 라벨 라인 생성
                    new_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    new_lines.append(new_line)
                
                # 변환된 라벨 저장
                if new_lines:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    converted_count += 1
        
        print(f"✅ 라벨 변환 완료: {converted_count}개 파일")
        return converted_count
    
    def update_yaml_paths(self, folder_mapping=None):
        """data.yaml의 경로를 절대경로로 업데이트 (폴더명 자동 매핑)"""
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 절대 경로로 업데이트
        base_path = Path(self.dataset_path).absolute()
        
        if folder_mapping:
            # 실제 폴더명으로 매핑
            if 'train' in folder_mapping:
                data['train'] = str(base_path / folder_mapping['train'] / 'images')
                print(f"✅ train 경로: {data['train']}")
            
            if 'val' in folder_mapping:
                data['val'] = str(base_path / folder_mapping['val'] / 'images')
                print(f"✅ val 경로: {data['val']}")
            
            if 'test' in folder_mapping:
                data['test'] = str(base_path / folder_mapping['test'] / 'images')
                print(f"✅ test 경로: {data['test']}")
        else:
            # 기본 폴더명 사용
            data['train'] = str(base_path / 'train' / 'images')
            data['val'] = str(base_path / 'valid' / 'images')
            
            if os.path.exists(base_path / 'test' / 'images'):
                data['test'] = str(base_path / 'test' / 'images')
        
        # 업데이트된 yaml 저장
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print("✅ data.yaml 경로가 업데이트되었습니다.")
        
        # 업데이트된 내용 확인
        print(f"\n📄 업데이트된 data.yaml 내용:")
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    def train_model(self, 
                   model_size='yolov8n',
                   epochs=100,
                   img_size=640,
                   batch_size=16,
                   device='auto',
                   project='yolov8_training',
                   name='exp'):
        """
        YOLOv8 모델 훈련
        
        Args:
            model_size (str): 모델 크기 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            epochs (int): 훈련 에포크 수
            img_size (int): 이미지 크기
            batch_size (int): 배치 크기
            device (str): 사용할 디바이스 ('auto', 'cpu', 'cuda', '0', '1', etc.)
            project (str): 프로젝트 폴더 이름
            name (str): 실험 이름
        """
        print(f"\n=== YOLOv8 훈련 시작 ===")
        print(f"모델: {model_size}")
        print(f"에포크: {epochs}")
        print(f"이미지 크기: {img_size}")
        print(f"배치 크기: {batch_size}")
        print(f"요청 디바이스: {device}")
        
        # GPU 사용 가능 여부 상세 확인
        print(f"\n🔍 GPU 상태 확인:")
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
        else:
            print("❌ CUDA를 사용할 수 없습니다!")
            print("GPU 드라이버와 CUDA 설치를 확인해주세요.")
        
        # 디바이스 설정 (GPU 우선 사용)
        if device == 'auto':
            if torch.cuda.is_available():
                device = '0'  # 첫 번째 GPU 사용
                print(f"✅ GPU 사용: cuda:{device}")
            else:
                device = 'cpu'
                print(f"⚠️  CPU 사용 (GPU 없음)")
        else:
            print(f"✅ 지정된 디바이스 사용: {device}")
        
        # GPU 메모리 확인 및 배치 크기 조정
        if device != 'cpu' and torch.cuda.is_available():
            try:
                # GPU 메모리 정보
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"📊 GPU 메모리: {gpu_memory:.1f}GB")
                
                # 메모리에 따른 배치 크기 권장
                if gpu_memory < 4:
                    recommended_batch = 8
                elif gpu_memory < 8:
                    recommended_batch = 16
                elif gpu_memory < 12:
                    recommended_batch = 32
                else:
                    recommended_batch = 64
                
                if batch_size > recommended_batch:
                    print(f"⚠️  배치 크기가 큽니다. 권장: {recommended_batch} (현재: {batch_size})")
                    print(f"메모리 부족시 배치 크기를 줄여보세요.")
                
            except Exception as e:
                print(f"GPU 메모리 확인 중 오류: {e}")
        
        # 모델 로드 (사전 훈련된 가중치 사용)
        print(f"\n📥 모델 로드 중: {model_size}.pt")
        model = YOLO(f'{model_size}.pt')
        
        # 모델을 명시적으로 GPU로 이동
        if device != 'cpu' and torch.cuda.is_available():
            try:
                model.model = model.model.to(f'cuda:{device}' if device.isdigit() else device)
                print(f"✅ 모델을 GPU로 이동: {device}")
            except Exception as e:
                print(f"⚠️  GPU 이동 실패: {e}")
                device = 'cpu'
        
        # 훈련 시작
        print(f"\n🚀 훈련 시작 - 디바이스: {device}")
        results = model.train(
            data=self.yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            save=True,
            save_period=10,  # 10 에포크마다 체크포인트 저장
            val=True,
            plots=True,
            verbose=True,
            # GPU 최적화 설정
            amp=True,        # Automatic Mixed Precision (GPU 가속)
            cache=False,     # 메모리 절약
            workers=8 if device != 'cpu' else 4,  # 데이터 로더 워커 수
        )
        
        print(f"\n🎉 훈련이 완료되었습니다!")
        print(f"📁 결과 폴더: {project}/{name}")
        print(f"🤖 최종 디바이스: {device}")
        
        return results
    
    def validate_model(self, model_path, project='yolov8_validation', name='exp'):
        """훈련된 모델 검증"""
        print(f"\n=== 모델 검증 시작 ===")
        
        model = YOLO(model_path)
        results = model.val(
            data=self.yaml_path,
            project=project,
            name=name,
            save_json=True,
            plots=True
        )
        
        print("검증이 완료되었습니다!")
        return results
    
    def predict_sample(self, model_path, image_path, conf=0.25, save=True):
        """샘플 이미지로 예측 테스트"""
        print(f"\n=== 예측 테스트 ===")
        
        model = YOLO(model_path)
        results = model.predict(
            source=image_path,
            conf=conf,
            save=save,
            show_labels=True,
            show_conf=True
        )
        
        print("예측이 완료되었습니다!")
        return results

def main():
    """메인 실행 함수"""
    # 설정
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    ZIP_PATH = os.path.join(BASE_PATH, "DSS_AI.v1i.yolov8.zip")
    EXTRACT_PATH = os.path.join(BASE_PATH, "dataset")
    
    # 경로 확인
    if not os.path.exists(ZIP_PATH):
        print(f"❌ ZIP 파일을 찾을 수 없습니다: {ZIP_PATH}")
        print("파일 경로를 확인해주세요.")
        return
    
    print(f"✅ ZIP 파일 확인됨: {ZIP_PATH}")
    
    # YOLOv8 훈련기 초기화
    trainer = YOLOv8Trainer(ZIP_PATH, EXTRACT_PATH)
    
    try:
        # 1. 데이터셋 압축 해제
        trainer.extract_dataset()
        
        # 2. 데이터셋 구조 확인
        folder_mapping = trainer.check_dataset_structure()
        
        # 2-1. train 폴더만 있는 경우 데이터 복사
        if 'val' not in folder_mapping and 'test' not in folder_mapping:
            print("\n⚠️  Valid/Test 폴더가 없습니다. Train 데이터를 복사합니다.")
            copy_result = trainer.copy_train_to_valid_test()
            
            # 복사 후 다시 폴더 구조 확인
            folder_mapping = trainer.check_dataset_structure()
        
        # 3. YAML 경로 업데이트 (폴더명 자동 매핑)
        trainer.update_yaml_paths(folder_mapping)
        
        # 4. 모델 훈련 (세그멘테이션 모델 - 수동 다운로드 대비)
        try:
            # 먼저 yolov8n-seg 시도
            model_name = 'yolov8n-seg'
            results = trainer.train_model(
                model_size=model_name,
                epochs=1000,            
                img_size=(640,480),
                batch_size=64,
                device='0',            
                project=os.path.join(BASE_PATH, 'DSS_AI_training'),
                name='DSS_experiment_1'
            )
        except Exception as seg_error:
            print(f"⚠️  세그멘테이션 모델 로드 실패: {seg_error}")
            print("🔄 기본 객체 감지 모델로 시도합니다...")
            
            # 세그멘테이션 실패시 기본 모델로 폴백
            model_name = 'yolov8n'
            results = trainer.train_model(
                model_size=model_name,
                epochs=100,            
                img_size=640,
                batch_size=32,
                device='0',            
                project=os.path.join(BASE_PATH, 'DSS_AI_training'),
                name='DSS_experiment_1_detection'
            )
        
        # 5. 최고 성능 모델로 검증
        best_model_path = os.path.join(BASE_PATH, 'DSS_AI_training', 'DSS_experiment_1', 'weights', 'best.pt')
        trainer.validate_model(best_model_path, 
                             project=os.path.join(BASE_PATH, 'DSS_AI_validation'),
                             name='DSS_validation_1')
        
        # 6. 샘플 예측 (선택사항)
        # 테스트 이미지가 있다면 아래 주석을 해제하고 경로를 설정하세요
        # sample_image = os.path.join(EXTRACT_PATH, "test", "images", "sample.jpg")
        # if os.path.exists(sample_image):
        #     trainer.predict_sample(best_model_path, sample_image)
        
        print("\n🎉 DSS AI 모델 훈련이 완료되었습니다!")
        print(f"📁 프로젝트 폴더: {BASE_PATH}")
        print(f"🤖 최종 모델: {best_model_path}")
        print(f"📊 훈련 결과: {os.path.join(BASE_PATH, 'DSS_AI_training', 'DSS_experiment_1')}")
        print(f"📈 검증 결과: {os.path.join(BASE_PATH, 'DSS_AI_validation', 'DSS_validation_1')}")
        
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("오류 해결을 위한 확인사항:")
        print("1. ZIP 파일 경로가 올바른지 확인")
        print("2. 충분한 디스크 공간이 있는지 확인")
        print("3. 필요한 패키지가 설치되었는지 확인")

if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    print("=" * 60)
    print("🚀 DSS AI Academy YOLOv8 훈련 시스템")
    print("=" * 60)
    print("📋 필요한 패키지:")
    print("   pip install ultralytics")
    print("   pip install torch torchvision")
    print("   pip install PyYAML")
    print("\n📂 프로젝트 경로: C:\\Project\\DSS\\AI_Academy\\yolov8")
    print("📦 데이터셋: DSS_AI.v1i.yolov8.zip")
    print("=" * 60 + "\n")
    
    main()

# ===== DSS 프로젝트 전용 유틸리티 함수들 =====
# ===== DSS 프로젝트 전용 유틸리티 함수들 =====

def resume_dss_training(base_path=r"C:\Project\DSS\AI_Academy\yolov8", epochs=100):
    """DSS 프로젝트의 중단된 훈련 재개"""
    checkpoint_path = os.path.join(base_path, 'DSS_AI_training', 'DSS_experiment_1', 'weights', 'last.pt')
    if os.path.exists(checkpoint_path):
        model = YOLO(checkpoint_path)
        results = model.train(resume=True, epochs=epochs)
        return results
    else:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None

def export_dss_model(base_path=r"C:\Project\DSS\AI_Academy\yolov8", format='onnx'):
    """DSS AI 모델을 다른 형식으로 내보내기"""
    model_path = os.path.join(base_path, 'DSS_AI_training', 'DSS_experiment_1', 'weights', 'best.pt')
    if os.path.exists(model_path):
        model = YOLO(model_path)
        model.export(format=format)
        print(f"✅ DSS AI 모델이 {format} 형식으로 내보내졌습니다.")
        export_path = model_path.replace('.pt', f'.{format}')
        print(f"📁 내보낸 파일: {export_path}")
    else:
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")

def benchmark_dss_model(base_path=r"C:\Project\DSS\AI_Academy\yolov8"):
    """DSS AI 모델 성능 벤치마크"""
    model_path = os.path.join(base_path, 'DSS_AI_training', 'DSS_experiment_1', 'weights', 'best.pt')
    if os.path.exists(model_path):
        model = YOLO(model_path)
        results = model.benchmark()
        return results
    else:
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return None

def predict_with_dss_model(image_path, base_path=r"C:\Project\DSS\AI_Academy\yolov8", conf=0.25):
    """DSS AI 모델로 이미지 예측"""
    model_path = os.path.join(base_path, 'DSS_AI_training', 'DSS_experiment_1', 'weights', 'best.pt')
    if os.path.exists(model_path) and os.path.exists(image_path):
        model = YOLO(model_path)
        results = model.predict(
            source=image_path,
            conf=conf,
            save=True,
            project=os.path.join(base_path, 'DSS_predictions'),
            name='prediction_results'
        )
        print(f"✅ 예측 완료! 결과: {os.path.join(base_path, 'DSS_predictions', 'prediction_results')}")
        return results
    else:
        if not os.path.exists(model_path):
            print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        if not os.path.exists(image_path):
            print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
        return None