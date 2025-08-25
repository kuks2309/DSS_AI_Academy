import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from pathlib import Path

class YOLOv8_ONNX:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        YOLOv8 ONNX 모델 초기화
        
        Args:
            model_path (str): ONNX 모델 파일 경로
            conf_threshold (float): 신뢰도 임계값
            iou_threshold (float): IoU 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # ONNX Runtime 세션 초기화 (GPU 사용 시도)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"사용중인 프로바이더: {self.session.get_providers()}")
        except Exception as e:
            print(f"GPU 프로바이더 실패, CPU 사용: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 모델 입출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 입력 크기 정보
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if input_shape[2] != -1 else 640
        self.input_width = input_shape[3] if input_shape[3] != -1 else 640
        
        print(f"모델 로드 완료: {model_path}")
        print(f"입력 크기: {self.input_width}x{self.input_height}")
        
        # COCO 클래스 이름 (필요에 따라 수정)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image):
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR)
        
        Returns:
            numpy.ndarray: 전처리된 이미지 배열
        """
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 현재 이미지 크기 확인
        current_height, current_width = image_rgb.shape[:2]
        
        # 크기가 같으면 리사이징 건너뛰기
        if current_width == self.input_width and current_height == self.input_height:
            processed_image = image_rgb
        else:
            # 리사이징
            processed_image = cv2.resize(image_rgb, (self.input_width, self.input_height))
        
        # 정규화 (0-1 범위)
        normalized = processed_image.astype(np.float32) / 255.0
        
        # 배치 차원 추가 및 채널 순서 변경 (BHWC -> BCHW)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess(self, outputs, original_shape):
        """
        후처리: NMS 적용 및 바운딩 박스 변환
        
        Args:
            outputs: 모델 출력
            original_shape: 원본 이미지 크기 (height, width)
        
        Returns:
            list: 검출된 객체 정보 [bbox, confidence, class_id]
        """
        predictions = outputs[0]
        
        # 예측값 형태 변환 (8400, 84) -> (84, 8400)
        if predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T
        
        # 바운딩 박스와 신뢰도 분리
        boxes = predictions[:4, :]  # x, y, w, h
        scores = predictions[4:, :]  # class confidences
        
        # 최고 신뢰도 클래스 찾기
        class_ids = np.argmax(scores, axis=0)
        confidences = np.max(scores, axis=0)
        
        # 신뢰도 임계값 적용
        valid_detections = confidences > self.conf_threshold
        boxes = boxes[:, valid_detections]
        confidences = confidences[valid_detections]
        class_ids = class_ids[valid_detections]
        
        if len(confidences) == 0:
            return []
        
        # 바운딩 박스 형식 변환 (중심점, 너비, 높이) -> (x1, y1, x2, y2)
        x_center, y_center, width, height = boxes[0], boxes[1], boxes[2], boxes[3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 원본 이미지 크기로 스케일링
        orig_height, orig_width = original_shape[:2]
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        
        # NMS 적용을 위한 바운딩 박스 형식 변환
        boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1])
        
        # NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                bbox = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
                confidence = float(confidences[i])
                class_id = int(class_ids[i])
                results.append([bbox, confidence, class_id])
        
        return results
    
    def detect(self, image):
        """
        객체 검출 수행
        
        Args:
            image: 입력 이미지
        
        Returns:
            list: 검출 결과
        """
        # 전처리
        input_tensor = self.preprocess(image)
        
        # 추론
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 후처리
        results = self.postprocess(outputs, image.shape)
        
        return results
    
    def draw_results(self, image, results):
        """
        검출 결과를 이미지에 그리기
        
        Args:
            image: 원본 이미지
            results: 검출 결과
        
        Returns:
            numpy.ndarray: 결과가 그려진 이미지
        """
        annotated_image = image.copy()
        
        for result in results:
            bbox, confidence, class_id = result
            x1, y1, x2, y2 = bbox
            
            # 클래스 이름
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # 텍스트
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_image


def find_test_images():
    """테스트용 이미지 파일들 찾기"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_dirs = [
        Path("C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1"),
        Path("C:/Project/DSS/test_images"),
        Path("./test_images"),
        Path("./images"),
        Path(".")
    ]
    
    image_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in image_extensions:
                image_files.extend(list(test_dir.glob(f"*{ext}")))
    
    return image_files


def main():
    # ONNX 모델 경로 설정
    model_dir = Path("C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1/weights/onnx_models")
    
    # ONNX 파일 찾기
    onnx_files = list(model_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"ONNX 파일을 찾을 수 없습니다: {model_dir}")
        return
    
    # 첫 번째 ONNX 파일 사용
    model_path = onnx_files[0]
    print(f"사용할 모델: {model_path}")
    
    # YOLOv8 ONNX 모델 초기화
    try:
        detector = YOLOv8_ONNX(str(model_path))
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 테스트 이미지 찾기
    image_files = find_test_images()
    
    if not image_files:
        print("테스트용 이미지를 찾을 수 없습니다.")
        print("다음 위치에 이미지 파일(.jpg, .png 등)을 배치해주세요:")
        print("- C:/Project/DSS/AI_Academy/yolov8/DSS_AI_training/DSS_experiment_1/")
        print("- 또는 현재 디렉토리에 test_images 폴더 생성")
        return
    
    print(f"찾은 이미지 파일 {len(image_files)}개:")
    for i, img_file in enumerate(image_files[:5]):  # 처음 5개만 표시
        print(f"  {i+1}. {img_file.name}")
    if len(image_files) > 5:
        print(f"  ... 외 {len(image_files)-5}개")
    
    print("\n이미지 처리 시작...")
    
    for img_file in image_files:
        print(f"\n처리 중: {img_file.name}")
        
        # 이미지 로드
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ✗ 이미지 로드 실패: {img_file}")
            continue
        
        print(f"  이미지 크기: {image.shape}")
        
        # 객체 검출
        start_time = time.time()
        try:
            results = detector.detect(image)
            inference_time = time.time() - start_time
            
            print(f"  추론 시간: {inference_time*1000:.1f}ms")
            print(f"  검출된 객체: {len(results)}개")
            
            # 결과 출력
            for i, result in enumerate(results):
                bbox, confidence, class_id = result
                class_name = detector.class_names[class_id] if class_id < len(detector.class_names) else f"Class_{class_id}"
                print(f"    {i+1}. {class_name}: {confidence:.2f}")
            
            # 결과 이미지 생성 및 저장
            annotated_image = detector.draw_results(image, results)
            
            # 결과 저장
            output_path = img_file.parent / f"result_{img_file.name}"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"  결과 저장: {output_path}")
            
            # 이미지 표시 (선택사항)
            cv2.imshow(f'Detection Result - {img_file.name}', annotated_image)
            print("  아무 키나 눌러서 다음 이미지로...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"  ✗ 검출 실패: {e}")
    
    print("\n모든 이미지 처리 완료!")


if __name__ == "__main__":
    main()
