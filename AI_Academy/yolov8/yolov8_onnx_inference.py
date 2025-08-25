import os
import cv2
import numpy as np
import onnxruntime as ort
import glob
from pathlib import Path

class YOLOv8ONNXSegmentationInference:
    def __init__(self, model_path, test_folder_path):
        """
        YOLOv8 ONNX 세그멘테이션 추론 클래스
        
        Args:
            model_path (str): ONNX 모델 경로 (best.onnx)
            test_folder_path (str): 테스트 이미지 폴더 경로
        """
        self.model_path = model_path
        self.test_folder_path = test_folder_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.load_model()
    
    def load_model(self):
        """ONNX 모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"📥 ONNX 모델 로드 중: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        
        # 입출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print("✅ ONNX 모델 로드 완료!")
        print(f"   입력: {self.session.get_inputs()[0].shape}")
        for i, output in enumerate(self.session.get_outputs()):
            print(f"   출력 {i}: {output.shape}")
    
    def get_test_images(self):
        """테스트 폴더에서 이미지 파일들 가져오기"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.test_folder_path, ext)
            image_files.extend(glob.glob(pattern))
            # 대소문자 구분 없이
            pattern_upper = os.path.join(self.test_folder_path, ext.upper())
            image_files.extend(glob.glob(pattern_upper))
        
        image_files = list(set(image_files))  # 중복 제거
        print(f"📁 발견된 이미지: {len(image_files)}개")
        
        return sorted(image_files)
    
    def preprocess_image(self, image):
        """
        이미지 전처리 (YOLOv8 ONNX용)
        
        Args:
            image (np.array): 원본 이미지 (BGR)
        
        Returns:
            np.array: 전처리된 텐서 (1, 3, 640, 640)
        """
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 640x640으로 리사이즈
        resized = cv2.resize(image_rgb, (640, 640))
        
        # 정규화: 0-255 -> 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가: (3, 640, 640) -> (1, 3, 640, 640)
        input_tensor = np.expand_dims(transposed, axis=0)
        
        return input_tensor
    
    def postprocess_detection(self, outputs, orig_w, orig_h, conf_threshold=0.5):
        """
        ONNX 모델 출력 후처리
        
        Args:
            outputs: 모델 출력 [detection, proto_masks]
            orig_w, orig_h: 원본 이미지 크기
            conf_threshold: 신뢰도 임계값
        
        Returns:
            tuple: (final_mask, detection_count)
        """
        # Detection 출력 처리
        detection_output = outputs[0]  # (1, 37, 8400)
        mask_protos = outputs[1] if len(outputs) > 1 else None  # (1, 32, 160, 160)
        
        # Detection 재구성
        detection = detection_output[0].T  # (8400, 37)
        
        # 박스, 신뢰도, 마스크 계수 분리
        boxes = detection[:, :4]  # (8400, 4) - x_center, y_center, width, height
        class_confidences = detection[:, 4]  # (8400,) - 클래스 신뢰도
        mask_coeffs = detection[:, 5:] if detection.shape[1] > 5 else None  # (8400, 32)
        
        # 적응형 임계값으로 유효한 검출 찾기
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.4]
        best_results = None
        best_threshold = conf_threshold
        
        for threshold in thresholds:
            valid_mask = class_confidences > threshold
            valid_count = valid_mask.sum()
            
            if valid_count > 0 and valid_count < 100:
                best_results = valid_mask
                best_threshold = threshold
                break
        
        # 유효한 검출이 없으면 상위 5개라도 시도
        if best_results is None or best_results.sum() == 0:
            top_5_indices = np.argsort(class_confidences)[::-1][:5]
            best_results = np.zeros(len(class_confidences), dtype=bool)
            best_results[top_5_indices] = True
            best_threshold = class_confidences[top_5_indices[-1]] if len(top_5_indices) > 0 else 0.001
        
        detection_count = best_results.sum()
        
        if detection_count == 0 or mask_protos is None or mask_coeffs is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8), 0
        
        # 유효한 검출 추출
        valid_boxes = boxes[best_results]
        valid_confidences = class_confidences[best_results]
        valid_mask_coeffs = mask_coeffs[best_results]
        
        # NMS 적용 (선택적)
        if len(valid_confidences) > 1:
            # 좌표 변환: center -> x1,y1,x2,y2
            x_center, y_center = valid_boxes[:, 0], valid_boxes[:, 1]
            width, height = valid_boxes[:, 2], valid_boxes[:, 3]
            
            x1 = (x_center - width / 2) * (orig_w / 640)
            y1 = (y_center - height / 2) * (orig_h / 640)
            x2 = (x_center + width / 2) * (orig_w / 640)
            y2 = (y_center + height / 2) * (orig_h / 640)
            
            # NMS용 박스 형식: [x, y, width, height]
            boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1])
            nms_indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(),
                valid_confidences.tolist(),
                best_threshold,
                0.4  # IoU threshold
            )
            
            if len(nms_indices) > 0:
                nms_indices = nms_indices.flatten()
                valid_mask_coeffs = valid_mask_coeffs[nms_indices]
                detection_count = len(nms_indices)
        
        # 세그멘테이션 마스크 생성
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        proto_masks = mask_protos[0]  # (32, 160, 160)
        
        for i in range(len(valid_mask_coeffs)):
            # 마스크 계수로 마스크 생성
            coeffs = valid_mask_coeffs[i]  # (32,)
            
            # 프로토타입 마스크와 계수 곱하기
            mask = np.zeros((160, 160), dtype=np.float32)
            for j in range(32):
                mask += coeffs[j] * proto_masks[j]
            
            # Sigmoid 활성화 및 이진화
            mask = 1.0 / (1.0 + np.exp(-mask))
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # 원본 크기로 리사이즈
            mask_resized = cv2.resize(mask_binary, (orig_w, orig_h))
            
            # 최종 마스크에 누적
            final_mask = np.maximum(final_mask, mask_resized * 255)
        
        return final_mask, detection_count
    
    def create_blue_overlay(self, image, mask, alpha=0.3, blue_color=(255, 0, 0)):
        """
        세그멘테이션 마스크에 투명한 파란색 오버레이 생성
        
        Args:
            image (np.array): 원본 이미지
            mask (np.array): 세그멘테이션 마스크
            alpha (float): 투명도 (0.0-1.0)
            blue_color (tuple): BGR 파란색 값
        
        Returns:
            np.array: 오버레이가 적용된 이미지
        """
        overlay_image = image.copy()
        
        if mask is not None and np.count_nonzero(mask) > 0:
            # 파란색 레이어 생성
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = blue_color  # BGR 형식 파란색
            
            # 알파 블렌딩
            overlay_image = cv2.addWeighted(overlay_image, 1-alpha, mask_colored, alpha, 0)
        
        return overlay_image
    
    def process_single_image(self, image_path, save_result=True, show_result=True):
        """
        단일 이미지 처리
        
        Args:
            image_path (str): 이미지 파일 경로
            save_result (bool): 결과 저장 여부
            show_result (bool): 결과 화면 표시 여부
        
        Returns:
            tuple: (overlay_image, should_exit, detection_info)
        """
        print(f"\n🔍 처리 중: {os.path.basename(image_path)}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return None, False, None
        
        orig_h, orig_w = image.shape[:2]
        print(f"   이미지 크기: {orig_w} x {orig_h}")
        
        # 전처리
        input_tensor = self.preprocess_image(image)
        
        # ONNX 추론
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 후처리
        mask, detection_count = self.postprocess_detection(outputs, orig_w, orig_h)
        
        mask_pixels = np.count_nonzero(mask)
        detection_info = {
            'detections': detection_count,
            'mask_pixels': mask_pixels,
            'has_lane': mask_pixels > 1000  # 1000픽셀 이상이면 유효한 레인으로 판단
        }
        
        if detection_info['has_lane']:
            print(f"✅ 레인 검출 성공: {detection_count}개 객체, {mask_pixels:,}픽셀")
            overlay_image = self.create_blue_overlay(image, mask)
        else:
            print(f"⚠️  레인 검출 미미: {detection_count}개 객체, {mask_pixels}픽셀")
            overlay_image = image
        
        # 결과 저장
        if save_result:
            output_dir = os.path.join(os.path.dirname(image_path), '..', 'onnx_results')
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # 오버레이 이미지 저장
            overlay_path = os.path.join(output_dir, f"{name}_onnx_segmented{ext}")
            cv2.imwrite(overlay_path, overlay_image)
            
            # 마스크만 저장
            mask_path = os.path.join(output_dir, f"{name}_mask{ext}")
            cv2.imwrite(mask_path, mask)
            
            print(f"💾 결과 저장: {overlay_path}")
            print(f"💾 마스크 저장: {mask_path}")
        
        # 결과 화면 표시
        should_exit = False
        if show_result:
            should_exit = self.display_result_cv2(image, overlay_image, 
                                                 os.path.basename(image_path), 
                                                 detection_info)
        
        return overlay_image, should_exit, detection_info
    
    def display_result_cv2(self, original, result, title, detection_info):
        """
        OpenCV를 사용한 결과 표시
        
        Args:
            original (np.array): 원본 이미지
            result (np.array): 처리된 이미지
            title (str): 제목
            detection_info (dict): 검출 정보
        """
        # 이미지 크기 조정 (화면에 맞게)
        height, width = original.shape[:2]
        max_height = 600
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_resized = cv2.resize(original, (new_width, new_height))
            result_resized = cv2.resize(result, (new_width, new_height))
        else:
            original_resized = original
            result_resized = result
        
        # 이미지를 나란히 배치
        combined = np.hstack([original_resized, result_resized])
        
        # 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Original: {title}', (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # 검출 결과에 따른 색상 변경
        status_color = (0, 255, 0) if detection_info['has_lane'] else (0, 165, 255)  # 초록 또는 주황
        status_text = f"Lane Detected: {detection_info['detections']} objs, {detection_info['mask_pixels']:,} pixels"
        cv2.putText(combined, status_text, (original_resized.shape[1] + 10, 30), 
                   font, 0.6, status_color, 2)
        
        # 하단에 안내 메시지
        cv2.putText(combined, 'Press SPACE for next image, ESC to exit', 
                   (10, combined.shape[0] - 40), font, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, 'Press S to save individual result', 
                   (10, combined.shape[0] - 15), font, 0.6, (0, 255, 255), 2)
        
        # 창 표시
        cv2.imshow('YOLOv8 ONNX Lane Segmentation (Blue)', combined)
        
        # 키 입력 대기
        status_emoji = "✅" if detection_info['has_lane'] else "⚠️"
        print(f"👆 {status_emoji} {title} - SPACE: 다음, ESC: 종료, S: 개별저장")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                return True
            elif key == ord(' '):  # SPACE
                return False
            elif key == ord('s') or key == ord('S'):  # S키로 개별 저장
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"manual_save_{timestamp}_{title}"
                cv2.imwrite(save_path, combined)
                print(f"📸 수동 저장: {save_path}")
                continue
    
    def process_all_images(self, max_images=None, save_results=True, show_results=True):
        """
        테스트 폴더의 모든 이미지 처리
        
        Args:
            max_images (int): 처리할 최대 이미지 수 (None이면 전체)
            save_results (bool): 결과 저장 여부
            show_results (bool): 결과 화면 표시 여부
        """
        image_files = self.get_test_images()
        
        if not image_files:
            print("❌ 테스트 이미지를 찾을 수 없습니다.")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n🚀 ONNX 이미지 처리 시작: {len(image_files)}개")
        print("=" * 60)
        
        if show_results:
            print("💡 조작법:")
            print("   - SPACE: 다음 이미지")
            print("   - ESC: 종료")
            print("   - S: 현재 결과 수동 저장")
            print("=" * 60)
        
        # 통계 정보
        stats = {
            'processed': 0,
            'successful_detections': 0,
            'total_objects': 0,
            'total_pixels': 0
        }
        
        try:
            for i, image_path in enumerate(image_files, 1):
                try:
                    print(f"\n[{i}/{len(image_files)}]", end=" ")
                    result, should_exit, detection_info = self.process_single_image(
                        image_path, 
                        save_result=save_results,
                        show_result=show_results
                    )
                    
                    if result is not None:
                        stats['processed'] += 1
                        if detection_info and detection_info['has_lane']:
                            stats['successful_detections'] += 1
                            stats['total_objects'] += detection_info['detections']
                            stats['total_pixels'] += detection_info['mask_pixels']
                    
                    # ESC로 종료
                    if should_exit:
                        print("\n👋 사용자가 종료를 선택했습니다.")
                        break
                        
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    continue
        
        finally:
            cv2.destroyAllWindows()
        
        # 최종 통계 출력
        print(f"\n🎉 처리 완료 통계:")
        print(f"   📊 처리된 이미지: {stats['processed']}/{len(image_files)}")
        print(f"   ✅ 성공적 검출: {stats['successful_detections']}/{stats['processed']}")
        if stats['successful_detections'] > 0:
            print(f"   🎯 평균 객체 수: {stats['total_objects']/stats['successful_detections']:.1f}")
            print(f"   🎨 평균 픽셀 수: {stats['total_pixels']/stats['successful_detections']:,.0f}")
        
        success_rate = (stats['successful_detections'] / stats['processed']) * 100 if stats['processed'] > 0 else 0
        print(f"   📈 성공률: {success_rate:.1f}%")

def main():
    """메인 실행 함수"""
    # 경로 설정
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    MODEL_PATH = os.path.join(BASE_PATH, "DSS_AI_training", "DSS_experiment_1", "weights", "onnx_models", "best.onnx")
    TEST_FOLDER = os.path.join(BASE_PATH, "dataset", "test", "images")
    
    print("=" * 60)
    print("🎯 YOLOv8 ONNX 차선 세그멘테이션 추론 시스템 (파란색 표시)")
    print("=" * 60)
    print(f"📂 ONNX 모델: {MODEL_PATH}")
    print(f"📁 테스트 폴더: {TEST_FOLDER}")
    print("=" * 60)
    
    # 경로 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("ONNX 변환이 완료된 후 다시 실행해주세요.")
        return
    
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ 테스트 폴더를 찾을 수 없습니다: {TEST_FOLDER}")
        return
    
    try:
        # ONNX 추론 시스템 초기화
        inference = YOLOv8ONNXSegmentationInference(MODEL_PATH, TEST_FOLDER)
        
        # 10개 이미지로 배치 테스트
        inference.process_all_images(
            max_images=10,      # 처음 10개 이미지만
            save_results=False,  # 결과 자동 저장
            show_results=True   # 화면에 표시
        )
        
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()

# ===== 사용 예시 =====
"""
# 1. 기본 사용법 (10개 이미지)
python yolov8_onnx_inference.py

# 2. 커스텀 사용법
inference = YOLOv8ONNXSegmentationInference(model_path, test_folder)
inference.process_all_images(max_images=20)  # 20개 이미지
inference.process_single_image("specific_image.jpg")  # 특정 이미지 하나

# 3. 자동 저장만 (화면 표시 없이)
inference.process_all_images(max_images=50, show_results=False)
"""