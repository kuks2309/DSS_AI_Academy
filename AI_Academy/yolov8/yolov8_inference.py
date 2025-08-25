import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class YOLOv8SegmentationInference:
    def __init__(self, model_path, test_folder_path):
        """
        YOLOv8 세그멘테이션 추론 클래스
        
        Args:
            model_path (str): 학습된 모델 경로 (best.pt)
            test_folder_path (str): 테스트 이미지 폴더 경로
        """
        self.model_path = model_path
        self.test_folder_path = test_folder_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """학습된 모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"📥 모델 로드 중: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✅ 모델 로드 완료!")
    
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
    
    def create_blue_overlay(self, image, mask, alpha=0.4, blue_color=(255, 0, 0)):
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
        # 이미지 복사
        overlay_image = image.copy()
        
        # 마스크가 있는 부분에 파란색 적용
        if mask is not None and mask.sum() > 0:
            # 마스크를 3채널로 확장
            if len(mask.shape) == 2:
                mask_3d = np.stack([mask] * 3, axis=-1)
            else:
                mask_3d = mask
            
            # 파란색 레이어 생성
            blue_layer = np.zeros_like(image)
            blue_layer[:, :] = blue_color  # BGR 순서로 파란색
            
            # 마스크 영역에만 파란색 적용 (알파 블렌딩)
            mask_bool = mask_3d > 0
            overlay_image[mask_bool] = (
                alpha * blue_layer[mask_bool] + 
                (1 - alpha) * overlay_image[mask_bool]
            ).astype(np.uint8)
        
        return overlay_image
    
    def process_single_image(self, image_path, save_result=True, show_result=True, use_cv2=True):
        """
        단일 이미지 처리
        
        Args:
            image_path (str): 이미지 파일 경로
            save_result (bool): 결과 저장 여부
            show_result (bool): 결과 화면 표시 여부
            use_cv2 (bool): OpenCV 표시 사용 여부 (True=빠름, False=matplotlib)
        """
        print(f"\n🔍 처리 중: {os.path.basename(image_path)}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return None, False
        
        # 모델 추론
        results = self.model(image_path, conf=0.25, verbose=False)
        
        if not results or not results[0].masks:
            print("⚠️  세그멘테이션 마스크가 감지되지 않았습니다.")
            overlay_image = image
        else:
            # 첫 번째 결과에서 마스크 추출
            masks = results[0].masks.data.cpu().numpy()
            
            # 모든 마스크를 하나로 합치기
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            for mask in masks:
                # 마스크를 원본 이미지 크기로 리사이즈
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
            
            # 파란색 오버레이 적용
            overlay_image = self.create_blue_overlay(
                image, 
                combined_mask, 
                alpha=0.4, 
                blue_color=(255, 0, 0)  # BGR 형식의 파란색
            )
            
            print(f"✅ 세그멘테이션 완료: {len(masks)}개 마스크 감지")
        
        # 결과 저장
        if save_result:
            output_dir = os.path.join(os.path.dirname(image_path), '..', 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_segmented{ext}")
            
            cv2.imwrite(output_path, overlay_image)
            print(f"💾 결과 저장: {output_path}")
        
        # 결과 화면 표시
        should_exit = False
        if show_result:
            if use_cv2:
                should_exit = self.display_result_cv2(image, overlay_image, os.path.basename(image_path))
            else:
                self.display_result(image, overlay_image, os.path.basename(image_path))
        
        return overlay_image, should_exit
    
    def display_result(self, original, result, title):
        """
        원본과 결과 이미지를 나란히 표시 (키 입력 대기)
        
        Args:
            original (np.array): 원본 이미지
            result (np.array): 처리된 이미지
            title (str): 제목
        """
        # BGR to RGB 변환 (matplotlib용)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # 화면 표시
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title(f'원본: {title}', fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_rgb)
        plt.title(f'차선 세그멘테이션 (파란색): {title}', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        
    def display_result_cv2(self, original, result, title):
        """
        OpenCV를 사용한 더 빠른 이미지 표시 (키 입력 대기)
        
        Args:
            original (np.array): 원본 이미지
            result (np.array): 처리된 이미지
            title (str): 제목
        """
        # 이미지 크기 조정 (화면에 맞게)
        height, width = original.shape[:2]
        max_height = 600
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            original = cv2.resize(original, (new_width, new_height))
            result = cv2.resize(result, (new_width, new_height))
        
        # 이미지를 나란히 배치
        combined = np.hstack([original, result])
        
        # 텍스트 추가
        cv2.putText(combined, f'Original: {title}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f'Blue Segmented: {title}', (original.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 하단에 안내 메시지
        cv2.putText(combined, 'Press any key for next image, ESC to exit', 
                   (10, combined.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 창 표시
        cv2.imshow('Lane Segmentation Results (Blue)', combined)
        
        # 키 입력 대기
        print(f"👆 {title} - 아무 키나 누르면 다음 이미지로... (ESC로 종료)")
        key = cv2.waitKey(0) & 0xFF
        
        return key == 27  # ESC 키 확인 (종료 신호)
    
    def process_all_images(self, max_images=None, save_results=True, show_results=True, use_cv2=True):
        """
        테스트 폴더의 모든 이미지 처리
        
        Args:
            max_images (int): 처리할 최대 이미지 수 (None이면 전체)
            save_results (bool): 결과 저장 여부
            show_results (bool): 결과 화면 표시 여부
            use_cv2 (bool): OpenCV 사용 여부 (빠른 키 입력 응답)
        """
        image_files = self.get_test_images()
        
        if not image_files:
            print("❌ 테스트 이미지를 찾을 수 없습니다.")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n🚀 이미지 처리 시작: {len(image_files)}개")
        print("=" * 60)
        
        if use_cv2 and show_results:
            print("💡 조작법:")
            print("   - 아무 키: 다음 이미지")
            print("   - ESC: 종료")
            print("=" * 60)
        
        processed_count = 0
        try:
            for i, image_path in enumerate(image_files, 1):
                try:
                    print(f"\n[{i}/{len(image_files)}]", end=" ")
                    result, should_exit = self.process_single_image(
                        image_path, 
                        save_result=save_results,
                        show_result=show_results,
                        use_cv2=use_cv2
                    )
                    
                    if result is not None:
                        processed_count += 1
                    
                    # ESC로 종료
                    if should_exit:
                        print("\n👋 사용자가 종료를 선택했습니다.")
                        break
                        
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    continue
        
        finally:
            if use_cv2:
                cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
        
        print(f"\n🎉 처리 완료: {processed_count}/{len(image_files)}개 성공")
    
    def process_video(self, video_path, output_path=None, show_live=True):
        """
        비디오 파일 처리 (보너스 기능)
        
        Args:
            video_path (str): 입력 비디오 경로
            output_path (str): 출력 비디오 경로
            show_live (bool): 실시간 표시 여부
        """
        if not os.path.exists(video_path):
            print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ 비디오를 열 수 없습니다.")
            return
        
        # 비디오 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 비디오 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
        
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 모델 추론
                results = self.model(frame, conf=0.25, verbose=False)
                
                if results and results[0].masks:
                    masks = results[0].masks.data.cpu().numpy()
                    
                    # 마스크 합치기
                    combined_mask = np.zeros((height, width), dtype=np.uint8)
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (width, height))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                    
                    # 파란색 오버레이 적용
                    frame = self.create_blue_overlay(frame, combined_mask)
                
                # 결과 저장
                if output_path:
                    out.write(frame)
                
                # 실시간 표시
                if show_live:
                    cv2.imshow('Lane Segmentation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 진행 상황 표시
                if frame_count % 30 == 0:
                    print(f"처리 중: {frame_count}/{total_frames} 프레임")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"🎉 비디오 처리 완료: {frame_count}프레임")

def main():
    """메인 실행 함수"""
    # 경로 설정
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    MODEL_PATH = os.path.join(BASE_PATH, "DSS_AI_training", "DSS_experiment_1", "weights", "best.pt")
    TEST_FOLDER = os.path.join(BASE_PATH, "dataset", "test", "images")
    
    print("=" * 60)
    print("🎯 YOLOv8 차선 세그멘테이션 추론 시스템 (파란색 표시)")
    print("=" * 60)
    print(f"📂 모델 경로: {MODEL_PATH}")
    print(f"📁 테스트 폴더: {TEST_FOLDER}")
    print("=" * 60)
    
    # 경로 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("훈련이 완료된 후 다시 실행해주세요.")
        return
    
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ 테스트 폴더를 찾을 수 없습니다: {TEST_FOLDER}")
        return
    
    try:
        # 추론 시스템 초기화
        inference = YOLOv8SegmentationInference(MODEL_PATH, TEST_FOLDER)
        
        # 모든 테스트 이미지 처리 (OpenCV 사용 - 빠른 키 입력)
        inference.process_all_images(
            max_images=10,      # 처음 10개 이미지만 처리 (전체는 None)
            save_results=True,  # 결과 저장
            show_results=True,  # 화면 표시
            use_cv2=True        # OpenCV 사용 (빠른 키 입력 응답)
        )
        
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()

# ===== 사용 예시 =====
"""
# 1. 기본 사용법
python yolov8_inference.py

# 2. 단일 이미지 처리
inference = YOLOv8SegmentationInference(model_path, test_folder)
inference.process_single_image("path/to/image.jpg")

# 3. 비디오 처리 (보너스)
inference.process_video("input_video.mp4", "output_video.mp4")

# 4. 색상 변경하고 싶다면:
# create_blue_overlay 함수에서 blue_color 값을 변경
# 예: (0, 255, 0) = 초록색, (0, 0, 255) = 빨간색, (255, 255, 0) = 시안색
"""