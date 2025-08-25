import os
import cv2
import numpy as np
import onnxruntime as ort
import time

def fixed_onnx_segmentation():
    """수정된 YOLOv8-seg ONNX 구현"""
    
    model_path = r"C:\Project\DSS\AI_Academy\yolov8\DSS_AI_training\DSS_experiment_1\weights\onnx_models\best.onnx"
    image_path = r"C:\Project\DSS\AI_Academy\image\img_20250719_175644.jpg"
    
    print("🚀 수정된 YOLOv8 세그멘테이션...")
    
    # ONNX 모델 로드
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 모델 입출력 정보 확인
    print(f"모델 입력: {session.get_inputs()[0].shape}")
    for i, output in enumerate(session.get_outputs()):
        print(f"모델 출력 {i}: {output.shape}")
    
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    orig_h, orig_w = image.shape[:2]
    print(f"원본 이미지 크기: {orig_w} x {orig_h}")
    
    # 전처리: BGR -> RGB -> 정규화 -> 640x640 리사이즈
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (640, 640))
    
    # 정규화: 0-255 -> 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    # CHW 형식으로 변환 (3, 640, 640)
    input_tensor = np.transpose(normalized, (2, 0, 1))
    
    # 배치 차원 추가 (1, 3, 640, 640)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    print(f"입력 텐서 크기: {input_tensor.shape}")
    print(f"입력 범위: {input_tensor.min():.3f} ~ {input_tensor.max():.3f}")
    
    # 추론 실행
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_tensor})
    
    print(f"\n📊 모델 출력 분석:")
    print(f"출력 개수: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"출력 {i} 크기: {out.shape}")
    
    # YOLOv8-seg 출력 구조:
    # 출력[0]: Detection (1, 37, 8400) -> (1, num_classes + 4 + 32, 8400)
    # 출력[1]: Proto masks (1, 32, 160, 160)
    
    detection_output = outputs[0]  # (1, 37, 8400)
    mask_protos = outputs[1] if len(outputs) > 1 else None  # (1, 32, 160, 160)
    
    # Detection 출력 재구성
    detection = detection_output[0]  # (37, 8400)
    detection = detection.T  # (8400, 37)로 전치
    
    print(f"\n🔍 Detection 분석:")
    print(f"Detection 형태: {detection.shape}")
    
    # YOLOv8-seg 구조: [x_center, y_center, width, height, class_conf, mask_coeffs...]
    # 1클래스의 경우: 4(bbox) + 1(class) + 32(mask) = 37
    
    num_detections = detection.shape[0]
    
    # 박스 좌표 (x_center, y_center, width, height)
    boxes = detection[:, :4]
    
    # 클래스 신뢰도 (1개 클래스)
    class_confidences = detection[:, 4]
    
    # 마스크 계수 (32개)
    if detection.shape[1] > 5:
        mask_coeffs = detection[:, 5:]  # (8400, 32)
        print(f"마스크 계수: {mask_coeffs.shape}")
    else:
        mask_coeffs = None
        print("❌ 마스크 계수가 없습니다")
    
    # 신뢰도 분석
    print(f"\n📈 신뢰도 분석:")
    print(f"신뢰도 범위: {class_confidences.min():.6f} ~ {class_confidences.max():.6f}")
    print(f"신뢰도 평균: {class_confidences.mean():.6f}")
    print(f"양수 신뢰도: {(class_confidences > 0).sum()}개")
    
    # 상위 신뢰도 출력
    top_indices = np.argsort(class_confidences)[::-1][:10]
    print(f"상위 10개 신뢰도:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {class_confidences[idx]:.6f}")
    
    # 매우 낮은 임계값으로 시작 (점진적으로 올림)
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    best_results = None
    best_threshold = None
    
    for threshold in thresholds:
        valid_mask = class_confidences > threshold
        valid_count = valid_mask.sum()
        print(f"\n임계값 {threshold}: {valid_count}개 검출")
        
        if valid_count > 0 and valid_count < 100:  # 너무 많지 않은 적당한 수
            best_results = valid_mask
            best_threshold = threshold
            break
    
    if best_results is None or best_results.sum() == 0:
        print("❌ 어떤 임계값에서도 유효한 검출이 없습니다")
        
        # 강제로 상위 5개라도 시도
        print("🔄 상위 5개 검출 강제 시도...")
        top_5_indices = np.argsort(class_confidences)[::-1][:5]
        best_results = np.zeros(len(class_confidences), dtype=bool)
        best_results[top_5_indices] = True
        best_threshold = class_confidences[top_5_indices[-1]]
    
    print(f"\n✅ 사용할 임계값: {best_threshold:.6f}")
    print(f"✅ 검출된 객체: {best_results.sum()}개")
    
    # 유효한 검출 결과 추출
    valid_boxes = boxes[best_results]
    valid_confidences = class_confidences[best_results]
    valid_mask_coeffs = mask_coeffs[best_results] if mask_coeffs is not None else None
    
    # 좌표 변환: center_x, center_y, w, h -> x1, y1, x2, y2
    x_center = valid_boxes[:, 0]
    y_center = valid_boxes[:, 1] 
    width = valid_boxes[:, 2]
    height = valid_boxes[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # 640x640에서 원본 크기로 스케일링
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    x1 = (x1 * scale_x).astype(int)
    y1 = (y1 * scale_y).astype(int)
    x2 = (x2 * scale_x).astype(int)
    y2 = (y2 * scale_y).astype(int)
    
    # 좌표 클리핑
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    print(f"\n🔲 바운딩 박스 정보:")
    for i in range(len(valid_confidences)):
        print(f"  박스 {i}: ({x1[i]}, {y1[i]}) -> ({x2[i]}, {y2[i]}), 신뢰도: {valid_confidences[i]:.4f}")
    
    # NMS 적용 (선택적)
    if len(valid_confidences) > 1:
        boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1])
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            valid_confidences.tolist(),
            best_threshold,
            0.4  # IoU threshold
        )
        
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            print(f"NMS 후: {len(nms_indices)}개 객체")
            
            # NMS 결과로 필터링
            x1 = x1[nms_indices]
            y1 = y1[nms_indices]
            x2 = x2[nms_indices]
            y2 = y2[nms_indices]
            valid_confidences = valid_confidences[nms_indices]
            if valid_mask_coeffs is not None:
                valid_mask_coeffs = valid_mask_coeffs[nms_indices]
    
    # 세그멘테이션 마스크 생성
    final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    if mask_protos is not None and valid_mask_coeffs is not None:
        print(f"\n🎭 마스크 생성 중...")
        
        proto_masks = mask_protos[0]  # (32, 160, 160)
        print(f"프로토 마스크: {proto_masks.shape}")
        
        for i in range(len(valid_confidences)):
            # 이 객체의 마스크 계수
            coeffs = valid_mask_coeffs[i]  # (32,)
            
            # 마스크 프로토타입과 계수로 마스크 생성
            mask = np.zeros((160, 160), dtype=np.float32)
            for j in range(32):
                mask += coeffs[j] * proto_masks[j]
            
            # Sigmoid 활성화
            mask = 1.0 / (1.0 + np.exp(-mask))
            
            # 이진화 (0.5 임계값)
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # 원본 크기로 리사이즈
            mask_resized = cv2.resize(mask_binary, (orig_w, orig_h))
            
            # 바운딩 박스 영역으로 마스킹 (선택적 - 더 정확한 결과를 위해)
            bbox_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if x1[i] < x2[i] and y1[i] < y2[i]:  # 유효한 박스인지 확인
                bbox_mask[y1[i]:y2[i], x1[i]:x2[i]] = 1
                mask_resized = mask_resized * bbox_mask
            
            # 최종 마스크에 누적
            final_mask = np.maximum(final_mask, mask_resized * 255)
            
            print(f"  객체 {i}: 마스크 픽셀 {np.count_nonzero(mask_resized)}개")
    
    total_mask_pixels = np.count_nonzero(final_mask)
    print(f"\n🏁 최종 마스크 픽셀: {total_mask_pixels}개")
    
    # 결과 시각화 및 저장
    if total_mask_pixels > 0:
        print("🎨 결과 시각화 중...")
        
        # 원본 이미지 복사
        result = image.copy()
        
        # 마스크 오버레이 (반투명 파란색)
        mask_colored = np.zeros_like(image)
        mask_colored[final_mask > 0] = [255, 0, 0]  # 파란색 (BGR)
        
        # 알파 블렌딩
        result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
        
        # 바운딩 박스 제거 - 마스크만 표시
        # for i in range(len(valid_confidences)):
        #     if x1[i] < x2[i] and y1[i] < y2[i]:  # 유효한 박스만
        #         cv2.rectangle(result, (x1[i], y1[i]), (x2[i], y2[i]), (0, 255, 0), 2)
        #         
        #         # 신뢰도 텍스트
        #         conf_text = f"lane: {valid_confidences[i]:.3f}"
        #         text_y = max(y1[i] - 10, 20)  # 텍스트가 화면 밖으로 나가지 않게
        #         cv2.putText(result, conf_text, (x1[i], text_y), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 결과 저장
        result_path = r"C:\Project\DSS\AI_Academy\fixed_segmentation_result.jpg"
        cv2.imwrite(result_path, result)
        
        # 마스크만 따로 저장
        mask_path = r"C:\Project\DSS\AI_Academy\segmentation_mask.jpg"
        cv2.imwrite(mask_path, final_mask)
        
        print(f"✅ 결과 저장 완료!")
        print(f"   - 전체 결과: {result_path}")
        print(f"   - 마스크만: {mask_path}")
        
        # 화면에 표시
        # 화면 크기에 맞게 리사이즈 (선택적)
        display_height = 800
        if result.shape[0] > display_height:
            scale = display_height / result.shape[0]
            new_width = int(result.shape[1] * scale)
            result_display = cv2.resize(result, (new_width, display_height))
        else:
            result_display = result
            
        cv2.imshow('Fixed Lane Segmentation', result_display)
        print("🖼️  결과 화면에 표시됨. 아무 키나 누르세요...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("❌ 최종 마스크가 비어있습니다")
        print("🔧 다른 이미지로 시도해보거나 모델 재학습을 고려해보세요")

if __name__ == "__main__":
    fixed_onnx_segmentation()
