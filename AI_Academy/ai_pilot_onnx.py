#ai_pilot_onnx.py

import sys
import time
import asyncio
import cv2
import numpy as np
import json
import signal
import logging
import os
from pathlib import Path
import time

# ONNX Runtime import
import onnxruntime as ort

# DSS SDK 관련 import
from dss_sdk.core.idsssdk import IDSSSDK
from dss_sdk.config.sdk_config import *
from dss_sdk.core.osi_manager import OSIManager
from dss_sdk.protobuf import dss_pb2

# =============== 로그 레벨 설정 ===============
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('nats').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('DSSSDK_PY').setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class YOLOv8ONNXLaneDetector:
    def __init__(self, model_path, conf_threshold=0.01):
        """
        YOLOv8 ONNX 차선 감지 클래스 (AI Pilot용)
        
        Args:
            model_path (str): ONNX 모델 경로
            conf_threshold (float): 신뢰도 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.load_model()
        
        # 성능 측정 변수
        self.inference_times = []
        self.max_inference_history = 30
    
    def load_model(self):
        """ONNX 모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            print(f"📥 YOLOv8 ONNX 모델 로드 중: {self.model_path}")
            
            # ONNX Runtime 세션 생성
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            
            # 입출력 정보 가져오기
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print("✅ YOLOv8 ONNX 모델 로드 완료!")
            print(f"   입력: {self.session.get_inputs()[0].shape}")
            for i, output in enumerate(self.session.get_outputs()):
                print(f"   출력 {i}: {output.shape}")
            
            return True
        except Exception as e:
            print(f"❌ ONNX 모델 로드 실패: {e}")
            return False
    
    def preprocess_image(self, image):
        """
        이미지 전처리 (YOLOv8 ONNX용)
        
        Args:
            image (np.array): 원본 이미지 (BGR)
        
        Returns:
            tuple: (전처리된 텐서, 원본 크기)
        """
        orig_h, orig_w = image.shape[:2]
        
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
        
        return input_tensor, (orig_w, orig_h)
    
    def postprocess_detection(self, outputs, orig_w, orig_h):
        """
        ONNX 모델 출력 후처리 (AI Pilot용 최적화)
        
        Args:
            outputs: 모델 출력 [detection, proto_masks]
            orig_w, orig_h: 원본 이미지 크기
        
        Returns:
            np.array: 최종 마스크 (원본 크기)
        """
        try:
            # Detection 출력 처리
            detection_output = outputs[0]  # (1, 37, 8400)
            mask_protos = outputs[1] if len(outputs) > 1 else None  # (1, 32, 160, 160)
            
            if mask_protos is None:
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            # Detection 재구성
            detection = detection_output[0].T  # (8400, 37)
            
            # 박스, 신뢰도, 마스크 계수 분리
            boxes = detection[:, :4]  # (8400, 4)
            class_confidences = detection[:, 4]  # (8400,)
            mask_coeffs = detection[:, 5:] if detection.shape[1] > 5 else None  # (8400, 32)
            
            if mask_coeffs is None:
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            # AI Pilot용 적응형 임계값 (더 정교한 검출)
            thresholds = [0.001, 0.003, 0.005, 0.01, 0.02]
            best_results = None
            
            for threshold in thresholds:
                valid_mask = class_confidences > threshold
                valid_count = valid_mask.sum()
                
                # AI Pilot은 더 많은 객체 처리 가능 (정확성 우선)
                if valid_count > 0 and valid_count < 80:
                    best_results = valid_mask
                    break
            
            # 유효한 검출이 없으면 상위 5개만 시도
            if best_results is None or best_results.sum() == 0:
                top_5_indices = np.argsort(class_confidences)[::-1][:5]
                best_results = np.zeros(len(class_confidences), dtype=bool)
                best_results[top_5_indices] = True
            
            detection_count = best_results.sum()
            
            if detection_count == 0:
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            # 유효한 검출 추출
            valid_mask_coeffs = mask_coeffs[best_results]
            
            # 세그멘테이션 마스크 생성 (AI Pilot용 - 더 많은 객체 처리)
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            proto_masks = mask_protos[0]  # (32, 160, 160)
            
            # AI Pilot은 최대 10개 객체까지 처리 (정확성 우선)
            max_objects = min(len(valid_mask_coeffs), 10)
            
            for i in range(max_objects):
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
            
            return final_mask
            
        except Exception as e:
            print(f"❌ ONNX 후처리 오류: {e}")
            return np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    def create_blue_overlay(self, image, mask, alpha=0.4, blue_color=(255, 0, 0)):
        """
        세그멘테이션 마스크에 투명한 파란색 오버레이 생성
        
        Args:
            image (np.array): 원본 이미지
            mask (np.array): 세그멘테이션 마스크
            alpha (float): 투명도
            blue_color (tuple): BGR 파란색 값
        
        Returns:
            np.array: 오버레이가 적용된 이미지
        """
        overlay_image = image.copy()
        
        if mask is not None and np.count_nonzero(mask) > 0:
            # 파란색 레이어 생성
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = blue_color
            
            # 알파 블렌딩
            overlay_image = cv2.addWeighted(overlay_image, 1-alpha, mask_colored, alpha, 0)
        
        return overlay_image
    
    def detect_car_region(self, image):
        """차량 영역 감지 (원본과 동일)"""
        height, width = image.shape[:2]
        
        car_region = {
            'x1': int(width * 0.25),
            'y1': int(height * 0.65),
            'x2': int(width * 0.75),
            'y2': int(height * 0.95)
        }
        
        return car_region
    
    def draw_anchor_line_excluding_car(self, image, y, width, car_region, color=(0, 0, 255), thickness=2):
        """차량 영역 제외하고 Anchor line 그리기 (원본과 동일)"""
        if y >= car_region['y1'] and y <= car_region['y2']:
            cv2.line(image, (0, y), (car_region['x1'], y), color, thickness)
            cv2.line(image, (car_region['x2'], y), (width, y), color, thickness)
        else:
            cv2.line(image, (0, y), (width, y), color, thickness)
    
    def get_average_inference_time(self):
        """평균 추론 시간 계산"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def detect_lanes_onnx(self, image):
        """
        ONNX 모델로 차선 감지
        
        Args:
            image (np.array): 입력 이미지
            
        Returns:
            tuple: (마스크, 추론 시간)
        """
        if self.session is None:
            return None, 0.0
        
        inference_start = time.time()
        
        try:
            # 전처리
            input_tensor, (orig_w, orig_h) = self.preprocess_image(image)
            
            # ONNX 추론
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # 후처리
            mask = self.postprocess_detection(outputs, orig_w, orig_h)
            
            # 추론 시간 기록
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # 최대 기록 수 제한
            if len(self.inference_times) > self.max_inference_history:
                self.inference_times.pop(0)
            
            return mask, inference_time
            
        except Exception as e:
            print(f"❌ ONNX 차선 감지 오류: {e}")
            return None, 0.0

class DSSYOLOONNXController:
    def __init__(self, model_path):
        """
        DSS + YOLOv8 ONNX AI Pilot 통합 컨트롤러
        
        Args:
            model_path (str): YOLOv8 ONNX 모델 경로
        """
        self.running = True
        self.rgb_image = None
        self.dss_instance = None
        self.last_key_pressed = None
        
        self.drive_state = {
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 0.0
        }
        
        # YOLOv8 ONNX 차선 감지기 초기화
        self.lane_detector = YOLOv8ONNXLaneDetector(model_path)
        
        # Anchor line 설정 (픽셀 값으로 관리, 위에서 0 기준)
        self.anchor_lines = []
        self.show_anchor_lines = True
        self.selected_line_index = 0
        self.edit_mode = False
        self.image_height = 480
        self.initialize_default_anchor_lines()
        
        # 프로그램 시작시 자동으로 JSON 파일 로드 시도
        self.load_anchor_config()
        
        # 성능 측정 변수
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.total_inference_time = 0.0
        self.frame_count = 0
        
        print("🚗 DSS + YOLOv8 ONNX AI Pilot 시스템 초기화 완료!")
    
    def initialize_default_anchor_lines(self):
        """기본 Anchor lines 초기화"""
        default_positions = [200, 250, 300, 350, 400, 450]
        self.anchor_lines = [pos for pos in default_positions]
        print(f"📏 기본 Anchor lines 초기화: {len(self.anchor_lines)}개 (픽셀 값)")
        self.print_anchor_positions()
    
    def update_image_height(self, height):
        """이미지 높이 업데이트"""
        if self.image_height != height:
            self.image_height = height
            print(f"📐 이미지 높이 업데이트: {height}px")
    
    def get_anchor_y_positions(self, image_height):
        """유효한 Anchor Y 위치 반환"""
        self.update_image_height(image_height)
        valid_positions = [y for y in self.anchor_lines if 0 <= y < image_height]
        return valid_positions
    
    def add_anchor_line(self, pixel_position=None):
        """Anchor line 추가"""
        if pixel_position is None:
            if self.anchor_lines:
                pixel_position = min(self.image_height - 20, max(self.anchor_lines) + 50)
            else:
                pixel_position = 300
        
        pixel_position = max(10, min(self.image_height - 10, pixel_position))
        
        self.anchor_lines.append(pixel_position)
        self.anchor_lines.sort()
        print(f"➕ Anchor line 추가: Y={pixel_position}px")
        self.print_anchor_positions()
    
    def remove_anchor_line(self, index=None):
        """Anchor line 제거"""
        if not self.anchor_lines:
            return
        
        if index is None:
            index = self.selected_line_index
        
        if 0 <= index < len(self.anchor_lines):
            removed_pos = self.anchor_lines.pop(index)
            print(f"➖ Anchor line 제거: Y={removed_pos}px")
            
            if self.selected_line_index >= len(self.anchor_lines):
                self.selected_line_index = max(0, len(self.anchor_lines) - 1)
            
            self.print_anchor_positions()
    
    def move_selected_line(self, direction, step=5):
        """선택된 라인 이동"""
        if not self.anchor_lines or not (0 <= self.selected_line_index < len(self.anchor_lines)):
            return
        
        current_pos = self.anchor_lines[self.selected_line_index]
        
        if direction == 'up':
            new_pos = max(10, current_pos - step)
        else:
            new_pos = min(self.image_height - 10, current_pos + step)
        
        self.anchor_lines[self.selected_line_index] = new_pos
        self.anchor_lines.sort()
        
        self.selected_line_index = self.anchor_lines.index(new_pos)
        
        print(f"📍 Line {self.selected_line_index + 1} 이동: Y={new_pos}px")
    
    def select_next_line(self):
        """다음 라인 선택"""
        if self.anchor_lines:
            self.selected_line_index = (self.selected_line_index + 1) % len(self.anchor_lines)
            selected_y = self.anchor_lines[self.selected_line_index]
            print(f"🎯 Line {self.selected_line_index + 1} 선택됨 (Y={selected_y}px)")
    
    def select_prev_line(self):
        """이전 라인 선택"""
        if self.anchor_lines:
            self.selected_line_index = (self.selected_line_index - 1) % len(self.anchor_lines)
            selected_y = self.anchor_lines[self.selected_line_index]
            print(f"🎯 Line {self.selected_line_index + 1} 선택됨 (Y={selected_y}px)")
    
    def toggle_edit_mode(self):
        """편집 모드 토글"""
        self.edit_mode = not self.edit_mode
        mode_text = "편집 모드" if self.edit_mode else "일반 모드"
        print(f"🔧 {mode_text} 전환")
        if self.edit_mode:
            self.print_anchor_positions()
    
    def print_anchor_status(self):
        """Anchor 상태 출력"""
        print(f"\n📏 Anchor Lines 상태 (이미지 높이: {self.image_height}px):")
        for i, pos in enumerate(self.anchor_lines):
            marker = "👉" if i == self.selected_line_index else "  "
            print(f"{marker} Line {i+1}: Y={pos}px")
        print(f"편집 모드: {'ON' if self.edit_mode else 'OFF'}")
        print()
    
    def print_anchor_positions(self):
        """Anchor 위치 출력"""
        if self.anchor_lines:
            positions_str = ", ".join([f"Y{pos}" for pos in self.anchor_lines])
            print(f"📍 Anchor 위치: {positions_str}")
    
    def save_anchor_config(self, filename="anchor_config.json"):
        """Anchor 설정 저장"""
        config = {
            'anchor_lines': self.anchor_lines,
            'show_anchor_lines': self.show_anchor_lines,
            'image_height': self.image_height,
            'format': 'pixels_from_top',
            'model_type': 'onnx'
        }
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Anchor 설정 저장: {filename}")
            self.print_anchor_positions()
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    def load_anchor_config(self, filename="anchor_config.json"):
        """Anchor 설정 로드"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            if config.get('format') == 'pixels_from_top':
                self.anchor_lines = config.get('anchor_lines', [])
            else:
                old_ratios = config.get('anchor_lines', [])
                self.anchor_lines = [int(ratio * self.image_height) for ratio in old_ratios]
                print("⚡ 기존 비율 설정을 픽셀 값으로 변환했습니다.")
            
            self.show_anchor_lines = config.get('show_anchor_lines', True)
            self.selected_line_index = 0
            print(f"📂 Anchor 설정 로드: {filename}")
            self.print_anchor_positions()
            
        except FileNotFoundError:
            print(f"⚠️  {filename} 파일이 없어서 기본 설정을 사용합니다.")
            self.initialize_default_anchor_lines()
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            self.initialize_default_anchor_lines()
    
    def find_available_onnx_models(self, base_path):
        """사용 가능한 ONNX 모델 찾기"""
        training_dir = os.path.join(base_path, "DSS_AI_training")
        available_models = []
        
        if os.path.exists(training_dir):
            for experiment in os.listdir(training_dir):
                onnx_model_path = os.path.join(training_dir, experiment, "weights", "onnx_models", "best.onnx")
                if os.path.exists(onnx_model_path):
                    available_models.append({
                        'name': f"{experiment}_onnx",
                        'path': onnx_model_path,
                        'size': os.path.getsize(onnx_model_path) / (1024*1024)
                    })
        
        return available_models
    
    def switch_onnx_model(self, key):
        """ONNX 모델 전환"""
        base_path = r"C:\Project\DSS\AI_Academy\yolov8"
        models = self.find_available_onnx_models(base_path)
        
        if key >= ord('1') and key <= ord('9'):
            model_index = key - ord('1')
            if model_index < len(models):
                new_model_path = models[model_index]['path']
                print(f"🔄 ONNX 모델 전환 중: {models[model_index]['name']}")
                
                old_detector = self.lane_detector
                self.lane_detector = YOLOv8ONNXLaneDetector(new_model_path)
                
                if self.lane_detector.session is not None:
                    print(f"✅ ONNX 모델 전환 완료: {models[model_index]['name']}")
                    del old_detector
                else:
                    print(f"❌ ONNX 모델 전환 실패, 기존 모델 유지")
                    self.lane_detector = old_detector
    
    def normalize_angle(self, angle):
        """각도 정규화 (–π ~ +π 범위)"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def compute_stanley_control(self, center_points, vehicle_position_x, heading, k=0.5):
        """Stanley Control 알고리즘 (원본과 동일)"""
        if len(center_points) < 3:
            return 0.0  # insufficient anchor points

        target_point = center_points[2]  # (x, y)

        dx = target_point[0] - vehicle_position_x  # 좌우 오차 (가로 방향)
        dy = self.image_height - target_point[1]   # 전방 거리 (세로 방향, 위->아래 flip 보정)

        print(f"[Stanley Debug] vehicle_x: {vehicle_position_x}, target_x: {target_point[0]}, dx: {dx}")
        print(f"[Stanley Debug] vehicle_x: {vehicle_position_x}, target_point: {target_point}, dx: {dx}")

        path_angle = np.arctan2(dx, 3*dy)  # ⬅️ 이제 dy가 양수
        heading_error = path_angle - heading

        heading_error = self.normalize_angle(path_angle - heading)

        cross_track_error = dx / max(1e-3, np.hypot(dx, dy))
        steer = heading_error + np.arctan2(k * cross_track_error, 200.0)
        steer_deg = np.degrees(steer)

         # 디버깅 출력 추가
        print(f"[Stanley Debug] dx: {dx:.2f}, dy: {dy:.2f}, path_angle(deg): {np.degrees(path_angle):.2f}, heading(deg): {np.degrees(heading):.2f}, heading_error(deg): {np.degrees(heading_error):.2f}")
        print(f"[Stanley] dx: {dx:.2f}, dy: {dy:.2f}, cross_track_error: {cross_track_error:.4f}, heading_error: {heading_error:.4f}, steer(deg): {steer_deg:.4f}")

        steer = np.clip(steer, -np.radians(30), np.radians(30))  # 제한 각도
        return steer

    def draw_anchor_lines_and_center(self, image, mask):
        """Anchor lines 그리기 및 중심점 계산"""
        if mask is None or mask.sum() == 0:
            return image, []

        result_image = image.copy()
        height, width = image.shape[:2]

        self.update_image_height(height)
        car_region = self.lane_detector.detect_car_region(image)
        y_positions = self.get_anchor_y_positions(height)

        center_points = []

        for i, y in enumerate(y_positions):
            if y < 0 or y >= height:
                continue

            line_mask = mask[y, :]
            lane_pixels = np.where(line_mask > 0)[0]

            if len(lane_pixels) > 0:
                left_most = lane_pixels[0]
                right_most = lane_pixels[-1]
                center_x = (left_most + right_most) // 2
                center_points.append((center_x, y))

                line_color = (0, 255, 255) if self.edit_mode and i == self.selected_line_index else (0, 0, 255)
                line_thickness = 4 if self.edit_mode and i == self.selected_line_index else 2
                self.lane_detector.draw_anchor_line_excluding_car(result_image, y, width, car_region, line_color, line_thickness)

                point_color = (255, 255, 0) if self.edit_mode and i == self.selected_line_index else (0, 255, 0)
                cv2.circle(result_image, (center_x, y), 4, point_color, -1)

                if self.edit_mode:
                    cv2.putText(result_image, f"L{i+1}:Y{y}", (center_x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if len(center_points) >= 2:
            for i in range(len(center_points) - 1):
                cv2.line(result_image, center_points[i], center_points[i + 1], (0, 255, 0), 3)

        return result_image, center_points
    
    def handle_key_control(self, key):
        """키보드 입력 처리"""
        step = 0.05
        
        if key >= ord('1') and key <= ord('9'):
            self.switch_onnx_model(key)
            return
        
        if self.edit_mode:
            if key == ord('w') or key == 72:
                self.move_selected_line('up')
                return
            elif key == ord('s') or key == 80:
                self.move_selected_line('down')
                return
            elif key == ord('d') or key == 77:
                self.select_next_line()
                return
            elif key == ord('a') or key == 75:
                self.select_prev_line()
                return
            elif key == ord('q'):
                self.add_anchor_line()
                return
            elif key == ord('e'):
                self.remove_anchor_line()
                return
            elif key == ord('p'):
                self.print_anchor_status()
                return
        
        if key == ord('=') or key == ord('+'):
            self.add_anchor_line()
            return
        elif key == ord('-') or key == ord('_'):
            self.remove_anchor_line()
            return
        elif key == ord('t'):
            self.show_anchor_lines = not self.show_anchor_lines
            status = "ON" if self.show_anchor_lines else "OFF"
            print(f"[ANCHOR] Anchor lines: {status}")
            return
        elif key == ord('g'):
            self.toggle_edit_mode()
            return
        elif key == ord('h'):
            self.print_help()
            return
        elif key == ord('m'):
            self.save_anchor_config()
            return
        elif key == ord('n'):
            self.load_anchor_config()
            return
        
        if not self.edit_mode:
            if key == ord('w'):
                self.drive_state['throttle'] = min(1.0, self.drive_state['throttle'] + step)
            elif key == ord('s'):
                self.drive_state['throttle'] = max(0.0, self.drive_state['throttle'] - step)
            elif key == ord('j'):
                self.drive_state['steer'] = max(-1.0, self.drive_state['steer'] - step)
            elif key == ord('l'):
                self.drive_state['steer'] = min(1.0, self.drive_state['steer'] + step)
            elif key == ord('k'):
                self.drive_state['steer'] = 0.0
            elif key == ord('x'):
                self.drive_state['brake'] = min(1.0, self.drive_state['brake'] + step)
            elif key == ord('z'):
                self.drive_state['brake'] = max(0.0, self.drive_state['brake'] - step)
            elif key == ord('r'):
                self.drive_state = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}
            
            print(f"[CONTROL] T:{self.drive_state['throttle']:.2f} S:{self.drive_state['steer']:.2f} B:{self.drive_state['brake']:.2f}")
    
    def print_help(self):
        """도움말 출력"""
        print("\n" + "="*60)
        print("🎮 ONNX AI Pilot 키보드 조작법")
        print("="*60)
        print("【차량 제어】")
        print("  W/S: 가속/감속")
        print("  J/L: 좌/우 조향")
        print("  K: 조향 중앙")
        print("  X/Z: 브레이크")
        print("  R: 모든 제어 리셋")
        print()
        print("【ONNX 모델 관리】")
        print("  1-9: ONNX 모델 전환")
        print()
        print("【Anchor Line 제어】")
        print("  T: Anchor line 표시 토글")
        print("  G: 편집 모드 토글")
        print("  +: 라인 추가")
        print("  -: 라인 제거")
        print("  M: 설정 저장")
        print("  N: 설정 로드")
        print("  H: 도움말")
        print()
        print("【편집 모드에서】")
        print("  W/S 또는 ↑/↓: 선택된 라인 위/아래 이동 (5px 단위)")
        print("  A/D 또는 ←/→: 이전/다음 라인 선택")
        print("  Q: 라인 추가")
        print("  E: 선택된 라인 제거")
        print("  P: 현재 상태 출력")
        print("="*60)
        print()
    
    def calculate_fps(self):
        """FPS 계산"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_overlay_info(self, image, inference_time=0.0):
        """이미지에 정보 오버레이 추가"""
        # FPS 표시
        cv2.putText(image, f'FPS: {self.fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ONNX 추론 시간 표시
        cv2.putText(image, f'ONNX Inf: {inference_time*1000:.1f}ms', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 평균 추론 시간 표시
        avg_inference = self.lane_detector.get_average_inference_time()
        cv2.putText(image, f'Avg: {avg_inference*1000:.1f}ms', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 차량 상태 표시
        info_text = f'T:{self.drive_state["throttle"]:.2f} S:{self.drive_state["steer"]:.2f} B:{self.drive_state["brake"]:.2f}'
        cv2.putText(image, info_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Anchor line 정보 표시
        anchor_info = f'Lines: {len(self.anchor_lines)} | Mode: {"EDIT" if self.edit_mode else "DRIVE"}'
        if self.edit_mode and self.anchor_lines:
            anchor_info += f' | Selected: {self.selected_line_index + 1}'
        cv2.putText(image, anchor_info, (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # ONNX AI Pilot 표시
        cv2.putText(image, 'ONNX AI PILOT', (image.shape[1] - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # 조작법 표시
        if self.edit_mode:
            cv2.putText(image, 'EDIT: W/S:Move A/D:Select Q:Add E:Del P:Status G:Exit H:Help', 
                       (10, image.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(image, 'ESC:Exit', 
                       (10, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(image, 'W/S:Throttle J/L:Steer K:Center X/Z:Brake R:Reset', 
                       (10, image.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(image, '1-9:ONNX Model T:Toggle G:Edit +/-:Lines M/N:Save/Load H:Help ESC:Exit', 
                       (10, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image
    
    def on_camera_data(self, image: dss_pb2.DSSImage):
        """카메라 데이터 콜백 (ONNX AI Pilot)"""
        if not self.running:
            return
        
        start_time = time.time()
        
        try:
            jpg_data = np.frombuffer(image.data, dtype=np.uint8)
            self.rgb_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)

            if self.rgb_image is None:
                return

            lane_detected_image = self.rgb_image
            inference_time = 0.0

            if self.lane_detector.session is not None:
                # ONNX 차선 감지
                combined_mask, inference_time = self.lane_detector.detect_lanes_onnx(self.rgb_image)
                
                # 성능 통계 업데이트
                self.total_inference_time += inference_time
                self.frame_count += 1
                
                if combined_mask is not None and combined_mask.sum() > 0:
                    overlay_image = self.lane_detector.create_blue_overlay(self.rgb_image, combined_mask, alpha=0.1)

                    if self.show_anchor_lines:
                        lane_detected_image, center_points = self.draw_anchor_lines_and_center(overlay_image, combined_mask)

                        # Stanley Control 적용
                        if len(center_points) >= 3:
                            vehicle_x = self.rgb_image.shape[1] // 2
                            heading = 0.0  # 정면 기준
                            steer_rad = self.compute_stanley_control(center_points, vehicle_x, heading)
                            steer_deg = np.degrees(steer_rad)
                            steer_cmd = steer_deg * 0.17  # 차량용 스케일 조향값
                            self.drive_state['steer'] = steer_cmd

                            # 디버깅 출력
                            print(f"[ONNX Stanley] steer_deg: {steer_deg:.2f}, steer_cmd: {steer_cmd:.2f}")
                        else:
                            lane_detected_image = overlay_image

            final_image = self.add_overlay_info(lane_detected_image, inference_time)
            self.calculate_fps()

            cv2.imshow('DSS Camera + ONNX AI Pilot', final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
                cv2.destroyAllWindows()
                
                # 최종 성능 통계 출력
                if self.frame_count > 0:
                    avg_inference = self.total_inference_time / self.frame_count
                    print(f"\n📊 ONNX AI Pilot 최종 성능 통계:")
                    print(f"   총 프레임: {self.frame_count}")
                    print(f"   평균 추론 시간: {avg_inference*1000:.2f}ms")
                    print(f"   최대 FPS: {1.0/avg_inference:.1f}")
                
            elif key != 255:
                self.last_key_pressed = chr(key) if 32 <= key <= 126 else str(key)
                self.handle_key_control(key)

        except Exception as e:
            print(f"❌ ONNX 카메라 데이터 처리 오류: {e}")
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            print(f"[⏱️ ONNX 처리 시간] on_camera_data duration: {duration_ms:.2f} ms")
    
    def on_lidar_data(self, lidar_data):
        """라이다 데이터 콜백"""
        if not self.running:
            return
    
    def on_imu_data(self, imu):
        """IMU 데이터 콜백"""
        if not self.running:
            return
    
    def on_gps_data(self, gps):
        """GPS 데이터 콜백"""
        if not self.running:
            return
    
    def on_odom_data(self, odom):
        """오도메트리 데이터 콜백"""
        if not self.running:
            return
    
    def on_ground_truth_data(self, gt_data):
        """그라운드 트루스 데이터 콜백"""
        if not self.running:
            return
    
    def on_sim_started(self):
        """시뮬레이션 시작 콜백"""
        print("🟢 시뮬레이션 시작!")
    
    def on_sim_ended(self):
        """시뮬레이션 종료 콜백"""
        print("🔴 시뮬레이션 종료!")
        self.running = False
    
    def on_sim_aborted(self):
        """시뮬레이션 중단 콜백"""
        print("⚠️ 시뮬레이션 중단!")
        self.running = False
    
    def on_sim_error(self):
        """시뮬레이션 오류 콜백"""
        print("❌ 시뮬레이션 오류!")
        self.running = False
    
    def update_vehicle_control(self):
        """차량 제어 업데이트"""
        if not self.running or not self.dss_instance:
            return
        
        control = DSSSDKCarControl(
            throttle=self.drive_state['throttle'],
            steer=self.drive_state['steer'],
            brake=self.drive_state['brake'],
            park_brake=False,
            target_gear=1
        )
        
        try:
            with SuppressOutput():
                self.dss_instance.set_car_control(control)
        except Exception as e:
            print(f"❌ 차량 제어 오류: {e}")
    
    def signal_handler(self, sig, frame):
        """시그널 핸들러"""
        print("\n🛑 ONNX AI Pilot 종료 중...")
        self.running = False
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
    
    def run(self, server_ip="127.0.0.1"):
        """메인 실행 함수"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        os.environ['PYTHONPATH'] = ''
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print(f"🔗 DSS 서버 연결 중: {server_ip}")
            
            with SuppressOutput():
                dss = IDSSSDK.create(
                    loop=loop,
                    nats_address=f'nats://{server_ip}:4222'
                )
                self.dss_instance = dss
                
                init_params = DSSSDKInitParams(
                    server=server_ip,
                    heartbeat_port=8886,
                    nats_port=4222
                )
                
                dss.initialize(init_params)
                dss.register_sensor_callback('camera', self.on_camera_data)
                dss.register_simulation_callback('started', self.on_sim_started)
                dss.register_simulation_callback('ended', self.on_sim_ended)
                dss.register_simulation_callback('aborted', self.on_sim_aborted)
                dss.register_simulation_callback('error', self.on_sim_error)
                
                dss.start()
            
            print("✅ DSS 연결 완료!")
            print("🎮 ONNX AI Pilot 키보드 조작법:")
            print("   W/S: 가속/감속")
            print("   J/L: 좌회전/우회전") 
            print("   K: 조향 중앙")
            print("   X/Z: 브레이크")
            print("   R: 리셋")
            print("   1-9: ONNX 모델 전환")
            print("   T: Anchor line 표시 토글")
            print("   G: 편집 모드 전환")
            print("   +/-: Anchor line 추가/제거")
            print("   M/N: 설정 저장/로드")
            print("   H: 도움말")
            print("   ESC: 종료")
            print("=" * 50)
            
            while self.running:
                self.update_vehicle_control()
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⌨️  키보드 인터럽트로 종료")
            self.running = False
        except Exception as e:
            print(f"❌ 실행 중 오류: {e}")
            self.running = False
        finally:
            if self.dss_instance:
                try:
                    with SuppressOutput():
                        self.dss_instance.cleanup()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            print("🏁 ONNX AI Pilot 종료 완료")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚗 DSS + YOLOv8 ONNX AI Pilot 자율주행 시스템")
    print("=" * 60)
    
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    EXPERIMENT_NAME = "DSS_experiment_1"
    MODEL_PATH = os.path.join(BASE_PATH, "DSS_AI_training", EXPERIMENT_NAME, "weights", "onnx_models", "best.onnx")
    
    print(f"📂 ONNX 모델 경로: {MODEL_PATH}")
    
    def find_onnx_models(base_path):
        """ONNX 모델 찾기"""
        training_dir = os.path.join(base_path, "DSS_AI_training")
        available_models = []
        if os.path.exists(training_dir):
            for experiment in os.listdir(training_dir):
                onnx_model_path = os.path.join(training_dir, experiment, "weights", "onnx_models", "best.onnx")
                if os.path.exists(onnx_model_path):
                    available_models.append({
                        'name': f"{experiment}_onnx",
                        'path': onnx_model_path,
                        'size': os.path.getsize(onnx_model_path) / (1024*1024)
                    })
        return available_models
    
    available_models = find_onnx_models(BASE_PATH)
    
    if available_models:
        print(f"\n📋 사용 가능한 ONNX 모델들:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}: {model['name']} ({model['size']:.1f}MB)")
        print("   실행 중 1-9 키로 ONNX 모델 전환 가능")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 YOLOv8 모델을 ONNX로 변환해주세요.")
        return
    
    server_ip = "127.0.0.1"
    
    try:
        controller = DSSYOLOONNXController(MODEL_PATH)
        controller.run(server_ip)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    finally:
        print("👋 ONNX AI Pilot 시스템 종료")

if __name__ == "__main__":
    main()