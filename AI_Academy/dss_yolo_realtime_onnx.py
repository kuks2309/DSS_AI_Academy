# dss_yolo_realtime_onnx.py

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
        YOLOv8 ONNX 차선 감지 클래스
        
        Args:
            model_path (str): ONNX 모델 경로
            conf_threshold (float): 신뢰도 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.execution_mode = "CPU"  # 현재 실행 모드 추가
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
            
            # ⚠️ 현재 CPU 모드로 설정됨 ⚠️
            # ONNX Runtime 세션 생성 - CPU 모드
            #self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            #self.execution_mode = "CPU"
            
            # GPU 모드로 변경하려면 아래 코드로 교체:
            self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.execution_mode = "GPU"
            
            # 입출력 정보 가져오기
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 현재 사용 중인 프로바이더 확인
            used_providers = self.session.get_providers()
            print("✅ YOLOv8 ONNX 모델 로드 완료!")
            print(f"🔧 실행 모드: {self.execution_mode} MODE")
            print(f"🖥️  사용 중인 프로바이더: {used_providers}")
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
        ONNX 모델 출력 후처리
        
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
            
            # 적응형 임계값으로 유효한 검출 찾기
            thresholds = [0.001, 0.005, 0.01, 0.05]
            best_results = None
            
            for threshold in thresholds:
                valid_mask = class_confidences > threshold
                valid_count = valid_mask.sum()
                
                if valid_count > 0 and valid_count < 50:  # 실시간용으로 더 적은 수
                    best_results = valid_mask
                    break
            
            # 유효한 검출이 없으면 상위 3개만 시도 (실시간 성능 우선)
            if best_results is None or best_results.sum() == 0:
                top_3_indices = np.argsort(class_confidences)[::-1][:3]
                best_results = np.zeros(len(class_confidences), dtype=bool)
                best_results[top_3_indices] = True
            
            detection_count = best_results.sum()
            
            if detection_count == 0:
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            # 유효한 검출 추출
            valid_mask_coeffs = mask_coeffs[best_results]
            
            # 세그멘테이션 마스크 생성
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            proto_masks = mask_protos[0]  # (32, 160, 160)
            
            # 최대 5개 객체만 처리 (실시간 성능 보장)
            max_objects = min(len(valid_mask_coeffs), 5)
            
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
            print(f"❌ 후처리 오류: {e}")
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
    
    def get_average_inference_time(self):
        """평균 추론 시간 계산"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def detect_lanes(self, image):
        """
        이미지에서 차선 감지 및 오버레이 적용
        
        Args:
            image (np.array): 입력 이미지
            
        Returns:
            tuple: (차선이 표시된 이미지, 추론 시간)
        """
        if self.session is None:
            return image, 0.0
        
        inference_start = time.time()
        
        try:
            # 전처리
            input_tensor, (orig_w, orig_h) = self.preprocess_image(image)
            
            # ONNX 추론
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # 후처리
            mask = self.postprocess_detection(outputs, orig_w, orig_h)
            
            # 파란색 오버레이 적용
            overlay_image = self.create_blue_overlay(image, mask, alpha=0.4)
            
            # 추론 시간 기록
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # 최대 기록 수 제한
            if len(self.inference_times) > self.max_inference_history:
                self.inference_times.pop(0)
            
            return overlay_image, inference_time
            
        except Exception as e:
            print(f"❌ ONNX 차선 감지 오류: {e}")
            return image, 0.0

class DSSYOLOONNXController:
    def __init__(self, model_path):
        """
        DSS + YOLOv8 ONNX 통합 컨트롤러
        
        Args:
            model_path (str): YOLOv8 ONNX 모델 경로
        """
        self.running = True
        self.rgb_image = None
        self.dss_instance = None
        self.last_key_pressed = None
        
        # 차량 제어 상태
        self.drive_state = {
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 0.0
        }
        
        # YOLOv8 ONNX 차선 감지기 초기화
        self.lane_detector = YOLOv8ONNXLaneDetector(model_path)
        
        # 성능 측정 변수
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.total_inference_time = 0.0
        self.frame_count = 0
        
        print("🚗 DSS + YOLOv8 ONNX 실시간 차선 감지 시스템 초기화 완료!")
    
    def handle_key_control(self, key):
        """키보드 입력 처리"""
        step = 0.05
        
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
        
        # 추론 시간 표시
        cv2.putText(image, f'Inference: {inference_time*1000:.1f}ms', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 평균 추론 시간 표시
        avg_inference = self.lane_detector.get_average_inference_time()
        cv2.putText(image, f'Avg Inf: {avg_inference*1000:.1f}ms', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 차량 상태 표시
        info_text = f'T:{self.drive_state["throttle"]:.2f} S:{self.drive_state["steer"]:.2f} B:{self.drive_state["brake"]:.2f}'
        cv2.putText(image, info_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 현재 실행 모드 표시 (CPU/GPU)
        mode_color = (0, 255, 255) if self.lane_detector.execution_mode == "CPU" else (0, 255, 0)
        cv2.putText(image, f'{self.lane_detector.execution_mode} MODE', (image.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # 조작법 표시
        cv2.putText(image, 'W/S:Throttle J/L:Steer K:Center X/Z:Brake R:Reset ESC:Exit', 
                   (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image
    
    def on_camera_data(self, image: dss_pb2.DSSImage):
        """카메라 데이터 콜백 (ONNX 차선 감지 포함)"""
        if not self.running:
            return
        
        try:
            # 이미지 디코딩
            jpg_data = np.frombuffer(image.data, dtype=np.uint8)
            self.rgb_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
            
            if self.rgb_image is None:
                return
            
            # YOLOv8 ONNX 차선 감지 적용
            lane_detected_image, inference_time = self.lane_detector.detect_lanes(self.rgb_image)
            
            # 성능 통계 업데이트
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            # 정보 오버레이 추가
            final_image = self.add_overlay_info(lane_detected_image, inference_time)
            
            # FPS 계산
            self.calculate_fps()
            
            # 화면 표시
            cv2.imshow('DSS Camera + ONNX Lane Detection', final_image)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.running = False
                cv2.destroyAllWindows()
                
                # 최종 성능 통계 출력
                if self.frame_count > 0:
                    avg_inference = self.total_inference_time / self.frame_count
                    print(f"\n📊 최종 성능 통계:")
                    print(f"   총 프레임: {self.frame_count}")
                    print(f"   평균 추론 시간: {avg_inference*1000:.2f}ms")
                    print(f"   최대 FPS: {1.0/avg_inference:.1f}")
                
            elif key != 255:
                self.last_key_pressed = chr(key) if 32 <= key <= 126 else str(key)
                self.handle_key_control(key)
                
        except Exception as e:
            print(f"❌ 카메라 데이터 처리 오류: {e}")
    
    def on_lidar_data(self, lidar_data: dss_pb2.DssLidarPointCloud):
        """라이다 데이터 콜백"""
        if not self.running:
            return
        # 필요시 라이다 데이터 처리 구현
    
    def on_imu_data(self, imu: dss_pb2.DSSIMU):
        """IMU 데이터 콜백"""
        if not self.running:
            return
        # 필요시 IMU 데이터 처리 구현
    
    def on_gps_data(self, gps: dss_pb2.DSSGPS):
        """GPS 데이터 콜백"""
        if not self.running:
            return
        # 필요시 GPS 데이터 처리 구현
    
    def on_odom_data(self, odom: dss_pb2.DSSOdom):
        """오도메트리 데이터 콜백"""
        if not self.running:
            return
        # 필요시 오도메트리 데이터 처리 구현
    
    def on_ground_truth_data(self, gt_data):
        """그라운드 트루스 데이터 콜백"""
        if not self.running:
            return
        # 필요시 그라운드 트루스 데이터 처리 구현
    
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
        print("\n🛑 프로그램 종료 중...")
        self.running = False
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
    
    def run(self, server_ip="127.0.0.1"):
        """메인 실행 함수"""
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 환경 설정
        os.environ['PYTHONPATH'] = ''
        
        # 비동기 이벤트 루프 설정
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print(f"🔗 DSS 서버 연결 중: {server_ip}")
            
            with SuppressOutput():
                # DSS SDK 초기화
                dss = IDSSSDK.create(
                    loop=loop,
                    nats_address=f'nats://{server_ip}:4222'
                )
                self.dss_instance = dss
                
                # 초기화 파라미터 설정
                init_params = DSSSDKInitParams(                   
                    server=server_ip,
                    heartbeat_port=8886,                    
                    nats_port=4222
                )
                
                # DSS 초기화 및 콜백 등록
                dss.initialize(init_params)
                dss.register_sensor_callback('camera', self.on_camera_data)
                dss.register_sensor_callback('lidar', self.on_lidar_data)
                dss.register_sensor_callback('imu', self.on_imu_data)
                dss.register_sensor_callback('gps', self.on_gps_data)
                dss.register_sensor_callback('odom', self.on_odom_data)
                dss.register_sensor_callback('ground_truth', self.on_ground_truth_data)
                dss.register_simulation_callback('started', self.on_sim_started)
                dss.register_simulation_callback('ended', self.on_sim_ended)
                dss.register_simulation_callback('aborted', self.on_sim_aborted)
                dss.register_simulation_callback('error', self.on_sim_error)
                
                # DSS 시작
                dss.start()
            
            print("✅ DSS 연결 완료!")
            print("🎮 키보드 조작법:")
            print("   W/S: 가속/감속")
            print("   J/L: 좌회전/우회전") 
            print("   K: 조향 중앙")
            print("   X/Z: 브레이크")
            print("   R: 리셋")
            print("   ESC: 종료")
            print("=" * 50)
            
            # 메인 루프
            while self.running:
                self.update_vehicle_control()
                time.sleep(0.1)  # 10Hz 제어 주기
                
        except KeyboardInterrupt:
            print("\n⌨️  키보드 인터럽트로 종료")
            self.running = False
        except Exception as e:
            print(f"❌ 실행 중 오류: {e}")
            self.running = False
        finally:
            # 정리
            if self.dss_instance:
                try:
                    with SuppressOutput():
                        self.dss_instance.cleanup()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            print("🏁 프로그램 종료 완료")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚗 DSS + YOLOv8 ONNX 실시간 차선 감지 주행 시스템")
    print("=" * 60)
    
    # ONNX 모델 경로 설정
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    MODEL_PATH = os.path.join(BASE_PATH, "DSS_AI_training", "DSS_experiment_1", "weights", "onnx_models", "best.onnx")
    
    print(f"📂 ONNX 모델 경로: {MODEL_PATH}")
    
    # ONNX 모델 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 YOLOv8 모델을 ONNX로 변환해주세요.")
        return
    
    # DSS 서버 IP (필요시 변경)
    server_ip = "127.0.0.1"
    
    try:
        # ONNX 컨트롤러 생성 및 실행
        controller = DSSYOLOONNXController(MODEL_PATH)
        controller.run(server_ip)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    finally:
        print("👋 ONNX 시스템 종료")

if __name__ == "__main__":
    main()