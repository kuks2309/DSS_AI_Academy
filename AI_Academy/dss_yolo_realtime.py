
# dss_yolo_realtime.py

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

# YOLOv8 관련 import
from ultralytics import YOLO

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

class YOLOv8LaneDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        """
        YOLOv8 차선 감지 클래스
        
        Args:
            model_path (str): 학습된 모델 경로
            conf_threshold (float): 신뢰도 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.inference_time = 0.0  # 추론 시간 저장
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            print(f"📥 YOLOv8 모델 로드 중: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ YOLOv8 모델 로드 완료!")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
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
        
        if mask is not None and mask.sum() > 0:
            # 마스크를 3채널로 확장
            if len(mask.shape) == 2:
                mask_3d = np.stack([mask] * 3, axis=-1)
            else:
                mask_3d = mask
            
            # 파란색 레이어 생성
            blue_layer = np.zeros_like(image)
            blue_layer[:, :] = blue_color
            
            # 마스크 영역에만 파란색 적용 (알파 블렌딩)
            mask_bool = mask_3d > 0
            overlay_image[mask_bool] = (
                alpha * blue_layer[mask_bool] + 
                (1 - alpha) * overlay_image[mask_bool]
            ).astype(np.uint8)
        
        return overlay_image
    
    def detect_lanes(self, image):
        """
        이미지에서 차선 감지 및 오버레이 적용
        
        Args:
            image (np.array): 입력 이미지
            
        Returns:
            np.array: 차선이 표시된 이미지
        """
        if self.model is None:
            return image
        
        try:
            # 추론 시간 측정 시작
            inference_start = time.time()
            
            # 모델 추론
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            # 추론 시간 측정 종료
            self.inference_time = (time.time() - inference_start) * 1000  # ms 단위
            
            if not results or not results[0].masks:
                return image
            
            # 마스크 추출 및 합치기
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            for mask in masks:
                # 마스크를 원본 이미지 크기로 리사이즈
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
            
            # 파란색 오버레이 적용
            overlay_image = self.create_blue_overlay(image, combined_mask, alpha=0.4)
            
            return overlay_image
            
        except Exception as e:
            print(f"❌ 차선 감지 오류: {e}")
            return image

class DSSYOLOController:
    def __init__(self, model_path):
        """
        DSS + YOLO 통합 컨트롤러
        
        Args:
            model_path (str): YOLOv8 모델 경로
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
        
        # YOLOv8 차선 감지기 초기화
        self.lane_detector = YOLOv8LaneDetector(model_path)
        
        # 성능 측정 변수
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
        print("🚗 DSS + YOLOv8 실시간 차선 감지 시스템 초기화 완료!")
    
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
        """FPS 계산 및 성능 정보 출력"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            print(f"[PERFORMANCE] FPS: {self.fps} | Inference: {self.lane_detector.inference_time:.1f}ms")
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_overlay_info(self, image):
        """이미지에 정보 오버레이 추가"""
        # FPS 표시
        cv2.putText(image, f'FPS: {self.fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 추론 시간 표시
        inference_text = f'Inference: {self.lane_detector.inference_time:.1f}ms'
        cv2.putText(image, inference_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 차량 상태 표시
        info_text = f'T:{self.drive_state["throttle"]:.2f} S:{self.drive_state["steer"]:.2f} B:{self.drive_state["brake"]:.2f}'
        cv2.putText(image, info_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 조작법 표시
        cv2.putText(image, 'W/S:Throttle J/L:Steer K:Center X/Z:Brake R:Reset ESC:Exit', 
                   (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image
    
    def on_camera_data(self, image: dss_pb2.DSSImage):
        """카메라 데이터 콜백 (차선 감지 포함)"""
        if not self.running:
            return
        
        try:
            # 이미지 디코딩
            jpg_data = np.frombuffer(image.data, dtype=np.uint8)
            self.rgb_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
            
            if self.rgb_image is None:
                return
            
            # YOLOv8 차선 감지 적용
            lane_detected_image = self.lane_detector.detect_lanes(self.rgb_image)
            
            # 정보 오버레이 추가
            final_image = self.add_overlay_info(lane_detected_image)
            
            # FPS 계산
            self.calculate_fps()
            
            # 화면 표시
            cv2.imshow('DSS Camera + Lane Detection', final_image)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.running = False
                cv2.destroyAllWindows()
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
    print("🚗 DSS + YOLOv8 실시간 차선 감지 주행 시스템")
    print("=" * 60)
    
    # 모델 경로 설정
    BASE_PATH = r"C:\Project\DSS\AI_Academy\yolov8"
    MODEL_PATH = os.path.join(BASE_PATH, "DSS_AI_training", "DSS_experiment_1", "weights", "best.pt")
    
    print(f"📂 모델 경로: {MODEL_PATH}")
    
    # 모델 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 YOLOv8 모델을 훈련해주세요.")
        return
    
    # DSS 서버 IP (필요시 변경)
    server_ip = "127.0.0.1"
    
    try:
        # 컨트롤러 생성 및 실행
        controller = DSSYOLOController(MODEL_PATH)
        controller.run(server_ip)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    finally:
        print("👋 시스템 종료")

if __name__ == "__main__":
    main()