#DSS_pilotnet_drive.py
import sys
import time
import asyncio
import cv2
import numpy as np
import signal
import logging
import os
import torch
import torch.nn as nn

# DSS SDK 관련 import
from dss_sdk.core.idsssdk import IDSSSDK
from dss_sdk.config.sdk_config import *
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

# PilotNet 모델 정의 (학습 시와 동일)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class PilotNetController:
    def __init__(self, model_path):
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
        
        # PilotNet 설정
        self.auto_driving_enabled = False
        self.steering_multiplier = 1.0  # 조향 감도 조정
        
        # 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # PilotNet 모델 로드
        self.model = None
        self.load_pilotnet_model(model_path)
        
        # 성능 측정
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.inference_time = 0
        
        # 크롭 영역 표시
        self.show_crop_area = True
        
        print("🚗 DSS PilotNet 자동 주행 시스템 초기화 완료!")
    
    def load_pilotnet_model(self, model_path):
        """PilotNet 모델 로드"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
                return False
            
            print(f"🧠 PilotNet 모델 로드 중: {model_path}")
            self.model = PilotNet()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ PilotNet 모델 로드 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def apply_center_crop(self, image):
        """중앙 크롭 적용 (학습 시와 동일)"""
        h, w = image.shape[:2]
        target_height = w // 3
        start_y = (h - target_height) // 2
        end_y = start_y + target_height
        cropped = image[start_y:end_y, 0:w]
        return cropped, (0, start_y, w, end_y)
    
    def preprocess_image(self, image):
        """PilotNet 입력을 위한 이미지 전처리"""
        # 중앙 크롭
        cropped, _ = self.apply_center_crop(image)
        
        # 200x66으로 리사이즈
        resized = cv2.resize(cropped, (200, 66))
        
        # RGB 변환 및 정규화
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image / 255.0
        
        # 텐서 변환: (H,W,C) -> (1,C,H,W)
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_steering(self, image):
        """PilotNet으로 조향각 예측"""
        if self.model is None:
            return 0.0
        
        try:
            start_time = time.time()
            
            # 이미지 전처리
            input_tensor = self.preprocess_image(image)
            
            # 모델 추론
            with torch.no_grad():
                steering_prediction = self.model(input_tensor).item()
            
            # 조향 감도 적용
            steering_command = steering_prediction * self.steering_multiplier
            
            # 조향각 제한 (-1.0 ~ 1.0)
            steering_command = np.clip(steering_command, -1.0, 1.0)
            
            self.inference_time = time.time() - start_time
            
            return steering_command
            
        except Exception as e:
            print(f"❌ 조향 예측 오류: {e}")
            return 0.0
    
    def draw_crop_area(self, image):
        """크롭 영역 표시"""
        if not self.show_crop_area:
            return image
        
        display_image = image.copy()
        h, w = image.shape[:2]
        
        # 크롭 영역 계산
        target_height = w // 3
        start_y = (h - target_height) // 2
        end_y = start_y + target_height
        
        # 핑크색 사각형 표시
        color = (255, 20, 147)  # 핑크색 (BGR)
        thickness = 3
        
        cv2.rectangle(display_image, (0, start_y), (w, end_y), color, thickness)
        
        # 모서리 강조
        corner_length = 30
        corner_thickness = thickness + 2
        
        # 4개 모서리에 L자 표시
        points = [(0, start_y), (w, start_y), (0, end_y), (w, end_y)]
        
        for i, (x, y) in enumerate(points):
            if i == 0:  # 왼쪽 상단
                cv2.line(display_image, (x, y), (x + corner_length, y), color, corner_thickness)
                cv2.line(display_image, (x, y), (x, y + corner_length), color, corner_thickness)
            elif i == 1:  # 오른쪽 상단
                cv2.line(display_image, (x, y), (x - corner_length, y), color, corner_thickness)
                cv2.line(display_image, (x, y), (x, y + corner_length), color, corner_thickness)
            elif i == 2:  # 왼쪽 하단
                cv2.line(display_image, (x, y), (x + corner_length, y), color, corner_thickness)
                cv2.line(display_image, (x, y), (x, y - corner_length), color, corner_thickness)
            else:  # 오른쪽 하단
                cv2.line(display_image, (x, y), (x - corner_length, y), color, corner_thickness)
                cv2.line(display_image, (x, y), (x, y - corner_length), color, corner_thickness)
        
        return display_image
    
    def add_overlay_info(self, image):
        """화면 정보 표시"""
        display_image = image.copy()
        
        # FPS 표시
        cv2.putText(display_image, f'FPS: {self.fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # PilotNet 상태 표시
        auto_status = "ON" if self.auto_driving_enabled else "OFF"
        status_color = (0, 255, 0) if self.auto_driving_enabled else (0, 0, 255)
        cv2.putText(display_image, f'PilotNet: {auto_status}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 차량 제어 상태 표시
        info_text = f'T:{self.drive_state["throttle"]:.2f} S:{self.drive_state["steer"]:.3f} B:{self.drive_state["brake"]:.2f}'
        cv2.putText(display_image, info_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 추론 시간 표시
        cv2.putText(display_image, f'Inference: {self.inference_time*1000:.1f}ms', (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 조향 감도 표시 (자동 주행 모드일 때만)
        if self.auto_driving_enabled:
            cv2.putText(display_image, f'Steering Gain: {self.steering_multiplier:.2f}', (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 크롭 영역 정보
        h, w = image.shape[:2]
        target_height = w // 3
        crop_info = f'Crop: {w}x{h} -> {w}x{target_height} -> 200x66'
        cv2.putText(display_image, crop_info, (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 147), 2)
        
        # 조작법 표시
        if self.auto_driving_enabled:
            controls = 'P:Auto OFF  W/S:Throttle  X/Z:Brake  I/O:Gain  C:Crop  R:Reset  ESC:Exit'
        else:
            controls = 'P:Auto ON  W/S:Throttle  J/L:Steer  K:Center  X/Z:Brake  R:Reset  ESC:Exit'
        
        cv2.putText(display_image, controls, (10, display_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return display_image
    
    def handle_key_control(self, key):
        """키보드 입력 처리 (기존 DSS와 동일한 키 맵핑)"""
        step = 0.05
        
        if key == ord('p'):
            # PilotNet 자동 주행 토글 (조향만 자동, 스로틀은 수동)
            self.auto_driving_enabled = not self.auto_driving_enabled
            status = "활성화" if self.auto_driving_enabled else "비활성화"
            print(f"🤖 PilotNet 자동 조향 {status} (스로틀은 수동 제어)")
            return
        
        elif key == ord('c'):
            # 크롭 영역 표시 토글
            self.show_crop_area = not self.show_crop_area
            status = "ON" if self.show_crop_area else "OFF"
            print(f"🔍 크롭 영역 표시: {status}")
            return
        
        # 조향 감도 조정 (자동 주행 모드에서만)
        elif key == ord('i') and self.auto_driving_enabled:
            self.steering_multiplier = max(0.1, self.steering_multiplier - 0.1)
            print(f"🎛️ 조향 감도: {self.steering_multiplier:.2f}")
            return
        elif key == ord('o') and self.auto_driving_enabled:
            self.steering_multiplier = min(3.0, self.steering_multiplier + 0.1)
            print(f"🎛️ 조향 감도: {self.steering_multiplier:.2f}")
            return
        
        # 스로틀 제어 (항상 수동 - 자동/수동 모드 공통)
        if key == ord('w'):
            self.drive_state['throttle'] = min(1.0, self.drive_state['throttle'] + step)
            print(f"[스로틀] {self.drive_state['throttle']:.2f}")
        elif key == ord('s'):
            self.drive_state['throttle'] = max(0.0, self.drive_state['throttle'] - step)
            print(f"[스로틀] {self.drive_state['throttle']:.2f}")
        
        # 브레이크 제어 (항상 수동 - 자동/수동 모드 공통)
        elif key == ord('x'):
            self.drive_state['brake'] = min(1.0, self.drive_state['brake'] + step)
            print(f"[브레이크] {self.drive_state['brake']:.2f}")
        elif key == ord('z'):
            self.drive_state['brake'] = max(0.0, self.drive_state['brake'] - step)
            print(f"[브레이크] {self.drive_state['brake']:.2f}")
        
        # 조향 제어 (자동 주행 OFF 시에만 수동)
        elif not self.auto_driving_enabled:
            if key == ord('j'):
                self.drive_state['steer'] = max(-1.0, self.drive_state['steer'] - step)
                print(f"[조향] {self.drive_state['steer']:.3f}")
            elif key == ord('l'):
                self.drive_state['steer'] = min(1.0, self.drive_state['steer'] + step)
                print(f"[조향] {self.drive_state['steer']:.3f}")
            elif key == ord('k'):
                self.drive_state['steer'] = 0.0
                print(f"[조향] 중앙: {self.drive_state['steer']:.3f}")
        
        # 전체 리셋
        if key == ord('r'):
            self.drive_state = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}
            print(f"[리셋] T:{self.drive_state['throttle']:.2f} S:{self.drive_state['steer']:.3f} B:{self.drive_state['brake']:.2f}")
    
    def calculate_fps(self):
        """FPS 계산"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def on_camera_data(self, image: dss_pb2.DSSImage):
        """카메라 데이터 처리"""
        if not self.running:
            return
        
        try:
            # 이미지 디코딩
            jpg_data = np.frombuffer(image.data, dtype=np.uint8)
            self.rgb_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
            
            if self.rgb_image is None:
                return
            
            # PilotNet 자동 주행
            if self.auto_driving_enabled and self.model is not None:
                steering_prediction = self.predict_steering(self.rgb_image)
                self.drive_state['steer'] = steering_prediction
                # 스로틀과 브레이크는 수동 제어 유지
            
            # 화면 표시
            display_image = self.draw_crop_area(self.rgb_image)
            final_image = self.add_overlay_info(display_image)
            
            self.calculate_fps()
            
            cv2.imshow('DSS PilotNet Autonomous Driving', final_image)
            
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
    
    def update_vehicle_control(self):
        """차량 제어 명령 전송"""
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
                dss.start()
            
            print("✅ DSS 연결 완료!")
            print("🤖 PilotNet 자동 주행 시스템 준비 완료!")
            print()
            print("🎮 조작법:")
            print("   P: PilotNet 자동 조향 ON/OFF (스로틀은 항상 수동)")
            print("   W/S: 스로틀 증가/감소 (모든 모드)")
            print("   X/Z: 브레이크 ON/OFF (모든 모드)")
            print("   자동 모드: I/O(조향 감도) C(크롭 표시)")
            print("   수동 모드: J/L(조향) K(중앙)")
            print("   R: 전체 리셋, ESC: 종료")
            print("=" * 60)
            
            while self.running:
                self.update_vehicle_control()
                time.sleep(0.01)  # 100Hz 제어 루프
                
        except KeyboardInterrupt:
            print("\n⌨️ 키보드 인터럽트로 종료")
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
            print("🏁 프로그램 종료 완료")

def main():
    print("=" * 60)
    print("🚗 DSS PilotNet 자동 주행 시스템")
    print("=" * 60)
    
    # PilotNet 모델 경로
    model_path = "DSS_pilotnet_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ PilotNet 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 PilotNet 모델을 학습해주세요.")
        return
    
    server_ip = "127.0.0.1"
    
    try:
        controller = PilotNetController(model_path)
        controller.run(server_ip)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    finally:
        print("👋 시스템 종료")

if __name__ == "__main__":
    main()