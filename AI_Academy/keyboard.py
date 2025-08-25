#keyboard.py

import sys
import time
import asyncio
import cv2
import numpy as np
import json
import signal
import logging
import os

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

running = True
rgb_image = None
dss_instance = None
loop_task = None
last_key_pressed = None

# 차량 제어 상태 변수
drive_state = {
    'throttle': 0.0,
    'steer': 0.0,
    'brake': 0.0
}

# 키보드 입력 → 제어 명령 매핑
def handle_key_control(key):
    step = 0.05
    global drive_state
    if key == ord('w'):
        drive_state['throttle'] = min(1.0, drive_state['throttle'] + step)
    elif key == ord('s'):
        drive_state['throttle'] = max(0.0, drive_state['throttle'] - step)
    elif key == ord('j'):
        drive_state['steer'] = max(-1.0, drive_state['steer'] - step)
    elif key == ord('l'):
        drive_state['steer'] = min(1.0, drive_state['steer'] + step)
    elif key == ord('k'):
        drive_state['steer'] = 0.0
    elif key == ord('x'):
        drive_state['brake'] = min(1.0, drive_state['brake'] + step)
    elif key == ord('z'):
        drive_state['brake'] = max(0.0, drive_state['brake'] - step)
    elif key == ord('r'):
        drive_state = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}

    print(f"[CONTROL] Throttle: {drive_state['throttle']:.2f}, Steer: {drive_state['steer']:.2f}, Brake: {drive_state['brake']:.2f}")

def signal_handler(sig, frame):
    global running
    running = False
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        pass

def on_camera_data(image: dss_pb2.DSSImage):
    global rgb_image, running, last_key_pressed
    if not running:
        return
    try:
        jpg_data = np.frombuffer(image.data, dtype=np.uint8)
        rgb_image = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
        if rgb_image is None:
            return

        cv2.imshow('DSS Camera', rgb_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False
            cv2.destroyAllWindows()
        elif key != 255:
            last_key_pressed = chr(key) if 32 <= key <= 126 else str(key)
            print(f"[KEYBOARD] Key pressed: {last_key_pressed}")
            handle_key_control(key)
    except Exception:
        pass

def on_lidar_data(lidar_data: dss_pb2.DssLidarPointCloud):
    global running
    if not running:
        return
    try:
        lidar = dss_pb2.DssLidarPointCloud()
        lidar.ParseFromString(lidar_data)
        num_points = len(lidar.data) // lidar.point_step
        if lidar.point_step > 0 and num_points > 0:
            np.frombuffer(lidar.data, dtype=np.uint8)
    except Exception:
        pass

def on_imu_data(imu: dss_pb2.DSSIMU):
    if not running:
        return

def on_gps_data(gps: dss_pb2.DSSGPS):
    if not running:
        return

def on_odom_data(odom: dss_pb2.DSSOdom):
    if not running:
        return

def on_ground_truth_data(gt_data):
    if not running:
        return
    try:
        gt = OSIManager.parse_ground_truth(gt_data)
        if gt:
            if gt.moving_object:
                for obj in gt.moving_object:
                    _ = obj.base.position
    except Exception:
        pass

def on_ground_truth_json_data(gt_data):
    if not running:
        return
    try:
        json.loads(gt_data)
    except Exception:
        pass

def on_sim_started():
    pass

def on_sim_ended():
    global running
    running = False

def on_sim_aborted():
    global running
    running = False

def on_sim_error():
    global running
    running = False

def update_vehicle_control(dss: IDSSSDK):
    if not running:
        return
    control = DSSSDKCarControl(
        throttle=drive_state['throttle'],
        steer=drive_state['steer'],
        brake=drive_state['brake'],
        park_brake=False,
        target_gear=1
    )
    try:
        with SuppressOutput():
            dss.set_car_control(control)
    except Exception:
        pass

def main():
    global running, dss_instance
    os.environ['PYTHONPATH'] = ''
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with SuppressOutput():
            dss = IDSSSDK.create(
                loop=loop,
                nats_address='nats://127.0.0.1:4222'
            )
            dss_instance = dss
            init_params = DSSSDKInitParams(
                server="127.0.0.1",
                heartbeat_port=8886,
	            nats_port=4222
            )
            dss.initialize(init_params)
            dss.register_sensor_callback('camera', on_camera_data)
            dss.register_sensor_callback('lidar', on_lidar_data)
            dss.register_sensor_callback('imu', on_imu_data)
            dss.register_sensor_callback('gps', on_gps_data)
            dss.register_sensor_callback('odom', on_odom_data)
            dss.register_sensor_callback('ground_truth', on_ground_truth_data)
            dss.register_simulation_callback('started', on_sim_started)
            dss.register_simulation_callback('ended', on_sim_ended)
            dss.register_simulation_callback('aborted', on_sim_aborted)
            dss.register_simulation_callback('error', on_sim_error)
            dss.start()

        while running:
            update_vehicle_control(dss)
            time.sleep(0.1)

    except KeyboardInterrupt:
        running = False
    except Exception:
        running = False
    finally:
        if dss_instance:
            try:
                with SuppressOutput():
                    dss_instance.cleanup()
            except Exception:
                pass
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()
