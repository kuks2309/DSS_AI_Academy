#visualize_data.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time

def load_driving_data(data_dir):
    """CSV 파일에서 이미지 경로와 조향각 데이터 로드"""
    csv_path = os.path.join(data_dir, 'driving_log.csv')
    
    images = []
    steerings = []
    
    try:
        with open(csv_path, 'r') as f:
            next(f)  # 헤더 건너뛰기
            for line_num, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    # 중앙 카메라 이미지 경로
                    center_img = os.path.basename(parts[0].strip())
                    img_path = os.path.join(data_dir, 'IMG', center_img)
                    
                    # 조향각
                    try:
                        steering = float(parts[3])
                        
                        # 파일이 존재하는지 확인
                        if os.path.exists(img_path):
                            images.append(img_path)
                            steerings.append(steering)
                    except ValueError:
                        continue
        
        print(f"✅ Total {len(images)} images loaded successfully.")
        return images, steerings
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return [], []

def draw_steering_wheel(ax, steering_angle, frame_info=""):
    """핸들 모양과 각도를 그리는 함수 (이미지와 같은 스타일)"""
    ax.clear()
    ax.set_facecolor('black')  # 검은 배경
    
    # 핸들 바깥쪽 원 (회색)
    circle_outer = Circle((0, 0), 0.9, fill=True, color='gray', alpha=0.8)
    ax.add_patch(circle_outer)
    
    # 핸들 안쪽 원 (검은색)
    circle_inner = Circle((0, 0), 0.3, fill=True, color='black')
    ax.add_patch(circle_inner)
    
    # 조향각을 각도로 변환
    angle_deg = steering_angle * 180 / np.pi
    angle_rad = np.radians(angle_deg)
    
    # 핸들 스포크 (흰색)
    spoke_length = 0.7
    for i in range(3):  # 3개 스포크
        spoke_angle = angle_rad + i * 2 * np.pi / 3
        spoke_x = spoke_length * np.sin(spoke_angle)
        spoke_y = spoke_length * np.cos(spoke_angle)
        ax.plot([0, spoke_x], [0, spoke_y], 'white', linewidth=8, alpha=0.9)
    
    # 중심 원 (검은색)
    center_circle = Circle((0, 0), 0.15, fill=True, color='black')
    ax.add_patch(center_circle)
    
    # 축 설정
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

def play_driving_video_auto(data_dir, fps=15):
    """이미지를 자동으로 동영상처럼 재생"""
    
    print("🚗 Loading PilotNet driving data...")
    images, steerings = load_driving_data(data_dir)
    
    if len(images) == 0:
        print("❌ No images found. Please check the path.")
        return
    
    print(f"🎬 Playing {len(images)} frames at {fps} FPS.")
    print("💡 Close the window to stop playback.")
    
    # matplotlib 설정 (이미지와 동일한 레이아웃)
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('navy')  # 배경색
    
    # 서브플롯 생성
    ax1 = plt.subplot(1, 2, 1)  # steering wheel
    ax2 = plt.subplot(1, 2, 2)  # scenario
    
    # 제목 설정
    ax1.set_title('steering wheel', color='white', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', alpha=0.7))
    ax2.set_title('scenario', color='white', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', alpha=0.7))
    
    # 전역 변수로 텍스트 객체 저장
    text_obj = None
    
    def update_frame(frame_idx):
        """각 프레임 업데이트"""
        nonlocal text_obj
        
        if frame_idx >= len(images):
            return
        
        # 이미지 로드
        img_path = images[frame_idx]
        steering = steerings[frame_idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 시나리오 이미지 표시 (오른쪽)
        ax2.clear()
        ax2.imshow(img)
        ax2.set_title('scenario', color='white', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', alpha=0.7))
        ax2.axis('off')
        
        # 핸들 각도 표시 (왼쪽)
        draw_steering_wheel(ax1, steering, f"Frame {frame_idx+1}")
        ax1.set_title('steering wheel', color='white', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', alpha=0.7))
        
        # 하단 정보 텍스트 (이미지와 동일한 스타일)
        if text_obj:
            text_obj.remove()
        
        # 각도를 도 단위로 변환
        angle_degrees = steering * 180 / np.pi
        info_text = f"Predicted steering angle: {angle_degrees:.9f} degrees\nScenario image size: 320 x 160"
        
        text_obj = fig.text(0.02, 0.02, info_text, 
                           color='white', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='purple', alpha=0.8))
        
        # Progress display
        progress = (frame_idx + 1) / len(images) * 100
        fig.suptitle(f'PilotNet Driving Video - Progress: {progress:.1f}% ({frame_idx+1}/{len(images)})', 
                    color='white', fontsize=14, fontweight='bold')
    
    # 애니메이션 생성
    interval = 1000 / fps  # milliseconds
    
    try:
        ani = animation.FuncAnimation(fig, update_frame, frames=len(images), 
                                    interval=interval, repeat=False, blit=False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # 텍스트 공간 확보
        plt.show()
        
        print("✅ Playback completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Playback interrupted.")

if __name__ == "__main__":
    data_dir = 'driving_data/data'
    
    print("🎬 PilotNet Driving Data Auto Player")
    print("=" * 40)
    
    # Start auto playback immediately (no selection needed)
    play_driving_video_auto(data_dir, fps=15)