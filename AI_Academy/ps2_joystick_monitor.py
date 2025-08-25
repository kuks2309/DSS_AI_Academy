import pygame
import time
import os
import sys

class PS2JoystickMonitor:
    def __init__(self):
        """PS2 조이스틱 모니터 초기화"""
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.running = True
        
    def initialize_joystick(self):
        """조이스틱 초기화 및 연결 확인"""
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count == 0:
            print("❌ 조이스틱이 연결되지 않았습니다.")
            print("PS2 조이스틱을 USB 어댑터로 연결한 후 다시 실행해주세요.")
            return False
        
        # 첫 번째 조이스틱 선택
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"✅ 조이스틱 연결됨: {self.joystick.get_name()}")
        print(f"📊 축(Axis) 개수: {self.joystick.get_numaxes()}")
        print(f"🔘 버튼 개수: {self.joystick.get_numbuttons()}")
        print(f"🎮 햇(Hat) 개수: {self.joystick.get_numhats()}")
        print("-" * 60)
        
        return True
    
    def clear_screen(self):
        """화면 지우기 (Windows)"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_axis_value(self, value):
        """축 값을 시각적 바로 표현"""
        # -1.0 ~ 1.0 값을 0~20 범위로 변환
        bar_length = 20
        normalized = int((value + 1.0) / 2.0 * bar_length)
        
        bar = "█" * normalized + "░" * (bar_length - normalized)
        return f"{value:6.3f} |{bar}|"
    
    def format_button_state(self, pressed):
        """버튼 상태를 시각적으로 표현"""
        return "🔴 ON " if pressed else "⚫ OFF"
    
    def format_hat_state(self, hat_value):
        """햇(D-pad) 상태를 방향으로 표현"""
        directions = {
            (0, 0): "⭕ 중립",
            (0, 1): "⬆️ 위", 
            (1, 1): "↗️ 우상",
            (1, 0): "➡️ 오른쪽",
            (1, -1): "↘️ 우하",
            (0, -1): "⬇️ 아래",
            (-1, -1): "↙️ 좌하",
            (-1, 0): "⬅️ 왼쪽",
            (-1, 1): "↖️ 좌상"
        }
        return directions.get(hat_value, f"❓ {hat_value}")
    
    def display_joystick_data(self):
        """조이스틱 데이터를 실시간으로 표시"""
        try:
            while self.running:
                pygame.event.pump()  # 이벤트 처리
                
                self.clear_screen()
                
                print("🎮 PS2 조이스틱 실시간 모니터")
                print(f"📱 장치명: {self.joystick.get_name()}")
                print("=" * 60)
                
                # 아날로그 스틱 및 트리거 값 표시
                print("📊 아날로그 축 (Analog Axes):")
                axis_names = [
                    "좌 스틱 X축 (Left X)",
                    "좌 스틱 Y축 (Left Y)", 
                    "우 스틱 X축 (Right X)",
                    "우 스틱 Y축 (Right Y)",
                    "L2 트리거 (L2)",
                    "R2 트리거 (R2)"
                ]
                
                for i in range(self.joystick.get_numaxes()):
                    axis_value = self.joystick.get_axis(i)
                    axis_name = axis_names[i] if i < len(axis_names) else f"축 {i}"
                    print(f"  {axis_name:20} {self.format_axis_value(axis_value)}")
                
                print("\n🔘 버튼 상태 (Buttons):")
                button_names = [
                    "X", "○", "□", "△",      # 0-3: 기본 버튼
                    "L1", "R1", "L2", "R2",   # 4-7: 어깨 버튼
                    "SELECT", "START",        # 8-9: 시스템 버튼
                    "L3", "R3"               # 10-11: 스틱 버튼
                ]
                
                # 버튼을 2열로 표시
                for i in range(0, self.joystick.get_numbuttons(), 2):
                    left_btn = i
                    right_btn = i + 1
                    
                    left_name = button_names[left_btn] if left_btn < len(button_names) else f"BTN{left_btn}"
                    left_state = self.format_button_state(self.joystick.get_button(left_btn))
                    
                    if right_btn < self.joystick.get_numbuttons():
                        right_name = button_names[right_btn] if right_btn < len(button_names) else f"BTN{right_btn}"
                        right_state = self.format_button_state(self.joystick.get_button(right_btn))
                        print(f"  {left_name:8} {left_state}    {right_name:8} {right_state}")
                    else:
                        print(f"  {left_name:8} {left_state}")
                
                # D-pad (햇) 상태 표시
                if self.joystick.get_numhats() > 0:
                    print("\n🎯 D-pad 상태:")
                    for i in range(self.joystick.get_numhats()):
                        hat_value = self.joystick.get_hat(i)
                        print(f"  D-pad {i}: {self.format_hat_state(hat_value)}")
                
                print("\n" + "=" * 60)
                print("💡 종료하려면 Ctrl+C를 누르세요")
                
                time.sleep(0.1)  # 100ms 간격으로 업데이트
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            self.running = False
    
    def run(self):
        """메인 실행 함수"""
        print("🎮 PS2 조이스틱 모니터 시작...")
        
        if not self.initialize_joystick():
            return
        
        try:
            self.display_joystick_data()
        finally:
            if self.joystick:
                self.joystick.quit()
            pygame.quit()

def main():
    """프로그램 진입점"""
    print("Windows 11 PowerShell - PS2 조이스틱 모니터")
    print("=" * 50)
    
    # 필요한 라이브러리 확인
    try:
        import pygame
    except ImportError:
        print("❌ pygame 라이브러리가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install pygame")
        return
    
    monitor = PS2JoystickMonitor()
    monitor.run()

if __name__ == "__main__":
    main()