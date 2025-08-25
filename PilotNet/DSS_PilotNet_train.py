#DSS_PilotNet_train.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PilotNet 모델 정의
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

# 데이터셋 클래스 정의
class DrivingDataset(Dataset):
    def __init__(self, images_paths, steering_angles, transform=None):
        self.images_paths = images_paths
        self.steering_angles = steering_angles
        self.transform = transform
        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 66))  # PilotNet 입력 크기
        
        # 이미지 정규화 (0-1 사이 값으로)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        
        steering = self.steering_angles[idx]
        
        return torch.FloatTensor(img), torch.FloatTensor([steering])

# 데이터 증강 함수
def augment_data(image, steering):
    # 랜덤 밝기 조정
    if np.random.rand() < 0.5:
        brightness_factor = np.random.uniform(0.5, 1.5)
        image = np.clip(image * brightness_factor, 0, 255)
    
    # 이미지 수평 뒤집기와 스티어링 각 반전
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    
    return image, steering

# 데이터 로드 및 전처리
def load_data(data_dir):
    images = []
    steerings = []
    
    driving_log_path = os.path.join(data_dir, 'driving_log.csv')
    
    print(f"📊 CSV 파일 읽는 중: {driving_log_path}")
    
    with open(driving_log_path, 'r') as f:
        next(f)  # 헤더 건너뛰기
        for line in f:
            parts = line.strip().split(',')
            # IMG/center_xxxxx.jpg에서 파일명만 추출
            center_img_name = parts[0].split('/')[-1]  # center_xxxxx.jpg
            center_img_path = os.path.join(data_dir, 'IMG', center_img_name)
            steering = float(parts[3])
            
            # 파일 존재 확인
            if os.path.exists(center_img_path):
                images.append(center_img_path)
                steerings.append(steering)
            else:
                print(f"⚠️ 파일 없음: {center_img_path}")
    
    print(f"✅ 데이터 로드 완료: {len(images)}개 이미지")
    return images, steerings

# 학습 함수
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 손실 그래프 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('DSS_pilotnet_loss_curves.png')
    plt.show()
    
    return model, train_losses, val_losses

# 모델 저장 함수
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

# 모델 로드 함수
def load_model(model, load_path, device='cuda'):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    return model

# 테스트 함수
def test_model(model, test_image_path, device='cuda'):
    model.eval()
    
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    
    with torch.no_grad():
        img_tensor = torch.FloatTensor(img).unsqueeze(0).to(device)
        steering = model(img_tensor).item()
    
    return steering

# 메인 함수
def main():
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️ Using device: {device}')
    
    # ★ DSS 데이터 디렉토리 설정
    data_dir = r'C:\Project\DSS\PilotNet\DSS_pilotnet_dataset'
    
    # ★ DSS 모델 저장 경로
    model_save_path = 'DSS_pilotnet_model.pth'
    
    # 하이퍼파라미터
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 100
    
    # 1. 모델 생성
    model = PilotNet()
    print(f"🧠 PilotNet 모델 생성 완료")
    
    # 2. 데이터 로드
    print("📁 DSS 데이터셋 로딩 중...")
    images, steerings = load_data(data_dir)
    
    if len(images) == 0:
        print("❌ 데이터가 없습니다! 데이터 수집을 먼저 해주세요.")
        return
    
    print(f"📊 데이터 통계:")
    print(f"   - 총 이미지 수: {len(images)}")
    print(f"   - 조향각 범위: {min(steerings):.3f} ~ {max(steerings):.3f}")
    print(f"   - 조향각 평균: {np.mean(steerings):.3f}")
    
    # 3. 학습/검증 데이터 분할
    train_imgs, val_imgs, train_steerings, val_steerings = train_test_split(
        images, steerings, test_size=0.2, random_state=42)
    
    print(f"🔄 데이터 분할:")
    print(f"   - 학습: {len(train_imgs)}개")
    print(f"   - 검증: {len(val_imgs)}개")
    
    # 4. 데이터셋 및 데이터로더 생성
    train_dataset = DrivingDataset(train_imgs, train_steerings)
    val_dataset = DrivingDataset(val_imgs, val_steerings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 6. 학습 시작
    print("🚀 PilotNet 학습 시작!")
    print("="*60)
    
    model, train_losses, val_losses = train(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # 7. 모델 저장
    print("💾 DSS PilotNet 모델 저장 중...")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 모델 저장 완료: {model_save_path}")
    
    # 8. 학습 결과 요약
    print("\n" + "="*60)
    print("📊 학습 완료 요약:")
    print(f"   - 최종 학습 손실: {train_losses[-1]:.6f}")
    print(f"   - 최종 검증 손실: {val_losses[-1]:.6f}")
    print(f"   - 모델 저장 위치: {model_save_path}")
    print(f"   - 손실 곡선 저장: DSS_pilotnet_loss_curves.png")
    print("="*60)

if __name__ == "__main__":
    main()