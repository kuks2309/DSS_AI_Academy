#pilotnet.py 파일

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
    
    with open(os.path.join(data_dir, 'driving_log.csv'), 'r') as f:
        next(f)  # 헤더 건너뛰기
        for line in f:
            parts = line.strip().split(',')
            center_img_path = os.path.join(data_dir, 'IMG', os.path.basename(parts[0]))
            steering = float(parts[3])
            
            images.append(center_img_path)
            steerings.append(steering)
    
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
    plt.savefig('loss_curves.png')
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
    print(f'Using device: {device}')
    
    # 데이터 디렉토리
    data_dir = 'driving_data/data'  # 데이터 폴더 경로
    
    # 모델 경로
    model_save_path = 'pilotnet_model.pth'
    
    # 하이퍼파라미터
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 50
    
    # 1. 모델 생성
    model = PilotNet()
    
    # 2. 데이터 로드
    print("Loading data...")
    images, steerings = load_data(data_dir)
    
    # 3. 학습/검증/테스트 데이터 분할
    train_imgs, val_imgs, train_steerings, val_steerings = train_test_split(
        images, steerings, test_size=0.2, random_state=42)
    
    # 4. 데이터셋 및 데이터로더 생성
    train_dataset = DrivingDataset(train_imgs, train_steerings)
    val_dataset = DrivingDataset(val_imgs, val_steerings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 6. 학습 또는 모델 로드
    train_model = True  # 학습 여부 설정
    
    if train_model:
        # 모델 학습
        print("Training model...")
        model, train_losses, val_losses = train(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=num_epochs, device=device
        )
        
        # 모델 저장 - 이 부분이 실행되도록 확인!
        print("Saving model...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        # 저장된 모델 로드
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.to(device)
    
    # 7. 테스트 이미지로 추론
    test_image = 'test_image.jpg'  # 테스트 이미지 경로
    if os.path.exists(test_image):
        steering = test_model(model, test_image, device)
        print(f'Predicted steering angle: {steering:.4f}')

if __name__ == "__main__":
    main()