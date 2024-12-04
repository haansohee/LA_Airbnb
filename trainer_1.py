import torch
from model_1 import PricePredictionModel
from utils_1 import load_and_preprocess_data, get_dataloaders
from train_1 import train_and_save_model

def main():
    # 데이터 파일 경로와 설정
    save_path = "best_model_3.pth"
    batch_size = 64
    learning_rate = 0.001

    # 데이터 전처리
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess_data()

    # 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val, batch_size)

    # 모델 초기화
    input_dim = X_train.shape[1]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricePredictionModel(input_dim)

    # # 옵티마이저 및 손실 함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # 모델 학습 및 저장
    train_losses, val_losses = train_and_save_model(train_loader, val_loader,
                                                    save_path, model, optimizer, criterion)

    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()
