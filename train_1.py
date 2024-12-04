import torch.optim as optim
import torch
import torch.nn as nn
from model_1 import PricePredictionModel
# from torch.utils.data import DataLoader
from utils_1 import load_and_preprocess_data, get_dataloaders
import matplotlib.pyplot as plt


def train_and_save_model(train_loader, val_loader,
                         save_path, model, criterion, optimizer):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(100):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(
            f'Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader)} | Val Loss: {val_loss / len(val_loader)}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid()

        # 그래프 저장
        plt.savefig('training_vs_validation_loss.png')  # PNG 형식으로 저장
        print("그래프가 'training_vs_validation_loss.png' 파일로 저장되었습니다.")

    # torch.save(model.state_dict(), 'model_8.pth')

    return train_losses, val_losses