import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from transforms import set_seed, get_data_loaders
def main():
    set_seed(42)
    train_loader, val_loader, _, class_names = get_data_loaders()
    # ... model ve eğitim kısmı ...

    # ==========================================
    # 2. MODEL MİMARİSİ VE OPTİMİZASYON
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 18)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Model Architecture Ready. It Is Starting...\n")

    # ==========================================
    # 3. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
    # ==========================================
    num_epochs = 10
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # --- EĞİTİM AŞAMASI ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')
        
        # --- DOĞRULAMA AŞAMASI ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val   -> Loss: {val_epoch_loss:.4f} | Accuracy: {val_epoch_acc:.4f}')
        
        # En iyi modeli kaydetme
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_resnet18_walnut.pth')
            print(">>> The new best model was saved to disc.!(best_resnet18_walnut.pth)")
            
        print()

    print(f'Training Completed. Best Verification Score: {best_acc:.4f}')

# ==========================================
# MULTIPROCESSING İÇİN ZORUNLU GİRİŞ NOKTASI
# ==========================================
if __name__ == '__main__':
    main()
