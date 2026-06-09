import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix

# Kendi yazdığımız modül
from transforms import set_seed, get_data_loaders

def main():
    # 1. Tohum Sabitleme
    set_seed(42)
    _, _, test_loader, class_names = get_data_loaders()

    # 2. Modelin Yüklenmesi ve En İyi Ağırlıkların Basılması
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 18)
    
    # Kaydedilen en iyi ağırlıkları yüklüyoruz
    model.load_state_dict(torch.load('best_resnet18_walnut.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Tahminlerin Toplanması
    all_preds = []
    all_labels = []

    print("The model is being evaluated on a test set....")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Akademik Raporlama (Classification Report)
    print("\n--- FINAL TEST SET PERFORMANCE REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Confusion Matrix Hesaplama ve Görselleştirme
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Walnut Species Classification - Complexity Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('Real Class')
    plt.xlabel('Predicted Class')
    
    # Matrisi rapor dosyanız için PNG olarak kaydeder
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(">>> The complexity matrix graph was successfully saved:'confusion_matrix.png'")

if __name__ == '__main__':
    main()
