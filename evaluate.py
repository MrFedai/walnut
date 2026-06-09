import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix
import random
import numpy as np

# REPRODUCIBILITY SEED
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. Test Verisi Hazırlığı
    data_dir = "dataset/dataset_ready" 
    test_dir = os.path.join(data_dir, 'test')

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    class_names = test_dataset.classes

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

    print("Model test seti üzerinde değerlendiriliyor...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Akademik Raporlama (Classification Report)
    print("\n--- NİHAİ TEST SETİ PERFORMANS RAPORU ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Confusion Matrix Hesaplama ve Görselleştirme
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ceviz Türleri Sınıflandırma - Karmaşıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    
    # Matrisi rapor dosyanız için PNG olarak kaydeder
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(">>> Karmaşıklık matrisi grafiği başarıyla kaydedildi: 'confusion_matrix.png'")

if __name__ == '__main__':
    main()
