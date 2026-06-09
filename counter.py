import os
#counts how much image is here 
dataset_path = "dataset/dataset_ready"
splits = ["train", "val", "test"]

for split in splits:
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.isdir(split_path):
        continue
        
    print(f"--- {split.upper()} SETI ---")
    
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)

        if os.path.isdir(cls_path):
            count = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"  {cls}: {count}")
            
    print() # Çıktıların okunabilirliği için boşluk
