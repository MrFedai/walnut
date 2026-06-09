import splitfolders

# Girdi ve çıktı klasör yolları
input_folder = "raw_data"
output_folder = "dataset/dataset_ready"

# Veri dağıtım işlemi
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42,               # Reproducibility (tekrarlanabilirlik) için sabit seed
    ratio=(0.8, 0.1, 0.1), # %80 Train, %10 Val, %10 Test
    group_prefix=None, 
    move=False             # Move=False olduğu için orijinal 'raw_data' bozulmaz, kopyalama yapar
)

print("Veri seti başarıyla dağıtıldı!")
