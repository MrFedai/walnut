import os
import requests
import zipfile

# Hocanın vereceği linki buraya koyun
DATASET_URL = "BURAYA_GDRIVE_VEYA_GITHUB_LINKINIZI_YAZIN"
OUTPUT_PATH = "dataset/dataset_ready"

def download_and_extract():
    print("Downloading dataset from cloud...")
    response = requests.get(DATASET_URL)
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    
    print("Extracting...")
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    
    os.remove("dataset.zip")
    print("Dataset ready!")

if __name__ == '__main__':
    download_and_extract()