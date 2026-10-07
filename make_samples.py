import json
import os
import urllib.request
import zipfile
import ssl
import shutil

def create_sample_dataset():
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    zip_path = "caption_datasets.zip"
    
    print("Downloading Karpathy annotations...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("dataset_coco.json", ".")
        
    os.remove(zip_path)
    
    sample_json = "datasets/coco/annotations/dataset_coco_sample.json"
    image_dir = "datasets/coco/images/val2014"
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs("datasets/coco/annotations", exist_ok=True)

    with open("dataset_coco.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    os.remove("dataset_coco.json") # cleanup full json so we don't commit it
        
    # Get 5 images from the test split
    sample_images = []
    count = 0
    for img in data['images']:
        if img['split'] == 'test':
            sample_images.append(img)
            count += 1
            if count == 5:
                break
                
    # Create sample data dictionary
    sample_data = {
        "dataset": "coco",
        "images": sample_images
    }
    
    # Write sample JSON
    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4)
    print(f"Created {sample_json} with {len(sample_images)} sample images.")
    
    # Download the 5 images
    for img in sample_images:
        filename = img['filename']
        filepath = img.get('filepath', 'val2014')
        url = f"http://images.cocodataset.org/{filepath}/{filename}"
        save_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, save_path)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
        else:
            print(f"{filename} already exists.")
            
if __name__ == "__main__":
    create_sample_dataset()
