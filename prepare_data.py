import pandas as pd
from pathlib import Path
from PIL import Image
import random
import shutil

base = Path(__file__).parent
oimhs_folder = base / "OIMHS"
images_folder = oimhs_folder / "Images"
excel_path = next(oimhs_folder.glob("*Demographics*.xlsx"))

output_base = base / "oimhs_data"
if output_base.exists():
    shutil.rmtree(output_base)  
output_base.mkdir(exist_ok=True)

df = pd.read_excel(excel_path)
print("Columns:", df.columns.tolist())

for stage in [3,4]:
    for split in ['train', 'val', 'test']:
        (output_base / split / f"stage{stage}").mkdir(parents=True, exist_ok=True)

processed = 0

for idx, row in df.iterrows():
    eye_id = str(row.get('Eye ID', row.iloc[1])).strip()
    stage_val = int(row.get('Stage', 4))
    if stage_val < 3:
        stage_val = 4  
    eye_folder = images_folder / eye_id
    if not eye_folder.exists():
        continue
    
    for img_file in eye_folder.glob("*.png"):
        try:
            img = Image.open(img_file)
            w = img.width // 2
            original = img.crop((0, 0, w, img.height))
            
            r = random.random()
            split = "train" if r < 0.7 else "val" if r < 0.85 else "test"
            
            target_dir = output_base / split / f"stage{stage_val}"
            target_path = target_dir / f"{eye_id}_{img_file.name}"
            
            original.save(target_path, "PNG")
            processed += 1
            if processed % 500 == 0:
                print(f"Processed {processed} images...")
        except:
            continue

print(f"Done! {processed} images prepared.")