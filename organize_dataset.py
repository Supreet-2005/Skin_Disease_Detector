import os
import pandas as pd
import shutil

csv_path = "HAM10000_metadata.csv"
image_dir = "images/"
output_dir = "dataset/"

df = pd.read_csv(csv_path)

label_map = {
    'mel': 'Melanoma',
    'nv': 'Nevus',
    'bcc': 'BCC',
    'akiec': 'AK',
    'bkl': 'BKL',
    'df': 'DF',
    'vasc': 'Vascular'
}

for _, row in df.iterrows():
    label = label_map[row['dx']]
    img_name = row['image_id'] + ".jpg"

    src = os.path.join(image_dir, img_name)
    dst_folder = os.path.join(output_dir, label)

    os.makedirs(dst_folder, exist_ok=True)

    if os.path.exists(src):
        shutil.copy(src, os.path.join(dst_folder, img_name))

print("DONE ✅ Dataset Ready")