import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Пути к папкам с данными
images_dir = Path('Data/Images')
labels_dir = Path('Data/Index')

# Списки файлов
pre_image_files = sorted(images_dir.glob('*_pre_disaster.png'))
post_image_files = sorted(images_dir.glob('*_post_disaster.png'))
post_label_files = sorted(labels_dir.glob('*_post_disaster.json'))
pre_label_files = sorted(labels_dir.glob('*_pre_disaster.json'))

# Предполагаем, что имена файлов согласованы между изображениями и масками
print(len(pre_image_files))
print(len(post_image_files))
print(len(post_label_files))
print(len(pre_label_files))
import os
import json
import pandas as pd
from pathlib import Path
data = []
for pre_image_file, post_image_file, post_label_file, pre_label_file in zip(pre_image_files, post_image_files, post_label_files, pre_label_files):
    if pre_image_file.exists() and post_label_file.exists() and pre_label_file.exists():
        with open(post_label_file, 'r') as post_file, open(pre_label_file, 'r') as pre_file:
            post_label_data = json.load(post_file)
            pre_label_data = json.load(pre_file)

            post_features = post_label_data.get('features', {})
            pre_features = pre_label_data.get('features', {})
            post_lng_lat_list = post_features.get('lng_lat', [])
            pre_lng_lat_list = pre_features.get('lng_lat', [])
            
            if post_lng_lat_list:
                post_properties = post_lng_lat_list[0].get('properties', {})
                damage_type = post_properties.get('subtype', 'Unknown')
            else:
                damage_type = 'Unknown'
                
            metadata = post_label_data.get('metadata', {})
            disaster = metadata.get('disaster', 'Unknown')
            disaster_type = metadata.get('disaster_type', 'Unknown')

        data.append({
            'pre_image': str(pre_image_file),
            'post_image': str(post_image_file),
            'pre_json': str(pre_label_file),
            'post_json': str(post_label_file),
            'damage_type': damage_type,
            'disaster': disaster,
            'disaster_type': disaster_type
        })

df = pd.DataFrame(data)

print(df.head())
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from shapely import wkt
import numpy as np
import cv2

# Константы
INPUT_IMAGE_KEY = "image"
INPUT_INDEX_KEY = "index"
INPUT_IMAGE_PRE_KEY = "image_pre"
INPUT_IMAGE_POST_KEY = "image_post"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_BUILDINGS_MASK_KEY = "buildings_mask"
INPUT_DAMAGE_MASK_KEY = "damage_mask"
OUTPUT_DAMAGE_MASK_KEY = "predicted_damage_mask"

def read_image_rgb(fname):
    image = cv2.imread(fname)[..., ::-1]
    if image is None:
        raise FileNotFoundError(fname)
    return image

def read_mask(fname):
    from PIL import Image
    mask = np.array(Image.open(fname))  # Read using PIL since it supports palletted image
    if len(mask.shape) == 3:
        mask = np.squeeze(mask, axis=-1)
    return mask

def overlay_damage_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 1] = (0, 255, 0)  # Regular building are green
    overlay[mask == 2] = (255, 255, 0)  # Destroyed buildings are yellow
    overlay[mask == 3] = (255, 69, 0)  # Destroyed buildings are orange
    overlay[mask == 4] = (255, 0, 0)  # Destroyed buildings are red
    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0, dtype=cv2.CV_8U)

def read_label(label_path):
    with open(label_path, 'r') as file:
        return json.load(file)

damage_dict = {
    "no-damage": (0, 255, 0, 50),
    "minor-damage": (0, 0, 255, 50),
    "major-damage": (255, 69, 0, 50),
    "destroyed": (255, 0, 0, 50),
    "un-classified": (255, 255, 255, 50)
}

def get_damage_type(properties):
    return properties.get('subtype', 'no-damage')

def annotate_img(draw, coords):
    for coord in coords:
        damage = get_damage_type(coord['properties'])
        polygon = wkt.loads(coord['wkt'])
        x, y = polygon.exterior.coords.xy
        draw.polygon(list(zip(x, y)), fill=damage_dict[damage])

# Отображение изображения с аннотацией или без
def display_img(image_path, json_path=None, annotated=True):
    img = Image.open(image_path)
    if annotated and json_path:
        image_json = read_label(json_path)
        draw = ImageDraw.Draw(img, 'RGBA')
        annotate_img(draw, image_json['features']['xy'])
    return img

def plot_images_from_dataframe(df, index):
    pre_image_path = df.loc[index, 'pre_image']
    post_image_path = df.loc[index, 'post_image']
    post_json_path = df.loc[index, 'post_json']
    pre_json_path = df.loc[index, 'pre_json']

    img_A = display_img(pre_image_path, annotated=False)
    img_B = display_img(post_image_path, annotated=False)
    img_C = display_img(pre_image_path, pre_json_path, annotated=True)
    img_D = display_img(post_image_path, post_json_path, annotated=True)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_A)
    ax[0].set_title('Pre Disaster')
    ax[0].axis('off')
    
    ax[1].imshow(img_B)
    ax[1].set_title('Post Disaster')
    ax[1].axis('off')
    
    ax[2].imshow(img_C)
    ax[2].set_title('Annotated Pre Disaster')
    ax[2].axis('off')

    ax[3].imshow(img_D)
    ax[3].set_title('Annotated Post Disaster')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

plot_images_from_dataframe(hurricane_df, 894)  
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import os
# Подготовка данных для стратификации
df['non_damaged_buildings'] = df['damage_type'].apply(lambda x: 1 if x == 'no-damage' else 0)
df['light_damaged_buildings'] = df['damage_type'].apply(lambda x: 1 if x == 'minor-damage' else 0)
df['medium_damaged_buildings'] = df['damage_type'].apply(lambda x: 1 if x == 'major-damage' else 0)
df['destroyed_buildings'] = df['damage_type'].apply(lambda x: 1 if x == 'destroyed' else 0)

df_all = df[df["damage_type"] != 'Unknown'].copy()
print("Samples with known damage types", len(df_all))

# Создание стратифицированных разбиений
mskf = MultilabelStratifiedKFold(n_splits=4, shuffle=True, random_state=0)
stratify_label = np.stack([
    df_all["non_damaged_buildings"].values,
    df_all["light_damaged_buildings"].values,
    df_all["medium_damaged_buildings"].values,
    df_all["destroyed_buildings"].values,
], axis=-1)

print(stratify_label.shape)

# Разделение данных на обучающую и валидационную выборки
train_index, valid_index = next(mskf.split(df_all, stratify_label))

train_df = df_all.iloc[train_index]
valid_df = df_all.iloc[valid_index]

# Сортировка и выборка топ-образцов
train_df = train_df.sort_values(by="destroyed_buildings", ascending=False)[:300]
valid_df = valid_df.sort_values(by="destroyed_buildings", ascending=False)[:60]
holdout_df = valid_df[50:]
valid_df = valid_df[:50]

print("Train samples", len(train_df))
print("Valid samples", len(valid_df))
print("Holdout samples", len(holdout_df))

# Подготовка списков путей к файлам
train_img_pre = train_df["pre_image"].tolist()
train_img_post = train_df["post_image"].tolist()
train_mask_post = train_df["post_json"].tolist()

valid_img_pre = valid_df["pre_image"].tolist()
valid_img_post = valid_df["post_image"].tolist()
valid_mask_post = valid_df["post_json"].tolist()

holdout_img_pre = holdout_df["pre_image"].tolist()
holdout_img_post = holdout_df["post_image"].tolist()
holdout_mask_post = holdout_df["post_json"].tolist()
