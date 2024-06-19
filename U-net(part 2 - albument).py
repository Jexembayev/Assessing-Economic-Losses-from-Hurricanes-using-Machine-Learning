import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from pathlib import Path
damage_dict = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4
}
def overlay_damage_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 1] = (0, 255, 0)  # No damage
    overlay[mask == 2] = (255, 255, 0)  # Minor damage
    overlay[mask == 3] = (255, 69, 0)  # Major damage
    overlay[mask == 4] = (255, 0, 0)  # Destroyed
    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0, dtype=cv2.CV_8U)
color_augmentations = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.5),
    A.RandomGamma(gamma_limit=(90, 110)),
    A.Normalize(),
])


spatial_augmentations = A.Compose([
    # Random crop training samples to 512x512 patches
    A.RandomSizedCrop((512-64, 512+64), 512, 512),
    # D4
    A.RandomRotate90(p=1),
    A.Transpose(p=0.5),
    # Random rotate
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
    ),    
])
from skimage.draw import polygon
import numpy as np

# Создание маски на основе данных из JSON
def create_mask(image_size, coords: List[Dict]) -> np.ndarray:
    mask = np.zeros(image_size[:2], dtype=np.uint8)
    for coord in coords:
        damage = get_damage_type(coord['properties'])
        damage_id = damage_dict[damage]
        polygon_shape = wkt.loads(coord['wkt'])
        x, y = polygon_shape.exterior.coords.xy
        rr, cc = polygon(np.clip(y, 0, image_size[0]-1), np.clip(x, 0, image_size[1]-1))  # Проверка диапазона координат
        mask[rr, cc] = damage_id
    return mask
# Класс датасета
class ImageMaskDataset(Dataset):
    def __init__(
        self,
        pre_image_filenames: List[str],
        post_image_filenames: List[str],
        post_json_filenames: List[str],
        spatial_transform: A.Compose,
        color_transform=None,
    ):
        assert len(pre_image_filenames) == len(post_image_filenames) == len(post_json_filenames)

        self.pre_image_filenames = pre_image_filenames
        self.post_image_filenames = post_image_filenames
        self.post_json_filenames = post_json_filenames

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.pre_image_filenames)

    def __getitem__(self, index):
        pre_image = read_image_rgb(self.pre_image_filenames[index])  # 1024x1024x3
        post_image = read_image_rgb(self.post_image_filenames[index])  # 1024x1024x3
        post_json = read_label(self.post_json_filenames[index])
        
        post_mask = create_mask(post_image.shape, post_json['features']['xy'])  # Создание маски для post_image
        
        if self.color_transform is not None:
            pre_image = self.color_transform(image=pre_image)["image"]
            post_image = self.color_transform(image=post_image)["image"]

        image = np.dstack([pre_image, post_image])  # 512x512x6
        
        if self.spatial_transform is not None:
            data = self.spatial_transform(image=image, mask=post_mask)
            image = data["image"]
            post_mask = data["mask"]
            
        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: Path(self.pre_image_filenames[index]).stem,
            INPUT_IMAGE_KEY: torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32),
            INPUT_BUILDINGS_MASK_KEY: torch.tensor(post_mask > 0, dtype=torch.long),
            INPUT_DAMAGE_MASK_KEY: torch.tensor(post_mask, dtype=torch.long)
        }

        return sample
      train_ds = ImageMaskDataset(
    train_img_pre,
    train_img_post,
    train_mask_post,
    spatial_transform=spatial_augmentations,
    color_transform=color_augmentations,
)

valid_ds = ImageMaskDataset(
    valid_img_pre,
    valid_img_post,
    valid_mask_post,
    color_transform=A.Normalize(),
    spatial_transform=None,
)

print(len(train_ds), len(valid_ds))
def show_sample(sample: Dict[str, Any]):    
    image_pre = sample[INPUT_IMAGE_KEY][0:3].permute(1, 2, 0).numpy().astype(np.uint8)
    image_post = sample[INPUT_IMAGE_KEY][3:6].permute(1, 2, 0).numpy().astype(np.uint8)
    damage_mask = sample[INPUT_DAMAGE_MASK_KEY].numpy()
    
    f, ax = plt.subplots(1, 3, figsize=(30, 9))
    ax[0].imshow(image_pre)
    ax[0].axis('off')
    ax[1].imshow(image_post)
    ax[1].axis('off')
    ax[2].imshow(overlay_damage_mask(image_post, damage_mask))
    ax[2].axis('off')
    f.tight_layout()
    plt.show()

# Пример использования
pre_image_files = df['pre_image'].tolist()
post_image_files = df['post_image'].tolist()
post_json_files = df['post_json'].tolist()

dataset = ImageMaskDataset(
    pre_image_filenames=pre_image_files,
    post_image_filenames=post_image_files,
    post_json_filenames=post_json_files,
    spatial_transform=spatial_augmentations,
    color_transform=color_augmentations
)

# Отображение примера
sample = train_ds[0]
show_sample(sample)
sample = train_ds[25]
show_sample(sample)
sample = train_ds[4]
show_sample(sample)
