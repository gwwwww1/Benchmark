import os
import glob
import pickle
import random
import re

import numpy as np
import torch
from PIL import Image

from clip_benchmark.engine.corrupt import ImageCorruptor

from clip_benchmark.histaug.augmentations import load_augmentations
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import h5py
import numpy
import json
import pandas as pd
from pathlib import Path
import io


def get_random_rotation():
    transformations = [
        "flip horizontal",
        "flip vertical",
        "rotate 90°",
        "rotate 180°",
        "rotate 270°",
        "rotate random angle"
    ]
    return random.choice(transformations)


# 27 augmentations and norm
def get_random_augmentation():
    transformations = [
        "Macenko",
        "low brightness",
        "high brightness",
        "low contrast",
        "high contrast",
        "low saturation",
        "high saturation",
        "colour jitter",
        "gamma 0.5",
        "gamma 2.0",
        "flip horizontal",
        "flip vertical",
        "rotate 90°",
        "rotate 180°",
        "rotate 270°",
        "rotate random angle",
        "zoom 1.75x",
        "zoom 2x",
        "affine",
        "warp perspective",
        "jigsaw",
        "Cutout",
        "AugMix",
        "sharpen",
        "gaussian blur",
        "median blur",
        "zoom 1.5x",
        "norm"
    ]
    return random.choice(transformations)


import torch
import kornia.geometry as Kgeo


def safe_augment(aug, img_tensor, recover_original_size=True, resize_mode='bicubic'):
    """
    安全地应用图像增强，使用缩放处理尺寸不兼容问题

    参数:
        aug: 增强器对象
        img_tensor: 输入图像张量 [B, C, H, W]
        recover_original_size: 是否在增强后恢复原始尺寸
        resize_mode: 缩放插值模式 ('bilinear', 'bicubic', 'nearest')
    """
    try:
        # 尝试直接应用增强,
        return aug(img_tensor)

    except RuntimeError as e:
        if "shape '[-1" in str(e) and "is invalid for input of size" in str(e):
            # print(f"警告: {aug.__class__.__name__} 增强时尺寸不兼容 - {e}")

            # 记录原始尺寸
            original_size = img_tensor.shape[2:]  # [H, W]

            # 尝试获取增强器的网格参数
            grid = (4, 4)  # 默认网格
            if hasattr(aug, 'flags') and 'grid' in aug.flags:
                grid = aug.flags['grid']
            elif hasattr(aug, 'p') and hasattr(aug.p, 'grid'):
                grid = aug.p.grid

            # 计算兼容尺寸（向上取整到最近的网格倍数）
            new_height = ((original_size[0] + grid[0] - 1) // grid[0]) * grid[0]
            new_width = ((original_size[1] + grid[1] - 1) // grid[1]) * grid[1]

            # 缩放图像到兼容尺寸
            if new_height != original_size[0] or new_width != original_size[1]:
                scale_factor = min(new_height / original_size[0], new_width / original_size[1])
                # print(f"缩放图像: 原始尺寸 {original_size} -> 目标尺寸 ({new_height}, {new_width}) "
                #       f"(缩放因子: {scale_factor:.4f})")

                img_resized = Kgeo.resize(
                    img_tensor,
                    (new_height, new_width),
                    interpolation=resize_mode,
                    align_corners=False if resize_mode in ['bilinear', 'bicubic'] else None
                )
                # print(img_resized.shape)
                try:
                    # 再次尝试应用增强
                    augmented = aug(img_resized)
                    # print(augmented)
                    # 恢复原始尺寸
                    if recover_original_size:
                        # print(f"将增强后的图像从 ({new_height}, {new_width}) 恢复到 {original_size}")
                        # print("hhhhhhhhhh")
                        # print(augmented[name].shape)
                        augmented = Kgeo.resize(
                            augmented,
                            original_size,
                            interpolation=resize_mode,
                            align_corners=False if resize_mode in ['bilinear', 'bicubic'] else None
                        )
                        # print(augmented[name].shape)
                    return augmented

                except Exception as e2:
                    print(f"二次尝试失败: {e2}")

        # 其他错误或二次尝试失败时返回原始图像
        return img_tensor


def get_aug_image(augmentation_types, image):
    # print(type(image))
    assert augmentation_types in ['stain normalisation', 'all augmentations', 'rotation_flipping']
    # print("augmentation type: ", augmentation_types)
    augmentations = load_augmentations()
    # print(augmentations)
    if augmentation_types == 'stain normalisation':
        import torchvision.transforms as T
        transform = T.ToTensor()
        inverse_transform = T.ToPILImage()
        img_tensor = transform(image).unsqueeze(0)
        aug_tensor = {aug_name: aug(img_tensor) for aug_name, aug in augmentations.items() if
                      aug_name == 'Macenko'}
        # print("augmentation: Macenko ")
        # print(aug_tensor)
        aug_img = inverse_transform(aug_tensor['Macenko'].squeeze(0))
    elif augmentation_types == 'rotation_flipping':
        rotation = get_random_rotation()
        # print("augmentation: ", rotation)
        aug_img = {aug_name: aug(image) for aug_name, aug in augmentations.items() if
                   aug_name == rotation}
        aug_img = aug_img[rotation]
    else:
        name = get_random_augmentation()
        if name == 'norm':
            aug_img = image
        else:
            import torchvision.transforms as T
            transform = T.ToTensor()
            inverse_transform = T.ToPILImage()
            img_tensor = transform(image).unsqueeze(0)
            # print(img_tensor.shape)
            # print(name)
            # print(type(img_tensor))
            aug_tensor = {aug_name: safe_augment(aug, img_tensor) for aug_name, aug in augmentations.items() if
                          aug_name == name}
            # print("augmentation: Macenko ")
            # print(aug_tensor)
            aug_img = inverse_transform(aug_tensor[name].squeeze(0))
            # plt.imshow(aug_img)
            # plt.savefig('output1.png', dpi=800)
        # print("augmentation: ", name)
    return aug_img


# single or mixture
# op
# op+lower
# test and train
# 已经跑
class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, corrupt=False, corrupt_level=None, corruption_types=None, transform=None,
                 train=True, val=False,
                 tumor=False, Optimization=True, lower=False, augmentation=False,
                 augmentation_types=None, task='zeroshot_classification'):

        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.lower = lower
        self.Optimization = Optimization
        self.corruption_types = corruption_types
        self.corrupt = corrupt
        self.corrupt_level = corrupt_level
        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)
        self.data_root = root

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train
        # 优化文本
        self.label_dict = {
            "nontumor_skin_necrosis_necrosis": "Non-tumor necrosis",
            "nontumor_skin_muscle_skeletal": "Non-tumor skeletal muscle",
            "nontumor_skin_sweatglands_sweatglands": "Non-tumor sweat glands",
            "nontumor_skin_vessel_vessel": "Non-tumor vessel",
            "nontumor_skin_elastosis_elastosis": "Non-tumor elastosis",
            "nontumor_skin_chondraltissue_chondraltissue": "Non-tumor chondral tissue",
            "nontumor_skin_hairfollicle_hairfollicle": "Non-tumor hair follicle",
            "nontumor_skin_epidermis_epidermis": "Non-tumor epidermis",
            "nontumor_skin_nerves_nerves": "Non-tumor nerves",
            "nontumor_skin_subcutis_subcutis": "Non-tumor subcutis",
            "nontumor_skin_dermis_dermis": "Non-tumor dermis",
            "nontumor_skin_sebaceousglands_sebaceousglands": "Non-tumor sebaceous glands",
            "tumor_skin_epithelial_sqcc": "Tumor epithelial squamous cell carcinoma",
            "tumor_skin_melanoma_melanoma": "Tumor melanoma",
            "tumor_skin_epithelial_bcc": "Tumor epithelial basal cell carcinoma",
            "tumor_skin_naevus_naevus": "Tumor naevus"
        }
        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.cat_to_num_map) if not self.tumor else list(self.tumor_map)
        if self.Optimization:
            self.classes = list(self.label_dict.values()) if not self.tumor else list(self.tumor_map)
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = ["An H&E image of {c}"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')
        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]
        # corrupt
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # aug
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# test and train
class PannukeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, corrupt=False, corrupt_level=None, corruption_types=None,
                 augmentation=False, augmentation_types=None, task='zeroshot_classification'):
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.root = root

        df = pd.read_csv(os.path.join(root, "PanNuke_all_binary.csv"))
        self.df = df[df['split'] == 'train'] if train else df[df['split'] == 'test']

        self.transform = transform

        self.classes = ["benign",
                        "malignant"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.df.iloc[index]['image'])
        image = Image.open(fpath).convert("RGB")
        label = 1 if 'malignant' in self.df.iloc[index]['caption'] else 0
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# test and train
class UnitopathoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, corrupt=False, corrupt_level=None, corruption_types=None,
                 augmentation=False, augmentation_types=None):
        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        if train:
            self.data = json.load(open(os.path.join(root, "images_train.json")))
        else:
            self.data = json.load(open(os.path.join(root, "images_test.json")))
        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}
        # NORM - Normal
        # tissue;
        # HP - Hyperplastic
        # Polyp;
        # TA.HG - Tubular
        # Adenoma, High - Grade
        # dysplasia;
        # TA.LG - Tubular
        # Adenoma, Low - Grade
        # dysplasia;
        # TVA.HG - Tubulo - Villous
        # Adenoma, High - Grade
        # dysplasia;
        # TVA.LG - Tubulo - Villous
        # Adenoma, Low - Grade
        # dysplasia.

        self.classes = ["Hyperplastic Polyp",
                        "Normal tissue",
                        "Tubular Adenoma, High-Grade dysplasia",
                        "Tubular Adenoma, Low-Grade dysplasia",
                        "Tubulo-Villous Adenoma, High-Grade dysplasia",
                        "Tubulo-Villous Adenoma, Low-Grade dysplasia"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")
        label = self.labels_dict[fpath.split("/")[-2]]
        if self.corrupt:
            # print("corrupting image")
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)
        return image, label


class UnitopathoRetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset for unitopatho image retrieval, using all samples.
    """

    def __init__(self, root, transform=None, train=True, corrupt=False, corrupt_level=None, corruption_types=None,
                 task='image retrival'):
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.data = json.load(open(os.path.join(root, "images.json")))

        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}

        # these prompts work better!
        self.classes = ["HP",
                        "NORM",
                        "TA.HG",
                        "TA.LG",
                        "TVA.HG",
                        "TVA.LG"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class PathMMUDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None):
        """
        Initialize the dataset by processing the JSON files and setting up the image roots.
        """
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.transform = transform
        self.root = Path(root)
        self.items = []

        # Process subsets
        self.img_root = {
            "Book": self.root / "Book/images",
            "WebPathology": self.root / "WebPathology/images",
            "Twitter": self.root / "Twitter/images",
        }

        self._process_json_files(self.root / "Book")
        self._process_json_files(self.root / "WebPathology")
        self._process_json_files(self.root / "Twitter")

    def _process_json_files(self, dir_path):
        """
        Process all JSON files in the given directory and extend the items list.
        """
        dir_path = dir_path / "GPTCaption"
        json_files = dir_path.glob("*.json")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)

                    data_list = list(data.values())
                    dataset_name = json_file.parts[-3]
                    for entry in data_list:
                        entry['dataset'] = dataset_name

                    self.items.extend(data_list)

                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {json_file}")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve an item by index, including the image and its corresponding caption.
        """

        item = self.items[idx]
        img_root = self.img_root[item['dataset']]
        img_path = img_root / item['img_path']

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            image = self.transform(image)

        return image, item['caption']


class BooksetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None):
        """
        Initialize the dataset by processing the JSON files and setting up the image roots.
        bookset的图像不全，不包含所有的uuid
        """
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.transform = transform
        self.root = Path(root)
        self.items = []
        # Process subsets
        self.img_root = {
            "books_set": self.root / "images",
        }
        self.uuid = [f_name.split('.')[0] for f_name in os.listdir(self.img_root['books_set'])]

        self._process_json_files(self.root)

    def _process_json_files(self, dir_path):
        """
        Process all JSON files in the given directory and extend the items list.
        """
        dir_path = dir_path
        print(dir_path)
        json_file = dir_path / "captions.json"
        print(json_file)
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                # print(data)
                bookset_captions_df = pd.DataFrame(data).T
                set(self.uuid).issubset(set(bookset_captions_df.uuid))
                # 找出 captions DataFrame 中存在但 images 文件夹中不存在的 UUID，
                # 即缺失图像的 UUID
                missing_image_uuids = set(bookset_captions_df.uuid) - set(self.uuid)
                # 输出缺失图像的 UUID 数量
                # print(len(missing_image_uuids))
                # print(missing_image_uuids)

                data_list = list(data.values())
                # print(data_list)
                dataset_name = json_file.parts[-2]
                # print(dataset_name)
                for entry in data_list:
                    entry['dataset'] = dataset_name
                self.items.extend(data_list)
                # 去除不存在的uuids
                self.items = [item for item in self.items if item['uuid'] not in missing_image_uuids]

            except json.JSONDecodeError:
                print(f"Error reading JSON file: {json_file}")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve an item by index, including the image and its corresponding caption.
        """

        item = self.items[idx]
        img_root = self.img_root[item['dataset']]
        img_path = img_root / (item['uuid'] + ".png")

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None
        # print('==========',self.corrupt)
        if self.corrupt:
            # print(self.corruption_types)
            # print(self.corrupt_level)
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            image = self.transform(image)

        return image, item['caption']


class PubmedsetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None):
        """
        Initialize the dataset by processing the JSON files and setting up the image roots.
        bookset的图像不全，不包含所有的uuid
        """
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.transform = transform
        self.root = Path(root)
        self.items = []
        # Process subsets
        self.img_root = {
            "pubmed_set": self.root / "images",
        }
        self.uuid = [f_name.split('.')[0] for f_name in os.listdir(self.img_root['pubmed_set'])]

        self._process_json_files(self.root)

    def _process_json_files(self, dir_path):
        """
        Process all JSON files in the given directory and extend the items list.
        """
        dir_path = dir_path
        print(dir_path)
        json_file = dir_path / "captions.json"
        # print(json_file)
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                # print(data)
                bookset_captions_df = pd.DataFrame(data).T
                set(self.uuid).issubset(set(bookset_captions_df.uuid))
                # 找出 captions DataFrame 中存在但 images 文件夹中不存在的 UUID，
                # 即缺失图像的 UUID
                # print(self.uuid)
                missing_image_uuids = set(bookset_captions_df.uuid) - set(self.uuid)
                # 输出缺失图像的 UUID 数量
                # print(len(missing_image_uuids))
                # print(missing_image_uuids)

                data_list = list(data.values())
                # print(data_list)
                dataset_name = json_file.parts[-2]
                # print(dataset_name)
                for entry in data_list:
                    entry['dataset'] = dataset_name
                self.items.extend(data_list)
                # 去除不存在的uuids
                self.items = [item for item in self.items if item['uuid'] not in missing_image_uuids]

            except json.JSONDecodeError:
                print(f"Error reading JSON file: {json_file}")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve an item by index, including the image and its corresponding caption.
        """

        item = self.items[idx]
        img_root = self.img_root[item['dataset']]
        img_path = img_root / (item['uuid'] + ".jpg")

        try:
            img_path = img_root / (item['uuid'] + ".jpg")
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            try:
                img_path = img_root / (item['uuid'] + ".png")
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"Image file not found: {img_path}")
                return None
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)

        return image, item['caption']


# single or mixture
# RGB!=RGB
# one all
class LC25000Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, lung=False, corrupt=False, corrupt_level=None, corruption_types=None
                 , augmentation=False,
                 augmentation_types=None):
        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        self.label_dict_colon = {
            'colon_n': 'normal colon tissue', 'colon_aca': 'colon adenocarcinoma'
        }
        self.label_index_colon_dict = {
            'colon_n': 0, 'colon_aca': 1
        }
        self.label_dict_lung = {
            'lung_n': 'benign lung tissue', 'lung_aca': 'lung adenocarcinoma',
            'lung_scc': 'lung squamous cell carcinomas'
        }
        self.label_index_lung_dict = {
            'lung_n': 0, 'lung_aca': 1, 'lung_scc': 2
        }
        self.lung = lung

        self.classes = list(self.label_dict_colon.values()) if not self.lung else list(self.label_dict_lung.values())
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        # self.templates = ["a histopathology slide showing {c}",
        #                   "histopathology image of {c}",
        #                   "pathology tissue showing {c}",
        #                   "presence of {c} tissue on image"]
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.jpeg']):
        path_list = []
        if self.lung:
            root_dir = os.path.join(root, 'lung_image_sets')
        else:
            root_dir = os.path.join(root, 'colon_image_sets')
        for dirpath, dirnames, files in os.walk(root_dir):
            for f in files:
                if any(f.lower().endswith(ft) for ft in file_types):
                    path_list.append(os.path.join(dirpath, f))
        # random.shuffle(path_list)
        self.img_list = path_list
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image1 = Image.open(img_path)
        # print('no  rgb', image.shape)
        image = image1.convert('RGB')
        # print("=====",image == image1)
        # print(image.shape)
        if self.lung:
            # print(os.path.basename(os.path.dirname(img_path)))
            label = self.label_index_lung_dict[os.path.basename(os.path.dirname(img_path))]
        else:
            label = self.label_index_colon_dict[os.path.basename(os.path.dirname(img_path))]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        return image, label


# revise 顺序
# single or mixture
# changed,RGB=RGB
# train and split
# label change
class PatchCamelyon(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, corrupt=False, split='test', corrupt_level=None, corruption_types=None
                 , augmentation=False,
                 augmentation_types=None, task='zeroshot_classification'):
        self.task = task
        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_npy = None
        self.label_npy = None
        self.split = split
        self.label_dict = {
            "lymph nodes without any metastases": 0,
            "lymph nodes containing metastases": 1
        }
        self.label_dict2 = {
            "lymph node": 0,
            "lymph node containing metastatic tumor tissue": 1
        }

        self.classes = list(self.label_dict)
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'an H&E patch of {c}'
        ]

    def __len__(self):
        return len(self.img_npy)

    def process_file(self, root, file_types=['.jpeg']):
        if self.split == 'test':
            rootx = os.path.join(root, 'camelyonpatch_level_2_split_test_x.h5')
            rooty = os.path.join(root, 'camelyonpatch_level_2_split_test_y.h5')
        elif self.split == 'train':
            rootx = os.path.join(root, 'camelyonpatch_level_2_split_train_x.h5')
            rooty = os.path.join(root, 'camelyonpatch_level_2_split_train_y.h5')
        else:
            rootx = os.path.join(root, 'camelyonpatch_level_2_split_valid_x.h5')
            rooty = os.path.join(root, 'camelyonpatch_level_2_split_valid_y.h5')
        with h5py.File(rootx, 'r') as hdf:
            # 获取数据集
            dataset_x = hdf['x'][:]  # 替换为实际的数据集名称
            # 将数据集转换为 NumPy 数组
            # print(dataset)
            # data_array_x = np.array(dataset)
        with h5py.File(rooty, 'r') as hdf:
            # 获取数据集
            dataset_y = hdf['y'][:]  # 替换为实际的数据集名称
            # 将数据集转换为 NumPy 数组
            # print(dataset)
            # data_array_y = np.array(dataset)
        self.label_npy = dataset_y
        self.img_npy = dataset_x
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_npy[idx]
        label = int(self.label_npy[idx])
        image = Image.fromarray(img_path)
        image = image.convert('RGB')
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# single or mixture
# changed,RGB!=RGB
# train and test
class CRC100K(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, lung=False, corrupt=False, corrupt_level=None, corruption_types=None,
                 augmentation=False,
                 augmentation_types=None,
                 split="test", task='zeroshot_classification', norm=False):
        self.norm = norm
        self.task = task
        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.split = split
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        self.label_dict = {
            'ADI': 'Adipose', 'BACK': 'background', 'DEB': 'debris', 'LYM': 'lymphocytes', 'MUC': 'mucus',
            'MUS': 'smooth muscle', 'NORM': 'normal colon mucosa', 'STR': 'cancer-associated stroma',
            'TUM': 'colorectal adenocarcinoma epithelium'
        }
        self.label_index_dict = {
            'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8
        }
        self.classes = list(self.label_dict.values())
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E patch of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.tif']):
        if self.norm:
            # norm
            if self.split == 'train':
                print('no Normalizing train')
                root = os.path.join(root, 'NCT-CRC-HE-100K-NONORM')
            else:
                print('no Normalizing test')
                root = os.path.join(root, 'CRC-VAL-HE-7K')
        else:
            if self.split == 'train':
                root = os.path.join(root, 'NCT-CRC-HE-100K')
            elif self.split == 'test':
                root = os.path.join(root, 'CRC-VAL-HE-7K')
            else:
                root = os.path.join(root, 'NCT-CRC-HE-100K')

        path_list = []
        for dirpath, dirnames, files in os.walk(root):
            for f in files:
                if any(f.lower().endswith(ft) for ft in file_types):
                    path_list.append(os.path.join(dirpath, f))
        # random.shuffle(path_list)
        self.img_list = path_list
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')
        # image = image1.convert('RGB')
        # print("============", image == image1)
        label = self.label_index_dict[os.path.basename(os.path.dirname(img_path))]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            # print("augmentation")
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


class BACH(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, lung=False, corrupt=False, corrupt_level=None, corruption_types=None):
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        self.label_dict_colon = {
            'colon_n': 'normal colon tissue', 'colon_aca': 'colon adenocarcinoma'
        }
        self.label_index_colon_dict = {
            'colon_n': 0, 'colon_aca': 1
        }
        self.label_dict_lung = {
            'lung_n': 'benign lung tissue', 'lung_aca': 'lung adenocarcinoma',
            'lung_scc': 'lung squamous cell carcinomas'
        }
        self.label_index_lung_dict = {
            'lung_n': 0, 'lung_aca': 1, 'lung_scc': 2
        }
        self.lung = lung

        self.classes = list(self.label_index_colon_dict) if not self.lung else list(self.label_index_lung_dict)
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.jpeg']):
        path_list = []
        if self.lung:
            root_dir = os.path.join(root, 'lung_image_sets')
        else:
            root_dir = os.path.join(root, 'colon_image_sets')
        for dirpath, dirnames, files in os.walk(root_dir):
            for f in files:
                if any(f.lower().endswith(ft) for ft in file_types):
                    path_list.append(os.path.join(dirpath, f))
        random.shuffle(path_list)
        self.img_list = path_list
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.lung:
            label = self.label_index_lung_dict[os.path.basename(os.path.dirname(img_path))]
        else:
            label = self.label_index_colon_dict[os.path.basename(os.path.dirname(img_path))]
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        return image, label


# single or mixture
# lower
# changed,RGB!=RGB
# one all
class Osteo(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False,
                 augmentation=False,
                 augmentation_types=None,
                 task='zeroshot_classification',
                 split='test'
                 ):
        self.split = split
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        self.label_index_dict = None
        self.cats_dict = {
            "Non-tumor": "Non-tumor",
            "Necrotic tumor": "Necrotic tumor",
            "Viable tumor": "Viable tumor"
        }
        self.label_dict = {
            'Non-tumor': 0,
            'Necrotic tumor': 1,
            'Viable tumor': 2
        }
        self.csv2_label_dict = {
            "Non-Tumor": "Non-tumor",
            "Viable": "Viable tumor",
            "viable: non-viable": "Viable tumor",
            "Non-Viable-Tumor": "Necrotic tumor",
        }
        self.classes = list(self.cats_dict.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = ["'An H&E image patch of {c}"]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.jpg']):
        # 使用 os.walk 函数遍历指定根目录 root 及其所有子目录
        # os.walk 函数返回一个生成器，每次迭代会返回一个三元组 (dirpath, dirnames, files)
        # dirpath 是当前正在遍历的目录的完整路径
        # dirnames 是当前目录下的所有子目录名称组成的列表
        # files 是当前目录下的所有文件名称组成的列表
        path_list, label_list = [], []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
            label_list = df['data_class'].tolist()
        else:
            i = 0
            root_dir = os.path.join(root, 'PKG - Osteosarcoma Tumor Assessment')
            for dirpath, dirnames, files in os.walk(root_dir):
                # 遍历当前目录下的所有文件名称
                i = i + 1
                if i >= 3:
                    try:
                        df = pd.read_csv(os.path.join(dirpath, "PathologistValidation.csv"), header=None,
                                         names=['image', 'label'])
                    except FileNotFoundError:
                        print(f"未找到文件: {os.path.join(dirpath, 'PathologistValidation.csv')}")
                        continue
                    # print("-------{}--------".format(dirpath))
                    for f in files:
                        # 检查当前文件名是否以 file_types 列表中的任意一种文件类型结尾
                        # f.lower() 将文件名转换为小写，实现不区分大小写的匹配
                        # f.lower().endswith(ft) 检查转换后的文件名是否以 ft 结尾
                        # any 函数用于判断在 file_types 列表中是否至少有一个 ft 满足上述条件
                        if any(f.lower().endswith(ft) for ft in file_types):
                            # print(f)
                            # 按 - 分割字符串
                            parts = f.split('-', 2)
                            part1 = parts[0]
                            part2 = parts[1]
                            part3 = parts[2]
                            basename = " ".join(parts)
                            # print(basename)
                            row = df[df['image'] == basename]
                            if row.empty:
                                part = part3.split('-', 1)
                                part4 = part[0]
                                part5 = part[1]
                                basename = part1 + ' ' + part2 + ' - ' + part4 + ' ' + part5
                                row = df[df['image'] == basename]
                                if row.empty:
                                    basename = part1 + ' ' + part2 + '-' + part3
                                    row = df[df['image'] == basename]
                            # print("row:{},basename:{}".format(row,basename))
                            label_list.append(row['label'].values[0])
                            path_list.append(os.path.join(dirpath, f))
        self.label_index_dict = {k: v for k, v in zip(path_list, label_list)}
        # random.shuffle(path_list)
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image1 = Image.open(img_path)
        image = image1.convert('RGB')
        label = self.label_dict[self.csv2_label_dict[self.label_index_dict[img_path]]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# single or mixture
# changed,RGB!=RGB
# one all train and test split 3:7
class WSSS4LUAD(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, split='test',
                 augmentation=False,
                 augmentation_types=None,
                 task='zeroshot_classification'
                 ):
        self.task = task
        self.augmentation_types = augmentation_types
        self.augmentation = augmentation
        self.split = split
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        self.label_dict = {
            "tumor": 0,
            "normal": 1
        }
        self.classes = list(self.label_dict)
        self.transform = transform
        self.data_root = root
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            "An H&E image of {c} tissue."
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.jpeg']):
        path_list = []
        if self.task == 'linear_probe':
            print(self.split)
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
                print('train')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            if self.split == 'train':
                root_dir = os.path.join(root, '1.training')
            elif self.split == 'test':
                root_dir = os.path.join(root, '1.training')
            path_list = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i.endswith('.png')]
            # random.shuffle(path_list)
        self.img_list = path_list
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image1 = Image.open(img_path)
        image = image1.convert('RGB')
        # print("=========",image1 == image)
        label_info = eval(img_path.split('/')[-1].split('-')[-1][:-4])
        label = 0 if label_info[0] == 1 else 1
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        # print(self.transform.__class__.__name__)
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # elif self.transform.__class__.__name__ == "VisionTransformer":
        #     print("Vision transformer")
        #     image = self.transform(image).unsqueeze(dim=0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# single or mixture
# changed,RGB ！= RGB
# train and test
class SICAPv2(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=True
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification'):
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_dict_map = {
            "NC": "Non-cancerous",
            "G3": "Gleason grade 3: Atrophic well differentiated and dense glandular regions",
            "G4": "Gleason grade 4: Cribriform, ill-formed, large-fused and papillary glandular patterns",
            "G5": "Gleason grade 5: Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns"
        }
        self.label_text_map = {
            "NC": "Non-cancerous, Benign, Normal tissue, Non-malignant, stroma",
            "G3": "Atrophic well differentiated and dense glandular regions, Low-grade cancer, Well-differentiated glands",
            "G4": "Cribriform, ill-formed, large-fused and papillary glandular patterns, Intermediate-grade cancer, Moderately differentiated glands",
            "G5": "Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns, High-grade cancer, Poorly differentiated or undifferentiated cells"
        }
        self.label_dict = {
            "NC": 0,
            "G3": 1,
            "G4": 2,
            "G5": 3,
        }

        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.jpg']):
        if self.split == 'train':
            path = root + '/partition/Test/Train.xlsx'
        elif self.split == 'test':
            path = root + '/partition/Test/Test.xlsx'
        data = pd.read_excel(path)
        data = data.drop(columns=['G4C'])

        label_list = ['NC', 'G3', 'G4', 'G5']
        label_dict = {}
        for idx, row in data.iterrows():
            img_name = row['image_name']
            labels = row.values[1:]
            if np.sum(labels) == 1:
                for label in labels:
                    if label == 1:
                        gt = label_list[np.where(labels == 1)[0][0]]
                        if gt not in label_dict:
                            label_dict[gt] = []
                        label_dict[gt].append(img_name)

        data_2_label_dict = {}
        img_list = []
        for key in label_dict:
            # random.shuffle(label_dict[key])
            for img in label_dict[key]:
                img = root + f'/images/{img}'
                img_list.append(img)
                data_2_label_dict[img] = key
        # random.shuffle(img_list)
        self.img_list = img_list
        self.data_2_label_dict = data_2_label_dict
        # print(self.data_2_label_dict)
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # print("-----------",type(image),type(image1))
        label = self.label_dict[self.data_2_label_dict[img_path]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


class PVQA(torch.utils.data.Dataset):
    """
    change
    question， image， ans
    """

    def __init__(self, root, transform=None, split="train", corrupt=False, corrupt_level=None, corruption_types=None):
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.img_list = None
        ans2label_path = os.path.join(root, 'qas', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(root, 'qas', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))  # ans--->num
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))  # num--->ans
        self.num_ans_candidates = len(self.ans2label)  # answer question number
        self.transform = transform
        self.data_root = root
        self.split = split
        self.process_file(root, split=self.split)

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, split, file_types=['.jpg']):
        if split == 'train':
            img_dir = os.path.join(self.data_root, 'qas', 'train', 'train_qa.pkl')
            id2dix = os.path.join(self.data_root, 'train_img_id2idx.pkl')
        elif split == 'test':
            img_dir = os.path.join(self.data_root, 'qas', 'test', 'test_qa.pkl')
            id2dix = os.path.join(self.data_root, 'test_img_id2idx.pkl')
        elif split == 'valid':
            split = 'val'
            img_dir = os.path.join(self.data_root, 'qas', 'val', 'val_qa.pkl')
            id2dix = os.path.join(self.data_root, 'val_img_id2idx.pkl')
        try:
            # 以二进制读取模式打开文件
            with open(img_dir, 'rb') as file:
                # 从文件中加载对象
                img_list = pickle.load(file)
                # print("成功加载文件内容:", data)
        except FileNotFoundError:
            print(f"未找到文件: {img_dir}")
        except pickle.UnpicklingError:
            print(f"无法反序列化文件: {img_dir}")
        except Exception as e:
            print(f"发生未知错误: {e}")
        i = 0
        for item in img_list:
            img_list[i]['image'] = os.path.join(root, "images", split, img_list[i]['image'] + '.jpg')
            i = i + 1
        self.img_list = img_list

        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]['image']
        image = Image.open(img_path).convert('RGB')
        quesion = self.img_list[idx]['question']
        answer = self.ans2label[self.img_list[idx]['answer']]
        if self.corrupt:
            print("--------------corrupt---------")
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            # print(image)
            image = self.transform(image)
        return image, quesion, answer


class BRACS_Rol(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, three=True, transform=None, corrupt=False, corrupt_level=None, corruption_types=None,
                 split='test'):
        self.split = split
        self.corruption_types = corruption_types
        self.three = three
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_dict_map = {
            "N": "Normal",
            "PB": "Pathological Benign",
            "UDH": "Usual Ductal Hyperplasia",
            "FEA": "Flat Epithelial Atypia",
            "ADH": "Atypical Ductal Hyperplasia",
            "DCIS": "Ductal Carcinoma In Situ",
            "IC": "Invasive Carcinoma",

        }
        self.label_dict = {
            "N": 0,
            "PB": 1,
            "UDH": 2,
            "FEA": 3,
            "ADH": 4,
            "DCIS": 5,
            "IC": 6
        }
        self.label_dict_map3 = {
            "N": "Benign Tumors",
            "PB": "Benign Tumors",
            "UDH": "Benign Tumors",
            "FEA": "Atypical Tumors",
            "ADH": "Atypical Tumors",
            "DCIS": "Malignant Tumors",
            "IC": "Malignant Tumors"
        }
        self.label_dict_3 = {
            "Benign Tumors": 0,
            "Atypical Tumors": 1,
            "Malignant Tumors": 2
        }
        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_dict_map.values()) if not self.three else list(self.label_dict_3)
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root, self.three)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        # self.single_template = [
        #     "An H&E image of {c} tissue."
        # ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, three, file_types=['.png']):
        print(self.split)
        if self.split == 'train':
            path = os.path.join(root, "latest_version", 'train')
        elif self.split == 'test':
            path = os.path.join(root, "latest_version", 'test')
        elif self.split == 'val':
            path = os.path.join(root, "latest_version", 'val')
        # print(path)
        path_list = []
        # print(path)
        for dirpath, dirnames, files in os.walk(path):
            # print("----------------")
            # print(files)
            for f in files:
                # print(f)
                if any(f.lower().endswith(ft) for ft in file_types):
                    path_list.append(os.path.join(dirpath, f))
        # random.shuffle(path_list)
        self.img_list = path_list
        # print(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        # print(img_path)
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')

        # print("-----------",type(image),type(image1))
        if self.three:
            label1 = os.path.basename(img_path).split('_')[2]
            label = self.label_dict_3[self.label_dict_map3[label1]]
        else:
            label = self.label_dict[os.path.basename(img_path).split('_')[2]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        return image, label


# one all
class GCHTID(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification'):
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_dict_map = {
            "ADI": "Adipose",
            "DEB": "Debtris",
            "LYM": "Lymphocytes",
            "MUC": "Mucus",
            "MUS": "Smooth Muscle",
            "NOR": "Normal Colon Mucosa",
            "STR": "Cancer-associated Stroma",
            "TUM": "Tumor",
        }
        # self.label_text_map = {
        #     "NC": "Non-cancerous, Benign, Normal tissue, Non-malignant, stroma",
        #     "G3": "Atrophic well differentiated and dense glandular regions, Low-grade cancer, Well-differentiated glands",
        #     "G4": "Cribriform, ill-formed, large-fused and papillary glandular patterns, Intermediate-grade cancer, Moderately differentiated glands",
        #     "G5": "Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns, High-grade cancer, Poorly differentiated or undifferentiated cells"
        # }
        self.label_dict = {
            "ADI": 0,
            "DEB": 1,
            "LYM": 2,
            "MUC": 3,
            "MUS": 4,
            "NOR": 5,
            "STR": 6,
            "TUM": 7,
        }

        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_dict_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            root = os.path.join(root, 'all_image')
            for dirpath, dirnames, files in os.walk(root):
                for f in files:
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[os.path.basename(os.path.dirname(img_path))]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# one all
class PXA(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification', rlabel=True):
        self.rlabel = rlabel
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_text_map = {
            "adjacent tissue": "adjacent tissue",
            "blood vessel": "blood vessel",
            "Necrosis": "Necrosis",
            "Tumor": "Tumor",
            "Red cells": "Red cells",
            "Tumor+blood vessel": "Tumor+blood vessel",
            "Tumor+calcification": "Tumor+calcification",
            "Tumor+giant cells": "Tumor+giant cells",
            "Tumor+Immune cells": "Tumor+Immune cells",
            "Tumor+Red cells": "Tumor+Red cells",
        }
        self.label_real_life_map = {
            "adjacent tissue": "adjacent non-tumor tissue",
            "blood vessel": "blood vessel structures",
            "Necrosis": "necrotic tissue",
            "Tumor": "tumor tissue",
            "Red cells": "red blood cells",
            "Tumor+blood vessel": "tumor tissue with blood vessels",
            "Tumor+calcification": "tumor tissue with calcification",
            "Tumor+giant cells": "tumor tissue with giant cells",
            "Tumor+Immune cells": "tumor tissue with immune cells",
            "Tumor+Red cells": "tumor tissue with red blood cells"
        }
        self.label_dict = {
            "adjacent tissue": 0,
            "blood vessel": 1,
            "Necrosis": 2,
            "Tumor": 3,
            "Red cells": 4,
            "Tumor+blood vessel": 5,
            "Tumor+calcification": 6,
            "Tumor+giant cells": 7,
            "Tumor+Immune cells": 8,
            "Tumor+Red cells": 9,
        }

        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        if self.rlabel:
            self.classes = list(self.label_real_life_map.values())
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            old_types = ['invalid_patches', '.ipynb_checkpoints']
            for dirpath, dirnames, files in os.walk(root):
                basename = os.path.basename(dirpath)
                if basename in old_types:
                    continue
                for f in files:
                    # print(f)
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[
            self.label_text_map[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


class PA(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification', rlabel=False):
        self.rlabel = rlabel
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_text_map = {
            "adjacent tissue": "adjacent tissue",
            "blood vessel": "blood vessel",
            "Necrosis": "Necrosis",
            "Red cells": "Red cells",
            "Tumor+blood vessel": "Tumor+blood vessel",
            "Tumor+cystic degeneration": "Tumor+cystic degeneration",
            "Tumor+Red cells": "Tumor+Red cells",
            "Tumor": "Tumor",
        }
        self.label_real_life_map = {
            "adjacent tissue": "adjacent non-tumor tissue",
            "blood vessel": "blood vessel structures",
            "Necrosis": "necrotic tissue",
            "Red cells": "red blood cells",
            "Tumor+blood vessel": "tumor tissue with blood vessels",
            "Tumor+cystic degeneration": "tumor tissue with cystic degeneration",
            "Tumor+Red cells": "tumor tissue with red blood cells",
            "Tumor": "tumor tissue"
        }
        self.label_dict = {
            "adjacent tissue": 0,
            "blood vessel": 1,
            "Necrosis": 2,
            "Red cells": 3,
            "Tumor+blood vessel": 4,
            "Tumor+cystic degeneration": 5,
            "Tumor+Red cells": 6,
            "Tumor": 7,
        }
        if self.task == 'linear_probe':
            # 去除类别过少的部分
            self.label_dict = {
                "adjacent tissue": 0,
                "blood vessel": 1,
                "Red cells": 2,
                "Tumor+blood vessel": 3,
                "Tumor+cystic degeneration": 4,
                "Tumor+Red cells": 5,
                "Tumor": 6,
            }
        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.rlabel:
            self.classes = list(self.label_real_life_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            old_types = ['invalid_patches', '.ipynb_checkpoints']
            for dirpath, dirnames, files in os.walk(root):
                basename = os.path.basename(dirpath)
                if basename in old_types:
                    continue
                for f in files:
                    # print(f)
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[
            self.label_text_map[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# one all
class SGCA(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification', rlabel=True):
        self.rlabel = rlabel
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_text_map = {
            "adjacent tissue": "adjacent tissue",
            "blood vessel": "blood vessel",
            "Red cells": "Red cells",
            "Tumor": "Tumor",
            "Tumor+blood vessel": "Tumor+blood vessel",
            "Tumor+calcification": "Tumor+calcification",
            "Tumor+cystic degeneration": "Tumor+cystic degeneration",
            "Tumor+giant cells": "Tumor+giant cells",
            "Tumor+Immune cells": "Tumor+Immune cells",
            "Tumor+Red cells": "Tumor+Red cells",
        }
        self.label_real_life_map = {
            "adjacent tissue": "adjacent non-tumor tissue",
            "blood vessel": "blood vessel structures",
            "Red cells": "red blood cells",
            "Tumor": "tumor tissue",
            "Tumor+blood vessel": "tumor tissue with blood vessels",
            "Tumor+calcification": "tumor tissue with calcification",
            "Tumor+cystic degeneration": "tumor tissue with cystic degeneration",
            "Tumor+giant cells": "tumor tissue with giant cells",
            "Tumor+Immune cells": "tumor tissue with immune cells",
            "Tumor+Red cells": "tumor tissue with red blood cells"
        }

        self.label_dict = {
            "adjacent tissue": 0,
            "blood vessel": 1,
            "Red cells": 2,
            "Tumor": 3,
            "Tumor+blood vessel": 4,
            "Tumor+calcification": 5,
            "Tumor+cystic degeneration": 6,
            "Tumor+giant cells": 7,
            "Tumor+Immune cells": 8,
            "Tumor+Red cells": 9,
        }

        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        if self.rlabel:
            self.classes = list(self.label_real_life_map.values())
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            old_types = ['invalid_patches', '.ipynb_checkpoints']
            for dirpath, dirnames, files in os.walk(root):
                basename = os.path.basename(dirpath)
                if basename in old_types:
                    continue
                for f in files:
                    # print(f)
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[
            self.label_text_map[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


# one all
class DIA(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification', rlabel=True):
        self.rlabel = rlabel
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_text_map = {
            "adjacent tissue": "adjacent tissue",
            "blood vessel": "blood vessel",
            "Tumor": "Tumor",
            "Tumor+blood vessel": "Tumor+blood vessel",
            "Tumor+Immune cells": "Tumor+Immune cells",
        }
        self.label_dict = {
            "adjacent tissue": 0,
            "blood vessel": 1,
            "Tumor": 2,
            "Tumor+blood vessel": 3,
            "Tumor+Immune cells": 4,
        }
        self.label_real_life_map = {
            "adjacent tissue": "adjacent non-tumor tissue",
            "blood vessel": "blood vessel structures",
            "Tumor": "tumor tissue",
            "Tumor+blood vessel": "tumor tissue with blood vessels",
            "Tumor+Immune cells": "tumor tissue with immune cells"
        }

        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.rlabel:
            self.classes = list(self.label_real_life_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            old_types = ['invalid_patches', '.ipynb_checkpoints']
            for dirpath, dirnames, files in os.walk(root):
                basename = os.path.basename(dirpath)
                if basename in old_types:
                    continue
                for f in files:
                    # print(f)
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[
            self.label_text_map[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label


class PMA(torch.utils.data.Dataset):
    """
    change
    """

    def __init__(self, root, transform=None, corrupt=False, corrupt_level=None, corruption_types=None, lower=False
                 , augmentation=False,
                 augmentation_types=None,
                 split='test', task='zeroshot_classification', rlabel=True):
        self.rlabel = rlabel
        self.task = task
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types
        self.split = split
        self.lower = lower
        self.corruption_types = corruption_types
        self.corrupt_level = corrupt_level
        self.corrupt = corrupt
        self.label_text_map = {
            "adjacent tissue": "adjacent tissue",
            "Necrosis": "Necrosis",
            "Tumor": "Tumor",
            "blood vessel": "blood vessel",
            "Tumor+blood vessel": "Tumor+blood vessel",
            "Tumor+cystic degeneration": "Tumor+cystic degeneration",
            "Tumor+mucoid degeneration": "Tumor+mucoid degeneration",
            "Tumor+Red cells": "Tumor+Red cells",
            "Red cells": "Red cells",
        }
        self.label_dict = {
            "adjacent tissue": 0,
            "Necrosis": 1,
            "Tumor": 2,
            "blood vessel": 3,
            "Tumor+blood vessel": 4,
            "Tumor+cystic degeneration": 5,
            "Tumor+mucoid degeneration": 6,
            "Tumor+Red cells": 7,
            "Red cells": 8,
        }
        self.label_real_life_map = {
            "adjacent tissue": "adjacent non-tumor tissue",
            "Necrosis": "necrotic tissue",
            "Tumor": "tumor tissue",
            "blood vessel": "blood vessel structures",
            "Tumor+blood vessel": "tumor tissue with blood vessels",
            "Tumor+cystic degeneration": "tumor tissue with cystic degeneration",
            "Tumor+mucoid degeneration": "tumor tissue with mucoid degeneration",
            "Tumor+Red cells": "tumor tissue with red blood cells",
            "Red cells": "red blood cells"
        }
        self.transform = transform
        self.data_root = root
        self.classes = list(self.label_text_map.values())
        if self.rlabel:
            self.classes = list(self.label_real_life_map.values())
        if self.lower:
            self.classes = [item.lower() if isinstance(item, str) else item for item in self.classes]
        self.img_list = []
        self.data_2_label_dict = {}
        self.process_file(self.data_root)
        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.single_template = [
            'An H&E image of {c}'
        ]

    def __len__(self):
        return len(self.img_list)

    def process_file(self, root, file_types=['.png']):
        path_list = []
        if self.task == 'linear_probe':
            if self.split == 'train':
                root_dir = os.path.join(root, 'train.csv')
            elif self.split == 'test':
                root_dir = os.path.join(root, 'test.csv')
            df = pd.read_csv(root_dir)
            path_list = df['data_path'].tolist()
        else:
            old_types = ['invalid_patches', '.ipynb_checkpoints']
            for dirpath, dirnames, files in os.walk(root):
                basename = os.path.basename(dirpath)
                if basename in old_types:
                    continue
                for f in files:
                    # print(f)
                    if any(f.lower().endswith(ft) for ft in file_types):
                        path_list.append(os.path.join(dirpath, f))
        self.img_list = path_list

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        # RGB版本
        image = image.convert('RGB')
        # image1 = []
        # print("-----------",type(image),type(image1))
        label = self.label_dict[
            self.label_text_map[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))]]
        if self.corrupt:
            image = ImageCorruptor.corrupt_image(
                image,
                corruption_types=self.corruption_types,
                severity=self.corrupt_level
            )
        if self.augmentation:
            aug_img = get_aug_image(self.augmentation_types, image)
            if self.transform is not None:
                if self.transform.__class__.__name__ == "CLIPProcessor":
                    image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                    aug_img = self.transform(images=aug_img, return_tensors="pt")['pixel_values'].squeeze(0)
                else:
                    image = self.transform(image)
                    aug_img = self.transform(aug_img)
            return image, aug_img, label
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            # print("CLIPProcessor already")
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            # print("other already")
            image = self.transform(image)
        if self.task == 'linear_probe':
            image1 = []
            return image, image1, label
        else:
            return image, label
