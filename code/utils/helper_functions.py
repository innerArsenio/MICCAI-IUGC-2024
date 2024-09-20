import glob 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import logging
import torch
import torch.nn as nn
import shutil
import uuid

import monai.transforms as T
from monai.transforms import (
    Compose,
    Resized,
    RandAffined,
    Rand2DElasticd,
    RandGaussianNoised,
    RandAdjustContrastd,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianSharpend
)
from monai.transforms import Transform
from monai.transforms import GridDistortion
from monai.utils import ensure_tuple_rep

def get_data_dict_2task(data_dir, pos_labeled_folder='Pos_Labeled', labeled_masks_folder='Labeled_Masks',
                        pos_unlabeled_folder='Pos_Unlabeled', synthetic_masks_folder='Synthetic_Masks', 
                        neg_folder='Neg', neg_masks_folder='Neg_Masks'):

    pos_labeled_images = sorted(glob.glob(os.path.join(data_dir, pos_labeled_folder, "*.png")))
    labeled_masks = sorted(glob.glob(os.path.join(data_dir, labeled_masks_folder, "*.png")))
    pos_unlabeled_images = sorted(glob.glob(os.path.join(data_dir, pos_unlabeled_folder, "*.png")))
    synthetic_masks = sorted(glob.glob(os.path.join(data_dir, synthetic_masks_folder, "*.png")))
    neg_images = sorted(glob.glob(os.path.join(data_dir, neg_folder, "*.png")))
    neg_masks = sorted(glob.glob(os.path.join(data_dir, neg_masks_folder, "*.png")))

    # Combine into a data dictionary
    data_dicts = []
    labeled_count = 0
    # Positive labeled
    for image, mask in zip(pos_labeled_images, labeled_masks):
        labeled_count += 1
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "labeled"})
    print(f"Number of labeled images: {labeled_count}")

    unlabeled_count = 0
    # Positive unlabeled (synthetic masks)
    for image, mask in zip(pos_unlabeled_images, synthetic_masks):
        unlabeled_count += 1
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "unlabeled"})
    print(f"Number of unlabeled images: {unlabeled_count}")
    

    neg_count = 0
    # Negative images with discard_gap logic
    discard_gap = 3
    for count, image in enumerate(neg_images):
        if count % discard_gap != 0:
            data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 0, "type": "negative"})
            neg_count += 1

    print(f"Number of negative images: {neg_count}")
    
    random.seed(42)
    random.shuffle(data_dicts)

    return data_dicts


def get_data_dict_2task_pt(data_dir, random_empty_mask_dir, pos_labeled_folder='Pos_Labeled', labeled_masks_folder='Labeled_Masks',
                           pos_unlabeled_folder='Pos_Unlabeled', synthetic_masks_folder='Synthetic_Masks', 
                           neg_folder='Neg'):

    pos_labeled_images = sorted(glob.glob(os.path.join(data_dir, pos_labeled_folder, "*.pt")))
    labeled_masks = sorted(glob.glob(os.path.join(data_dir, labeled_masks_folder, "*.pt")))
    pos_unlabeled_images = sorted(glob.glob(os.path.join(data_dir, pos_unlabeled_folder, "*.pt")))
    synthetic_masks = sorted(glob.glob(os.path.join(data_dir, synthetic_masks_folder, "*.pt")))
    neg_images = sorted(glob.glob(os.path.join(data_dir, neg_folder, "*.pt")))

    # Combine into a data dictionary
    data_dicts = []
    labeled_count = 0

    # Positive labeled
    for image, mask in zip(pos_labeled_images, labeled_masks):
        labeled_count += 1
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "labeled"})
    print(f"Number of labeled images: {labeled_count}")

    unlabeled_count = 0
    # Positive unlabeled (synthetic masks)
    for image, mask in zip(pos_unlabeled_images, synthetic_masks):
        unlabeled_count += 1
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "unlabeled"})
    print(f"Number of unlabeled images: {unlabeled_count}")
    
    neg_count = 0
    # Negative images (include all negative images)
    for image in neg_images:
        data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 0, "type": "negative"})
        neg_count += 1

    print(f"Number of negative images: {neg_count}")
    
    random.seed(42)
    random.shuffle(data_dicts)

    return data_dicts

#######

def get_val_dict_2task(data_dir, random_empty_mask_dir, pos_labeled_folder='pos_labeled', labeled_masks_folder='masks',
                        pos_folder='pos', neg_folder='neg'):

    pos_labeled_images = sorted(glob.glob(os.path.join(data_dir, pos_labeled_folder, "*.pt")))
    pos_images = sorted(glob.glob(os.path.join(data_dir, pos_folder, "*.pt")))
    labeled_masks = sorted(glob.glob(os.path.join(data_dir, labeled_masks_folder, "*.pt")))
    neg_images = sorted(glob.glob(os.path.join(data_dir, neg_folder, "*.pt")))

    # Combine into a data dictionary
    data_dicts = []
    
    # Positive labeled
    for image, mask in zip(pos_labeled_images, labeled_masks):
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "labeled"})

    # Positive unlabeled (synthetic masks)
    for image in pos_images:
        data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 1, "type": "unlabeled"})

    # Negative images
    for image in neg_images:
        data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 0, "type": "negative"})
        
    random.seed(42)
    random.shuffle(data_dicts)

    return data_dicts


####

def get_val_dict_2task_pt(data_dir, random_empty_mask_dir, pos_labeled_folder='pos_labeled', labeled_masks_folder='masks',
                        pos_folder='pos', neg_folder='neg'):

    pos_labeled_images = sorted(glob.glob(os.path.join(data_dir, pos_labeled_folder, "*.pt")))
    pos_images = sorted(glob.glob(os.path.join(data_dir, pos_folder, "*.pt")))
    labeled_masks = sorted(glob.glob(os.path.join(data_dir, labeled_masks_folder, "*.pt")))
    neg_images = sorted(glob.glob(os.path.join(data_dir, neg_folder, "*.pt")))

    # Combine into a data dictionary
    data_dicts = []
    
    # Positive labeled
    for image, mask in zip(pos_labeled_images, labeled_masks):
        data_dicts.append({"image": image, "label": mask, "class": 1, "type": "labeled"})

    # Positive unlabeled (synthetic masks)
    for image in pos_images:
        data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 1, "type": "unlabeled"})

    # Negative images
    for image in neg_images:
        data_dicts.append({"image": image, "label": random_empty_mask_dir, "class": 0, "type": "negative"})
        
    random.seed(42)
    random.shuffle(data_dicts)

    return data_dicts


def get_sample_weights(data_dicts):
    """
    Calculate sample weights for balanced sampling based on 'type'.

    Args:
        data_dicts (List[Dict]): List of dictionaries containing data information.

    Returns:
        List[float]: List of sample weights.
    """
    type_to_idx = {'negative': 0, 'labeled': 1, 'unlabeled': 2}
    type_indices = [type_to_idx[data['type']] for data in data_dicts]

    # Calculate type weights
    type_sample_counts = [type_indices.count(i) for i in range(len(type_to_idx))]
    type_weights = [1.0 / count for count in type_sample_counts]
    type_weights = list(np.array(type_weights)*np.array([0.5, 0.25, 0.25]))

    # Calculate sample weights
    sample_weights = [type_weights[idx] for idx in type_indices]

    return sample_weights
        
 
# Function to save checkpoint safely
def save_checkpoint(state, filename):
    temp_filename = filename + ".tmp"
    torch.save(state, temp_filename)
    shutil.move(temp_filename, filename)
    logging.info(f"Checkpoint saved: {filename}")


# Function to load checkpoint
def load_checkpoint(filename, model, optimizer, scaler):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', -1)
        best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
        CLS_TRAIN = checkpoint.get('CLS_TRAIN')
        START_BEST = checkpoint.get('START_BEST', 999)
        logging.info(f"Checkpoint loaded: {filename} (epoch {epoch})")
        return epoch, best_metric, best_metric_epoch, CLS_TRAIN, START_BEST
    else:
        logging.info(f"No checkpoint found at: {filename}")
        return 0, -1, -1, False, 999


def f1_recall_precision(tp,fp,fn):
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return f1_score, recall, precision


class LoadTensorsFromPath(Transform):
    def __init__(self, image_path_key='image', label_path_key='label'):
        self.image_path_key = image_path_key
        self.label_path_key = label_path_key

    def __call__(self, data):
        image_path = data[self.image_path_key]
        image = torch.load(image_path)

        if data['type'] == 'negative':
            label = torch.zeros((1,512,512))
        else:
            label_path = data[self.label_path_key]
            label = torch.load(label_path)

        return {self.image_path_key: image, self.label_path_key: label, 'class': data['class'], 'type' : data['type']}
    
class CustomRandGridDistortion(Transform):
    def __init__(self, image_key='image', label_key='label', num_cells=5, distort_limit=(-0.05, 0.05), mode='bilinear', padding_mode='border', prob=0.3):
        self.image_key = image_key
        self.label_key = label_key
        self.num_cells = num_cells
        self.distort_limit = distort_limit if isinstance(distort_limit, tuple) else (-distort_limit, distort_limit)
        self.mode = mode
        self.padding_mode = padding_mode
        self.prob = prob

    def randomize(self, spatial_shape):
        self.distort_steps = [
            1.0 + np.random.uniform(low=self.distort_limit[0], high=self.distort_limit[1], size=n_cells + 1)
            for n_cells in ensure_tuple_rep(self.num_cells, len(spatial_shape))
        ]

    def __call__(self, data):
        d = dict(data)

        # Apply the random distortion with a certain probability
        if np.random.rand() < self.prob:
            self.randomize(d[self.image_key].shape[1:])  # Assuming shape is (C, H, W)

            # Distort both image and label
            d[self.image_key] = self.apply_distortion(d[self.image_key])
            d[self.label_key] = self.apply_distortion(d[self.label_key])

        return d

    def apply_distortion(self, tensor):
        grid_distortion = GridDistortion(num_cells=self.num_cells, distort_steps=self.distort_steps, mode=self.mode, padding_mode=self.padding_mode)
        return grid_distortion(tensor)

        
class CustomRandColorJitter(Transform):
    def __init__(self, image_key='image', brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), saturation_limit=(-0.2, 0.2), hue_limit=(-0.1, 0.1), prob=0.15):
        self.image_key = image_key
        self.brightness_limit = brightness_limit if isinstance(brightness_limit, tuple) else (-brightness_limit, brightness_limit)
        self.contrast_limit = contrast_limit if isinstance(contrast_limit, tuple) else (-contrast_limit, contrast_limit)
        self.saturation_limit = saturation_limit if isinstance(saturation_limit, tuple) else (-saturation_limit, saturation_limit)
        self.hue_limit = hue_limit if isinstance(hue_limit, tuple) else (-hue_limit, hue_limit)
        self.prob = prob

    def randomize(self):
        self.brightness_factor = 1.0 + np.random.uniform(low=self.brightness_limit[0], high=self.brightness_limit[1])
        self.contrast_factor = 1.0 + np.random.uniform(low=self.contrast_limit[0], high=self.contrast_limit[1])
        self.saturation_factor = 1.0 + np.random.uniform(low=self.saturation_limit[0], high=self.saturation_limit[1])
        self.hue_factor = np.random.uniform(low=self.hue_limit[0], high=self.hue_limit[1])

    def __call__(self, data):
        d = dict(data)

        # Apply the color jitter with a certain probability
        if np.random.rand() < self.prob:
            self.randomize()
            d[self.image_key] = self.apply_color_jitter(d[self.image_key])

        return d

    def apply_color_jitter(self, tensor):
        tensor = torch.clamp(tensor * self.brightness_factor, 0, 1)
        tensor = torch.clamp((tensor - 0.5) * self.contrast_factor + 0.5, 0, 1)
        tensor = torch.clamp(tensor * self.saturation_factor, 0, 1)
        tensor = tensor + self.hue_factor  # Simple hue adjustment, consider a more complex one if necessary
        return torch.clamp(tensor, 0, 1)



def transformations():
    train_transforms = Compose(
        [
            LoadTensorsFromPath(image_path_key='image', label_path_key='label'),  # Load tensors from paths
            Resized(keys=['image', 'label'], spatial_size=(512, 512)), 
            RandFlipd(keys=["image", "label"], prob= 0.5,spatial_axis=1),
            RandAffined(keys=['image', 'label'], prob=0.3,mode='bilinear', rotate_range=[np.pi/12, np.pi/12], scale_range=[0.2,0.2], translate_range=(-15,15), shear_range=[0.1,0.1] ), 
            Rand2DElasticd(keys=['image', 'label'], prob=0.3, spacing=(15,15), magnitude_range=(0.3,0.3),spatial_size=(512, 512)), # Elastic deform
            CustomRandGridDistortion(image_key='image', label_key='label'),
            CustomRandColorJitter(image_key='image'),
            RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.1),
            RandGaussianSharpend(keys=['image'], prob=0.1),    # Add Gaussian noise
            RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(1.5, 4.5)),     # Adjust contrast
            NormalizeIntensityd(keys=['image']),
        ]
    )

    # NOTE: No random cropping in the validation data,
    # we will evaluate the entire image using a sliding window.
    val_transforms = Compose(
        [
            LoadTensorsFromPath(image_path_key='image', label_path_key='label'),  # Load tensors from paths
            Resized(keys=['image', 'label'], spatial_size=(512, 512)),          # Resize images and labels
            NormalizeIntensityd(keys=['image']),                                # Normalize intensity
        ]
    )
    return train_transforms, val_transforms


if __name__ == '__main__':
    dicts = get_val_dict_2task_pt(data_dir='/l/users/salem.alnasi/FrameDataset_pt_val',random_empty_mask_dir = '/l/users/salem.alnasi/FrameDataset_pt/Neg_Masks/20190726T095643_0_1.pt')
    



