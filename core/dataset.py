import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from core.utils import (create_random_shape_with_random_motion, create_random_shape_with_random_motion_zoom_rotation, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip)
from pillow_lut import load_hald_image, rgb_color_enhance


def detect_edges_train(mask, kernel_size=3, dilation_iteration=3, threshold1=0, threshold2=0):
    edge_mask = cv2.Canny(mask.astype(np.uint8), threshold1=threshold1, threshold2=threshold2)

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # dilate (zoom) the edge mask
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iteration)

    return edge_mask

def generate_random_lut(choose_rate=0.2, param_range=None, min_sample=0):
    if param_range is None:
        param_range = {
            'brightness': (-0.5, 0.5),  
            'exposure': (-2.5, 2.5),
            'contrast': (-0.5, 2.5),
            'warmth': (-0.5, 0.5),
            'saturation': (-0.5, 2.5),
            'vibrance': (-0.5, 2.5),
            'hue': (0, 1.0),
            'gamma': (0.5, 5),
        }
    
    # Initially select parameters based on choose_rate
    params = {param: random.uniform(*rng) for param, rng in param_range.items() if random.random() < choose_rate}

    # Ensure at least min_sample parameters are selected
    while len(params) < min_sample:
        param_to_add = random.choice(list(param_range.keys()))
        if param_to_add not in params:
            params[param_to_add] = random.uniform(*param_range[param_to_add])

    return rgb_color_enhance(16, **params)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])
        self.norm_01 = args['norm_01'] if 'norm_01' in args else False
        
        json_path = args['meta_json_path']

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        # self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        random_lut = generate_random_lut()
        # read video frames
        frames = []
        frame_distorts = []
        masks = []
        mask_edges = []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)
            masks.append(all_masks[idx])
            # add freecolor input 
            kernel_size = np.random.randint(3, 5)
            dilation_iteration = np.random.randint(1,16)
            mask_edge = Image.fromarray(detect_edges_train(np.array(all_masks[idx]), kernel_size=kernel_size, dilation_iteration=dilation_iteration, threshold1=0, threshold2=0))
            mask_edges.append(mask_edge)
            img_distort = img.filter(random_lut)
            frame_distorts.append(img_distort)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    mask_edges.reverse()
                    frame_distorts.reverse()
        
        # normalizate, to tensors
        if self.norm_01:
            frame_tensors = self._to_tensors(frames)
            frame_distort_tensors = self._to_tensors(frame_distorts)
        else:
            frame_tensors = self._to_tensors(frames)* 2.0 - 1.0
            frame_distort_tensors = self._to_tensors(frame_distorts) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        mask_edge_tensors = self._to_tensors(mask_edges)

        return frame_tensors, frame_distort_tensors, mask_tensors, mask_edge_tensors, 'None', 'None', video_name, self.norm_01
