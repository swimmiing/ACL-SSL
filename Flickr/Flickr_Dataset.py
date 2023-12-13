import torch
from torch.utils.data import Dataset

import numpy as np
import torchaudio
from torchvision import transforms as vt
from PIL import Image
import os
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union


def load_all_bboxes(annotation_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Load all bounding boxes from XML annotations.

    Args:
        annotation_dir (str): Directory containing XML annotation files.

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary containing bounding boxes for each file.
    """
    gt_bboxes = {}
    anno_files = os.listdir(annotation_dir)
    for filename in anno_files:
        file = filename.split('.')[0]
        gt = ET.parse(os.path.join(annotation_dir, filename)).getroot()
        bboxes = []
        for child in gt:
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index, ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text) / 256))
                bboxes.append(np.array(bbox))
        gt_bboxes[file] = bboxes

    return gt_bboxes


def bbox2gtmap(bboxes: List[List[int]]) -> np.ndarray:
    """
    Convert bounding boxes to numpy ground truth map.

    Args:
        bboxes (List[List[int]]): List of bounding boxes.

    Returns:
        np.ndarray: Ground truth map.
    """
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp
    gt_map = gt_map / 2
    gt_map[gt_map > 1] = 1

    return gt_map


class FlickrDataset(Dataset):
    def __init__(self, data_path: str, split: str, is_train: bool = True, set_length: int = 10,
                 input_resolution: int = 224):
        """
        Initialize Flickr SoundNet Dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Dataset split (Use csv file name in metadata directory).
            is_train (bool, optional): Whether it's a training set. Default is True.
            set_length (int, optional): Duration of input audio. Default is 10.
            input_resolution (int, optional): Resolution of input images. Default is 224.
        """
        super(FlickrDataset, self).__init__()

        self.SAMPLE_RATE = 16000
        self.split = split
        self.set_length = set_length
        self.csv_dir = 'Flickr/metadata/' + split + '.csv'
        self.data_path = data_path
        self.is_trainset = True

        if split.split('.')[0].split('_')[-1] == 'test':
            self.data_path = os.path.join(data_path, 'test')
            self.is_trainset = False

        ''' Audio files '''
        self.audio_path = os.path.join(self.data_path, 'audio')
        audio_files = set([fn.split('.wav')[0] for fn in os.listdir(self.audio_path) if fn.endswith('.wav')])

        ''' Image files '''
        self.image_path = os.path.join(self.data_path, 'frames')
        image_files = set([fn.split('.jpg')[0] for fn in os.listdir(self.image_path)])

        ''' Ground truth (Bounding box) '''
        gt_path = os.path.join(self.data_path, 'Annotations')
        if is_train:
            self.bbox_dict = None
        else:
            self.bbox_dict = load_all_bboxes(gt_path)

        ''' Ground truth (Text label) '''
        self.label_dict = {item[0]: item[1] for item in csv.reader(open(self.csv_dir))}

        ''' Available files'''
        subset = set([item[0] for item in csv.reader(open(self.csv_dir))])
        self.file_list = list(audio_files.intersection(image_files).intersection(subset))
        self.file_list = sorted(self.file_list)

        ''' Transform '''
        if is_train:
            self.image_transform = vt.Compose([
                vt.Resize((int(input_resolution * 1.1), int(input_resolution * 1.1)), vt.InterpolationMode.BICUBIC),
                vt.ToTensor(),
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
                vt.RandomCrop((input_resolution, input_resolution)),
                vt.RandomHorizontalFlip(),
            ])
        else:
            self.image_transform = vt.Compose([
                vt.Resize((input_resolution, input_resolution), vt.InterpolationMode.BICUBIC),
                vt.ToTensor(),
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
            ])

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.file_list)

    def get_audio(self, item: int) -> torch.Tensor:
        """
        Get audio data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            torch.Tensor: Audio data.
        """
        audio_file, _ = torchaudio.load(os.path.join(self.audio_path, self.file_list[item] + '.wav'))
        audio_file = audio_file.squeeze(0)

        # slicing or padding based on set_length
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * self.set_length):
            audio_file = audio_file[:self.SAMPLE_RATE * self.set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * self.set_length):
            pad_len = (self.SAMPLE_RATE * self.set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)

        return audio_file

    def get_image(self, item: int) -> Image.Image:
        """
        Get image data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Image.Image: Image data.
        """
        if self.is_trainset:
            file = os.listdir(os.path.join(self.image_path, self.file_list[item]))[0]  # Get first frame for train
            image_file = Image.open(os.path.join(self.image_path, self.file_list[item], file))
        else:
            image_file = Image.open(os.path.join(self.image_path, self.file_list[item] + '.jpg'))
        return image_file

    def get_bbox(self, item: int) -> Optional[torch.Tensor]:
        """
        Get ground truth data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Optional[torch.Tensor]: Ground truth data.
        """
        # Bounding Box
        if self.bbox_dict is None:
            return None
        else:
            bbox = self.bbox_dict[self.file_list[item]]
            return vt.ToTensor()(bbox2gtmap(bbox))

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str, str]]:
        """
        Get item from the dataset.

        Args:
            item (int): Index of the item.

        Returns:
            Dict[str, Union[torch.Tensor, torch.Tensor, Optinal[torch.Tensor], str, str]]: Data example
        """
        file_id = self.file_list[item]

        ''' Load data '''
        audio_file = self.get_audio(item)
        image_file = self.get_image(item)
        label = self.label_dict[self.file_list[item]].replace('_', ' ')
        bboxes = self.get_bbox(item)

        ''' Transform '''
        audio = audio_file
        image = self.image_transform(image_file)

        out = {'images': image, 'audios': audio, 'bboxes': bboxes, 'labels': label, 'ids': file_id}
        out = {key: value for key, value in out.items() if value is not None}
        return out


class ExtendFlickrDataset(Dataset):
    def __init__(self, data_path: str, set_length: int = 10, input_resolution: int = 224):
        """
        Initialize Extended Flickr Dataset.

        Args:
            data_path (str): Path to the dataset.
            set_length (int, optional): Duration of input audio. Default is 10.
            input_resolution (int, optional): Resolution of input images. Default is 224.
        """
        super(ExtendFlickrDataset, self).__init__()

        self.SAMPLE_RATE = 16000
        self.set_length = set_length
        self.data_path = os.path.join(data_path, 'test')
        self.csv_dir = 'Flickr/metadata/flickr_test.csv'
        self.extend_csv_dir = 'Flickr/metadata/flickr_test_plus_silent.csv'
        self.split = 'exflickr'

        ''' Audio files '''
        self.audio_path = os.path.join(data_path, 'extend_audio')
        audio_files = set([fn.split('.wav')[0] for fn in os.listdir(self.audio_path) if fn.endswith('.wav')])

        ''' Image files '''
        self.image_path = os.path.join(data_path, 'extend_frames')
        image_files = set([fn.split('.jpg')[0] for fn in os.listdir(self.image_path) if fn.endswith('.jpg')])

        ''' Ground truth (Bounding box) '''
        gt_path = os.path.join(self.data_path, 'Annotations')
        self.bbox_dict = load_all_bboxes(gt_path)

        ''' Ground truth (Text label) '''
        self.label_dict = {item[0]: item[1] for item in csv.reader(open(self.csv_dir))}

        ''' Available files'''
        subset = set([item[0] for item in csv.reader(open(self.csv_dir))])
        file_list = sorted(list(audio_files.intersection(image_files).intersection(subset)))
        self.image_files = [dt + '.jpg' for dt in file_list]
        self.audio_files = [dt + '.wav' for dt in file_list]
        self.bboxes = [self.bbox_dict[dt] for dt in file_list]
        self.labels = [self.label_dict[dt] for dt in file_list]

        ''' Add non-sounding files '''
        for item in csv.reader(open(self.extend_csv_dir)):
            if item[2] == 'non-sounding':
                self.image_files.append(f'{item[0]}.jpg')
                self.audio_files.append(f'{item[1]}.wav')
                self.bboxes.append([])
                self.labels.append('non-sounding')

        ''' Transform '''
        self.image_transform = vt.Compose([
            vt.Resize((input_resolution, input_resolution), vt.InterpolationMode.BICUBIC),
            vt.ToTensor(),
            vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
        ])

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.image_files)

    def get_audio(self, item: int) -> torch.Tensor:
        """
        Get audio data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            torch.Tensor: Audio data.
        """
        audio_file, _ = torchaudio.load(os.path.join(self.audio_path, self.audio_files[item]))
        audio_file = audio_file.squeeze(0)

        # slicing or padding based on set_length
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * self.set_length):
            audio_file = audio_file[:self.SAMPLE_RATE * self.set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * self.set_length):
            pad_len = (self.SAMPLE_RATE * self.set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)

        return audio_file

    def get_image(self, item: int) -> Image.Image:
        """
        Get image data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Image.Image: Image data.
        """
        image_file = Image.open(os.path.join(self.image_path, self.image_files[item]))
        return image_file

    def get_bbox(self, item: int) -> Optional[torch.Tensor]:
        """
        Get ground truth data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Optional[torch.Tensor]: Ground truth data.
        """
        # Bounding Box
        if len(self.bboxes[item]) == 0:
            return vt.ToTensor()(np.zeros([224, 224]))
        else:
            bbox = self.bboxes[item]
            return vt.ToTensor()(bbox2gtmap(bbox))

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str, str]]:
        """
        Get item from the dataset.

        Args:
            item (int): Index of the item.

        Returns:
            Dict[str, Union[torch.Tensor, torch.Tensor, Optinal[torch.Tensor], str, str]]: Data example
        """
        ''' Load data '''
        audio_file = self.get_audio(item)
        image_file = self.get_image(item)
        bboxes = self.get_bbox(item)
        label = self.labels[item]
        file_id = self.image_files[item].split('.')[0] + '_' + self.audio_files[item].split('.')[0]

        ''' Transform '''
        audio = audio_file
        image = self.image_transform(image_file)

        out = {'images': image, 'audios': audio, 'bboxes': bboxes, 'labels': label, 'ids': file_id}
        out = {key: value for key, value in out.items() if value is not None}
        return out
