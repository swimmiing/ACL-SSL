import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms as vt
from PIL import Image
import os
import csv
from typing import Dict, Optional, Union, List


class AVSBenchDataset(Dataset):
    def __init__(self, data_path: str, split: str, is_train: bool = True, set_length: int = 10,
                 input_resolution: int = 224) -> None:
        """
        Initialize AVSBench Dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Dataset split (Use csv file name in metadata directory).
            is_train (bool, optional): Whether it's a training set. Default is True.
            set_length (int, optional): Duration of input audio. Default is 10.
            input_resolution (int, optional): Resolution of input images. Default is 224.
        """
        super(AVSBenchDataset, self).__init__()

        self.SAMPLE_RATE = 16000
        self.split = split
        self.set_length = set_length
        self.csv_dir = 'AVSBench/metadata/' + split + '.csv'
        self.setting = split.split('_')[1]

        ''' Audio files '''
        self.audio_path = os.path.join(data_path, self.setting, 'audio_wav')
        audio_files = set([fn.split('.wav')[0] for fn in os.listdir(self.audio_path) if fn.endswith('.wav')])

        ''' Image files '''
        self.image_path = os.path.join(data_path, self.setting, 'visual_frames')
        image_files = set([fn.split('.png')[0] for fn in os.listdir(self.image_path) if fn.endswith('.png')])

        ''' Ground truth (Bounding box) '''
        if is_train:
            self.gt_path = None
        else:
            self.gt_path = os.path.join(data_path, self.setting, 'gt_masks')

        ''' Ground truth (Text label) '''
        self.label_dict = {item[0]: item[1] for item in csv.reader(open(self.csv_dir))}

        ''' Available files'''
        subset = set([item[0] for item in csv.reader(open(self.csv_dir))])
        self.file_list = sorted(list(image_files.intersection(subset)))

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
        audio_file, _ = torchaudio.load(os.path.join(self.audio_path, self.file_list[item][:-2] + '.wav'))
        audio_file = torch.concat([audio_file[0], audio_file[1]], dim=0)  # Stereo 5 sec -> 10 sec
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
        image_file = Image.open(os.path.join(self.image_path, self.file_list[item] + '.png'))
        return image_file

    def get_gt(self, item: int) -> Optional[torch.Tensor]:
        """
        Get ground truth data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Optional[torch.Tensor]: Ground truth data.
        """
        # Ground truth
        if self.gt_path is None:
            return None
        else:
            gt = vt.ToTensor()(
                Image.open(os.path.join(self.gt_path, self.file_list[item] + '.png')).convert('1')).float()
            return gt

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
        gts = self.get_gt(item)

        ''' Transform '''
        audio = audio_file
        image = self.image_transform(image_file)

        out = {'images': image, 'audios': audio, 'gts': gts, 'labels': label, 'ids': file_id}
        out = {key: value for key, value in out.items() if value is not None}
        return out
