import torch

import numpy as np
import random
import os
from typing import Tuple, Optional


def get_prompt_template(mode: str = 'default') -> Tuple[str, int, int]:
    '''
    Generate a prompt template based on the specified mode.

    Args:
        mode (str, optional): The mode for selecting the prompt template. Default is 'default'.

    Returns:
        Tuple[str, int, int]: A tuple containing the generated prompt template, the position of the placeholder '{}',
                             and the length of the prompt.

    Notes:
        If the mode is 'random', a random prompt template is chosen from a predefined list.
    '''
    prompt_template = 'A photo of {}'

    if mode == 'random':
        prompt_templates = [
            'a photo of a {}', 'a photograph of a {}', 'an image of a {}', '{}',
            'a cropped photo of a {}', 'a good photo of a {}', 'a photo of one {}',
            'a bad photo of a {}', 'a photo of the {}', 'a photo of {}', 'a blurry photo of a {}',
            'a picture of a {}', 'a photo of a scene where {}'
        ]
        prompt_template = random.choice(prompt_templates)

    # Calculate prompt length and text position
    prompt_length = 1 + len(prompt_template.split(' ')) + 1 - 1  # eos, sos => 1 + 1, {} => -1
    text_pos_at_prompt = 1 + prompt_template.split(' ').index('{}')

    return prompt_template, text_pos_at_prompt, prompt_length


# Reproducibility
def fix_seed(seed: int = 0) -> None:
    '''
    Set seeds for random number generators to ensure reproducibility.

    Args:
        seed (int, optional): The seed value. Default is 0.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id: int) -> None:
    '''
    Set a seed for a worker process to ensure reproducibility in PyTorch DataLoader.

    Args:
        worker_id (int): The ID of the worker process.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)