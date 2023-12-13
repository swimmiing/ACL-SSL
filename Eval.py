import torch
import os
import cv2

import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Optional

from torchvision import transforms as vt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from util import get_prompt_template
from viz_utils import draw_overall, draw_overlaid

import VGGSS.eval_utils as vggss_eval
import VGGSS.extend_eval_utils as exvggss_eval
import Flickr.eval_utils as flickr_eval
import Flickr.extend_eval_utils as exflickr_eval
import AVSBench.eval_utils as avsbench_eval
from typing import List, Optional, Tuple, Dict


@torch.no_grad()
def eval_vggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> Dict[str, float]:
    '''
    Evaluate provided  model on VGG-SS (VGG Sound Source) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        result_dict (Dict): Best AUC value (threshold optimized)

    Notes:
        The evaluation includes threshold optimization for VGG-SS.
    '''

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [vggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate VGG-SS({test_split}) dataset...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(
                (224, 224))
            gt_image = vt.ToPILImage()(bboxes[j]).resize((224, 224)).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    best_AUC = 0.0

    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict['cIoU'], audio_loc_dict['AUC'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/{test_split}({thr})', audio_loc_dict, epoch)

        best_AUC = audio_loc_dict['AUC'] if best_AUC < audio_loc_dict['AUC'] else best_AUC

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    result_dict = {'epoch': epoch, 'best_AUC': best_AUC}

    return result_dict


@torch.no_grad()
def eval_avsbench_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.setting

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [avsbench_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate AVSBench dataset({test_split})...")):
        images, audios, gts, labels, name = data['images'], data['audios'], data['gts'], data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], gts.to(model.device), thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.png')).resize(
                (224, 224))
            gt_image = Image.open(os.path.join(test_dataloader.dataset.gt_path, name[j] + '.png')).resize(
                (224, 224)).point(
                lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    rst_path = os.path.join(f'{result_dir}', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'mIoU={}, F={}\n'.format(audio_loc_dict['mIoU'], audio_loc_dict['Fmeasure'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/avs({test_split})({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_flickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [flickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Flickr dataset...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = (out_dict['heatmap'][j:j+1])
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(
                (224, 224))
            gt_image = vt.ToPILImage()(bboxes[j]).resize((224, 224)).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result (aggressive)
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict['cIoU'], audio_loc_dict['AUC'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/flickr({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exvggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [exvggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Extend VGG-SS dataset...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, labels, confs, name, thr)

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP={}, Max-F1={}\n'.format(audio_loc_dict['AP'], audio_loc_dict['Max-F1'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/exvggss({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exflickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [exflickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Extend Flickr dataset...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, labels, confs, name, thr)

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP={}, Max-F1={}\n'.format(audio_loc_dict['AP'], audio_loc_dict['Max-F1'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/exflickr({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()
