import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def draw_overall(result_dir: str, original_image: Image.Image, gt_image: Image.Image, heatmap_image: Image.Image,
                 seg_image: Image.Image, label: str, name: str) -> None:
    '''
    Draw an overall result figure with original, ground truth, heatmap, and binarized heatmap images.

    Args:
        result_dir (str): Directory to save the figure.
        original_image (Image.Image): Original image.
        gt_image (Image.Image): Ground truth image.
        heatmap_image (Image.Image): Heatmap image.
        seg_image (Image.Image): Binarized heatmap image.
        label (str): Label information.
        name (str): Name identifier.

    Returns:
        None
    '''
    result_box_shape = (2, 2)

    # Calculate IoU
    np_gt = 1 - (np.array(gt_image) / 255)
    np_seg = 1 - np.array(seg_image) / 255
    seg_iou = (np_seg * np_gt).sum() / (((np_seg + np_gt) > 0).sum() + 1e-6)

    # Draw overall result figure
    image_width, image_height = 224, 224
    padding = 10
    canvas_width = (image_width * result_box_shape[1]) + (padding * (result_box_shape[1] + 1))
    canvas_height = (image_height * result_box_shape[0]) + (padding * (result_box_shape[0] + 1))
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    out_text = [f'Label: {label}', f'IoU: {seg_iou:.2f}']

    resized_images = [original_image, gt_image, heatmap_image, seg_image]
    for i in range(np.prod(result_box_shape)):
        row = i % 2
        col = i // 2
        x = (image_width + padding) * col
        y = (image_height + padding) * row
        canvas.paste(resized_images[i], (x, y))

        if row == 1:
            text = out_text[i // 2]
            text_x = (image_width + padding) * col
            text_y = (image_height + padding) * row + image_height + padding
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    # save fig
    output_path = os.path.join(result_dir, 'overall')
    os.makedirs(output_path, exist_ok=True)
    canvas.save(os.path.join(output_path, f'{name}.jpg'))


def draw_overlaid(result_dir: str, original_image: Image.Image, heatmap_image: Image.Image, name: str) -> None:
    '''
    Draw an overlaid figure with the original image and heatmap.

    Args:
        result_dir (str): Directory to save the figure.
        original_image (Image.Image): Original image.
        heatmap_image (Image.Image): Heatmap image.
        name (str): Name identifier.

    Returns:
        None
    '''
    heatmap_image = cv2.applyColorMap(np.array(heatmap_image), cv2.COLORMAP_JET)
    overlaid_image = cv2.addWeighted(np.array(original_image), 0.5, heatmap_image, 0.5, 0)
    overlaid_image = Image.fromarray(overlaid_image)

    # save fig
    output_path = os.path.join(result_dir, 'overlaid')
    os.makedirs(output_path, exist_ok=True)
    overlaid_image.save(os.path.join(output_path, f'{name}.jpg'))
