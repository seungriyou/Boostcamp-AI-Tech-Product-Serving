import io
from typing import List, Dict, Any

import torch, torchvision
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import pandas as pd
import numpy as np
from PIL import Image
import io
import os


def create_trash_label_colormap(class_colormap):
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]
    
    return colormap


def label_to_color_image(label, class_colormap):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap(class_colormap)

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/0511-upernet_swin_l_full_pl.py')
checkpoint_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/epoch_46.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device):
    model = init_segmentor(config_file, checkpoint_file, device=device)
    print('Model loaded :)')
    return model
