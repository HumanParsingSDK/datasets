import os
import cv2
import numpy as np
import scipy.io

from piepline import BasicDataset
from pietoolbelt.datasets.common import get_root_by_env


class ClothingCoParsingDataset(BasicDataset):
    def __init__(self):
        root = get_root_by_env("CLOTHIG_CO_PARSING_DATASET")
        img_dir = os.path.join(root, 'photos')
        masks_dir = os.path.join(root, 'annotations')

        items = []
        for it in os.listdir(img_dir):
            img_name = os.path.splitext(it)[0]
            target_path = os.path.join(masks_dir, 'pixel-level', img_name + '.mat')
            if os.path.exists(target_path):
                items.append({'data': os.path.join(img_dir, it), 'target': target_path})
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        img = cv2.imread(item['data'])
        mask = scipy.io.loadmat(item['target'])['groundtruth']
        mask[mask > 0] = 1
        return {'data': cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'target': mask.astype(np.float32)}
