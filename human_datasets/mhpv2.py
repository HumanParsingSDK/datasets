import os

import cv2
from pietoolbelt.mask_composer import MasksComposer
from piepline import BasicDataset
from pietoolbelt.datasets.common import get_root_by_env


class MHPV2(BasicDataset):
    def __init__(self):
        root = get_root_by_env('MHPV2_DATASET')

        items = []
        for set_dir in ['train', 'val']:
            cur_dir = os.path.join(root, set_dir)
            cur_images_dir = os.path.join(cur_dir, 'images')
            cur_annos_dir = os.path.join(cur_dir, 'parsing_annos')

            annotations = {}
            for ann in os.listdir(cur_annos_dir):
                idx = ann.split('_')[0]
                if idx not in annotations:
                    annotations[idx] = []
                annotations[idx].append(os.path.join(cur_annos_dir, ann))

            for file in os.listdir(cur_images_dir):
                items.append({'data': os.path.join(cur_images_dir, file), 'target': annotations[os.path.splitext(file)[0]]})
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        img = cv2.imread(item['data'])
        mask = MasksComposer((img.shape[0], img.shape[1]))

        for it in item['target']:
            cur_mask = cv2.cvtColor(cv2.imread(it, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
            cur_mask[cur_mask > 0] = 1
            mask.add_mask(cur_mask, cls=0)

        return {'data': img, 'target': mask.compose()}
