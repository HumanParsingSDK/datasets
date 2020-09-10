import os

import cv2
import numpy as np
from piepline import BasicDataset
from pietoolbelt.datasets.common import get_root_by_env


class CHIP(BasicDataset):
    def __init__(self, enable_masks: bool = False, enable_bbox: bool = False):
        self._enable_masks, self._enable_bbox = enable_masks, enable_bbox

        root = get_root_by_env('CHIP_DATASET')

        items = []
        for set_dir in ['Training', 'Validation']:
            cur_images_dir = os.path.join(root, set_dir, "Images")
            cur_annos_dir = os.path.join(root, set_dir, "Category_ids")
            cur_instances_dir = os.path.join(root, set_dir, "Human_ids")

            for img_file in os.listdir(cur_images_dir):
                mask_file = os.path.splitext(img_file)[0] + '.png'
                items.append({'data': os.path.join(cur_images_dir, img_file), 'mask': os.path.join(cur_annos_dir, mask_file),
                              'instances': os.path.join(cur_instances_dir, mask_file)})
        super().__init__(items)

    @staticmethod
    def _get_mask(item) -> np.ndarray:
        mask = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)
        mask[mask > 0] = 1
        return mask

    @staticmethod
    def _get_bboxes(item) -> np.ndarray:
        mask = cv2.imread(item['instances'], cv2.IMREAD_UNCHANGED)

        bboxes = []
        for i in range(1, mask.max() + 1):
            cur_mask = np.where(mask == i, mask, 0)
            x, y, w, h = cv2.boundingRect(cur_mask)
            bboxes.append([x, y, x + w, y + h])
        return np.array(bboxes)

    def _interpret_item(self, item) -> any:
        img = cv2.imread(item['data'])
        res = {'data': cv2.cvtColor(img, cv2.COLOR_BGR2RGB)}

        if self._enable_bbox or self._enable_masks:
            res['target'] = {}

        if self._enable_masks:
            res['target']['mask'] = self._get_mask(item)
        if self._enable_bbox:
            res['target']['bbox'] = self._get_bboxes(item)
        return res
