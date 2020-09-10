import json
import os

import cv2
import numpy as np
from pietoolbelt.datasets.common import get_root_by_env, BasicDataset

__all__ = ['OCHumanDataset']


class OCHumanDataset(BasicDataset):
    def __init__(self, enable_masks: bool = False, enable_bbox: bool = False):
        self._base_dir = get_root_by_env('OCHUMAN_DATASET')

        with open(os.path.join(self._base_dir, 'ochuman.json'), 'r') as data_in:
            data = json.load(data_in)

        items = {}
        for it in data['images']:
            if it['file_name'] not in items:
                items[it['file_name']] = {'ann': [], 'width': it['width'], 'height': it['height']}
            items[it['file_name']]['ann'].extend(it['annotations'])

        self._enable_masks, self._enable_bbox = enable_masks, enable_bbox

        items = [dict({'file_name': k}, **v) for k, v in items.items()]
        super().__init__(items)

    @staticmethod
    def _get_mask(item) -> np.ndarray:
        res_mask = np.zeros((item['height'], item['width']), dtype=np.uint8)
        for ann in item['ann']:
            if ann['segms'] is None:
                continue
            for cntr in ann['segms']['outer']:
                res_mask = cv2.fillPoly(res_mask, np.array(cntr, dtype=np.int).reshape((1, len(cntr) // 2, 2)), 255)

        for ann in item['ann']:
            if ann['segms'] is None:
                continue
            for cntr in ann['segms']['inner']:
                res_mask = cv2.fillPoly(res_mask, np.array(cntr, dtype=np.int).reshape((1, len(cntr) // 2, 2)), 0)

        return res_mask

    @staticmethod
    def _get_bbox(item) -> np.ndarray:
        return np.array([ann['bbox'] for ann in item['ann']])

    def _interpret_item(self, item) -> any:
        img = cv2.cvtColor(cv2.imread(os.path.join(self._base_dir, 'images', item['file_name'])), cv2.COLOR_BGR2RGB)
        result = {'data': img}

        if self._enable_masks or self._enable_bbox:
            result['target'] = {}
        if self._enable_masks:
            result['target']['mask'] = self._get_mask(item)
        if self._enable_bbox:
            result['target']['bbox'] = self._get_bbox(item)
        return result
