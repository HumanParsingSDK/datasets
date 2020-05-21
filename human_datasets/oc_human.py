import json
import os


import cv2
import numpy as np
from pietoolbelt.datasets.common import get_root_by_env, BasicDataset

__all__ = ['OCHumanDataset']


class OCHumanDataset(BasicDataset):
    def __init__(self):
        self._base_dir = get_root_by_env('OCHUMAN_DATASET')

        with open(os.path.join(self._base_dir, 'ochuman.json'), 'r') as data_in:
            data = json.load(data_in)

        items = {}
        for it in data['images']:
            if it['file_name'] not in items:
                items[it['file_name']] = {'ann': [], 'width': it['width'], 'height': it['height']}
            items[it['file_name']]['ann'].extend(it['annotations'])

        # with open(os.path.join(self._base_dir, 'ochuman_coco_format_test_range_0.00_1.00.json'), 'r') as data_in:
        #     data = json.load(data_in)

        items = [dict({'file_name': k}, **v) for k, v in items.items()]
        super().__init__(items)

    def _interpret_item(self, item) -> any:
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

        img = cv2.cvtColor(cv2.imread(os.path.join(self._base_dir, 'images', item['file_name'])), cv2.COLOR_BGR2RGB)
        return {'data': img, 'target': res_mask}
