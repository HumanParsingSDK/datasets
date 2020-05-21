import os

import cv2
from piepline import BasicDataset
from pietoolbelt.datasets.common import get_root_by_env


class CHIP(BasicDataset):
    def __init__(self):
        root = get_root_by_env('CHIP_DATASET')

        items = []
        for set_dir in ['Training', 'Validation']:
            cur_images_dir = os.path.join(root, set_dir, "Images")
            cur_annos_dir = os.path.join(root, set_dir, "Category_ids")

            for img_file in os.listdir(cur_images_dir):
                mask_file = os.path.splitext(img_file)[0] + '.png'
                items.append({'data': os.path.join(cur_images_dir, img_file), 'target': os.path.join(cur_annos_dir, mask_file)})
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        img = cv2.imread(item['data'])
        mask = cv2.imread(item['target'], cv2.IMREAD_UNCHANGED)
        mask[mask > 0] = 1
        return {'data': img, 'target': mask}
