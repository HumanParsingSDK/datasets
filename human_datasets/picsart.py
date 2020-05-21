import os
import cv2
from pietoolbelt.datasets.common import get_root_by_env, BasicDataset

__all__ = ['PicsartDataset']


class PicsartDataset(BasicDataset):
    def __init__(self):
        base_dir = get_root_by_env('PICSART_DATASET')

        images_dir = os.path.join(base_dir, 'train')
        masks_dir = os.path.join(base_dir, 'train_mask')

        images_pathes = os.listdir(images_dir)
        images_pathes = sorted(images_pathes, key=lambda p: int(os.path.splitext(p)[0]))

        items = []
        for p in images_pathes:
            name = os.path.splitext(p)[0]
            mask_img = os.path.join(masks_dir, name + '.png')
            if os.path.exists(mask_img):
                path = {'data': os.path.join(images_dir, p), 'target': mask_img}
                items.append(path)

        super().__init__(items)

    def _interpret_item(self, item) -> any:
        img = cv2.cvtColor(cv2.imread(item['data']), cv2.COLOR_BGR2RGB)
        return {'data': img, 'target': cv2.imread(item['target'], cv2.IMREAD_UNCHANGED)}
