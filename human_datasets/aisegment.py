import os
import cv2
from pietoolbelt.datasets.common import get_root_by_env, BasicDataset

__all__ = ['AISegmentDataset']


class AISegmentDataset(BasicDataset):
    def __init__(self):
        base_dir = get_root_by_env('AISEGMENT_DATASET')

        items = []

        for root, dirs, files in os.walk(os.path.join(base_dir, 'matting')):
            for file in files:
                if os.path.splitext(file)[1] in ['.png', '.jpg']:
                    mask_img = os.path.join(root, file)
                    img = os.path.join(base_dir, 'clip_img', os.path.basename(os.path.dirname(os.path.dirname(mask_img))),
                                       os.path.basename(os.path.dirname(mask_img)).replace('matting', 'clip'), file.replace('png', 'jpg'))

                    if os.path.exists(mask_img) and os.path.exists(img):
                        items.append({'mask': mask_img, 'img': img})

        super().__init__(items)

    def _interpret_item(self, item) -> any:
        mask = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)[:, :, 3]
        return {'data': cv2.cvtColor(cv2.imread(item['img']), cv2.COLOR_BGR2RGB), 'target': mask}
