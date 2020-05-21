import base64
import json
import os
import zlib
import numpy as np
import cv2

from pietoolbelt.datasets.common import get_root_by_env, BasicDataset

__all__ = ['SuperviselyPersonDataset']


class SuperviselyPersonDataset(BasicDataset):
    def __init__(self, include_not_marked_people: bool = False, include_neutral_objects: bool = False):
        path = get_root_by_env('SUPERVISELY_DATASET')

        items = {}
        for root, path, files in os.walk(path):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext == '.json':
                    item_type = 'target'
                    name = os.path.splitext(name)[0]
                elif ext == '.png' or ext == '.jpg':
                    item_type = 'data'
                else:
                    continue

                if name in items:
                    items[name][item_type] = os.path.join(root, file)
                else:
                    items[name] = {item_type: os.path.join(root, file)}

        final_items = []
        for item, data in items.items():
            if 'data' in data and 'target' in data:
                final_items.append(data)

        final_items = self._filter_items(final_items, include_not_marked_people, include_neutral_objects)
        self._use_border_as_class = False
        self._border_thikness = None
        super().__init__(final_items)

    def _interpret_item(self, item) -> any:
        return {'data': cv2.imread(item['data']),
                'target': {'masks': [SuperviselyPersonDataset._object_to_mask(obj) for obj in item['target']['objects']],
                           'size': item['target']['size']}}

    @staticmethod
    def _object_to_mask(obj):
        obj_mask, origin = None, None

        if obj['bitmap'] is not None:
            z = zlib.decompress(base64.b64decode(obj['bitmap']['data']))
            n = np.fromstring(z, np.uint8)

            origin = np.array([obj['bitmap']['origin'][0], obj['bitmap']['origin'][1]], dtype=np.uint16)
            obj_mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
            obj_mask[obj_mask > 0] = 1

        elif len(obj['points']['interior']) + len(obj['points']['exterior']) > 0:
            pts = np.array(obj['points']['exterior'], dtype=np.int)
            origin = pts.min(axis=0)
            shape = pts.max(axis=0) - origin
            obj_mask = cv2.drawContours(np.zeros((shape[1], shape[0]), dtype=np.uint8), [pts - origin], -1, 1, cv2.FILLED)

            if len(obj['points']['interior']) > 0:
                for pts in obj['points']['interior']:
                    pts = np.array(pts, dtype=np.int)
                    obj_mask = cv2.drawContours(obj_mask, [pts - origin], -1, 0, cv2.FILLED)

        origin = np.array([origin[1], origin[0]], dtype=np.int)
        return obj_mask, origin

    @staticmethod
    def _filter_items(items, include_not_marked_people: bool, include_neutral_objects: bool) -> list:
        res = []
        for item in items:
            with open(item['target'], 'r') as file:
                target = json.load(file)

            if not include_not_marked_people and ('not-marked-people' in [n['name'] for n in target['tags'] if 'value' in n]):
                continue

            if not include_neutral_objects:
                res_objects = []
                for obj in target['objects']:
                    if obj['classTitle'] != 'neutral':
                        res_objects.append(obj)
                target['objects'] = res_objects

            res.append({'data': item['data'], 'target': target})

        return res
