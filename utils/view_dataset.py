import argparse
import os
import sys

import cv2
from pietoolbelt.datasets.utils import InstanceSegmentationDataset
from pietoolbelt.viz import ContourVisualizer, ColormapVisualizer, MulticlassColormapVisualizer
import numpy as np

from chip import CHIP
from human_datasets.aisegment import AISegmentDataset
from human_datasets.clothing_co_parsing import ClothingCoParsingDataset
from human_datasets.picsart import PicsartDataset
from human_datasets.oc_human import OCHumanDataset
from human_datasets.supervisely_person import SuperviselyPersonDataset
from lip import LIP
from mhpv2 import MHPV2


def vis_picsart(vis_type: str):
    dataset = PicsartDataset()

    vis = ContourVisualizer(thickness=1)
    for i, it in enumerate(dataset):
        img = vis.process_img(cv2.cvtColor(it['data'], cv2.COLOR_RGB2BGR), it['target'])

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_ochuman(vis_type: str):
    dataset = OCHumanDataset()

    # vis = ContourVisualizer(thickness=1)
    vis = ColormapVisualizer(proportions=[0.3, 0.7])
    for i, it in enumerate(dataset):
        img = vis.process_img(cv2.cvtColor(it['data'], cv2.COLOR_RGB2BGR), it['target'].astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_aisegment(vis_type: str):
    dataset = AISegmentDataset()

    # vis = ContourVisualizer(thickness=1)
    vis = ColormapVisualizer(proportions=[0.3, 0.7])
    for i, it in enumerate(dataset):
        img = vis.process_img(cv2.cvtColor(it['data'], cv2.COLOR_RGB2BGR), it['target'].astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_supervisely(vis_type: str):
    dataset = InstanceSegmentationDataset(SuperviselyPersonDataset()).enable_border(3, 0)
    vis = MulticlassColormapVisualizer(main_class=0, proportions=[0.5, 0.5],
                                       other_colors=[[255, 255, 255], [255, 255, 255], [255, 255, 255]])
    for i, it in enumerate(dataset):
        img = vis.process_img(cv2.cvtColor(it['data'], cv2.COLOR_RGB2BGR), (255 * it['target']).astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_clothing_co_parsing(vis_type: str):
    dataset = ClothingCoParsingDataset()
    vis = ColormapVisualizer(proportions=[0.5, 0.5])

    for i, it in enumerate(dataset):
        img = vis.process_img(it['data'], (255 * it['target']).astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_mhpv2(vis_type: str):
    dataset = MHPV2()
    vis = ColormapVisualizer(proportions=[0.5, 0.5])

    for i, it in enumerate(dataset):
        img = vis.process_img(it['data'], (255 * it['target']).astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_lip(vis_type: str):
    dataset = LIP()
    vis = ColormapVisualizer(proportions=[0.5, 0.5])

    for i, it in enumerate(dataset):
        img = vis.process_img(it['data'], (255 * it['target']).astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


def vis_chip(vis_type: str):
    dataset = CHIP()
    vis = ColormapVisualizer(proportions=[0.5, 0.5])

    for i, it in enumerate(dataset):
        img = vis.process_img(it['data'], (255 * it['target']).astype(np.uint8))

        if vis_type == 'window':
            cv2.imshow('img', img)
            cv2.waitKey()
        elif vis_type == 'file':
            cv2.imwrite(os.path.join('img', "{}.jpg".format(i)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dataset')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to visualize', required=True,
                        choices=['picsart', 'supervisely_person', 'ochuman', 'aisegment', 'clothing_co_parsing', 'mhpv2', 'lip',
                                 'chip'])
    parser.add_argument('-v', '--vis', type=str, help='How to visualize dataset', required=True,
                        choices=['window', 'file'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.vis == 'file':
        if not os.path.exists('img') or not os.path.isdir('img'):
            os.makedirs('img')

    if args.dataset == 'picsart':
        vis_picsart(args.vis)
    elif args.dataset == 'ochuman':
        vis_ochuman(args.vis)
    elif args.dataset == 'aisegment':
        vis_aisegment(args.vis)
    elif args.dataset == 'supervisely_person':
        vis_supervisely(args.vis)
    elif args.dataset == 'clothing_co_parsing':
        vis_clothing_co_parsing(args.vis)
    elif args.dataset == 'mhpv2':
        vis_mhpv2(args.vis)
    elif args.dataset == 'lip':
        vis_lip(args.vis)
    elif args.dataset == 'chip':
        vis_chip(args.vis)
    else:
        print("Dataset {} doesn't implemented".format(args.dataset))
