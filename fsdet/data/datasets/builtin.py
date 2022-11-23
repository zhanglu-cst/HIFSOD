# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""
import copy
import os

from global_var import DATASET_DIR
from .builtin_meta import _get_coco_fewshot_instances_meta_general
from .meta_coco import register_meta_coco


def register_all_coco(root = "/"):
    bird_anns = [
        ('bird_base_train', DATASET_DIR['train_img_dir'], DATASET_DIR['train_anns']),
        ('bird_test_novel', DATASET_DIR['val_img_dir'], DATASET_DIR['val_anns']),
        ('bird_test_base', DATASET_DIR['val_img_dir'], DATASET_DIR['val_anns']),
        ('bird_test_all', DATASET_DIR['val_img_dir'], DATASET_DIR['val_anns'])
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                bird_anns.append((name, DATASET_DIR['train_img_dir'], ""))
    # base_ann_path = r'xxx/xxx//bird_fsod/annotations/base_train.json'
    base_ann_path = DATASET_DIR['cats_base_ann']
    # novel_ann_path = r'xxx/xxx//bird_fsod/annotations/novel.json'
    novel_ann_path = DATASET_DIR['cats_novel_ann']
    meta_info = _get_coco_fewshot_instances_meta_general(coco_path_base = base_ann_path,
                                                         coco_path_novel = novel_ann_path)
    for name, imgdir, annofile in bird_anns:
        copy_meta_info = copy.deepcopy(meta_info)
        register_meta_coco(
                name,
                copy_meta_info,
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
        )


# Register them all under "./datasets"
register_all_coco()
# register_all_lvis()
# register_all_pascal_voc()
