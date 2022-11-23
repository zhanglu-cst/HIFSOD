import argparse
import os

import torch
from pycocotools.coco import COCO

path_base_json = r'xxx/xxx/xxx/dataset/annotations/base_train.json'
path_novel_json = r'xxx/xxx/xxx/dataset/annotations/novel.json'


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
            "--src1", type = str,
            default = "xxx/xxx/xxx/hifsod/checkpoints/origin_version/R_101_FPN_base/model_final.pth",
            help = "Path to the main checkpoint"
    )
    parser.add_argument(
            "--src2",
            type = str,
            default = "xxx/xxx/xxx/hifsod/checkpoints/origin_version/2_shot_novel/model_final.pth",
            help = "Path to the secondary checkpoint (for combining)",
    )
    parser.add_argument(
            "--save-dir", type = str,
            default = "xxx/xxx/xxx/hifsod/checkpoints/origin_version/model_files/",
            help = "Save directory"
    )
    parser.add_argument(
            "--tar-name",
            type = str,
            default = "model_combine",
            help = "Name of the new ckpt",
    )
    # Surgery method
    parser.add_argument(
            "--method",
            choices = ["combine", "remove", "randinit"],
            required = True,
            default = 'combine',
            help = "Surgery method. combine = "
                   "combine checkpoints. remove = for fine-tuning on "
                   "novel dataset, remove the final layer of the "
                   "base detector. randinit = randomly initialize "
                   "novel weights.",
    )
    # Targets
    parser.add_argument(
            "--param-name",
            type = str,
            nargs = "+",
            default = [
                "roi_heads.box_predictor.cls_score",
                "roi_heads.box_predictor.bbox_pred",
            ],
            help = "Target parameter names",
    )

    # Dataset
    args = parser.parse_args()
    return args


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """

    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2 = None):
        weight_name = param_name + (".weight" if is_weight else ".bias")
        pretrained_weight = ckpt["model"][weight_name]
        prev_cls = pretrained_weight.size(0)
        if "cls_score" in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)

        for i, c in enumerate(BASE_CLASSES):
            idx = i
            if "cls_score" in param_name:
                new_weight[IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[
                IDMAP[c] * 4: (IDMAP[c] + 1) * 4
                ] = pretrained_weight[idx * 4: (idx + 1) * 4]

        if "cls_score" in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        ckpt["model"][weight_name] = new_weight

    surgery_loop(args, surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. Feature extractor weights are
    from the base detector. Only the final layer weights are combined.
    """

    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2 = None):
        if not is_weight and param_name + ".bias" not in ckpt["model"]:
            return
        weight_name = param_name + (".weight" if is_weight else ".bias")
        pretrained_weight = ckpt["model"][weight_name]  # [61, 1024]
        len_ckpt_1 = pretrained_weight.size(0)  # 61
        if "cls_score" in param_name:
            len_ckpt_1 -= 1  # 60
        if is_weight:
            feat_size = pretrained_weight.size(1)  # 1024
            new_weight = torch.rand((tar_size, feat_size))  # [81,1024]
        else:
            new_weight = torch.zeros(tar_size)

        for i, c in enumerate(BASE_CLASSES):
            idx = i
            if "cls_score" in param_name:
                new_weight[IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[
                IDMAP[c] * 4: (IDMAP[c] + 1) * 4
                ] = pretrained_weight[idx * 4: (idx + 1) * 4]

        ckpt2_weight = ckpt2["model"][weight_name]

        for i, c in enumerate(NOVEL_CLASSES):
            if "cls_score" in param_name:
                new_weight[IDMAP[c]] = ckpt2_weight[i]
            else:
                new_weight[
                IDMAP[c] * 4: (IDMAP[c] + 1) * 4
                ] = ckpt2_weight[i * 4: (i + 1) * 4]
        if "cls_score" in param_name:
            new_weight[-1] = pretrained_weight[-1]
        # else:
        #     if "cls_score" in param_name:
        #         new_weight[len_ckpt_1:-1] = ckpt2_weight[:-1]
        #         new_weight[-1] = pretrained_weight[-1]
        #     else:
        #         new_weight[len_ckpt_1:] = ckpt2_weight
        ckpt["model"][weight_name] = new_weight

    surgery_loop(args, surgery)


def surgery_loop(args, surgery):
    # Load checkpoints
    print('load scr1')
    ckpt = torch.load(args.src1)
    print('load scr1 finish')
    if args.method == "combine":
        ckpt2 = torch.load(args.src2)
        save_name = args.tar_name + "_combine.pth"
    else:
        ckpt2 = None
        save_name = (
                args.tar_name
                + "_"
                + ("remove" if args.method == "remove" else "surgery")
                + ".pth"
        )
    if args.save_dir == "":
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok = True)
    reset_ckpt(ckpt)

    # Remove parameters
    if args.method == "remove":
        for param_name in args.param_name:
            del ckpt["model"][param_name + ".weight"]
            if param_name + ".bias" in ckpt["model"]:
                del ckpt["model"][param_name + ".bias"]
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
    for idx, (param_name, tar_size) in enumerate(
            zip(args.param_name, tar_sizes)
    ):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    # Save to file
    save_ckpt(ckpt, save_path)


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print("save changed ckpt to {}".format(save_name))


def reset_ckpt(ckpt):
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0


def get_catIDs(path_json):
    coco = COCO(path_json)
    cat_ids = coco.getCatIds()
    return cat_ids


if __name__ == "__main__":
    args = parse_args()

    NOVEL_CLASSES = get_catIDs(path_json = path_novel_json)
    BASE_CLASSES = get_catIDs(path_json = path_base_json)

    ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES
    IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
    TAR_SIZE = len(ALL_CLASSES)

    if args.method == "combine":
        combine_ckpts(args)
    else:
        ckpt_surgery(args)
