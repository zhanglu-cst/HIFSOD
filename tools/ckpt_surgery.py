import argparse
import os

import torch
from pycocotools.coco import COCO

from global_var import DATASET_DIR

path_base_json = DATASET_DIR['cats_base_ann']
path_novel_json = DATASET_DIR['cats_novel_ann']


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
            "--src1", type = str,
            # default = "xxx/xxx/TFA/checkpoints/bird/multi_layer/train_base_unfreeze_random_loss_beta_1/model_0079999.pth",
            default = '',
            # default = 'checkpoints/bird/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth',
            help = "Path to the main checkpoint"
    )
    parser.add_argument(
            "--src2",
            type = str,
            default = " ",
            help = "Path to the secondary checkpoint (for combining)",
    )
    parser.add_argument(
            "--save-dir", type = str,
            default = "",
            help = "Save directory"
    )
    parser.add_argument(
            "--tar-name",
            type = str,
            default = "xxx.pth",
            help = "Name of the new ckpt",
    )
    # Surgery method
    parser.add_argument(
            "--method",
            choices = ["combine", "remove", "randinit"],
            required = True,
            help = "Surgery method. combine = "
                   "combine checkpoints. remove = for fine-tuning on "
                   "novel dataset, remove the final layer of the "
                   "base detector. randinit = randomly initialize "
                   "novel weights.",
    )
    # Targets
    parser.add_argument(
            "--param-name-start",
            type = str,
            nargs = "+",
            default = 'roi_heads.box_predictor',
            help = "Target parameter names",
    )

    args = parser.parse_args()
    return args


def combine_multi_layer(ckpt1, ckpt2):
    root_param_names = ['roi_heads.box_predictor_dict.root.cls_score', 'roi_heads.box_predictor_dict.root.bbox_pred']
    for item_param_name in root_param_names:
        for is_weight in [True, False]:
            weight_name = item_param_name + (".weight" if is_weight else ".bias")
            print('cur weight name:{}'.format(weight_name))
            pretrained_weight = ckpt1["model"][weight_name]
            len_ckpt_1 = pretrained_weight.size(0)
            if "cls_score" in item_param_name:
                len_ckpt_1 -= 1
                len_ckpt_2 = ckpt2['model'][weight_name].size(0) - 1
                tar_size = len_ckpt_1 + len_ckpt_2 + 1
            else:
                len_ckpt_2 = ckpt2['model'][weight_name].size(0)
                tar_size = len_ckpt_1 + len_ckpt_2

            print('target len:{} name:{}'.format(tar_size, weight_name))
            if is_weight:
                feat_size = pretrained_weight.size(1)  # 1024
                new_weight = torch.rand((tar_size, feat_size))  # [81,1024]
            else:
                new_weight = torch.zeros(tar_size)

            weight_1 = ckpt1["model"][weight_name]  # [8,1024]
            weight_2 = ckpt2['model'][weight_name]  # [26,1024]
            if ('cls_score' in item_param_name):
                backgroud = weight_1[-1]
                weight_1 = weight_1[:-1]
                weight_2 = weight_2[:-1]
                new_weight[-1] = backgroud

            new_weight[:len_ckpt_1] = weight_1
            if ('cls_score' in item_param_name):
                new_weight[len_ckpt_1:-1] = weight_2
            else:
                new_weight[len_ckpt_1:] = weight_2
            ckpt1['model'][weight_name] = new_weight

    # ---------
    count = 0
    for item_key in ckpt2['model']:
        if (item_key not in ckpt1['model']):
            ckpt1['model'][item_key] = ckpt2['model'][item_key]
            print('add 2->1 :{}'.format(item_key))
            count += 1
    print('add count:{}'.format(count))


def main(args):
    # Load checkpoints
    print('load')
    ckpt = torch.load(args.src1)
    print('finish load')
    save_name = args.tar_name
    if args.method == "combine":
        ckpt2 = torch.load(args.src2)
    else:
        ckpt2 = None
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
        start_flag = args.param_name_start
        assert isinstance(start_flag, str)
        all_param_keys = list(ckpt['model'].keys())
        count = 0
        for param_name in all_param_keys:
            print(param_name)
            if (param_name.startswith(start_flag)):
                del ckpt["model"][param_name]
                print('del:{}'.format(param_name))
                count += 1
        print('total del count:{}'.format(count))

    elif (args.method == 'combine'):
        combine_multi_layer(ckpt, ckpt2)
    else:
        raise Exception
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
    print(args)

    main(args)
