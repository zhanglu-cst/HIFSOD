import logging
import random

from fsdet.utils.tree_handle import Tree


def resample_train_dataset_dict_to_balance_tree(cfg, dataset_dicts):
    if (cfg.BALANCE_TRAIN_TREE.ENABLE and "Hierarchical" in cfg.MODEL.ROI_HEADS.NAME):
        balancer = Balance_Tree_Data(cfg)
        res_dataset_dict = balancer(dataset_dicts)
        logger = logging.getLogger(__name__)
        logger.info('perfrom resample to balance the tree, origin dataset dict:{}, new:{}'.format(len(dataset_dicts),
                                                                                                  len(res_dataset_dict)))
        return res_dataset_dict
    else:
        return dataset_dicts


class Balance_Tree_Data():
    def __init__(self, cfg):

        cat_path = cfg.CATS_UES_JSON
        self.tree = Tree(cat_path)

        self.max_resample_rate = cfg.BALANCE_TRAIN_TREE.MAX_SAMPLE_RATE

        self.target_sample_count = {}
        self.leaf_count = {}
        self.cof_up = {}
        self.leaf_sum_cof = {}

    def infer_target_sample_count(self, cur_node_name, deep):
        if (deep == 4):
            self.target_sample_count[cur_node_name] = self.leaf_count[cur_node_name]
            return self.leaf_count[cur_node_name]
        else:
            cur_children_names = self.tree.name_to_node[cur_node_name].children_name
            max_count = 0
            for item_child in cur_children_names:
                this_count = self.infer_target_sample_count(item_child, deep = deep + 1)
                max_count = max(max_count, this_count)
            cur_target_count = max_count * len(cur_children_names)
            self.target_sample_count[cur_node_name] = cur_target_count
            return cur_target_count

    def infer_cof_up(self, cur_node_name, deep):
        if (deep == 4):
            return
        cur_children_names = self.tree.name_to_node[cur_node_name].children_name
        cur_child_max = 0
        for item_child in cur_children_names:
            this_value = self.target_sample_count[item_child]
            cur_child_max = max(cur_child_max, this_value)
        for item_child in cur_children_names:
            cof = cur_child_max / self.target_sample_count[item_child]
            self.cof_up[item_child] = cof
        for item_child in cur_children_names:
            self.infer_cof_up(item_child, deep = deep + 1)

    def infer_leaf_cof(self, cur_node_name, deep, cur_sum_cof):
        if (deep == 4):
            cur_leaf_cof = self.cof_up[cur_node_name]
            target = cur_sum_cof * cur_leaf_cof
            self.leaf_sum_cof[cur_node_name] = target
        else:
            all_children_names = self.tree.name_to_node[cur_node_name].children_name
            for item_child in all_children_names:
                this_cof = self.cof_up[item_child]
                self.infer_leaf_cof(cur_node_name = item_child, deep = deep + 1, cur_sum_cof = cur_sum_cof * this_cof)

    def __call__(self, dataset_dicts):

        for item_record in dataset_dicts:
            annotations = item_record['annotations']
            for obj in annotations:
                continue_id = obj['category_id']
                leaf_node_name = self.tree.get_leaf_node_name_with_continuous_id(continue_id)
                if (leaf_node_name not in self.leaf_count):
                    self.leaf_count[leaf_node_name] = 0
                self.leaf_count[leaf_node_name] += 1

        self.infer_target_sample_count('root', deep = 0)
        self.infer_cof_up('root', deep = 0)
        self.infer_leaf_cof('root', deep = 0, cur_sum_cof = 1)

        samples_each_cats = {}
        for item_record in dataset_dicts:
            annotations = item_record['annotations']
            for obj in annotations:
                continue_id = obj['category_id']
                leaf_node_name = self.tree.get_leaf_node_name_with_continuous_id(continue_id)
                if (leaf_node_name not in samples_each_cats):
                    samples_each_cats[leaf_node_name] = []
                samples_each_cats[leaf_node_name].append(item_record)

        res_dataset_dicts = []
        for leaf_node_name, list_samples_this_cat in samples_each_cats.items():
            cof = self.leaf_sum_cof[leaf_node_name]
            assert cof >= 1
            rate = int(cof)
            rate = min(rate, self.max_resample_rate)
            res_cur_cat_samples = list_samples_this_cat * rate

            res_dataset_dicts += res_cur_cat_samples
        random.shuffle(res_dataset_dicts)
        return res_dataset_dicts
