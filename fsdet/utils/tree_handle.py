import torch
from pycocotools.coco import COCO


class Tree_Node():
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth
        self.parent_name = None
        self.id_in_cur_classify = None
        self.class_id = None
        self.continuous_class_id = None
        self.is_leaf = False
        self.children_name = []

    def add_child(self, child_name):
        if (child_name not in self.children_name):
            self.children_name.append(child_name)

    def __repr__(self):
        res = 'name:{}, depth:{}, parent:{}, class_id:{}'.format(self.name, self.depth, self.parent_name, self.class_id)
        return res


def get_node_name_list_from_cat_name(cat_name):
    all_nodes_cur_cat = cat_name.split('.')
    res = []
    for index, item_name in enumerate(all_nodes_cur_cat, start = 1):
        item_name = item_name + '_' + str(index)
        res.append(item_name)
    return res


def check_all_equal_base_plus_novel(path_base, path_novel, path_all):
    coco_base = COCO(path_base)
    coco_novel = COCO(path_novel)
    coco_all = COCO(path_all)
    cats_base = coco_base.getCatIds()
    cats_novel = coco_novel.getCatIds()
    cats_all = coco_all.getCatIds()
    combine = cats_base + cats_novel
    for item_combine, item_all in zip(combine, cats_all):
        assert item_combine == item_all


class Tree():
    def __init__(self, path_ann):
        # current_mode = cfg.CURRENT_MODE
        # if (current_mode == 'base'):
        #     path_ann = cfg.PATH.BASE_ANN
        # elif (current_mode == 'novel'):
        #     path_ann = cfg.PATH.NOVEL_ANN
        # elif (current_mode == 'all'):
        #     path_ann = cfg.PATH.ALL_TRAIN
        # else:
        #     raise Exception
        # check_all_equal_base_plus_novel(cfg.PATH.BASE_ANN, cfg.PATH.NOVEL_ANN, cfg.PATH.ALL_TRAIN)

        self.coco = COCO(path_ann)
        cat_ids = self.coco.getCatIds()

        origin_id_to_continuous_id = {k: v for v, k in enumerate(cat_ids)}

        self.name_to_node = {}
        root = Tree_Node(name = 'root', depth = 0)
        self.name_to_node['root'] = root
        self.origin_id_to_continuous_id = origin_id_to_continuous_id
        self.continuous_id_to_origin = {}
        for origin_id, continuous_id in self.origin_id_to_continuous_id.items():
            self.continuous_id_to_origin[continuous_id] = origin_id
        self.origin_id_to_nodename = {}

        self.build_tree()

    def build_tree(self):

        all_cat_info = self.coco.loadCats(self.coco.getCatIds())
        all_cat_names = []
        cat_name_to_classID = {}

        for item in all_cat_info:
            all_cat_names.append(item['name'])
            class_id = item['id']
            cat_name_to_classID[item['name']] = class_id

        for item_cat_name in all_cat_names:
            all_nodes_cur_cat = get_node_name_list_from_cat_name(item_cat_name)
            assert len(all_nodes_cur_cat) == 4
            for deep_node, name_node in enumerate(all_nodes_cur_cat, start = 1):
                if (name_node in self.name_to_node):
                    continue
                cur_node = Tree_Node(name = name_node, depth = deep_node)
                if (deep_node == 1):
                    cur_node.parent_name = 'root'
                else:
                    cur_node.parent_name = all_nodes_cur_cat[deep_node - 2]
                self.name_to_node[cur_node.parent_name].add_child(name_node)

                if (deep_node == 4):
                    class_id = cat_name_to_classID[item_cat_name]
                    cur_node.class_id = class_id
                    cur_node.continuous_class_id = self.origin_id_to_continuous_id[class_id]
                    self.origin_id_to_nodename[class_id] = name_node

                self.name_to_node[name_node] = cur_node

        for item_name, item_node in self.name_to_node.items():
            for child_class_id, item_childname in enumerate(item_node.children_name):
                self.name_to_node[item_childname].id_in_cur_classify = child_class_id
            if (len(item_node.children_name) == 0):
                assert item_node.class_id is not None
                item_node.is_leaf = True

        leaf_count = 0
        for item_name, item_node in self.name_to_node.items():
            if (item_node.is_leaf):
                leaf_count += 1
        assert leaf_count == self.get_number_of_classes()

    def get_layer_deep(self, item_classifier_name):
        if (item_classifier_name == 'root'):
            return 0
        deep = item_classifier_name.split('_')[-1]
        deep = int(deep)
        return deep

    def judge_is_leaf_classifier(self, item_node_name):
        deep = self.get_layer_deep(item_node_name)
        if (deep == 3):
            return True
        else:
            return False

    def get_number_of_classes(self):
        return len(self.continuous_id_to_origin)

    def get_all_classifier_names_children_number(self):
        res = []
        for item_name, item_node in self.name_to_node.items():
            if (len(item_node.children_name) > 0):
                assert item_node.class_id is None
                res.append([item_name, len(item_node.children_name)])
        return res

    def generate_labels(self, gt_labels, number_class):
        if (isinstance(gt_labels, torch.Tensor)):
            gt_labels = gt_labels.tolist()
        label_each_classifier = {}
        classifier_name_number_class = self.get_all_classifier_names_children_number()
        for (item_classifier_name, item_number_class) in classifier_name_number_class:
            pre_label_cur_classifier = [item_number_class] * len(gt_labels)
            label_each_classifier[item_classifier_name] = pre_label_cur_classifier

        for box_index, item_label in enumerate(gt_labels):
            if (item_label != number_class):
                trace = self.get_all_training_classify_name_GT_label(label_continuous = item_label)
                for (item_classifier_name, item_label_cur_cls) in trace:
                    label_each_classifier[item_classifier_name][box_index] = item_label_cur_cls
        # print(label_each_classifier)
        return label_each_classifier

    def get_children_number(self, name_node):
        return len(self.name_to_node[name_node].children_name)

    def get_all_training_classify_name_GT_label(self, label_continuous):
        assert len(self.name_to_node) > 1, 'need to build tree first'
        if (isinstance(label_continuous, torch.Tensor)):
            label_continuous = label_continuous.tolist()
        assert isinstance(label_continuous, int), 'type:{}'.format(type(label_continuous))
        origin_id = self.continuous_id_to_origin[label_continuous]
        leaf_node_name = self.origin_id_to_nodename[origin_id]
        cur_node = self.name_to_node[leaf_node_name]
        res = []
        while True:
            parent_name = cur_node.parent_name
            label_cur_name = cur_node.id_in_cur_classify
            assert label_cur_name is not None
            item_cls = (parent_name, label_cur_name)
            res.append(item_cls)
            cur_node = self.name_to_node[parent_name]
            if (parent_name == 'root'):
                break
        assert len(res) == 4
        res.reverse()
        return res

    def get_leaf_node_name_with_continuous_id(self, label_continuous):
        assert isinstance(label_continuous, int), 'type:{}'.format(type(label_continuous))
        origin_id = self.continuous_id_to_origin[label_continuous]
        leaf_node_name = self.origin_id_to_nodename[origin_id]
        return leaf_node_name

    def get_all_sub_children(self, cur_node, deep = None):
        if (deep is None):
            deep = self.get_layer_deep(item_classifier_name = cur_node)
        if (deep == 4):
            cls_id_continue = self.name_to_node[cur_node].continuous_class_id
            return [(cur_node, cls_id_continue)]
        else:
            res = []
            node_cur = self.name_to_node[cur_node]
            children = node_cur.children_name
            for item_child in children:
                item_res = self.get_all_sub_children(item_child, deep + 1)
                res += item_res
            return res


if __name__ == '__main__':
    from fsdet.config import get_cfg, set_global_cfg
    from fsdet.engine import default_argument_parser, default_setup

    args = default_argument_parser().parse_args()


    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        if args.opts:
            cfg.merge_from_list(args.opts)
        cfg.freeze()
        set_global_cfg(cfg)
        default_setup(cfg, args)
        return cfg


    cfg = setup(args)
    print(cfg)
    tree = Tree(cfg)
    res = tree.get_all_training_classify_name_GT_label(label_continuous = 100)
    print(res)
