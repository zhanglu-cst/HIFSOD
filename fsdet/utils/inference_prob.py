import numpy
import torch
from torch.nn import functional as F

from fsdet.utils.tree_handle import Tree


class Multi_Layer_Inference_Distributed():
    def __init__(self, tree: Tree, rm_bg_scores_each_classifier, pred_proposal_deltas):
        self.tree = tree
        self.scores_each_classifier = rm_bg_scores_each_classifier  # Dict, item: classifier-> [1000,Ki]
        self.pred_proposal_deltas = pred_proposal_deltas.numpy()  # [1000, 7*4]
        self.number_boxes = len(self.scores_each_classifier['root'])  # 1000
        self.number_root_classes = self.scores_each_classifier['root'].shape[1]  # 7
        self.scores_dist_each_box = numpy.zeros((self.number_boxes, self.tree.get_number_of_classes() + 1),
                                                dtype = numpy.float64)  # [1000,1432]
        self.deltas_each_box = numpy.zeros((self.number_boxes, self.tree.get_number_of_classes(), 4),
                                           dtype = numpy.float64)

    def do_inference(self, cur_node_name, cur_sum_score, cur_box_deltas, depth, cur_box_index):
        if (depth < 4):
            scores_cur_node = self.scores_each_classifier[cur_node_name][cur_box_index]
            child_names = self.tree.name_to_node[cur_node_name].children_name
            assert len(child_names) == len(scores_cur_node)
            for item_score, child_name in zip(scores_cur_node, child_names):
                self.do_inference(cur_node_name = child_name, cur_sum_score = cur_sum_score * item_score,
                                  cur_box_deltas = cur_box_deltas, depth = depth + 1, cur_box_index = cur_box_index)
        else:
            cur_node = self.tree.name_to_node[cur_node_name]
            assert cur_node.continuous_class_id is not None and len(cur_node.children_name) == 0
            cur_continuous_class_id = cur_node.continuous_class_id
            self.scores_dist_each_box[cur_box_index][cur_continuous_class_id] = cur_sum_score
            self.deltas_each_box[cur_box_index][cur_continuous_class_id] = cur_box_deltas

    def inference(self):
        scores_root_classifier = self.scores_each_classifier['root']  # tensor [1000,7]
        for index_box, (item_box_root_scores, item_box_deltas) in enumerate(zip(scores_root_classifier,
                                                                                self.pred_proposal_deltas)):
            item_box_deltas = item_box_deltas.reshape(-1, 4)  # [7,4]
            assert len(item_box_root_scores) == len(item_box_deltas)
            assert len(item_box_root_scores) == self.number_root_classes  # 7
            root_children_names = self.tree.name_to_node['root'].children_name
            assert len(root_children_names) == len(item_box_root_scores)
            for item_root_score_cur_class, item_box_delta_cur_class, child_name in zip(item_box_root_scores,
                                                                                       item_box_deltas,
                                                                                       root_children_names):
                self.do_inference(cur_node_name = child_name, cur_sum_score = item_root_score_cur_class,
                                  cur_box_deltas = item_box_delta_cur_class, depth = 1, cur_box_index = index_box)
        # assert numpy.sum(self.scores_dist_each_box == 0) == self.number_boxes, 'scores_dist_each_box:{}'.format(
        #         numpy.sum(self.scores_dist_each_box == 0))
        self.deltas_each_box = self.deltas_each_box.reshape((self.number_boxes, -1))
        # assert numpy.sum(self.deltas_each_box == 0) == 0, numpy.sum(self.deltas_each_box == 0)
        return self.scores_dist_each_box, self.deltas_each_box


class Multi_Layer_Inference_Max():
    def __init__(self, tree: Tree, rm_bg_scores_each_classifier, pred_proposal_deltas, proposals):
        self.tree = tree
        self.scores_each_classifier = rm_bg_scores_each_classifier  # Dict, item: classifier-> [1000,Ki]
        # only root contains bg col
        self.pred_proposal_deltas = pred_proposal_deltas.numpy()  # [1000, 7*4]
        self.number_boxes = len(self.scores_each_classifier['root'])  # 1000
        self.proposals = proposals
        self.scores_record = []
        self.class_ids_record = []
        self.box_deltas_record = []

    def do_inference(self, cur_node_name, cur_sum_score, cur_box_deltas, depth, cur_box_index):
        if (depth < 4):
            scores_cur_node = self.scores_each_classifier[cur_node_name][cur_box_index]
            child_names = self.tree.name_to_node[cur_node_name].children_name
            assert len(child_names) == len(scores_cur_node)
            assert scores_cur_node.dim() == 1
            max_score, max_index = torch.max(scores_cur_node, dim = 0)
            next_child_name = child_names[max_index]
            self.do_inference(cur_node_name = next_child_name, cur_sum_score = cur_sum_score * max_score,
                              cur_box_deltas = cur_box_deltas, depth = depth + 1, cur_box_index = cur_box_index)
        else:
            cur_node = self.tree.name_to_node[cur_node_name]
            assert cur_node.continuous_class_id is not None and len(cur_node.children_name) == 0
            cur_continuous_class_id = cur_node.continuous_class_id
            self.scores_record.append(cur_sum_score)
            self.class_ids_record.append(cur_continuous_class_id)
            self.box_deltas_record.append(cur_box_deltas)

    def inference(self):
        scores_root_classifier = self.scores_each_classifier['root']  # tensor [1000,8]
        scores_root_classifier = scores_root_classifier[:, :-1]
        values_root_cls, indexs_root_cls = torch.max(scores_root_classifier, dim = 1)
        # values_root_cls: [1000]  indexs_root_cls: [1000]  range from 0~len(class)
        root_children_names = self.tree.name_to_node['root'].children_name
        assert len(values_root_cls) == len(indexs_root_cls)
        assert len(values_root_cls) == len(self.pred_proposal_deltas)
        assert len(values_root_cls) == len(self.proposals)
        res_proposals = []
        for index_box, (item_root_score, item_index_root_cls, item_box_deltas, item_proposal) in enumerate(
                zip(values_root_cls, indexs_root_cls, self.pred_proposal_deltas, self.proposals)):
            # if (item_index_root_cls == self.number_root_classes - 1):
            #     continue
            # item_root_score: int  item_index_root_cls: int   item_box_deltas:
            max_childname = root_children_names[item_index_root_cls]
            item_box_deltas = item_box_deltas.reshape(-1, 4)  # [7,4]
            max_box_delta = item_box_deltas[item_index_root_cls]
            self.do_inference(cur_node_name = max_childname, cur_sum_score = item_root_score,
                              cur_box_deltas = max_box_delta, depth = 1, cur_box_index = index_box)
            res_proposals.append(item_proposal)

        assert len(res_proposals) == len(self.scores_record)
        res_scores = torch.tensor(self.scores_record).float().cuda()  # [K]
        res_class_id = torch.tensor(self.class_ids_record).long().cuda()  # [K]
        res_box_deltas = torch.tensor(self.box_deltas_record).float().cuda()  # [K,4]

        assert res_box_deltas.dim() == 2, 'res_box_deltas:{}'.format(res_box_deltas)
        assert len(res_class_id) == len(res_scores), 'len(res_class_id):{},len(res_scores):{}'.format(len(res_class_id),
                                                                                                      len(res_scores))
        assert len(res_scores) == len(res_box_deltas), 'len(res_scores):{},len(res_box_deltas):{}'.format(
                len(res_scores), len(res_box_deltas))

        return res_scores, res_class_id, res_box_deltas, res_proposals


class Multi_Layer_Inference_TopK():
    def __init__(self, tree: Tree, rm_bg_scores_each_classifier, pred_proposal_deltas, proposals):
        self.tree = tree
        self.scores_each_classifier = rm_bg_scores_each_classifier  # Dict, item: classifier-> [1000,Ki]
        # only root contains bg col
        self.pred_proposal_deltas = pred_proposal_deltas.numpy()  # [1000, 7*4]
        self.number_boxes = len(self.scores_each_classifier['root'])  # 1000
        self.proposals = proposals
        self.scores_record = []
        self.class_ids_record = []
        self.box_deltas_record = []
        self.res_proposals = []
        self.topK = 2

    def do_inference(self, cur_node_name, cur_sum_score, cur_box_deltas, depth, cur_box_index, item_proposal):
        if (depth < 4):
            scores_cur_node = self.scores_each_classifier[cur_node_name][cur_box_index]
            child_names = self.tree.name_to_node[cur_node_name].children_name
            assert len(child_names) == len(scores_cur_node)
            assert scores_cur_node.dim() == 1
            # max_score, max_index = torch.max(scores_cur_node, dim = 0)
            sorted_scores_classifier, indices_sort = torch.sort(scores_cur_node, dim = -1,
                                                                descending = True)

            for j in range(min(self.topK, len(indices_sort))):
                next_index = indices_sort[j]
                next_score = sorted_scores_classifier[j]
                next_child_name = child_names[next_index]
                self.do_inference(cur_node_name = next_child_name, cur_sum_score = cur_sum_score * next_score,
                                  cur_box_deltas = cur_box_deltas, depth = depth + 1, cur_box_index = cur_box_index,
                                  item_proposal = item_proposal)
        else:
            cur_node = self.tree.name_to_node[cur_node_name]
            assert cur_node.continuous_class_id is not None and len(cur_node.children_name) == 0
            cur_continuous_class_id = cur_node.continuous_class_id
            self.scores_record.append(cur_sum_score)
            self.class_ids_record.append(cur_continuous_class_id)
            self.box_deltas_record.append(cur_box_deltas)
            self.res_proposals.append(item_proposal)

    def inference(self):
        scores_root_classifier = self.scores_each_classifier['root']  # tensor [1000,8]
        scores_root_classifier = scores_root_classifier[:, :-1]  # tensor [1000,7]

        sorted_scores_root_classifier, indices_sort_root = torch.sort(scores_root_classifier, dim = -1,
                                                                      descending = True)
        # sorted_scores_root_classifier: [1000,7]    indices_sort_root :[1000,7]
        root_children_names = self.tree.name_to_node['root'].children_name  # [7]

        for index_box, (item_sort_score_cur_box, item_sort_index_cur_box, item_box_deltas, item_proposal) in enumerate(
                zip(sorted_scores_root_classifier, indices_sort_root, self.pred_proposal_deltas, self.proposals)):
            item_box_deltas = item_box_deltas.reshape(-1, 4)  # [7,4]
            for j in range(self.topK):
                cur_index = item_sort_index_cur_box[j]
                cur_score = item_sort_score_cur_box[j]
                cur_childname = root_children_names[cur_index]
                cur_box_deltas = item_box_deltas[cur_index]  # [4]
                self.do_inference(cur_node_name = cur_childname, cur_sum_score = cur_score,
                                  cur_box_deltas = cur_box_deltas, depth = 1, cur_box_index = index_box,
                                  item_proposal = item_proposal)

        assert len(self.res_proposals) == len(self.scores_record)
        res_scores = torch.tensor(self.scores_record).float().cuda()  # [K]
        res_class_id = torch.tensor(self.class_ids_record).long().cuda()  # [K]
        res_box_deltas = torch.tensor(self.box_deltas_record).float().cuda()  # [K,4]

        assert res_box_deltas.dim() == 2, 'res_box_deltas:{}'.format(res_box_deltas)
        assert len(res_class_id) == len(res_scores), 'len(res_class_id):{},len(res_scores):{}'.format(len(res_class_id),
                                                                                                      len(res_scores))
        assert len(res_scores) == len(res_box_deltas), 'len(res_scores):{},len(res_box_deltas):{}'.format(
                len(res_scores), len(res_box_deltas))

        return res_scores, res_class_id, res_box_deltas, self.res_proposals


class Multi_Layer_Inference_TopK_Fuse():
    def __init__(self, cfg, tree: Tree, rm_bg_scores_each_classifier, pred_flat_scores, pred_flat_box_deltas,
                 proposals):
        self.tree = tree
        self.cfg = cfg
        self.scores_each_classifier = rm_bg_scores_each_classifier  # Dict, item: classifier-> [1000,Ki]
        # only root contains bg col
        pred_flat_box_deltas = pred_flat_box_deltas.cpu()  # [1000, 7*4]
        self.pred_flat_box_deltas = pred_flat_box_deltas.reshape(pred_flat_box_deltas.shape[0], -1, 4)  # [1000,1432,4]
        pred_flat_scores = pred_flat_scores.cpu()  # [1000,1432]
        self.pred_flat_scores = F.softmax(pred_flat_scores, dim = -1)[:, :-1]
        self.number_boxes = len(self.scores_each_classifier['root'])  # 1000
        self.proposals = proposals
        self.scores_record = []
        self.class_ids_record = []
        self.box_deltas_record = []
        self.res_proposals = []
        self.topK = 2

    def do_inference(self, cur_node_name, cur_sum_score, depth, cur_box_index, item_proposal):
        if (depth < 4):
            scores_cur_node = self.scores_each_classifier[cur_node_name][cur_box_index]
            child_names = self.tree.name_to_node[cur_node_name].children_name
            assert len(child_names) == len(scores_cur_node)
            assert scores_cur_node.dim() == 1
            # max_score, max_index = torch.max(scores_cur_node, dim = 0)
            sorted_scores_classifier, indices_sort = torch.sort(scores_cur_node, dim = -1,
                                                                descending = True)

            for j in range(min(self.topK, len(indices_sort))):
                next_index = indices_sort[j]
                next_score = sorted_scores_classifier[j]
                next_child_name = child_names[next_index]
                self.do_inference(cur_node_name = next_child_name, cur_sum_score = cur_sum_score * next_score,
                                  depth = depth + 1, cur_box_index = cur_box_index,
                                  item_proposal = item_proposal)
        else:
            cur_node = self.tree.name_to_node[cur_node_name]
            assert cur_node.continuous_class_id is not None and len(cur_node.children_name) == 0
            cur_continuous_class_id = cur_node.continuous_class_id
            cur_box_deltas = self.pred_flat_box_deltas[cur_box_index][cur_continuous_class_id]
            self.scores_record.append(cur_sum_score)
            self.class_ids_record.append(cur_continuous_class_id)
            self.box_deltas_record.append(cur_box_deltas)
            self.res_proposals.append(item_proposal)

    def add_flat_items(self):
        for box_index, (scores_cur_box, box_deltas_cur_box, cur_proposal) in enumerate(zip(self.pred_flat_scores,
                                                                                           self.pred_flat_box_deltas,
                                                                                           self.proposals)):
            max_index = torch.argmax(scores_cur_box)
            max_score = scores_cur_box[max_index]
            max_deltas = box_deltas_cur_box[max_index]
            self.scores_record.append(max_score)
            self.class_ids_record.append(max_index)
            self.box_deltas_record.append(max_deltas)
            self.res_proposals.append(cur_proposal)

    def add_flat_items_thr(self):
        for box_index, (scores_cur_box, box_deltas_cur_box, cur_proposal) in enumerate(zip(self.pred_flat_scores,
                                                                                           self.pred_flat_box_deltas,
                                                                                           self.proposals)):
            for class_index, (item_score, item_delta) in enumerate(zip(scores_cur_box, box_deltas_cur_box)):
                if (item_score > 0.05):
                    self.scores_record.append(item_score)
                    self.class_ids_record.append(class_index)
                    self.box_deltas_record.append(item_delta)
                    self.res_proposals.append(cur_proposal)

    def inference(self):
        scores_root_classifier = self.scores_each_classifier['root']  # tensor [1000,8]
        scores_root_classifier = scores_root_classifier[:, :-1]  # tensor [1000,7]

        sorted_scores_root_classifier, indices_sort_root = torch.sort(scores_root_classifier, dim = -1,
                                                                      descending = True)
        # sorted_scores_root_classifier: [1000,7]    indices_sort_root :[1000,7]
        root_children_names = self.tree.name_to_node['root'].children_name  # [7]

        for index_box, (item_sort_score_cur_box, item_sort_index_cur_box, item_proposal) in enumerate(
                zip(sorted_scores_root_classifier, indices_sort_root, self.proposals)):
            for j in range(self.topK):
                cur_index = item_sort_index_cur_box[j]
                cur_score = item_sort_score_cur_box[j]
                cur_childname = root_children_names[cur_index]
                self.do_inference(cur_node_name = cur_childname, cur_sum_score = cur_score, depth = 1,
                                  cur_box_index = index_box, item_proposal = item_proposal)

        if (self.cfg.TEST.ADD_FLAT_SCORE):
            self.add_flat_items()

        assert len(self.res_proposals) == len(self.scores_record)

        res_scores = torch.tensor(self.scores_record).float().cuda()  # [K]
        res_class_id = torch.tensor(self.class_ids_record).long().cuda()  # [K]
        # res_box_deltas = torch.tensor(self.box_deltas_record).float().cuda()  # [K,4]
        res_box_deltas = torch.stack(self.box_deltas_record, dim = 0).cuda()

        assert res_box_deltas.dim() == 2, 'res_box_deltas:{}'.format(res_box_deltas)
        assert len(res_class_id) == len(res_scores), 'len(res_class_id):{},len(res_scores):{}'.format(len(res_class_id),
                                                                                                      len(res_scores))
        assert len(res_scores) == len(res_box_deltas), 'len(res_scores):{},len(res_box_deltas):{}'.format(
                len(res_scores), len(res_box_deltas))

        return res_scores, res_class_id, res_box_deltas, self.res_proposals
