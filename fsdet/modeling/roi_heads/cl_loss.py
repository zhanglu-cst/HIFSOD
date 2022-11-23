import math

import torch
from torch import nn


class HiCL(nn.Module):
    def __init__(self, tree, feature_size, number_classes, temperature = 0.2):
        super(HiCL, self).__init__()
        self.tree = tree
        self.node_name_to_memory = dict()
        all_node_names = self.tree.name_to_node.keys()
        for item_name in all_node_names:
            item_memory = nn.Parameter(data = torch.zeros(feature_size), requires_grad = False).cuda()
            self.node_name_to_memory[item_name] = item_memory
        self.temperature = temperature
        self.number_classes = number_classes

    def get_update_radio(self, deep):
        radio = 1 - math.pow(0.1, 5 - deep)
        return radio

    def get_layer_weight(self, deep):
        return deep

    def update_memory(self, gt_labels, box_features):
        for item_label, item_feature in zip(gt_labels, box_features):
            trace = self.tree.get_all_training_classify_name_GT_label(item_label, including_leaf = True)
            for depth, (item_name, _) in enumerate(trace):
                fj = self.get_update_radio(depth)
                origin_feature = self.node_name_to_memory[item_name]
                new_feature = origin_feature * fj + item_feature * (1 - fj)
                self.node_name_to_memory[item_name] = new_feature

    def get_sim(self, item_feature, item_memory):
        sim = torch.dot(item_feature, item_memory) / self.temperature
        item_sim = torch.exp(sim)
        return item_sim

    def loss(self, gt_labels, box_features):
        loss_sum = 0
        for item_label, item_feature in zip(gt_labels, box_features):
            sum_gj = 0
            trace = self.tree.get_all_training_classify_name_GT_label(item_label, including_leaf = True)
            fenmu = 0
            for item_memory in self.node_name_to_memory.values():
                fenmu += self.get_sim(item_feature, item_memory)

            cur_loss = 0
            for depth, (item_name, _) in enumerate(trace):
                item_memory = self.node_name_to_memory[item_name]
                item_sim = self.get_sim(item_feature, item_memory)
                gj = self.get_layer_weight(depth)
                item_loss = torch.log(item_sim / fenmu) * gj
                sum_gj += gj
                cur_loss += item_loss
            cur_loss = -cur_loss / sum_gj
            loss_sum += cur_loss
        loss_sum = loss_sum * 0.001
        return loss_sum

    def one_step(self, gt_labels, box_features):
        if (isinstance(gt_labels, torch.Tensor)):
            gt_labels = gt_labels.tolist()
        filter_bg_labels = []
        filter_bg_features = []
        for item_label, item_feature in zip(gt_labels, box_features):
            if (item_label != self.number_classes):
                filter_bg_labels.append(item_label)
                filter_bg_features.append(item_feature)

        loss = self.loss(filter_bg_labels, filter_bg_features)
        self.update_memory(filter_bg_labels, filter_bg_features)
        return loss
