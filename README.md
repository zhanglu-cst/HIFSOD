## Hierarchical Few-Shot Object Detection: Problem, Benchmark and Method


### Dataset
HiFSOD-Bird is the first dataset for HIFSOD problem.
You can download the dataset from [Google Driver](https://drive.google.com/drive/folders/1w_gqllWURJuYvzV85nTISW53cY6Rsv4f?usp=share_link).


### Code
Our code is built on detectron2, following TFA and FSCE.

#### Installation
* You need to install the following packages first:
```shell
pytorch==1.10.0
torchvision
fvcore
pycocotools
albumentations
```
(It is worth noting that later versions of pytorch may cause the loss to be NaN, which is due to the update of CrossEntroyLoss in different iimplement of ignre_index)

* Then, run the following command to install   
```shell
pip install -e .
```

* Then, set the dataset path to `global_var.py`.


#### Running
run the following command the training the model.
```shell
python -u tools/train_net.py --config-file configs/base_training.yml --num-gpus 4 # base training
python -u tools/ckpt_surgery.py --src1 checkpoints/base_pretraining/model_final.pth --save-dir checkpoints/model_files/ --tar-name model_pretrain_reset.pth --method remove
python -u tools/train_net.py --config-file configs/base_hierarchical.yml --num-gpus 4
python -u tools/train_net.py --config-file configs/2shot/2shot_novel.yml --num-gpus 4
python -u tools/ckpt_surgery.py --src1 checkpoints/base_hierarchical/model_final.pth --scr2 checkpoints/2shot/ft_2shot_novel/model_final.pth --save-dir checkpoints/model_files/ --tar-name model_combine_2shot.pth --method combine
python -u tools/train_net.py --config-file configs/2shot/2shot_all_fc.yml --num-gpus 4
```


### Cite
If you find the dataset or code useful, please cite the following articles.
```
@inproceedings{10.1145/3503161.3548412,
author = {Zhang, Lu and Wang, Yang and Zhou, Jiaogen and Zhang, Chenbo and Zhang, Yinglu and Guan, Jihong and Bian, Yatao and Zhou, Shuigeng},
title = {Hierarchical Few-Shot Object Detection: Problem, Benchmark and Method},
year = {2022},
isbn = {9781450392037},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3503161.3548412},
doi = {10.1145/3503161.3548412},
abstract = {Few-shot object detection (FSOD) is to detect objects with a few examples. However, existing FSOD methods do not consider hierarchical fine-grained category structures of objects that exist widely in real life. For example, animals are taxonomically classified into orders, families, genera and species etc. In this paper, we propose and solve a new problem called hierarchical few-shot object detection (Hi-FSOD), which aims to detect objects with hierarchical categories in the FSOD paradigm. To this end, on the one hand, we build the first large-scale and high-quality Hi-FSOD benchmark dataset HiFSOD-Bird, which contains 176,350 wild-bird images falling to 1,432 categories. All the categories are organized into a 4-level taxonomy, consisting of 32 orders, 132 families, 572 genera and 1,432 species. On the other hand, we propose the first Hi-FSOD method HiCLPL, where a hierarchical contrastive learning approach is developed to constrain the feature space so that the feature distribution of objects is consistent with the hierarchical taxonomy and the model's generalization power is strengthened. Meanwhile, a probabilistic loss is designed to enable the child nodes to correct the classification errors of their parent nodes in the taxonomy. Extensive experiments on the benchmark dataset HiFSOD-Bird show that our method HiCLPL outperforms the existing FSOD methods.},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {2002–2011},
numpages = {10},
keywords = {hierarchical classification, benchmark, hierarchical few-shot object detection, few-shot object detection},
location = {Lisboa, Portugal},
series = {MM '22}
}
```