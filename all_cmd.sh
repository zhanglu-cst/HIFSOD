python -u tools/train_net.py --config-file configs/base_training.yml --num-gpus 4 # base training
python -u tools/ckpt_surgery.py --src1 checkpoints/base_pretraining/model_final.pth --save-dir checkpoints/model_files/ --tar-name model_pretrain_reset.pth --method remove
python -u tools/train_net.py --config-file configs/base_hierarchical.yml --num-gpus 4
python -u tools/train_net.py --config-file configs/2shot/2shot_novel.yml --num-gpus 4
python -u tools/ckpt_surgery.py --src1 checkpoints/base_hierarchical/model_final.pth --scr2 checkpoints/2shot/ft_2shot_novel/model_final.pth --save-dir checkpoints/model_files/ --tar-name model_combine_2shot.pth --method combine
python -u tools/train_net.py --config-file configs/2shot/2shot_all_fc.yml --num-gpus 4
