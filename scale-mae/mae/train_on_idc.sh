torchrun --nproc_per_node=4 main_pretrain.py --config config/2m.yaml --eval_path /data/2m/val --eval_dataset 2m --skip_knn_eval
