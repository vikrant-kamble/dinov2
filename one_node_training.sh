export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/cnvrg
torchrun \
    --nproc_per_node=4 \
    dinov2/train/train.py \
    --config-file=/cnvrg/dinov2/configs/train/vitl16_short.yaml \
    --output-dir=/data/output \
    train.dataset_path=Cape:split=TRAIN:root=/data/2m:extra=/data/2m \
    train.batch_size_per_gpu=128
