#mkdir /cnvrg/output/dino_checkpoint;
#gsutil cp gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/eval/training_112499/teacher_checkpoint.pth /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth;
#gsutil cp gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/config.yaml /cnvrg/output/dino_checkpoint/config.yaml;
#python /cnvrg/scripts/dino_embedding_creation.py --data /data/dino_fixed_rg_evaluation_imagenet --config /cnvrg/output/dino_checkpoint/config.yaml --checkpoint /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth --outpath /cnvrg/output/rgevaluation;
pip install xformers==0.0.18
python /cnvrg/scripts/dino_embedding_creation.py --data /data/dino_fixed_rg_evaluation_imagenet --config $1 --checkpoint $2 --outpath /cnvrg/output/rgevaluation;
cd /cnvrg/output/rgevaluation/embeddings;
cnvrgv2 dataset create -n $3;
cnvrgv2 dataset put -n $3 -f .

# $1  /data/dino_models/30epochswd01_config.yaml
# $2  /data/dino_models/30epochswd01_87499.pth
# $3 rg_embeddings_dino_30epochswd01_87499
