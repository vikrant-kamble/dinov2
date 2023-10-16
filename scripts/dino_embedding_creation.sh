mkdir /cnvrg/output/dino_checkpoint;
gsutil cp $1 /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth;
gsutil cp $2 /cnvrg/output/dino_checkpoint/config.yaml;
pip install fvcore
pip install xformers==0.0.18
python /cnvrg/scripts/dino_embedding_creation.py --data /data/dino_fixed_rg_evaluation_imagenet --config /cnvrg/output/dino_checkpoint/config.yaml --checkpoint /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth --outpath /cnvrg/output/rgevaluation;
cd /cnvrg/output/rgevaluation/embeddings;
cnvrgv2 dataset create -n $3;
cnvrgv2 dataset put -n $3 -f .

# $2  gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/config.yaml
# $1  gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/eval/training_112499/teacher_checkpoint.pth
# $3 rg_embeddings_dino_145_112499

"""
bash /cnvrg/scripts/dino_embedding_creation.sh 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/eval/training_112499/teacher_checkpoint.pth' 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/config.yaml' rg_embeddings_dino_145_112499
bash /cnvrg/scripts/dino_embedding_creation.sh 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-121/eval/training_124999/teacher_checkpoint.pth' 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-121/config.yaml' rg_embeddings_dino_121_124999


"""