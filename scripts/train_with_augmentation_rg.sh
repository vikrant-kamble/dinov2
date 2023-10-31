mkdir /cnvrg/output/dino_checkpoint;
gsutil cp $1 /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth;
gsutil cp $2 /cnvrg/output/dino_checkpoint/config.yaml;
pip install fvcore;
pip install xformers==0.0.18;
pip install neptune;
cd /cnvrg;
git clone https://github.com/capeanalytics/pj_cape_foundation_eval.git;
pip install /cnvrg/pj_cape_foundation_eval/pj_cape_foundation_eval;
python /cnvrg/scripts/train_head_with_augmentation.py --data /data/dino_fixed_rg_evaluation_imagenet --outpath /cnvrg/output/train_with_aug --useaugmentation True --weightdecay 0 --epochs 10 --labelsmoothing 0 --batchsize 512 --equalclassdist False --equalclassdistrcr False --checkpoint '/cnvrg/output/dino_checkpoint/teacher_checkpoint.pth' --config '/cnvrg/output/dino_checkpoint/config.yaml' --embeddingdim $3 --subsetsize $4;



# $2  gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/config.yaml
# $1  gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/eval/training_112499/teacher_checkpoint.pth
# $3 embeddingdim
# $4 subsetsize

"""
bash /cnvrg/scripts/dino_embedding_creation.sh 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/eval/training_112499/teacher_checkpoint.pth' 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-145/config.yaml' rg_embeddings_dino_145_112499
bash /cnvrg/scripts/dino_embedding_creation.sh 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-121/eval/training_124999/teacher_checkpoint.pth' 'gs://cape-ml-projects-data/data_stores/dinov2/experiments/a100x4/DIN-121/config.yaml' rg_embeddings_dino_121_124999


"""