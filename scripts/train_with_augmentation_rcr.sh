mkdir /cnvrg/output/dino_checkpoint;
gsutil cp $1 /cnvrg/output/dino_checkpoint/teacher_checkpoint.pth;
gsutil cp $2 /cnvrg/output/dino_checkpoint/config.yaml;
gsutil -m -q cp -r gs://cape-ml-projects-data/data_stores/dinov2/rcr_v3_train_test_imagenet /data
pip install fvcore;
pip install xformers==0.0.18;
pip install neptune;
cd /cnvrg;
git clone https://github.com/capeanalytics/pj_cape_foundation_eval.git;
pip install /cnvrg/pj_cape_foundation_eval/pj_cape_foundation_eval;
python /cnvrg/scripts/train_head_with_augmentation.py --data /data/rcr_v3_train_test_imagenet --outpath /cnvrg/output/train_with_aug --useaugmentation True --weightdecay 0 --epochs 20 --labelsmoothing 0 --batchsize 512 --equalclassdist False --equalclassdistrcr False --checkpoint '/cnvrg/output/dino_checkpoint/teacher_checkpoint.pth' --config '/cnvrg/output/dino_checkpoint/config.yaml' --embeddingdim $3 --subsetsize $4;


