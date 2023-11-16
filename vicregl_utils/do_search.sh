rm -rf db.sqlite

for f in 0.05 0.1 0.2 0.3 0.5 0.75 1
do
    optuna create-study --study-name "checkpoint90_rcr_frac_${f}" --storage "sqlite:///db.sqlite" --direction maximize
    
    python hyperparam_search.py --model vicregl --train_data /data/rcr_v3_train --val_data /data/rcr_v3_test --arch small --checkpoint /cnvrg/vicregl_checkpoints/epoch90.pth --config config.json --train-fraction ${f} --storage 'sqlite:///db.sqlite' --study "checkpoint90_rcr_frac_${f}" --n-trials 30

done