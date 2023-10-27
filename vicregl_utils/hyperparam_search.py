import argparse
from pathlib import Path
import json
import subprocess
import json
import os

import optuna
from optuna.integration import FastAIPruningCallback


from train_head_with_fastai import train


def objective(trial, config, study_name):
    
#     batch_size_power = trial.suggest_int("batch_size_power", 7, 10)
    config['lr'] = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    config['linear_layer_dropout'] = trial.suggest_float("linear_layer_dropout", 0, 0.7)
#     n_epochs = trial.suggest_int("n_epochs", 2, 30)
    
#     config['batch_size'] = 2**batch_size_power
#     config['lr'] = lr
#     config['n_epochs'] = n_epochs
    
#     dp1 = trial.suggest_float("dp1", 0, 1)
#     dp2 = trial.suggest_float("dp2", 0, 1)
#     size_of_linear_layer = trial.suggest_int("size_of_linear_layer", 128, 2048, log=True)
    
#    config['poor_severe_weight'] = trial.suggest_float("poor_severe_weight", 0.5, 3)

    os.makedirs(study_name, exist_ok=True)
    
    with open(f"{study_name}/{trial.number}_input.json", "w+") as fp:
        json.dump(config, fp)
    
    subprocess.run(
        [
            "accelerate", 
            "launch",
            "train_head_with_fastai.py",
            "--config", f"{study_name}/{trial.number}_input.json",
            "--model", config['model_type'],
            "--train_data", config["train_path"],
            "--train-fraction", str(config["train_fraction"]),
            "--val_data", config["val_path"],
            "--arch", config['convnext_arch'],
            "--checkpoint", config['checkpoint'],
            "--outdir", study_name,
            "--run-id", str(trial.number)
        ]
    )
    
    with open(f"{study_name}/{trial.number}_results.json") as fp:
        results = json.load(fp)
    
    return results['accuracy']

#     return train(
#         config,
#         run_id=trial.number,
# #         callbacks=[FastAIPruningCallback(trial, monitor="accuracy")],
#         save_checkpoint=False
#     )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--storage', type=str, required=True)
    parser.add_argument('--study', type=str, required=True)
    
    parser.add_argument("--config", type=Path, help="Configuration file", required=True)
    parser.add_argument('--model', choices=['swin_base_patch4_window7_224_in22k', 'vicregl'], required=True)
    parser.add_argument('--train_data', type=Path, required=True, help="Path to root of Bedrock-like dataset")
    parser.add_argument('--train-fraction', type=float, default=1, required=False, help="Fraction of training to use")
    parser.add_argument('--val_data', type=Path, required=True, help="Path to root of Bedrock-like dataset")
    parser.add_argument('--arch', choices=['tiny', 'small', 'base', 'large', 'xlarge'], required=False, default='small')
    parser.add_argument('--checkpoint', type=Path, required=False)
    parser.add_argument('--fix-lr', action='store_true', help="Divide input LR by the number of GPUs")
    parser.add_argument('--finetune', action='store_true', help="Finetune the entire model")
    parser.add_argument('--run-id', type=int, required=False, default=0, help="ID of the run (useful for hyperparam searches)")
    parser.add_argument('--n-trials', type=int, required=False, default=20, help="Number of trials for this search")

    args = parser.parse_args()
        
    if args.model == 'vicregl' and not args.checkpoint:
        raise argparse.ArgumentError(args.checkpoint, '--checkpoint is required when model is vicregl')
    
    with open(args.config) as fp:
        config = json.load(fp)
        
    config['train_path'] = str(args.train_data.absolute())
    config['val_path'] = str(args.val_data.absolute())
    config['model_type'] = args.model
    config['checkpoint'] = str(args.checkpoint)
    config['convnext_arch'] = args.arch
    config['train_fraction'] = args.train_fraction
    config['finetune'] = args.finetune
    config['run_id'] = args.run_id

#     pruner = optuna.pruners.MedianPruner()
    
    study = optuna.load_study(
        study_name=args.study, storage=args.storage
    )
    study.optimize(
        lambda trial: objective(trial, config, args.study), 
        n_trials=args.n_trials  # trials per process
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))