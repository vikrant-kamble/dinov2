import argparse

import optuna
from optuna.integration import FastAIPruningCallback


from train_head_with_fastai import train


def objective(trial):
    
    batch_size = trial.suggest_int("batch_size", 64, 1024, log=True)
    lr = trial.suggest_float("lr", 1e-5, 0.5, log=True)
    n_epochs = trial.suggest_int("n_epochs", 5, 50)
    dp1 = trial.suggest_float("dp1", 0, 1)
    dp2 = trial.suggest_float("dp2", 0, 1)
    size_of_linear_layer = trial.suggest_int("size_of_linear_layer", 128, 2048, log=True)
    
    return train(
        str(args.data), 
        args.model, 
        args.checkpoint, 
        args.arch, 
        batch_size=batch_size,
        lr=lr, 
        n_epochs=n_epochs, 
        dp1=dp1, 
        dp2=dp2, 
        size_of_linear_layer=size_of_linear_layer, 
        run_id=trial.number,
        callbacks=[FastAIPruningCallback(trial, monitor="accuracy")]
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--storage', type=str, required=True)
    parser.add_argument('--study', type=str, required=True)
    parser.add_argument('--model', choices=['swin_base_patch4_window7_224_in22k', 'vicregl'], required=True)
    parser.add_argument('--data', type=str, required=True, help="Path to root of Imagenet-like dataset")
    parser.add_argument('--arch', choices=['tiny', 'small', 'base', 'large', 'xlarge'], required=False, default='small')
    parser.add_argument('--checkpoint', type=str, required=False)

    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner()
    
    study = optuna.load_study(
        study_name=args.study, storage=args.storage
    )
    study.optimize(
        objective, 
        n_trials=20  # trials per process
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))