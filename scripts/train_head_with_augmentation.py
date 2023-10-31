import os

from pj_cape_foundation_eval.models.classifier_head import ImageClassifier
from pj_cape_foundation_eval.utils.embedding_data_module import EmbeddingImageNetDataModule

from pj_cape_foundation_eval.models.classifier_full import ImageClassifierFull, build_model_for_eval

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import sys
from neptune.types import File

sys.path.append("/cnvrg/")
from dinov2.configs import dinov2_default_config
import os.path
from omegaconf import OmegaConf

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    import argparse

    torch.multiprocessing.set_start_method('spawn')
    pl.utilities.seed.seed_everything(seed=0, workers=True)

    parser = argparse.ArgumentParser(description='Run Pythorch Lightning training')
    parser.add_argument(
        '--data',
        help='Path to an imagenet dataset with train and test directories (containing embeddings or images depending on the useaugmentation flag)',
        required=True
    )


    parser.add_argument(
        '--outpath',
        help='Path to the output directory',
        required=True
    )

    parser.add_argument(
        '--checkpoint',
        help='Path to the backbone checkpoint (if augmentation is used and no precalculated embeddings)',
        required=False,
        default=None
    )

    parser.add_argument(
        '--config',
        help='Path to the config of the backbone checkpoint (if augmentation is used and no precalculated embeddings)',
        required=False,
        default=None
    )

    parser.add_argument(
        '--subsetsize',
        help='Percentage of data to use from the total train set.',
        required=False,
        type=float,
        default=1
    )

    parser.add_argument("--equalclassdist", type=str2bool, nargs='?',
                        const=False, default=False,
                        help="Assure an equal distribution of classes in the training set.")

    parser.add_argument("--equalclassdistrcr", type=str2bool, nargs='?',
                        const=False, default=False,
                        help="Assure an equal distribution of classes in the training set for rcr (treat all unknown as one class).")

    parser.add_argument("--useaugmentation", type=str2bool, nargs='?',
                        const=False, default=False,
                        help="Whether to use augmentation during training (that way you operate on images instead of embeddings).")

    parser.add_argument(
        '--batchsize',
        help='Batch size for model training',
        required=False,
        type=int,
        default=1024
    )

    parser.add_argument(
        '--weightdecay',
        help='Weight decay for model training',
        required=False,
        type=float,
        default=0
    )

    parser.add_argument(
        '--dropout',
        help='Dropout for classifier model training',
        required=False,
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--labelsmoothing',
        help='Label smoothing for classifier model training',
        required=False,
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--epochs',
        help='Number of epochs for classifier model training',
        required=False,
        type=int,
        default=10
    )

    parser.add_argument(
        '--embeddingdim',
        help='Dimension of the embedding on which this classifier is trained',
        required=False,
        type=int,
        default=2048
    )



    parser.set_defaults(swap=False)

    args = parser.parse_args()


    # add output directory if it doesn't exist
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    # 3. Training
    data_dir = args.data
    data_module = EmbeddingImageNetDataModule(data_dir, args)
    data_module.setup()
    if args.useaugmentation:
        default_cfg = OmegaConf.create(dinov2_default_config)
        cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(default_cfg, cfg)

        backbone_model = build_model_for_eval(cfg, args.checkpoint, cuda=True)
        classifier_model = ImageClassifierFull(learning_rate=0.001, backbone_model = backbone_model, embedding_dim=args.embeddingdim,
                                       n_classes=data_module.n_classes, weight_decay=args.weightdecay,
                                       dropout=args.dropout, label_smoothing=args.labelsmoothing,
                                       steps_per_epoch=len(data_module.train_dataloader()),
                                       num_epochs=args.epochs)
    else:
        classifier_model = ImageClassifier(learning_rate=0.001, embedding_dim=args.embeddingdim,
                                       n_classes=data_module.n_classes, weight_decay=args.weightdecay,
                                       dropout=args.dropout, label_smoothing=args.labelsmoothing,
                                       steps_per_epoch=len(data_module.train_dataloader()),
                                       num_epochs=args.epochs)

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODViNzIxOC1jMjUyLTRhMGMtOWYwNi1kYjgxMGFkM2FhMjAifQ==",
        project="cape/dinov2",
        tags=['mlp', 'franziska'],
        description=os.getenv('CNVRG_JOB_URL'),
        log_model_checkpoints=False  # otherwise Neptune saves _every_ checkpoint for a total of 100 Gb
    )
    # record the additional metadata
    neptune_logger.experiment['CNVRG_JOB_NAME'] = os.getenv('CNVRG_JOB_NAME', 'local')
    neptune_logger.experiment['CNVRG_JOB_URL'] = os.getenv('CNVRG_JOB_URL', 'local')
    # in order to continue writing into this neptune run,
    # we need to set the run_id to the one from the environment
    run_id = neptune_logger.experiment['sys/id'].fetch()
    # for non-master GPUs run_id will be None
    if run_id:
        os.environ["NEPTUNE_RUN_ID"] = run_id

    """checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=2,
        monitor="epoch",
        mode="max",
        dirpath=args.outpath,
        filename="dinohead-{epoch:02d}-{global_step}",
    )"""

    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_acc',
        mode="max",
        save_top_k=3,
        dirpath=args.outpath,
        filename='dinohead-{epoch:02d}-{val_acc:.2f}')

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=2,
        logger=neptune_logger,
        callbacks=[lr_monitor, checkpoint_callback_best],
    )

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(classifier_model, datamodule=data_module, max_lr=0.02)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print(f"Using found lr {new_lr}")

    # update hparams of the model
    classifier_model.hparams.learning_rate = new_lr
    classifier_model.learning_rate = new_lr

    neptune_logger.experiment["parameters"] = args
    neptune_logger.experiment["lr_finder_result"].upload(File.as_image(fig))

    trainer.fit(classifier_model, datamodule=data_module)
    classifier_model.eval()

    results = trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path, dataloaders=data_module.test_dataloader())
    print(results)

    # test_weights = [3331,3331,1105,913,2577,526]
"""
python /cnvrg/pj_cape_foundation_eval/scripts/train_head.py --data /cnvrg/output/rgevaluation_embeddings_no_aug/embeddings --outpath /cnvrg/output/rgevaluation --weightdecay 0.005 --epochs 80 --labelsmoothing 0.1
python /cnvrg/pj_cape_foundation_eval/scripts/train_head.py --data /cnvrg/output/rgevaluation_embeddings_slimtp/embeddings --outpath /cnvrg/output/rgevaluation --weightdecay 0.0005 --epochs 20 --labelsmoothing 0.1 --embeddingdim 1408 --batchsize 4096 --subsetsize 0.5 --equalclassdist


python /cnvrg/pj_cape_foundation_eval/scripts/train_head.py --data /cnvrg/output/rcr_evaluation_embeddings_slimtp_e2e_new/embeddings --outpath /cnvrg/output/rcr_evaluation_e2e_embeddings --batchsize 4096 --epochs 20 --equalclassdist False --equalclassdistrcr False --labelsmoothing 0 --subsetsize 1 --weightdecay 0 --embeddingdim 1408
python /cnvrg/scripts/train_head_with_augmentation.py --data /data/dino_fixed_rg_evaluation_imagenet --outpath /cnvrg/output/rcr_evaluation_dino_145_324999_with_aug --useaugmentation True --weightdecay 0 --epochs 10 --labelsmoothing 0 --embeddingdim 2048 --batchsize 4096 --subsetsize 1 --equalclassdist False --equalclassdistrcr False --checkpoint '/cnvrg/output/dino_checkpoint/teacher_checkpoint.pth' --config '/cnvrg/output/dino_checkpoint/config.yaml'

"""