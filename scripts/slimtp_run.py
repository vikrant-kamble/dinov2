from data_science_tools.mlops.hydra import hydra_main

from slimtp.modules import (
    AMDataModule,
    AMDataset,
    AMBedrockDataset,
    AMExport,
    AMLightningModule,
    AMMultiHead,
)
from slimtp.pipelines import (
    AMEvaluation,
    AMExportValidation,
    AMInference,
    AMPreprocessing,
    AMTraining,
)
import os
from typing import List
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

class NeptuneTrainer(AMTraining):

    def _get_loggers(self) -> List:
        loggers = super()._get_loggers()
        logger_config = self.config.get("loggers", {})
        neptune_logger = logger_config.get("neptune")
        if neptune_logger:

            # some metadata for neptune
            CNVRG_EMAIL = os.getenv("CNVRG_EMAIL", "local_job")
            default_neptune_tags = [CNVRG_EMAIL]
            neptune_logger = NeptuneLogger(
                api_key=os.getenv("NEPTUNE_API_TOKEN", neptune_logger["api_token"]),
                project=neptune_logger.get("project", "cape/dinov2"),
                description=os.getenv('CNVRG_JOB_URL'),
                tags=neptune_logger.get('tags', []) + default_neptune_tags,
                log_model_checkpoints=False)
            # record the additional metadata
            neptune_logger.experiment['CNVRG_JOB_NAME'] = os.getenv('CNVRG_JOB_NAME', 'local')
            neptune_logger.experiment['CNVRG_JOB_URL'] = os.getenv('CNVRG_JOB_URL', 'local')
            # in order to continue writing into this neptune run,
            # we need to set the run_id to the one from the environment
            run_id = neptune_logger.experiment['sys/id'].fetch()
            # for non-master GPUs run_id will be None
            if run_id:
                os.environ["NEPTUNE_RUN_ID"] = run_id
            loggers.append(neptune_logger)
        return loggers

    def _get_callbacks(self):
        if not self.config.get("save_checkpoint_every_n_epochs"):
            return super()._get_callbacks()
        else:
            regular_checkpoint_callback = ModelCheckpoint(
                every_n_epochs=self.config["save_checkpoint_every_n_epochs"],
                monitor="val_loss",
                dirpath=f"{self.config['output_directory']}/training",
                filename="{epoch:02d}-regular",
            )
            last_checkpoint_callback = ModelCheckpoint(
                save_top_k=3,
                monitor="epoch",
                mode="max",
                dirpath=f"{self.config['output_directory']}/training",
                filename="{epoch:02d}-{global_step}",
            )

            callbacks = super()._get_callbacks()
            callbacks.append(regular_checkpoint_callback)
            callbacks.append(last_checkpoint_callback)
            return callbacks


@hydra_main(cnvrg_compatible=True, config_name="config", config_path="../slimtp_configs")
def main(config):  # type: ignore
    """
    Do everything.
    """

    preprocessing = AMPreprocessing(config)
    preprocessing.run()

    training = NeptuneTrainer(config)
    training.run(AMBedrockDataset, AMDataModule, AMMultiHead, AMLightningModule)

    inference = AMInference(config)
    inference.run(AMBedrockDataset, AMDataModule, AMLightningModule)

    evaluation = AMEvaluation(config)
    evaluation.run()

if __name__ == "__main__":
    main()  # pylint: disable=E1120
