"""
Reads the training_metrics.json file and upload the metrics to Neptune.ai.

Every line is as follows:

{
    "iteration": 530, 
    "iter_time": 1.2649617195129395, 
    "data_time": 0.0003526330110616982, 
    "lr": 2.9417995537003342e-05, 
    "wd": 0.040015371729541005, 
    "mom": 0.9920003415939898, 
    "last_layer_lr": 0.0, 
    "current_batch_size": 16.0, 
    "total_loss": 13.727184295654297, 
    "dino_local_crops_loss": 9.806875228881836, 
    "dino_global_crops_loss": 1.2285749912261963, 
    "koleo_loss": -0.00846099853515625, 
    "ibot_loss": 2.704326629638672
}
"""

import argparse
from pathlib import Path
import json
import os
import neptune
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Disable annoying warnings from Neptune
logging.getLogger("neptune.internal.operation_processors.async_operation_processor").setLevel(logging.CRITICAL)



def upload_to_neptune(run_id: str, filepath: Path):

    # Make sure the project is set
    assert os.environ.get('NEPTUNE_PROJECT') is not None, "You have to set the NEPTUNE_PROJECT environment variable"
        
    # Try to fetch the run if it exist
    with neptune.init_run(
        custom_run_id=run_id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False
        ) as run:

        with open(filepath, "r") as f:
            lines = [json.loads(x) for x in f.readlines()]
        
        for line in lines:
            for k, v in line.items():
                if k in ['iteration', 'iter_time', 'data_time']:
                    continue
                else:
                    run[k].append(value=v, step=int(line['iteration'])+1)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Upload training metrics to Neptune.ai")
    parser.add_argument("--run_id", type=str, required=True, help="The Neptune.ai run id")
    parser.add_argument("--filepath", type=str, required=True, help="The path to the training_metrics.json file")
    args = parser.parse_args()

    # Upload to Neptune.ai
    upload_to_neptune(args.run_id, Path(args.filepath))
