from dinov2.models.cape_backbone import CapeBackbone

PRETRAIN_DIR = "pretrained_models"

import os

print(os.environ.get('CONDA_PREFIX'))

model_name = "efficientnet-b2"
add_pooling = False
checkpoint_path = "/home/franziska/PycharmProjects/dinov2/mtmv_last.ckpt"
container = CapeBackbone(model_name, pretrained=True, in_channels=3)
model= container.get_features(add_pooling, checkpoint_path)
model = model.eval()
print(model)

#!/usr/bin/env bash




