#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from data_store import DataStore
from data_store.data_containers import Experiments, Dataset
from data_store.storage import GCSStorage
import glob
import os
import sys

try:
    dataset_name = sys.argv[1]
except IndexError:
    dataset_name = 'rcr_2M_chips'

try:
    destination = sys.argv[2]
except IndexError:
    
    destination = '/data/chips'
# %%


gs = GCSStorage(
    # endpoint="http://0.0.0.0:4443", 
    path=f"gs://cape-ml-projects-data/data_stores"
)
ds = DataStore(name="dinov2", storage=gs)


# dataset = ds[dataset_name]


# dataset.clone(destination)

exp = ds['experiments/scale_mae']
e = exp.get("laced-universe-9")
e.clone("/data/laced-universe-9")