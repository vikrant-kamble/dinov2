#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from data_store import DataStore
from data_store.data_containers import Experiments, Dataset
from data_store.storage import GCSStorage
import glob
import os


# %%


gs = GCSStorage(
    # endpoint="http://0.0.0.0:4443", 
    path=f"gs://cape-ml-projects-data/data_stores"
)
ds = DataStore(name="dinov2", storage=gs)


dataset = ds['rcr_2M_chips']


dataset.clone("/data/chips")
