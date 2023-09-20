#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_store import DataStore
from data_store.data_containers import Experiments, Dataset
from data_store.storage import GCSStorage
import glob
import os

import nest_asyncio
nest_asyncio.apply()


# In[2]:


gs = GCSStorage(
    # endpoint="http://0.0.0.0:4443", 
    path=f"gs://cape-ml-projects-data/data_stores"
)
ds = DataStore(name="dinov2", storage=gs)


# In[3]:

try:
    ds['rcr_2M_chips'] = Dataset(description="More than 2M chips with RCR pseudo labels in ImageNet format. It is stratified so that severe is twice more prevalent than the other labels")
except ValueError:
    pass


# In[4]:


dataset = ds['rcr_2M_chips']


# In[6]:


files = [x for x in glob.glob("/data/chips/**/*", recursive=True) if os.path.isfile(x)]
print(f"Found {len(files)} files")


# In[ ]:


chunk_size = 100000

for i in range(0, len(files), chunk_size):
    print(f"\n=======================\nchunk {i}\n")
    chunk = files[i:i + chunk_size]
    
    dataset.upload(chunk, root_path='/data/chips/')


# In[ ]:




