import os
import sys

os.environ['TLDR_DATA_STORAGE_URL'] = 's3://cape-data-east/datascience_storage/tldr-datasets'
os.environ['TLDR_SECRET_NAME'] = 'useless'
os.environ['NEPTUNE_API_TOKEN'] = 'useless'
os.environ['TLDR_LOG_LEVEL'] = 'debug'

from tldr.dvc.dataset import Dataset

ds = Dataset(sys.argv[1])
ds.clone(path="/home/ec2-user/")
