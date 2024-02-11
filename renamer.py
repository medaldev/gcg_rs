

import os
import uuid
import sys

def rename_subdirs(rootdir):
    for subdir in os.listdir(rootdir):
        full_path = os.path.join(rootdir, subdir)
        if os.path.isdir(full_path):
            new_uuid = str(uuid.uuid4())
            os.rename(full_path, os.path.join(rootdir, new_uuid))

args = sys.argv[1:]

rootdir = args[0]
#rootdir = '/home/amedvedev/Downloads/data/content/datasets/gcg19/train/calculations'

rename_subdirs(rootdir)
