import math
import time

from pathlib import Path

import numpy as np
import h5py

db_ffp = Path("/home/claudy/dev/work/data/nn_gmm/training_data/training_data.hdf5")


with h5py.File(db_ffp, "r") as db:
    columns = list(db.keys())
    length = db[columns[0]].shape[0]


batch_size = 512
n_steps = math.ceil(length / 512)

ind = np.random.permutation(length)

start_time = time.time()
with h5py.File(db_ffp, "r") as db:
    cur_data_dict = {}
    for ix in range(5):
        for cur_col in columns:
            db[cur_col][np.sort(ind[ix:(ix+1) * batch_size])]

        print("wtf")

print(f"Took {time.time() - start_time}")
print("wtf")