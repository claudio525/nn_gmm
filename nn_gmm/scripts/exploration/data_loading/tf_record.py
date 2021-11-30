import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import ml_tools
import nn_gmm

# data_dir = Path("/home/claudy/dev/work/tmp/gmm_record_test")
data_dir = Path("/home/claudy/dev/work/data/nn_gmm/training_data/train")

# ds = nn_gmm.load_dataset([data_dir], nn_gmm.load_feature_details(data_dir), 512, block_size=512)
ds = nn_gmm.load_dataset([data_dir], nn_gmm.load_feature_details(data_dir), 512, block_size=256, shuffle_buffer=int(5e6),
                         n_open_files=256, file_filter="*.tfrecord")

ds = ds.prefetch(tf.data.AUTOTUNE)

start_time = time.time()
load_time = []
pre_start = time.time()
for cur_batch in ds:
    load_time.append(time.time() - pre_start)
    pre_start = time.time()
print(f"Took {time.time() - start_time}")

ml_tools.utils.write_np_array(np.asarray(load_time), data_dir / "load_times.npy", clobber=True)

exit()