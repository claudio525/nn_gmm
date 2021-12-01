import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import ml_tools

csv_dir = Path("/home/claudy/dev/work/tmp/gmm_csv_data_test")

# ds = tf.data.experimental.make_csv_dataset(
#     file_pattern=str(csv_dir / "*.csv"),
#     batch_size=512,
#     num_epochs=1,
#     shuffle=True,
#     shuffle_buffer_size=100000,
#     # shuffle=False,
#     # shuffle_buffer_size=None,
#     num_parallel_reads=30,
# )

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            continue
    print("Execution time:", time.perf_counter() - start_time)


def convert_types(np_dtypes):
    tf_types = []
    for cur_dtype in np_dtypes:
        if cur_dtype is np.dtype(np.float64):
            tf_types.append(tf.float32)
        elif cur_dtype is np.dtype(np.int64):
            tf_types.append(tf.int32)
        elif cur_dtype is np.dtype(object):
            tf_types.append(tf.string)
        elif cur_dtype is np.dtype(bool):
            tf_types.append(tf.string)
        else:
            raise NotImplementedError()
    return tf_types

csv_ffps = [str(cur_ffp) for cur_ffp in csv_dir.glob("A*.csv")]

# Get the types
t = pd.read_csv(csv_ffps[0])
column_types = convert_types(t.dtypes)

# ds = tf.data.experimental.CsvDataset(csv_ffps, column_types, header=True)

ds = tf.data.Dataset.list_files(csv_ffps, shuffle=True).interleave(
    lambda f: tf.data.experimental.CsvDataset(f, column_types, header=True),
    num_parallel_calls=tf.data.AUTOTUNE,
    cycle_length=60,
    block_length=10000,
    deterministic=False,
)

# ds = ds.shuffle(int(5e5)).batch(512).repeat(1).prefetch(tf.data.AUTOTUNE)
ds = ds.batch(512).repeat(1).prefetch(60000)


start_time = time.time()
load_time = []
pre_start = time.time()
for cur_batch in ds:
    load_time.append(time.time() - pre_start)
    pre_start = time.time()
print(f"Took {time.time() - start_time}")


ml_tools.utils.write_np_array(np.asarray(load_time), Path("/home/claudy/dev/work/tmp/load_times.npy"), clobber=True)