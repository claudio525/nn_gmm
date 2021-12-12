import multiprocessing as mp
import time
import copy
from pathlib import Path


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import nn_gmm

train_data_dir = Path("/home/claudy/dev/work/data/nn_gmm/training_data/train")
val_data_dir = Path("/home/claudy/dev/work/data/nn_gmm/training_data/val")

model_dir = Path("/home/claudy/dev/work/data/nn_gmm/results/test/1201_1105/best_model")



# r = nn_gmm.get_realisation_residuals([val_data_dir], gmm, abs_residual=False)

# start_time = time.time()
# sim_df, mean_df, _ = gmm.predict_dirs([val_data_dir])
# print(f"Took {time.time() - start_time}")

def _run(cur_data):
    gmm = nn_gmm.NeuralNetworkGMM.load(model_dir)
    return gmm.model.predict(cur_data)

def main():
    ds = nn_gmm.load_dataset(
        [val_data_dir], nn_gmm.load_feature_details(train_data_dir), batch_size=int(1e6), shuffle_buffer=None
    ).prefetch(10)


    gmm = nn_gmm.NeuralNetworkGMM.load(model_dir)
    ds = nn_gmm.preprocess_ds(ds, feature_config=nn_gmm.convert_to_transform_fn(copy.copy(gmm.feature_config)))
    #
    # # start_time = time.time()
    # # r = gmm.model.predict(ds)
    # # print(f"Took {time.time() - start_time}")
    #




    start_time = time.time()
    with mp.Pool(8) as p:
        results = p.starmap(_run, [(cur_data, )  for cur_data in ds.as_numpy_iterator()])
    print(f"Took {time.time() - start_time}")


# start_time = time.time()
# for cur_data in ds.as_numpy_iterator():
#     gmm.model.predict(cur_data)
# print(f"Took {time.time() - start_time}")

if __name__ == '__main__':
    main()
    exit()