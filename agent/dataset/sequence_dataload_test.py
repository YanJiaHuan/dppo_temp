import numpy as np
import torch
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

# Batch = namedtuple("Batch", "actions conditions")
# Transition = namedtuple("Transition", "actions conditions rewards dones")
# TransitionWithReturn = namedtuple(
#     "Transition", "actions conditions rewards dones reward_to_gos"
# )


# dataset_path = '/home/zcai/dppo/data/furniture/one_leg_low/train.npz'
# if dataset_path.endswith(".npz"):
#     dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
# pass

dataset_path = '/home/zcai/jh_workspace/furniture_bench_gen72/furniture-bench/teleop_sim_demo/one_leg_processed_test/2024-10-23-10:33:08.pkl'
if dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
pass