"""Read the field ranking emitted by the official ERASE search code."""
import numpy as np

def ranking_from_npy(path):
    arr = np.load(path, allow_pickle=True)
    return [str(x) for x in arr[0]]
