import numpy as np

def tensor_array(tensor):
    t = tensor.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist())
