
import numpy as np

def split_by_hue(img):
    if np.mean(img) > 0.5:
        return 0
    else:
        return 1