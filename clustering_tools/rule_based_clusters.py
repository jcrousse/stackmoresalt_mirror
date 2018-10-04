
import numpy as np

def split_by_hue(img, img_id):
    if np.mean(img) > 0.5:
        return 0
    else:
        return 1

def no_clustering(img, img_id):
    return 0