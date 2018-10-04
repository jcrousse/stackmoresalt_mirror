from models.UNet import unet_128, unet_768, unet_256
from clustering_tools.rule_based_clusters import split_by_hue, no_clustering

import os
import json
import pickle

size_to_model= {
    128 : unet_128,
    256 : unet_256,
    768 : unet_768
}


out_model = "pickle"
cluster_pickle = os.path.join(out_model, "depth_cluster_1.pkl")

clustering_methods = {
    'hue' : split_by_hue,
    'no' : no_clustering,
    'depth_a': pickle.load( open( cluster_pickle , "rb" ) )
}