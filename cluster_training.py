from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle
import json
import os


with open('config.json') as f:
    paths = json.load(f)

depth_path = paths['root_data_path']
out_model_path = paths['out_model_path']

depth_df = pd.read_csv(os.path.join(depth_path, 'depths.csv'))
depth_df = depth_df.set_index('id')
z_array = depth_df.z.values
X = np.reshape(z_array, (-1,1))

kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(X)

depth_df['depth_group'] = kmeans

depth_dict = depth_df[['depth_group']].to_dict(orient='index')
depth_dict = {key : value['depth_group'] for key, value in depth_dict.items()}


cluster_pickle = os.path.join(out_model_path, "depth_cluster_1.pkl")
pickle.dump( depth_dict, open( cluster_pickle, "wb" ) )