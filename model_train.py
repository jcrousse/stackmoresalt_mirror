from helpers import performance_test
from keras.optimizers import Adam
from sklearn import model_selection
from helpers.dictionaries import *
from helpers.model_management import get_model_memory_usage
from helpers.data_generator_functions import get_image_dict
from helpers.class_generators import data_generator
from clustering_tools.apply_clustering import apply_clustering

import pickle
import json
import os

BATCH_SIZE = 100
EPOCHS = 300
MODEL_SIZE = 128
DROPOUT = 0.3
FLIP_FLAG = True
VAL_SPLIT = 0.1
CLUSTER_METHOD = 'depth_a'

#############################################



model = size_to_model[MODEL_SIZE](drop_rate= DROPOUT)

paths={}

with open('config.json') as f:
    paths = json.load(f)

data_path = paths['train_images_path']
mask_path = paths['train_masks_path']
out_model = paths['out_model_path']

# floydhub paths
if not os.path.isfile(".iamlocal"): #if running on floydhub
    data_path = "/train/images"
    mask_path = "/train/masks"
    out_model = "/output"


train_img_dict = get_image_dict(data_path)
mask_img_dict = get_image_dict(mask_path)

img_ids = list(train_img_dict.keys())

img_ids_list = apply_clustering(img_ids, train_img_dict, clustering_methods[CLUSTER_METHOD])

for idx, cluster in enumerate(img_ids_list):
    train_ids, validation_ids = model_selection.train_test_split(cluster, random_state=1, test_size=VAL_SPLIT)

    train_generator = data_generator(list_IDs=train_ids,
                                     train_dict=train_img_dict,
                                     label_dict=mask_img_dict,
                                     batch_size=BATCH_SIZE,
                                     flip=FLIP_FLAG
                                     )
    validation_generator = data_generator(list_IDs=validation_ids,
                                     train_dict=train_img_dict,
                                     label_dict=mask_img_dict,
                                     batch_size=int(BATCH_SIZE*VAL_SPLIT),
                                     flip=False
                                     )

    model.compile(
        loss=performance_test.bce_dice_loss,
        optimizer=Adam(lr=1e-4),
        metrics=[performance_test.dice_coef, 'mean_squared_error'])

    print(model.summary())
    print(get_model_memory_usage(BATCH_SIZE, model))


    model.fit_generator(train_generator, validation_data=validation_generator, epochs=EPOCHS)

    # model name: U128_t1_b40_e20 = Unet 128, train set 1, Batch size 40, epochs: 20.
    model_name = 'U'+ str(MODEL_SIZE) +\
                 '_b' + str(BATCH_SIZE) +\
                 '_e' + str(EPOCHS) +\
                 '_c' + CLUSTER_METHOD +\
                 '_' + str(idx) +'.model'
    model_save_path = os.path.join(out_model, model_name)
    model.save(model_save_path)
