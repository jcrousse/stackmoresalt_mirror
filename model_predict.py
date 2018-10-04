from helpers.mask_manipulation import binary_to_mask
from helpers.data_generator_functions import generate_test_batch
from helpers.class_generators import data_generator
from models.UNet import unet_128, unet_768, unet_256
from helpers.data_generator_functions import *
from helpers.dictionaries import *
from clustering_tools.apply_clustering import apply_clustering
from imageio import imsave
import numpy as np
import pandas as pd
import json
import os.path

#### model details
BATCH_SIZE = 120
EPOCHS = 300
MODEL_SIZE = 128
CLUSTER_METHOD = 'hue'

# prediction parameters
PRED_BATCH_SIZE = 100 # must divide length of test set for full prediction
SAVE_PREDICTION_IMG = True


model_path = "trained_models"
csv_name = "sub_temp.csv"


paths={}
with open('config.json') as f:
    paths = json.load(f)
predict_path = paths['train_images_path']
out_csv_path = os.path.join(paths['out_csv_path'], csv_name)

# floydhub paths
if not os.path.isfile(".iamlocal"): #if running on floydhub
    predict_path = "/test"
    model_path = "/models"
    out_csv_path = "/output/submission.csv"
    SAVE_PREDICTION_IMG = False



predict_img_dict = get_image_dict(predict_path)
predict_ids = list(predict_img_dict.keys())

img_ids_list = apply_clustering(predict_ids, predict_img_dict, clustering_methods[CLUSTER_METHOD])

# predict_ids = ['0a31d7553c', '0a41de5e3b', '0a1742c740', '0a19821a16']

submission_df_list = []
for idx, prediction_sublist in enumerate(img_ids_list):
    model_name = 'U' + str(MODEL_SIZE) + \
                  '_b' + str(BATCH_SIZE) + \
                  '_e' + str(EPOCHS) + \
                  '_' + str(idx) + '.model'

                    #todo add                   '_c' + CLUSTER_METHOD + \

    train_model_path = os.path.join(model_path, model_name)

    model = size_to_model[MODEL_SIZE]()

    model.load_weights(train_model_path)

    test_generator = data_generator(
        list_IDs=prediction_sublist,
        train_dict=predict_img_dict,
        label_dict={},
        batch_size=PRED_BATCH_SIZE,
        shuffle=False,
        train=False
    )

    # steps = len(predict_ids) // BATCH_SIZE
    predictions = model.predict_generator(test_generator,  verbose=1, max_q_size = 1)
    pred_list = [array_to_image(image) for image in predictions]

    rle_submission = [binary_to_mask(pred) for pred in pred_list]
    submission_df = pd.DataFrame({'id': predict_ids[0:len(rle_submission)],
                                  'rle_mask': rle_submission}
                                 )
    submission_df_list.append(submission_df)

    if SAVE_PREDICTION_IMG:
        for idx, image in enumerate(prediction_sublist):
            imsave("out_images/" + prediction_sublist[idx] + "_fl.png", predictions[idx])
            imsave("out_images/" + prediction_sublist[idx] + "_bin.png", np.rint(predictions[idx]))



submission_df = pd.concat(submission_df_list)
submission_df.to_csv(out_csv_path, index=False)





