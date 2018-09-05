from helpers.mask_manipulation import binary_to_mask
from helpers.data_generator_functions import generate_test_batch
from helpers.class_generators import data_generator
from models.UNet import unet_128, unet_768
from helpers.data_generator_functions import *
from imageio import imsave
import numpy as np
import pandas as pd
import json
import os.path


model = unet_768()
model_name = "U768_t1_b40_e20.model"
model_path = "trained_models"
csv_name = "sub_temp.csv"

BATCH_SIZE = 1 # must divide length of test set for full prediction
SAVE_PREDICTION_IMG = True

paths={}
with open('config.json') as f:
    paths = json.load(f)
test_path = paths['test_images_path']
out_csv_path = os.path.join(paths['out_csv_path'], csv_name)

# floydhub paths
if not os.path.isfile(".iamlocal"): #if running on floydhub
    test_path = "/test"
    model_path = "/models"
    out_csv_path = "/output/submission.csv"
    SAVE_PREDICTION_IMG = False

train_model_path = os.path.join(model_path,model_name)
model.load_weights(train_model_path)

test_img_dict = get_image_dict(test_path)
test_ids = list(test_img_dict.keys())

test_ids = ['60764d08d3', '21c2d4e863', 'a63fd8e2a8', '56d600b706', '2191c9ab21']

test_generator = data_generator(
    list_IDs=test_ids,
    train_dict=test_img_dict,
    label_dict={},
    batch_size=BATCH_SIZE,
    shuffle=False,
    train=False
)

steps = len(test_ids)//BATCH_SIZE
predictions = model.predict_generator(test_generator, steps=steps, verbose=1, max_q_size = 1)
pred_list = [array_to_image(image) for image in predictions]

rle_submission = [binary_to_mask(pred) for pred in pred_list]

submission_df = pd.DataFrame({'id': test_ids[0:len(rle_submission)],
                              'rle_mask': rle_submission}
                             )
submission_df.to_csv(out_csv_path, index=False)


if SAVE_PREDICTION_IMG:
    for idx, image in enumerate(test_ids):
        imsave("out_images/" + test_ids[idx] + "_fl.png", predictions[idx])
        imsave("out_images/" + test_ids[idx]+"_bin.png", np.rint(predictions[idx]))


