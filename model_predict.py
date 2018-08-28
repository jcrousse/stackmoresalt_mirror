from helpers.mask_manipulation import binary_to_mask
from helpers.data_generators import generate_test_batch
from models.UNet import unet_128
from helpers.data_generators import *
from imageio import imsave
import numpy as np
import pandas as pd
import json

model = unet_128()
model.load_weights('trained_fh.model')

BATCH_SIZE = 100
SAVE_PREDICTION_IMG = False

paths={}
with open('config.json') as f:
    paths = json.load(f)

test_path = paths['test_images_path']

test_img_dict = get_image_dict(test_path)
test_ids = list(test_img_dict.keys())

test_generator = generate_test_batch(test_ids, test_img_dict, BATCH_SIZE)

steps = len(test_ids)//BATCH_SIZE
steps = 1

predictions = model.predict_generator(test_generator, steps=steps, verbose=1)
pred_list = [array_to_image(image) for image in predictions]

rle_submission = [binary_to_mask(pred) for pred in pred_list]

submission_df = pd.DataFrame({'id': test_ids[0:len(rle_submission)],
                              'rle': rle_submission}
                             )
submission_df.to_csv("submission.csv", index=False)


if SAVE_PREDICTION_IMG:
    for idx, image in enumerate(test_ids):
        imsave("out/" + test_ids[idx] + "_fl.png", predictions[idx])
        imsave("out/" + test_ids[idx]+"_bin.png", np.rint(predictions[idx]))


