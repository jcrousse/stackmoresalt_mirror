from helpers.mask_manipulation import binary_to_mask
from models.UNet import unet_128
from helpers.data_generators import *
import numpy as np
import json

model = unet_128()
model.load_weights('temp.model')

predict_ids=['0a41de5e3b', '0a1742c740']

paths={}
with open('config.json') as f:
    paths = json.load(f)

data_path = paths['train_images_path']
mask_path = paths['train_masks_path']


train_img_dict = get_image_dict(data_path)
mask_img_dict = get_image_dict(mask_path)

img_list = np.asarray([load_resize_bw_image(train_img_dict[predict_id]) for predict_id in predict_ids])
predictions = model.predict(img_list)
pred_list = [array_to_image(image) for image in predictions]

rle_submission = [binary_to_mask(pred) for pred in pred_list]


from imageio import imsave
imsave('temp_pct.png', predictions[0])
imsave('temp_class.png', np.rint(predictions[0]))
