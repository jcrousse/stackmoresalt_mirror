from helpers import performance_test
from keras.optimizers import Adam
from sklearn import model_selection
from models.UNet import unet_128, unet_768
from helpers.model_management import get_model_memory_usage
from helpers.data_generator_functions import *
from helpers.class_generators import data_generator
import json


paths={}

with open('config.json') as f:
    paths = json.load(f)

data_path = paths['train_images_path']
mask_path = paths['train_masks_path']


# floydhub paths
if not os.path.isfile(".iamlocal"): #if running on floydhub
    data_path = "/train/images"
    mask_path = "/train/masks"

BATCH_SIZE = 32

train_img_dict = get_image_dict(data_path)
mask_img_dict = get_image_dict(mask_path)

img_ids = list(train_img_dict.keys())
train_ids, validation_ids = model_selection.train_test_split(img_ids, random_state=1, test_size=0.20)

train_generator = data_generator(list_IDs=train_ids,
                                  train_dict=train_img_dict,
                                  label_dict=mask_img_dict,
                                  batch_size=BATCH_SIZE
                                  )

model = unet_768()
model.compile(
    loss=performance_test.bce_dice_loss,
    optimizer=Adam(lr=1e-4),
    metrics=[performance_test.dice_coef])

print(model.summary())
print(get_model_memory_usage(BATCH_SIZE, model))

num_epochs = 20
steps_per_epoch = int(len(img_ids) * 0.8/BATCH_SIZE)

model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs)

# model name: U128_t1_b40_e20 = Unet 128, train set 1, Batch size 40, epochs: 20.
model.save('/output/U768_t1_b32_e20.model')
