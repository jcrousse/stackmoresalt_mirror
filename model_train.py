from helpers import performance_test
from keras.optimizers import Adam
from sklearn import model_selection
from models.UNet import unet_128
from helpers.model_management import get_model_memory_usage
from helpers.data_generators import *

import json


paths={}
with open('config.json') as f:
    paths = json.load(f)

data_path = paths['train_images_path']
mask_path = paths['train_masks_path']

BATCH_SIZE = 1

train_img_dict = get_image_dict(data_path)
mask_img_dict = get_image_dict(mask_path)

img_ids = list(train_img_dict.keys())
train_ids, validation_ids = model_selection.train_test_split(img_ids, random_state=1, test_size=0.20)

train_generator = generate_training_batch(train_ids, train_img_dict, mask_img_dict, BATCH_SIZE)

model = unet_128()
model.compile(
    loss=performance_test.bce_dice_loss,
    optimizer=Adam(lr=1e-4),
    metrics=[performance_test.dice_coef])

print(model.summary())
print(get_model_memory_usage(BATCH_SIZE, model))

num_epochs = 2
steps_per_epoch = int(len(img_ids) * 0.8/BATCH_SIZE)

model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs)

model.save('temp.model')
