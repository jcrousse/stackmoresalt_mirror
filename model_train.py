from helpers import performance_test
from keras.optimizers import Adam
from sklearn import model_selection
from models.UNet import unet_128, unet_768, unet_256
from helpers.model_management import get_model_memory_usage
from helpers.data_generator_functions import *
from helpers.class_generators import data_generator
from clustering_tools.apply_clustering import apply_clustering
from clustering_tools.rule_based_clusters import split_by_hue
import json

BATCH_SIZE = 100
EPOCHS = 300
MODEL_SIZE = 128
DROPOUT = 0.3
FLIP_FLAG = True

#############################################

size_to_model= {
    128 : unet_128,
    256 : unet_256,
    768 : unet_768
}


model = size_to_model[MODEL_SIZE](drop_rate= DROPOUT)
unet_128(drop_rate=0.3)

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

img_ids_list = apply_clustering(img_ids, train_img_dict, split_by_hue)

for idx, cluster in enumerate(img_ids_list):
    train_ids, validation_ids = model_selection.train_test_split(img_ids, random_state=1, test_size=0.20)

    train_generator = data_generator(list_IDs=train_ids,
                                     train_dict=train_img_dict,
                                     label_dict=mask_img_dict,
                                     batch_size=BATCH_SIZE,
                                     flip=FLIP_FLAG
                                     )

    model.compile(
        loss=performance_test.bce_dice_loss,
        optimizer=Adam(lr=1e-4),
        metrics=[performance_test.dice_coef])

    print(model.summary())
    print(get_model_memory_usage(BATCH_SIZE, model))


    model.fit_generator(train_generator,
                        epochs=EPOCHS)

    # model name: U128_t1_b40_e20 = Unet 128, train set 1, Batch size 40, epochs: 20.
    model_name = 'U'+ str(MODEL_SIZE) +\
                 '_b' + str(BATCH_SIZE) +\
                 '_e' + str(EPOCHS) +\
                 '_' + str(idx) + '.model'
    model_save_path = os.path.join(out_model, model_name)
    model.save(model_save_path)
