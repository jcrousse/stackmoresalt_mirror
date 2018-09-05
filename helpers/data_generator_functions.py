import os
from cv2 import resize
from imageio import imread
import numpy as np

def get_image_dict(image_path):
    return {
        os.path.splitext(img_filename)[0]:
            os.path.join(image_path, img_filename)
        for img_filename in os.listdir(image_path) if os.path.splitext(img_filename)[1]}

def load_resize_bw_image(img_path, target_shape=(128, 128, 1), normalize=True):
    bw_image_data = imread(img_path, pilmode='L')
    resized_image = resize(bw_image_data, (target_shape[0], target_shape[1]))
    array_image = np.reshape(resized_image, target_shape)
    if normalize:
        return array_image/255
    return array_image

def array_to_image(img_array, target_shape=(101, 101)):
    resized_image = resize(img_array, (target_shape))
    array_image = np.reshape(resized_image, target_shape)
    return array_image

def generate_training_batch(data, train_dict, mask_dict, batch_size, target_shape=(128, 128,
                                                                                   1)):
    while True:
        X_batch = []
        Y_batch = []
        batch_ids = np.random.choice(data,
                                     size=batch_size,
                                     replace=False)
        for idx, img_id in enumerate(batch_ids):
            X_batch.append(
                load_resize_bw_image(train_dict[img_id], target_shape)
            )
            Y_batch.append(
                load_resize_bw_image(mask_dict[img_id], target_shape)
            )
        X = np.asarray(X_batch, dtype=np.float32)
        Y = np.asarray(Y_batch, dtype=np.float32)
        yield X, Y

def generate_test_batch(data, test_dict, batch_size, target_shape=(128, 128, 1)):

    X_batch = []

    l=len(data)

    for ndx in range(0, l, batch_size):
        batch_ids = data[ndx:min(ndx + batch_size, l)]
        for idx, img_id in enumerate(batch_ids):
            X_batch.append(
                load_resize_bw_image(test_dict[img_id], target_shape)
            )
        X = np.asarray(X_batch, dtype=np.float32)

        yield X