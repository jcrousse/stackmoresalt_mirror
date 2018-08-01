from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np


def add_mask_to_img(base_image, mask_image):
    """

    :param base_image: path to base image
    :param mask_image: path to mask image
    :return: fusion of image and mask with funky colors
    """
    base_array =img_to_array(load_img(base_image))
    mask_array = img_to_array(load_img(mask_image, grayscale=True))

    img_masked = np.dstack(
        (mask_array[:,:,0],base_array[:,:,1],base_array[:,:,2])
    )

    return img_masked

