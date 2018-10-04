
from keras.layers import *
from keras.models import Model, load_model


"""
Shamelessly stolen U-Net models from 
https://github.com/malhotraa/carvana-image-masking-challenge/blob/master/notebooks/model_cnn.ipynb
https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
"""


def down(filters, input_, drop_rate=0.0):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_ = Activation('relu')(down_)
    # down_ = Dropout(drop_rate)(down_)
    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    # down_ = Dropout(drop_rate)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    down_pool = Dropout(0.3)(down_pool)
    return down_pool, down_res

def up(filters, input_, down_, drop_rate=0.0):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Dropout(0.3)(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Dropout(drop_rate)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Dropout(drop_rate)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    # up_ = Dropout(drop_rate)(up_)
    up_ = Activation('relu')(up_)
    return up_


def unet_768(input_shape=(128, 128, 1), num_classes=1, drop_rate=0.0):
    inputs = Input(shape=input_shape)

    down0a, down0a_res = down(24, inputs, drop_rate)
    down0, down0_res = down(64, down0a, drop_rate)
    down1, down1_res = down(128, down0, drop_rate)
    down2, down2_res = down(256, down1, drop_rate)
    down3, down3_res = down(512, down2, drop_rate)
    down4, down4_res = down(768, down3, drop_rate)

    center = Conv2D(768, (3, 3), padding='same')(down4)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)


    center = Conv2D(768, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up4 = up(768, center, down4_res, drop_rate)
    up3 = up(512, up4, down3_res, drop_rate)
    up2 = up(256, up3, down2_res, drop_rate)
    up1 = up(128, up2, down1_res, drop_rate)
    up0 = up(64, up1, down0_res, drop_rate)
    up0a = up(24, up0, down0a_res, drop_rate)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model

def unet_256(input_shape=(128, 128, 1), num_classes=1, drop_rate=0.0):
    inputs = Input(shape=input_shape)

    down0a, down0a_res = down(16, inputs, drop_rate)
    down0, down0_res = down(32, down0a, drop_rate)
    down1, down1_res = down(64, down0, drop_rate)
    down2, down2_res = down(128, down1, drop_rate)
    down3, down3_res = down(256, down2, drop_rate)

    center = Conv2D(256, (3, 3), padding='same')(down3)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)


    center = Conv2D(256, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up3 = up(256, center, down3_res, drop_rate)
    up2 = up(128, up3, down2_res, drop_rate)
    up1 = up(64, up2, down1_res, drop_rate)
    up0 = up(32, up1, down0_res, drop_rate)
    up0a = up(16, up0, down0a_res, drop_rate)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model

def unet_128(input_shape=(128, 128, 1), num_classes=1, drop_rate=0.0):
    inputs = Input(shape=input_shape)

    down0a, down0a_res = down(24, inputs, drop_rate)
    down0, down0_res = down(48, down0a, drop_rate)
    down1, down1_res = down(90, down0, drop_rate)
    down2, down2_res = down(128, down1, drop_rate)

    center = Conv2D(128, (3, 3), padding='same')(down2)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)


    center = Conv2D(128, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up2 = up(128, center, down2_res, drop_rate)
    up1 = up(90, up2, down1_res, drop_rate)
    up0 = up(48, up1, down0_res, drop_rate)
    up0a = up(24, up0, down0a_res, drop_rate)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model