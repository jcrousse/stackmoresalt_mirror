from keras.utils import Sequence
from helpers.data_generator_functions import load_resize_bw_image
import numpy as np


class data_generator(Sequence):
    def __init__(self, list_IDs, train_dict, label_dict, batch_size=32, dim=(128, 128, 1),
                 shuffle=True, train=True):
        self.dim = dim
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train_dict = train_dict
        self.train = train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp, self.train_dict)
        if self.train:
            y = self.__data_generation(list_IDs_temp, self.label_dict)
            return X,y
        else:
            return X

        # X, y = self.__train_data_generation(list_IDs_temp)
        # return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, dict_path):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        out_array = np.empty((self.batch_size, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            out_array[i,] = load_resize_bw_image(self.train_dict[ID], self.dim)

        return out_array

    def __train_data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = X

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = load_resize_bw_image(self.train_dict[ID], self.dim)
            y[i,] = load_resize_bw_image(self.label_dict[ID], self.dim)

        return X, y