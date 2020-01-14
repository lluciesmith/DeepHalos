import numpy as np
import tensorflow
from tensorflow.keras.layers import Input, Dense, Conv3D
from tensorflow.keras.models import load_model
path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/standardize_wdropout/"
model_file = path_model + "/model/weights.80.hdf5"
old_model = load_model(model_file)
weights = old_model.layers[1].get_weights()[0]
biases = old_model.layers[1].get_weights()[1]
model = tensorflow.keras.Sequential()
model.add(Conv3D(1, (3, 3, 3), activation='relu', padding='valid', data_format="channels_last", strides=2,
                 input_shape=(51, 51, 51, 1)))
model.layers[0].set_weights([0*weights[:,:,:,:,0].reshape((3, 3, 3, 1, 1)), np.array([0])])
model.layers[0].set_weights([weights[:,:,:,:,1].reshape((3, 3, 3, 1, 1)), np.array([biases[0]])])
data = np.random.normal(0, 1, size=(1, 51, 51, 51, 1))
yhat = model.predict(data).reshape(25, 25, 25)