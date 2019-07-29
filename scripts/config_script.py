import tensorflow
import tensorflow.keras as keras

config = tensorflow.ConfigProto(log_device_placement=True)
config.intra_op_parallelism_threads = 80
config.inter_op_parallelism_threads = 80
sess = tensorflow.Session(config=config)
keras.backend.set_session(sess)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)