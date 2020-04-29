import sys
sys.path.append("/home/luisals/DeepHalos")
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump, load
from tensorflow.keras.models import load_model
from collections import OrderedDict

# class MyCustomCallback(callbacks.Callback):
#
#   def on_epoch_end(self, epoch, logs='logs_samples.log'):
#       l = self.model.evaluate()
#       err =
#
#
# def custom_loss():
#     dof = K.variable(1, dtype='float64', name='dof', trainable=True)


# if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse2/"

# First you will have to load the simulation

all_sims = ["0", "1", "2", "4", "5", "6"]
s = tn.SimulationPreparation(all_sims)

train_sims = all_sims[:-1]
val_sim = all_sims[-1]

params_inputs = {'batch_size': 100,
                 'rescale_mean': 1.005,
                 'rescale_std': 0.05050,
                 'dim': (31, 31, 31),
                 # 'shuffle': True
                 }

# training set

training_particle_IDs = load(open(path_model + 'training_set.pkl', 'rb'))
training_labels_particle_IDS = load(open(path_model + 'labels_training_set.pkl', 'rb'))
s_output = load(open(path_model + 'scaler_output.pkl', "rb"))

generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic, **params_inputs)

# errors

dict_err = OrderedDict([(key, []) for key in training_particle_IDs])
y_truth = np.array([training_labels_particle_IDS[key] for key in training_particle_IDs])

epochs = ["%.2d" % i for i in range(1, 51)]

for epoch in epochs[:7]:
    print(epoch)
    model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5")
    pred_tr = model_epoch.predict_generator(generator_training, verbose=1, use_multiprocessing=False, workers=1,
                                            max_queue_size=1)
    err = pred_tr.flatten() - y_truth
    for j, key in enumerate(training_particle_IDs):
        dict_err[key].append(err[j])
    del model_epoch

dump(dict_err, open(path_model + 'error_training.pkl', "wb"))

# # Resume computation
#
# dict_err = load(open(path_model + 'error_training.pkl', "wb"))
# particle_example = dict_err.keys()[0]
#
# epoch_final = len(dict_err[particle_example])
new_epochs = epochs[16:]

for epoch in new_epochs:
    print(epoch)
    model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5")
    pred_tr = model_epoch.predict_generator(generator_training, verbose=1, use_multiprocessing=False, workers=1,
                                            max_queue_size=1)
    err = pred_tr.flatten() - y_truth
    for j, key in enumerate(training_particle_IDs):
        dict_err[key].append(err[j])
    del model_epoch
