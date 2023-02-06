import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from dlhalos_code_tf2 import evaluation as evalu


class RegularizerCallback(Callback):
    def __init__(self, layer, alpha_check):
        super(Callback, self).__init__()
        self.layer = layer
        self.alpha_check = alpha_check

    def on_epoch_end(self, epoch, logs=None):
        print("\nUpdated gamma to value %.5f" % float(K.get_value(self.layer.gamma)))
        if self.alpha_check is True:
            print("Updated log-alpha to value %.5f" % float(K.get_value(self.layer.alpha)))


class BetaCallback(Callback):
    def __init__(self, beta, params):
        super(BetaCallback, self).__init__()
        self.beta = beta
        self.beta0 = params.beta

    def on_epoch_end(self, epoch, logs=None):
        new_value = self.beta0 * (1 + 1 / ((self.beta0/0.0002) + np.exp(9 - epoch / 3.2)))
        # new_value = self.beta0 * (1 + 1/(1+np.exp(15 - epoch/1.8)))
        self.beta.assign(new_value)
        print("\nEpoch %s, beta = %.5f" % (epoch, float(self.beta.numpy())))


class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())


class AucCallback(Callback):
    def __init__(self, training_dataset, validation_dataset, name_training="0", names_val="1"):
        print("WARNING: Probably more efficient to use tf.keras.metrics.AUC.")
       
        self.training_dataset = training_dataset
        self._validation_dataset = validation_dataset

        self.names_training = name_training
        self.names_val = names_val

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        name_train = "auc_train_" + str(self.names_training)
        logs[name_train] = self.get_auc(self.training_dataset)
        
        name_val = "auc_val_" + str(self.names_val)
        logs[name_val] = self.get_auc(self._validation_dataset)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def get_auc(self, dataset):
        t0 = time.time()
        
        pred_ds = []
        labels_ds = []
        for batch, (box,label) in enumerate(dataset):
            pred = self.model.predict(box, verbose=1)
            truth = label.numpy()            
            pred_ds.append(pred.flatten())
            labels_ds.append(truth.flatten())        
        y_pred = np.concatenate(pred_ds)
        labels = np.concatenate(labels_ds)        

        y_pred_proba = np.column_stack((1 - y_pred[:, 0], y_pred[:, 0]))
        auc_score = evalu.roc(y_pred_proba, labels, true_class=1, auc_only=True)

        t1 = time.time()
        print("AUC computation for a single dataset took " + str((t1 - t0) / 60) + " minutes.")
        print("AUC = %s" % auc_score)
        return auc_score


class LossCallback(Callback):
    def __init__(self, validation_dataset, names_val="1"):
        self.validation_dataset = validation_dataset
        self.names_val = names_val

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        name_val = "loss_val_" + str(self.names_val)
        logs[name_val] = self.model.evaluate(self.validation_dataset)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return