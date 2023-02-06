import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
import collections
from tensorflow.keras.models import load_model
import dlhalos_code_tf2.data_processing as tn
from pickle import load


def evaluate_model(simulation_id, model_path,
                   params_inputs=None, epochs="all", save=True, save_name="loss.npy",
                   load_ids_sim=False, random_subset_each_sim=None, random_subset_all=None):

    if params_inputs is None:
        params_inputs = {'batch_size': 80,
                         'rescale_mean': 1.005,
                         'rescale_std': 0.05050,
                         'dim': (75, 75, 75)
                     }
    s_output = load(open(model_path + 'scaler_output.pkl', 'rb'))

    s_val = tn.SimulationPreparation(simulation_id)
    validation_set = tn.InputsPreparation(simulation_id, load_ids=load_ids_sim,
                                          random_subset_each_sim=random_subset_each_sim,
                                          random_subset_all=random_subset_all, scaler_output=s_output, shuffle=False)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                            s_val.sims_dic, **params_inputs)
    if epochs == "all":
        epochs = [5*i for i in range(1, 21)]
        epochs = np.array(epochs).astype('str')
        epochs[0] = "05"

        loss = np.zeros((len(epochs), 2))
        loss[:, 0] = [5*i for i in range(1, 21)]

        for i, epoch in enumerate(epochs):
            model_epoch = load_model(model_path + "model/weights." + epoch + ".hdf5")
            loss[i, 1] = model_epoch.evaluate_generator(generator_validation, use_multiprocessing=False, workers=1,
                                                        verbose=1)
            del model_epoch

    else:
        model_epoch = load_model(model_path + "model/weights." + epochs + ".hdf5")
        loss = model_epoch.evaluate_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)

    if save is True:
        np.save(model_path + save_name, loss)

    return loss



def predict_model(simulation_id, model_path,
                  params_inputs=None, epochs="100", save=True,
                  load_ids_sim=False, random_subset_each_sim=None, random_subset_all=None):
    if params_inputs is None:
        params_inputs = {'batch_size': 80,
                         'rescale_mean': 1.005,
                         'rescale_std': 0.05050,
                         'dim': (75, 75, 75)
                         }
    s_output = load(open(model_path + 'scaler_output.pkl', 'rb'))

    s_val = tn.SimulationPreparation(simulation_id)
    validation_set = tn.InputsPreparation(simulation_id, load_ids=load_ids_sim,
                                          random_subset_each_sim=random_subset_each_sim,
                                          random_subset_all=random_subset_all, scaler_output=s_output, shuffle=False)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                            s_val.sims_dic, **params_inputs)

    model = load_model(model_path + "model/weights." + epochs + ".hdf5")

    pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])

    h_m_pred = s_output.inverse_transform(pred).flatten()
    true = s_output.inverse_transform(truth_rescaled).flatten()

    if save is True:
        np.save(model_path + "predicted" + simulation_id + "_" + epochs + ".npy", h_m_pred)
        np.save(model_path + "true" + simulation_id + "_" + epochs + ".npy", true)
    return h_m_pred, true


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS,
                       val_IDs, val_labels_IDs, scaler, path_model, predict_train=True):
    if predict_train:
        pred = model.predict_generator(gen_train, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
        truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
        h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
        np.save(path_model + "predicted_training_"+ epoch + ".npy", h_m_pred)
        np.save(path_model + "true_training_"+ epoch + ".npy", true)
    pred = model.predict_generator(gen_val, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([val_labels_IDs[ID] for ID in val_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_val_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_val_" + epoch + ".npy", true)


def plot_true_vs_predict(true, predicted):
    f = plt.figure()
    plt.plot(true, true, color="dimgrey")
    plt.scatter(true, predicted, s=1)
    plt.xlabel("True log mass")
    plt.ylabel("Predicted log mass")
    plt.subplots_adjust(bottom=0.14)
    return f


def plot_loss(history, val_data=True):
    epochs = history.epoch
    loss_training = history.history["loss"]
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_training, label="training")

    if val_data is True:
        loss_val = history.history["val_loss"]
        plt.plot(epochs, loss_val, label="validation")

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel(history.model.loss, fontsize=18)
    plt.subplots_adjust(bottom=0.14)
    plt.legend(loc="best", fontsize=15)
    return fig


def plot_metric(history, val_data=True, ylabel="mae"):
    epochs = history.epoch
    mae_training = history.history["mean_absolute_error"]

    fig = plt.figure()
    plt.plot(epochs, mae_training, label="training")

    if val_data is True:
        mae_val = history.history["val_mean_absolute_error"]
        plt.plot(epochs, mae_val, label="validation")

    plt.xlabel("Epoch")
    if history.model.metrics_names[1] == "mean_absolute_error":
        plt.ylabel("mae")
    else:
        plt.ylabel(ylabel)
    plt.subplots_adjust(bottom=0.14)
    plt.legend(loc="best")
    return fig

def roc_plot(fpr, tpr, auc, labels=[" "],
             figsize=(8, 6),
             add_EPS=False, fpr_EPS=None, tpr_EPS=None, label_EPS="EPS",
             add_ellipsoidal=False, fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_ellipsoidal="ST ellipsoidal",
             frameon=False, fontsize_labels=20, cols=None):
    """Plot a ROC curve given the false positive rate(fpr), true positive rate(tpr) and Area Under Curve (auc)."""

    if figsize is not None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure, ax = plt.subplots()

    ax.pr(fpr, tpr, lw=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=fontsize_labels)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize_labels)

    # Robust against the possibility of AUC being a single number instead of list
    if not isinstance(auc, (collections.Sequence, list, np.ndarray)):
        auc = [auc]

    if len(labels) > 0:
        labs = []
        for i in range(len(labels)):
            labs.append(labels[i] + " (AUC = " + ' %.3f' % (auc[i]) + ")")
    else:
        labs = np.array(range(len(ax.lines)), dtype='str')
        for i in range(len(labs)):
            labs[i] = (labs[i] + " (AUC = " + ' %.3f' % (auc[i]) + ")")

    if add_EPS is True:
        plt.scatter(fpr_EPS, tpr_EPS, color="k", s=30)
        if label_EPS is not None:
            labs.append(label_EPS)

    if add_ellipsoidal is True:
        if len([fpr_ellipsoidal]) > 1:
            plt.scatter(fpr_ellipsoidal[0], tpr_ellipsoidal[0], color="k", marker="^", s=30)
            plt.scatter(fpr_ellipsoidal[1], tpr_ellipsoidal[1], color="r", marker="^", s=30)
            labs.append("ST ellipsoidal, a=0.75")
            labs.append("ST ellipsoidal, a=0.707")
        else:
            plt.scatter(fpr_ellipsoidal, tpr_ellipsoidal, color="k", marker="^", s=30)
            if label_ellipsoidal is not None:
                labs.append(label_ellipsoidal)

    ax.legend(labs, loc=4,
              # fontsize=18,
              bbox_to_anchor=(0.95, 0.05), frameon=frameon)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    return figure


def roc(y_proba, y_true, true_class=1, auc_only=False):
    """
    Produce the false positive rate and true positive rate required to pr a ROC curve. Compute the Area Under the
    Curve(auc) using the trapezoidal rule.
    True class is 'in' label.

    Args:
        y_proba (np.array): An array of probability scores, either a 1d array of size N_samples or an nd array,
            in which case the column corresponding to the true class will be used.
            You can produce array of probability scores by doing predict_with_proba and then for each sample you get
            its probability to be in and its probability to be out. (Should be column_0 = in and column_1 = out). You
            can give this array checking that true_class is 'in' class probabilities.).
        y_true (np.array): An array of class labels, of size (N_samples,)
        true_class (int): Which class is taken to be the "true class". Default is 1.

    Returns:
        fpr (array): An array containing the false positive rate at each probability threshold
        tpr (array): An array containing the true positive rate at each probability threshold
        auc (array): The area under the ROC curve.

    Raises:
        IOError: "Predicted and true class arrays are not same length."

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    """
    if len(y_proba) != len(y_true):
        raise IOError("Predicted and true class arrays are not same length.")

    if len(y_proba.shape) > 1:

        if true_class == 1:
            proba_in_class = y_proba[:, 1]  # The order of classes in y_proba is first column - 1 and second +1
        elif true_class == -1:
            proba_in_class = y_proba[:, 0]

    else:
        proba_in_class = y_proba

    # 50 evenly spaced numbers between 0,1.
    threshold = np.linspace(0., 1., 50)

    # This creates an array where each column is the prediction for each threshold. It checks if predicted probability
    # of being "in" is greater than probability threshold. If yes it returns True, if no it returns False.
    preds = np.tile(proba_in_class, (len(threshold), 1)).T >= np.tile(threshold, (len(proba_in_class), 1))

    # Make y_true a boolean vector - array that returns True if particle is "in" and False if particle is "out".
    # It is true for all values of threshold ( hence it is rearranged as (len(threshold),1).T).
    y_bool = (y_true == true_class)
    y_bool = np.tile(y_bool, (len(threshold), 1)).T

    # These arrays compare predictions to Y_bool array at each threshold. Sum(axis=0) counts the number of "True"
    # samples at each threshold.
    # If both the predictions and Y_bool return true (or false) it is a true positive (or true negative).
    # If predictions return true and y_bool is false, then it is a false positive.
    # If predictions return false and y_book is true, then it is a false negative.
    TP = (preds & y_bool).sum(axis=0)
    FP = (preds & ~y_bool).sum(axis=0)
    TN = (~preds & ~y_bool).sum(axis=0)
    FN = (~preds & y_bool).sum(axis=0)

    # True positive rate is defined as true positives / (true positives + false negatives), i.e. true positives out
    # of all positives. False positive rate is defined as false positives / (false positives + true negatives),
    # i.e. false positives out of all negatives.
    tpr = np.zeros(len(TP))
    tpr[TP != 0] = TP[TP != 0] / (TP[TP != 0] + FN[TP != 0])
    fpr = FP / (FP + TN)  # Make sure you have included from __future__ import division for this if using python 2!

    # Reverse order of fpr and tpr so that thresholds go from high to low.
    fpr = np.array(fpr)[::-1]
    tpr = np.array(tpr)[::-1]
    threshold = threshold[::-1]

    # Compute Area Under Curve according to trapezoidal rule, using :func:`trapz` in :mod:`scipy.integrate`.
    auc = trapz(tpr, fpr)

    if auc_only is True:
        return auc
    else:
        return fpr, tpr, auc, threshold


def get_roc_curve(y_proba, y_true, true_class=1, label=[" "]):
    """Get ROC pr given predicted probability scores of classes and true classes for samples."""
    fpr, tpr, auc, threshold = roc(y_proba, y_true, true_class=true_class)
    roc_plot(fpr, tpr, auc, labels=label)

