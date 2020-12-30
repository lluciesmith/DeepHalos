from dlhalos_code import loss_functions as lf
from dlhalos_code import CNN
import numpy as np
import dlhalos_code.data_processing as tn
from pickle import load
from dlhalos_code import custom_regularizers as reg


def perturbed_model(model, parameters, training_generator):
    w = model.get_weights()
    alpha, beta = parameters
    d0 = create_random_direction(w)
    d1 = create_random_direction(w)
    L = []
    dx = []
    dy = []
    for a in alpha:
        for b in beta:
            print(b)
            w2 = w + a*np.array(d0) + b*np.array(d1)
            model.set_weights(w2)
            dx.append(a)
            dy.append(b)
            L.append(loss(model, training_generator))
    return L, (dx, dy)


def new_weights(model, parameters):
    w = model.get_weights()
    alpha, beta = parameters
    d0 = create_random_direction(w)
    d1 = create_random_direction(w)
    for a in alpha:
        for b in beta:
            print(b)
            w2 = np.array(w) + a * np.array(d0) + b * np.array(d1)
    return w2


def loss(model, training_generator):
    return model.evaluate(training_generator, verbose=1, steps=100)


def loss2(y_true, model, training_generator):
    y_predicted = model.predict_generator(training_generator)
    return lf.cauchy_selection_loss_fixed_boundary(gamma=0.2, y_max=1, y_min=-1)(y_true, y_predicted)


def normalize_direction(direction, weights):
    count = 1
    for d, w in zip(direction, weights):
        print(count, end="\t")
        if len(w.shape) == 1:
            # print("Ignore biases")
            d *= 0
        elif len(w.shape) == 2:
            # for num_neuron in range(len(w)):
            #     w_norm = np.linalg.norm(w[num_neuron, :])
            #     d_norm = np.linalg.norm(d[num_neuron, :])
            #     d[num_neuron, :] *= w_norm / (d_norm + 1e-10)
            d *= 0
        else:
            for num_kernel in range(len(w)):
                w_norm = np.linalg.norm(w[:, :, :, 0, num_kernel])
                #print(w_norm)
                d_norm = np.linalg.norm(d[:, :, :, 0, num_kernel])
                #print(d_norm)
                d[:, :, :, 0, num_kernel] *= w_norm/(d_norm + 1e-10)
        count += 1


def get_weights(model):
    return model.get_weights()


def get_random_weights(weights):
    return [np.random.normal(0, 1, weights[i].shape) for i in range(len(weights))]


def create_random_direction(weights):
    direction = get_random_weights(weights)
    print(direction[0][:, :, :, 0, 0])
    normalize_direction(direction, weights)
    print(direction[0][:, :, :, 0, 0])
    return direction

if __name__ == "__main__":
    path = "/mnt/beegfs/work/ati/pearl037/regression/test/fermi_training/"
    path_sims = "/mnt/beegfs/work/ati/pearl037/"

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims, path="/mnt/beegfs/work/ati/pearl037/")

    p_ids = load(open(path + 'training_set.pkl', 'rb'))
    l_ids = load(open(path + 'labels_training_set.pkl', 'rb'))
    v_ids = load(open(path + 'validation_set.pkl', 'rb'))
    l_v_ids = load(open(path + 'labels_validation_set.pkl', 'rb'))

    dim = (51, 51, 51)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(p_ids, l_ids, s.sims_dic, shuffle=False, **params_tr)
    generator_validation = tn.DataGenerator(v_ids, l_v_ids, s.sims_dic, shuffle=False, **params_tr)

    # Convolutional layers parameters

    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(10**-3.5)}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l2_norm(10**-3.5)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}
    reg_params = {'init_gamma': 0.2}


    # Define model

    path1 = path + "run_2_fermi/"
    weights = path1 + "model/weights.10.h5"
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                          validation_generator=generator_validation, num_epochs=20, dim=generator_training.dim,
                          max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001,
                          save_summary=False, path_summary=path1, validation_freq=1, train=False, compile=True)

    # Get loss landscape

    alpha = np.linspace(-0.001, 0.001, num=5, endpoint=True)
    beta = np.linspace(-0.001, 0.001, num=5, endpoint=True)
    params = (alpha, beta)

    y_true = np.array([generator_training.labels[ID] for ID in generator_training.list_IDs]).reshape(-1, 1)

    l, (x, y) = perturbed_model(Model.model, params, generator_training)
