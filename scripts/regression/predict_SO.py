import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from utils.old import generators_training as gbc
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    path_model = "/lfstev/deepskies/luisals/regression/test_spherical_overden/standardize/scale_34/"
    ph = "/lfstev/deepskies/luisals/"
    model_file = path_model + "/model/weights.25.hdf5"
    index_scale = 34
    batch_size = 80

    rescale_mean = 1.004
    rescale_std = 0.05

    model = load_model(model_file)
    f = "random_training_set.txt"

    ids_0, so_0 = gbc.get_ids_and_SO_labels(sim="0", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
    ids_3, so_3 = gbc.get_ids_and_SO_labels(sim="3", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
    ids_4, so_4 = gbc.get_ids_and_SO_labels(sim="4", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
    ids_5, so_5 = gbc.get_ids_and_SO_labels(sim="5", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)

    ran = np.load(path_model + "ran.npy")
    output_ids, output_scaler = gbc.get_standard_scaler_and_transform([so_0[ran], so_3[ran], so_4[ran], so_5[ran]])
    ids_1, so_1 = gbc.get_ids_and_SO_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler, index=index_scale)
    generator_1 = gbc.create_generator_sim(ids_1, so_1, batch_size=batch_size,
                                           rescale_mean=rescale_mean, rescale_std=rescale_std,
                                           path=ph + "reseed1_simulation/training_set/")
    ids_3, so_3 = gbc.get_ids_and_SO_labels(sim="3", ids_filename=f, fitted_scaler=output_scaler, index=index_scale)
    generator_3 = gbc.create_generator_sim(ids_3, so_3, batch_size=batch_size,
                                           rescale_mean=rescale_mean, rescale_std=rescale_std,
                                           path=ph + "reseed3_simulation/training_set/")

    pred_1 = model.predict_generator(generator_1, verbose=1)
    so_pred1 = output_scaler.inverse_transform(pred_1).flatten()
    np.save(path_model + "/predictions/pred1_25.npy", so_pred1)
    np.save(path_model + "/predictions/truth1.npy", output_scaler.inverse_transform(so_1).flatten())

    pred_3 = model.predict_generator(generator_3, verbose=1)
    so_pred3 = output_scaler.inverse_transform(pred_3).flatten()
    np.save(path_model + "/predictions/pred3_25.npy", so_pred3)
    np.save(path_model + "/predictions/truth3.npy", output_scaler.inverse_transform(so_3).flatten())

    # pred_1 = model.predict_generator(generator_1, verbose=1)
    # so_pred1 = so_scaler.inverse_transform(pred_1).flatten()
    # np.save(path_model + "/predictions/pred1_80.npy", so_pred1)
    # np.save(path_model + "/predictions/truth1.npy", so_scaler.inverse_transform(so_1.reshape(-1,1)).flatten())
    #
    # pred_2 = model.predict_generator(generator_2, verbose=1)
    # so_pred2 = so_scaler.inverse_transform(pred_2).flatten()
    # np.save(path_model + "/predictions/pred2_80.npy", so_pred2)
    # np.save(path_model + "/predictions/truth2.npy", so_scaler.inverse_transform(so_2.reshape(-1,1)).flatten())
