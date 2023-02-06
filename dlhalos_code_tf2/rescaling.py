import sklearn


def rescale_output(output_array, scaler_type, output_range, scaler=None):
    if scaler is None:
        print("Fitting and transforming")
        output_ids, scaler = fit_scaler_and_transform(output_array, scaler_type, output_range)
    else:
        print("Transforming only")
        output_ids = transform_array_given_scaler(scaler, output_array)
    return output_ids, scaler


def fit_scaler_and_transform(array_outputs, scaler_type, output_range):
    """ This rescaling is for 1D arrays """
    assert array_outputs.ndim == 1, "The array you would like to scale has shape " + str(array_outputs.shape) + \
                                    " and is " + str(array_outputs)

    # maybe can change below scalers to use tf
    if scaler_type == "standard":
        norm_scaler = sklearn.preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        norm_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=output_range)
    elif scaler_type == "robust":
        norm_scaler = sklearn.preprocessing.RobustScaler()
    else:
        raise NameError("Choose between 'standard' and 'minmax' scalers")

    print(norm_scaler)
    norm_scaler.fit(array_outputs.reshape(-1, 1))
    rescaled_out = transform_array_given_scaler(norm_scaler, array_outputs)
    return rescaled_out, norm_scaler


def transform_array_given_scaler(scaler, array):
    """ This rescaling is for 1D arrays """
    assert array.ndim == 1
    scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
    return scaled_array