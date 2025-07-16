from model.SAGE_batch import SAGEBATCH

def init_model(model_name, feature_channel, num_class, hidden_channel):
    model = None
    if model_name == "sage_batch":
        model = SAGEBATCH(in_channels = feature_channel, out_channels = num_class, hidden = hidden_channel)
    else:
        print('Model is not supported')

    return model