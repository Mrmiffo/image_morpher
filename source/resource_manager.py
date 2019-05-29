import os
from keras.models import save_model, load_model

def save(model, name, path="models/", overwrite=True):
    if (".h5" not in name):
        name = name + ".h5"
    save_model(model, path+name, overwrite=overwrite)

def load(path):
    if (".h5" not in path):
        path = path + ".h5"
    return load_model(path)