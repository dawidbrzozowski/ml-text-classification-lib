import json
import pickle
import numpy as np


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def write_json_file(out_path, data):
    with open(out_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def write_pickle(out_path, obj):
    pickle.dump(obj, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    return pickle.load(open(path, 'rb'))


def write_numpy(path, np_obj):
    np.save(path, np_obj)


def read_numpy(path):
    return np.load(path)
