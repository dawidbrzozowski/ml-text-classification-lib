import json


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def write_json_file(out_path, data):
    with open(out_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
